#include "clothManager.h"
#include "LevelSet3D.h"
#include "clothPiece.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include "PROGRESSING_BAR.h"
#include "Renderable\ObjMesh.h"
#define ENABLE_DEBUG_DUMPING

namespace ldp
{
	PROGRESSING_BAR g_debug_save_bar(0);
	template<class T> static void debug_save_gpu_array(DeviceArray<T> D, std::string filename)
	{
		std::vector<T> H;
		D.download(H);
		std::ofstream stm(filename);
		if (stm.fail())
			throw std::exception(("IOError: " + filename).c_str());
		for (const auto& v : H)
		{
			stm << v << std::endl;
			g_debug_save_bar.Add();
		}
		stm.close();
	}

	ClothManager::ClothManager()
	{
		m_bodyMesh.reset(new ObjMesh);
		m_bodyLvSet.reset(new LevelSet3D);
		m_simulationMode = SimulationNotInit;
		m_fps = 0.f;
		m_avgArea = 0;
	}

	ClothManager::~ClothManager()
	{
		clear();
	}

	void ClothManager::clear()
	{
		simulationDestroy();

		m_clothVertBegin.clear();		
		m_X.clear();
		m_V.clear();
		m_T.clear();
		m_allE.clear();
		m_allVV.clear();
		m_allVL.clear();
		m_allVW.clear();
		m_allVC.clear();
		m_allVV_num.clear();
		m_fixed.clear();
		m_edgeWithBendEdge.clear();

		m_bodyMesh->clear();
		m_clothPieces.clear();
		m_bodyLvSet->clear();

		m_fps = 0.f;
		m_avgArea = 0;
	}

	void ClothManager::dragBegin(DragInfo info)
	{
		if (m_simulationMode != SimulationOn)
			return;
		// convert drag_info to global index
		m_curDragInfo.vert_id = -1;
		m_curDragInfo.dir = 0;
		m_curDragInfo.target = 0;
		for (size_t iCloth = 0; iCloth < m_clothPieces.size(); iCloth++)
		{
			if (info.selected_cloth == &m_clothPieces[iCloth]->mesh3d())
			{
				m_curDragInfo.vert_id = info.selected_vert_id + m_clothVertBegin[iCloth];
				m_curDragInfo.target = info.target;
				break;
			}
		} // end for iCloth
		resetMoreFixed();
	}

	void ClothManager::dragMove(ldp::Float3 target)
	{
		if (m_simulationMode != SimulationOn)
			return;
		m_curDragInfo.target = target;
	}

	void ClothManager::dragEnd()
	{
		if (m_simulationMode != SimulationOn)
			return;
		m_curDragInfo.vert_id = -1;
		m_curDragInfo.dir = 0;
		m_curDragInfo.target = 0;
		resetMoreFixed();
	}

	void ClothManager::simulationInit()
	{
		if (m_clothPieces.size() == 0)
			return;
		for (size_t i = 0; i < m_clothPieces.size(); i++)
			m_clothPieces[i]->mesh3d().cloneFrom(&m_clothPieces[i]->mesh3dInit());
		buildTopology();
		allocateGpuMemory();
		buildNumerical();
		copyToGpuMatrix();
		m_simulationMode = SimulationPause;
	}

	void ClothManager::simulationUpdate()
	{
		if (m_simulationMode != SimulationOn)
			return;
		if (m_X.size() == 0)
			return;

		gtime_t t_begin = ldp::gtime_now();

		for (int oiter = 0; oiter < m_simulationParam.out_iter; oiter++)
		{
			// 1. process dragging info
			if (m_curDragInfo.vert_id != -1)
			{
				m_curDragInfo.dir = m_curDragInfo.target - m_X[m_curDragInfo.vert_id];
				float dir_length = m_curDragInfo.dir.length();
				m_curDragInfo.dir.normalizeLocal();
				if (dir_length>0.1)	dir_length = 0.1;
				m_curDragInfo.dir *= dir_length;
			}

			// backup
			m_dev_X.copyTo(m_dev_old_X);

			// 2. laplacian damping, considering the air damping
			laplaceDamping();
			updateAfterLap();
			constrain0();

			//cudaThreadSynchronize();
			//ldp::tic();

			// 3. perform inner loops
			ValueType omega = 0;
			for (int iter = 0; iter<m_simulationParam.inner_iter; iter++)
			{
				constrain1();

				// chebshev param
				if (iter <= 5)		
					omega = 1;
				else if (iter == 6)	
					omega = 2 / (2 - ldp::sqr(m_simulationParam.rho));
				else			
					omega = 4 / (4 - ldp::sqr(m_simulationParam.rho)*omega);

				constrain2(omega);

				m_dev_X.swap(m_dev_prev_X);
				m_dev_X.swap(m_dev_next_X);
			} // end for iter

			//cudaThreadSynchronize();
			//ldp::toc();

			constrain3();

#ifdef ENABLE_SELF_COLLISION
			collider.Run(dev_old_X, dev_X, dev_V, number, dev_T, t_number, X, 1/t);
#endif

			constrain4();
			m_dev_X.download((ValueType*)m_X.data());
		} // end for oiter

		// finally we update normals and bounding boxes
		for (size_t iCloth = 0; iCloth < m_clothPieces.size(); iCloth++)
		{
			int vb = m_clothVertBegin[iCloth];
			auto& mesh = m_clothPieces[iCloth]->mesh3d();
			mesh.vertex_list.assign(m_X.begin() + vb, m_X.begin() + vb + mesh.vertex_list.size());
			mesh.updateNormals();
			mesh.updateBoundingBox();
		} // end for iCloth

		gtime_t t_end = ldp::gtime_now();
		m_fps = 1 / ldp::gtime_seconds(t_begin, t_end);
	}

	void ClothManager::simulationDestroy()
	{
		m_simulationMode = SimulationNotInit;
		updateInitialClothsToCurrent();
		//releaseGpuMemory();
	}

	void ClothManager::setSimulationMode(SimulationMode mode)
	{
		m_simulationMode = mode;
	}

	void ClothManager::setSimulationParam(SimulationParam param)
	{
		auto lastParam = m_simulationParam;
		m_simulationParam = param;
		m_simulationParam.spring_k = m_simulationParam.spring_k_raw / m_avgArea;
#if 0
		printf("simulaton param:\n");
		printf("\t rho          = %f\n", m_simulationParam.rho);
		printf("\t under_relax  = %f\n", m_simulationParam.under_relax);
		printf("\t lap_damping  = %d\n", m_simulationParam.lap_damping);
		printf("\t air_damping  = %f\n", m_simulationParam.air_damping);
		printf("\t bending_k    = %f\n", m_simulationParam.bending_k);
		printf("\t spring_k     = %f\n", m_simulationParam.spring_k);
		printf("\t spring_k_raw = %f\n", m_simulationParam.spring_k_raw);
		printf("\t out_iter     = %d\n", m_simulationParam.out_iter);
		printf("\t inner_iter   = %d\n", m_simulationParam.inner_iter);
		printf("\t time_step    = %f\n", m_simulationParam.time_step);
		printf("\t control_mag  = %f\n", m_simulationParam.control_mag);
		printf("\t gravity      = %f %f %f\n", m_simulationParam.gravity[0], 
			m_simulationParam.gravity[1], m_simulationParam.gravity[2]);
#endif
		
		bool matrixNumericUpdate = false;
		if (fabs(lastParam.spring_k - m_simulationParam.spring_k) >= std::numeric_limits<float>::epsilon())
			matrixNumericUpdate = true;
		if (fabs(lastParam.bending_k - m_simulationParam.bending_k) >= std::numeric_limits<float>::epsilon())
			matrixNumericUpdate = true;

		// some parameter changes will have effects on the precompuated matrix..
		if (matrixNumericUpdate)
		{
			buildNumerical();
			m_dev_all_VL.upload(m_allVL);
			m_dev_all_VW.upload(m_allVW);
			m_dev_all_VC.upload(m_allVC);
		} // end if spring_k updated
	}

	SimulationParam::SimulationParam()
	{
		setDefaultParam();
	}

	void SimulationParam::setDefaultParam()
	{
		rho = 0.996;
		under_relax = 0.5;
		lap_damping = 4;
		air_damping = 0.999;
		bending_k = 10;
		spring_k_raw = 1000;
		spring_k = 0;//will be updated after built topology
		out_iter = 8;
		inner_iter = 40;
		time_step = 1.0 / 240.0;
		control_mag = 400;
		gravity = ldp::Float3(0, 0, -9.8);
	}

	void ClothManager::updateCurrentClothsToInitial()
	{
		for (auto& cloth : m_clothPieces)
		{
			cloth->mesh3dInit().cloneFrom(&cloth->mesh3d());
		}
	}

	void ClothManager::updateInitialClothsToCurrent()
	{
		for (auto& cloth : m_clothPieces)
		{
			cloth->mesh3d().cloneFrom(&cloth->mesh3dInit());
		}
	}
	//////////////////////////////////////////////////////////////////////////////////
	void ClothManager::buildTopology()
	{
		// collect all cloth pieces
		m_X.clear();
		m_T.clear();
		m_clothVertBegin.resize(numClothPieces() + 1, 0);
		m_avgArea = 0;
		int fcnt = 0;
		for (int iCloth = 0; iCloth < numClothPieces(); iCloth++)
		{
			const int vid_s = m_clothVertBegin[iCloth];
			const auto& mesh = clothPiece(iCloth)->mesh3d();
			m_clothVertBegin[iCloth + 1] = vid_s + (int)mesh.vertex_list.size();
			for (const auto& v : mesh.vertex_list)
				m_X.push_back(v);
			for (const auto& f : mesh.face_list)
			{
				m_T.push_back(ldp::Int3(f.vertex_index[0], f.vertex_index[1], f.vertex_index[2]));
				ldp::Float3 v[3] = { mesh.vertex_list[f.vertex_index[0]], mesh.vertex_list[f.vertex_index[1]],
					mesh.vertex_list[f.vertex_index[2]] };
				m_avgArea += sqrt(Area_Squared(v[0].ptr(), v[1].ptr(), v[2].ptr()));
				fcnt++;
			}
		} // end for iCloth
		m_avgArea /= fcnt;

		m_simulationParam.spring_k = m_simulationParam.spring_k_raw / m_avgArea;

		m_V.resize(m_X.size());
		std::fill(m_V.begin(), m_V.end(), ValueType(0));
		m_fixed.resize(m_X.size());
		std::fill(m_fixed.begin(), m_fixed.end(), ValueType(0));

		// build connectivity
		ldp::BMesh bmesh;
		bmesh.init_triangles((int)m_X.size(), m_X.data()->ptr(), (int)m_T.size(), m_T.data()->ptr());

		// set up all edges + bending edges
		m_edgeWithBendEdge.clear();
		m_allE.clear();
		BMESH_ALL_EDGES(e, eiter, bmesh)
		{
			ldp::Int2 vori(bmesh.vofe_first(e)->getIndex(),
				bmesh.vofe_last(e)->getIndex());
			m_allE.push_back(vori);
			m_allE.push_back(ldp::Int2(vori[1], vori[0]));
			if (bmesh.fofe_count(e) > 2)
				throw std::exception("error: non-manifold mesh found!");
			ldp::Int2 vbend = -1;
			int vcnt = 0;
			BMESH_F_OF_E(f, e, fiter, bmesh)
			{
				int vsum = 0;
				BMESH_V_OF_F(v, f, viter, bmesh)
				{
					vsum += v->getIndex();
				}
				vsum -= vori[0] + vori[1];
				vbend[vcnt++] = vsum;
			} // end for fiter
			if (vbend[0] >= 0 && vbend[1] >= 0)
			{
				m_allE.push_back(vbend);
				m_allE.push_back(ldp::Int2(vbend[1], vbend[0]));
			}
			m_edgeWithBendEdge.push_back(ldp::Int4(vori[0], vori[1], vbend[0], vbend[1]));
		} // end for all edges
		std::sort(m_allE.begin(), m_allE.end());
		m_allE.resize(std::unique(m_allE.begin(), m_allE.end()) - m_allE.begin());

		// setup one-ring vertex info
		size_t eIdx = 0;
		m_allVV_num.reserve(m_X.size()+1);
		for (size_t i = 0; i<m_X.size(); i++)
		{
			m_allVV_num.push_back(m_allVV.size());
			for (; eIdx<m_allE.size(); eIdx++)
			{
				const auto& e = m_allE[eIdx];
				if (e[0] != i)						
					break;		// not in the right vertex
				if (eIdx != 0 && e[1] == e[0])	
					continue;	// duplicate
				m_allVV.push_back(e[1]);
			} 
		} // end for i
		m_allVV_num.push_back(m_allVV.size());
	}

	void ClothManager::buildNumerical()
	{
		// compute matrix related values
		m_allVL.resize(m_allVV.size());
		m_allVW.resize(m_allVV.size());
		m_allVC.resize(m_X.size());
		std::fill(m_allVL.begin(), m_allVL.end(), ValueType(-1));
		std::fill(m_allVW.begin(), m_allVW.end(), ValueType(0));
		std::fill(m_allVC.begin(), m_allVC.end(), ValueType(0));
		for (size_t iv = 0; iv < m_edgeWithBendEdge.size(); iv++)
		{
			const auto& v = m_edgeWithBendEdge[iv];

			// first, handle spring length			
			ValueType l = (m_X[v[0]] - m_X[v[1]]).length();
			m_allVL[findNeighbor(v[0], v[1])] = l;
			m_allVL[findNeighbor(v[1], v[0])] = l;
			m_allVC[v[0]] += m_simulationParam.spring_k;
			m_allVC[v[1]] += m_simulationParam.spring_k;
			m_allVW[findNeighbor(v[0], v[1])] -= m_simulationParam.spring_k;
			m_allVW[findNeighbor(v[1], v[0])] -= m_simulationParam.spring_k;

			// ignore boundary edges for bending
			if (v[2] == -1 || v[3] == -1)
				continue;

			// second, handle bending weights
			ValueType c01 = Cotangent(m_X[v[0]].ptr(), m_X[v[1]].ptr(), m_X[v[2]].ptr());
			ValueType c02 = Cotangent(m_X[v[0]].ptr(), m_X[v[1]].ptr(), m_X[v[3]].ptr());
			ValueType c03 = Cotangent(m_X[v[1]].ptr(), m_X[v[0]].ptr(), m_X[v[2]].ptr());
			ValueType c04 = Cotangent(m_X[v[1]].ptr(), m_X[v[0]].ptr(), m_X[v[3]].ptr());
			ValueType area0 = sqrt(Area_Squared(m_X[v[0]].ptr(), m_X[v[1]].ptr(), m_X[v[2]].ptr()));
			ValueType area1 = sqrt(Area_Squared(m_X[v[0]].ptr(), m_X[v[1]].ptr(), m_X[v[3]].ptr()));
			ValueType weight = 1 / (area0 + area1);
			ValueType k[4];
			k[0] = c03 + c04;
			k[1] = c01 + c02;
			k[2] = -c01 - c03;
			k[3] = -c02 - c04;

			for (int i = 0; i<4; i++)
			for (int j = 0; j<4; j++)
			{
				if (i == j)
					m_allVC[v[i]] += k[i] * k[j] * m_simulationParam.bending_k*weight;
				else
					m_allVW[findNeighbor(v[i], v[j])] += k[i] * k[j] * m_simulationParam.bending_k*weight;
			}
		} // end for all edges
	}

	int ClothManager::findNeighbor(int i, int j)const
	{
		for (int index = m_allVV_num[i]; index<m_allVV_num[i + 1]; index++)
		if (m_allVV[index] == j)	
			return index;
		printf("ERROR: failed to find the neighbor in all_VV.\n"); getchar();
		return -1;
	}

	void ClothManager::allocateGpuMemory()
	{
		const int nverts = (int)m_X.size();
		const int ntris = (int)m_T.size();
		const int nvv = m_allVV_num.back();
		m_dev_X.create(nverts * 3);
		m_dev_old_X.create(nverts * 3);
		m_dev_next_X.create(nverts * 3);
		m_dev_prev_X.create(nverts * 3);
		m_dev_fixed.create(nverts);
		m_dev_more_fixed.create(nverts);
		m_dev_V.create(nverts * 3);
		m_dev_init_B.create(nverts * 3);
		m_dev_T.create(ntris * 3);
		m_dev_all_VV.create(nvv);
		m_dev_all_vv_num.create(nverts + 1);
		m_dev_all_VL.create(nvv);
		m_dev_all_VW.create(nvv);
		m_dev_all_VC.create(nverts);
		m_dev_new_VC.create(nverts);
		m_dev_phi.create(m_bodyLvSet->sizeXYZ());
	}

	void ClothManager::releaseGpuMemory()
	{
		m_dev_X.release();			
		m_dev_old_X.release();
		m_dev_next_X.release();
		m_dev_prev_X.release();
		m_dev_fixed.release();
		m_dev_more_fixed.release();
		m_dev_V.release();
		m_dev_init_B.release();
		m_dev_T.release();
		m_dev_all_VV.release();
		m_dev_all_vv_num.release();
		m_dev_all_VL.release();
		m_dev_all_VW.release();
		m_dev_all_VC.release();
		m_dev_new_VC.release();
		m_dev_phi.release();
	}

	void ClothManager::copyToGpuMatrix()
	{
		m_dev_X.upload((const ValueType*)m_X.data(), m_X.size() * 3);
		m_dev_old_X.upload((const ValueType*)m_X.data(), m_X.size() * 3);
		m_dev_next_X.upload((const ValueType*)m_X.data(), m_X.size() * 3);
		m_dev_prev_X.upload((const ValueType*)m_X.data(), m_X.size() * 3);
		m_dev_fixed.upload(m_fixed);
		cudaMemset(m_dev_more_fixed, 0, m_dev_fixed.sizeBytes());
		m_dev_V.upload((const ValueType*)m_V.data(), m_V.size() * 3);
		cudaMemset(m_dev_init_B.ptr(), 0, m_dev_init_B.sizeBytes());
		m_dev_T.upload((const int*)m_T.data(), m_T.size() * 3);
		m_dev_all_VV.upload(m_allVV);
		m_dev_all_vv_num.upload(m_allVV_num);
		m_dev_all_VL.upload(m_allVL);
		m_dev_all_VW.upload(m_allVW);
		m_dev_all_VC.upload(m_allVC);
		cudaMemset(m_dev_new_VC.ptr(), 0, m_dev_new_VC.sizeBytes());
		m_dev_phi.upload(m_bodyLvSet->value(), m_bodyLvSet->sizeXYZ());
	}

	void ClothManager::debug_save_values()
	{
#ifdef ENABLE_DEBUG_DUMPING
		static int a = 0;
		if (a != 0)
			return;
		a++;
		printf("begin debug saving all variables..\n");
		g_debug_save_bar.sample_number = m_dev_X.size() + m_dev_next_X.size()
			+ m_dev_prev_X.size() + m_dev_V.size()
			+ m_dev_init_B.size() + m_dev_T.size() + m_dev_all_VV.size()
			+ m_dev_all_VC.size() + m_dev_all_VW.size() + m_dev_all_VL.size()
			+ m_dev_new_VC.size() + m_dev_all_vv_num.size();// +m_dev_phi.size();
		g_debug_save_bar.Start();
		debug_save_gpu_array(m_dev_X, "tmp/X.txt");
		debug_save_gpu_array(m_dev_next_X, "tmp/next_X.txt");
		debug_save_gpu_array(m_dev_prev_X, "tmp/prev_X.txt");
		debug_save_gpu_array(m_dev_fixed, "tmp/fixed.txt");
		debug_save_gpu_array(m_dev_more_fixed, "tmp/more_fixed.txt");
		debug_save_gpu_array(m_dev_V, "tmp/V.txt");
		debug_save_gpu_array(m_dev_init_B, "tmp/init_B.txt");
		debug_save_gpu_array(m_dev_T, "tmp/fixed_T.txt");
		debug_save_gpu_array(m_dev_all_VV, "tmp/all_VV.txt");
		debug_save_gpu_array(m_dev_all_VC, "tmp/all_VC.txt");
		debug_save_gpu_array(m_dev_all_VW, "tmp/all_VW.txt");
		debug_save_gpu_array(m_dev_all_VL, "tmp/all_VL.txt");
		debug_save_gpu_array(m_dev_new_VC, "tmp/new_VC.txt");
		debug_save_gpu_array(m_dev_all_vv_num, "tmp/all_vv_num.txt");
		//debug_save_gpu_array(m_dev_phi, "tmp/phi.txt");
		g_debug_save_bar.End();
#endif
	}
}