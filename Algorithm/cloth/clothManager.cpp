#include "clothManager.h"
#include "LevelSet3D.h"
#include "clothPiece.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include "PROGRESSING_BAR.h"
#include "Renderable\ObjMesh.h"
#include <eigen\Dense>
#include <eigen\Sparse>
#include "svgpp\SvgManager.h"
#include "svgpp\SvgPolyPath.h"
#include "ldputil.h"
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
		m_fps = 0;
		m_avgArea = 0;
		m_avgEdgeLength = 0;
		m_curStitchRatio = 0;
		m_shouldMergePieces = false;
		m_shouldTopologyUpdate = false;
		m_shouldNumericUpdate = false;
		m_shouldStitchUpdate = false;
	}

	ClothManager::~ClothManager()
	{
		clear();
	}

	void ClothManager::clear()
	{
		simulationDestroy();

		m_V.clear();
		m_allE.clear();
		m_allVV.clear();
		m_allVL.clear();
		m_allVW.clear();
		m_allVC.clear();
		m_allVV_num.clear();
		m_fixed.clear();
		m_edgeWithBendEdge.clear();

		m_bodyMesh->clear();
		m_bodyLvSet->clear();

		m_fps = 0;

		clearSewings();
		clearClothPieces();
	}

	void ClothManager::dragBegin(DragInfo info)
	{
		if (m_simulationMode != SimulationOn)
			return;
		// convert drag_info to global index
		m_curDragInfo.vert_id = -1;
		m_curDragInfo.dir = 0;
		m_curDragInfo.target = 0;
		if (m_clothVertBegin.find(info.selected_cloth) != m_clothVertBegin.end())
		{
			m_curDragInfo.vert_id = info.selected_vert_id + m_clothVertBegin.at(info.selected_cloth);
			m_curDragInfo.target = info.target;
		}
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
		m_dev_phi.upload(m_bodyLvSet->value(), m_bodyLvSet->sizeXYZ());
		m_shouldMergePieces = true;
		m_curStitchRatio = 1;
		mergePieces();
		buildTopology();
		buildNumerical();
		buildStitch();
		m_simulationMode = SimulationPause;
	}

	void ClothManager::simulationUpdate()
	{
		if (m_simulationMode != SimulationOn)
			return;
		if (m_X.size() == 0)
			return;

		if (m_shouldMergePieces)
			mergePieces();
		if (m_shouldTopologyUpdate)
			buildTopology();
		if (m_shouldNumericUpdate)
			buildNumerical();
		if (m_shouldStitchUpdate)
			buildStitch();

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

			m_curStitchRatio = std::max(0.f, m_curStitchRatio - 
				m_simulationParam.stitch_ratio * m_simulationParam.time_step);

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
			auto& mesh = m_clothPieces[iCloth]->mesh3d();
			int vb = m_clothVertBegin.at(&mesh);
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
		m_simulationParam.stitch_k = m_simulationParam.stitch_k_raw / m_avgArea;
#if 0
		printf("simulaton param:\n");
		printf("\t rho          = %f\n", m_simulationParam.rho);
		printf("\t under_relax  = %f\n", m_simulationParam.under_relax);
		printf("\t lap_damping  = %d\n", m_simulationParam.lap_damping);
		printf("\t air_damping  = %f\n", m_simulationParam.air_damping);
		printf("\t bending_k    = %f\n", m_simulationParam.bending_k);
		printf("\t spring_k     = %f\n", m_simulationParam.spring_k);
		printf("\t spring_k_raw = %f\n", m_simulationParam.spring_k_raw);
		printf("\t stitch_k     = %f\n", m_simulationParam.stitch_k);
		printf("\t stitch_k_raw = %f\n", m_simulationParam.stitch_k_raw);
		printf("\t out_iter     = %d\n", m_simulationParam.out_iter);
		printf("\t inner_iter   = %d\n", m_simulationParam.inner_iter);
		printf("\t time_step    = %f\n", m_simulationParam.time_step);
		printf("\t control_mag  = %f\n", m_simulationParam.control_mag);
		printf("\t gravity      = %f %f %f\n", m_simulationParam.gravity[0], 
			m_simulationParam.gravity[1], m_simulationParam.gravity[2]);
#endif
		
		if (fabs(lastParam.spring_k - m_simulationParam.spring_k) >= std::numeric_limits<float>::epsilon())
			m_shouldNumericUpdate = true;
		if (fabs(lastParam.bending_k - m_simulationParam.bending_k) >= std::numeric_limits<float>::epsilon())
			m_shouldNumericUpdate = true;
		if (fabs(lastParam.stitch_k - m_simulationParam.stitch_k) >= std::numeric_limits<float>::epsilon())
			m_shouldStitchUpdate = true;
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
	void ClothManager::clearClothPieces() 
	{
		m_bmesh.reset((BMesh*)nullptr);
		m_bmeshVerts.clear();
		m_clothPieces.clear();
		m_clothVertBegin.clear();
		m_X.clear();
		m_T.clear();
		m_avgArea = 0;
		m_avgEdgeLength = 0;
		m_shouldMergePieces = false;
	}

	void ClothManager::addClothPiece(std::shared_ptr<ClothPiece> piece) 
	{ 
		m_clothPieces.push_back(piece); 
		m_shouldMergePieces = true;
	}

	void ClothManager::removeClothPiece(int i) 
	{ 
		m_clothPieces.erase(m_clothPieces.begin() + i);
		m_shouldMergePieces = true;
	}

	void ClothManager::mergePieces()
	{
		if (!m_shouldMergePieces)
			return;
		m_X.clear();
		m_T.clear();
		m_clothVertBegin.clear();
		m_avgArea = 0;
		m_avgEdgeLength = 0;
		int fcnt = 0;
		int ecnt = 0;
		int vid_s = 0;
		for (int iCloth = 0; iCloth < numClothPieces(); iCloth++)
		{
			const auto& mesh = clothPiece(iCloth)->mesh3d();
			m_clothVertBegin.insert(std::make_pair(&mesh, vid_s));
			for (const auto& v : mesh.vertex_list)
				m_X.push_back(v);
			for (const auto& f : mesh.face_list)
			{
				m_T.push_back(ldp::Int3(f.vertex_index[0], f.vertex_index[1],
					f.vertex_index[2]) + vid_s);
				ldp::Float3 v[3] = { mesh.vertex_list[f.vertex_index[0]], mesh.vertex_list[f.vertex_index[1]],
					mesh.vertex_list[f.vertex_index[2]] };
				m_avgArea += sqrt(Area_Squared(v[0].ptr(), v[1].ptr(), v[2].ptr()));
				fcnt++;

				m_avgEdgeLength += (v[0] - v[1]).length() + (v[0] - v[2]).length() + (v[1] - v[2]).length();
				ecnt += 3;
			}
			vid_s += mesh.vertex_list.size();
		} // end for iCloth
		m_avgArea /= fcnt;
		m_avgEdgeLength /= ecnt;

		// build connectivity
		m_bmesh.reset(new ldp::BMesh);
		m_bmesh->init_triangles((int)m_X.size(), m_X.data()->ptr(), (int)m_T.size(), m_T.data()->ptr());
		m_bmeshVerts.clear();
		BMESH_ALL_VERTS(v, viter, *m_bmesh)
		{
			m_bmeshVerts.push_back(v);
		}

		m_shouldMergePieces = false;
		m_shouldTopologyUpdate = true;
	}

	void ClothManager::clearSewings()
	{
		m_sewings.clear();
		m_stitches.clear();
		m_stitchVV.clear();
		m_stitchVV_num.clear();
		m_stitchVC.clear();
		m_stitchVW.clear();
		m_stitchVL.clear();
	}

	void ClothManager::addStitchVert(const ClothPiece* cloth1, int mesh_vid1, const ClothPiece* cloth2, int mesh_vid2)
	{
		if (m_shouldMergePieces)
			mergePieces();

		Int2 vid;

		// convert from mesh id to global id
		vid[0] = mesh_vid1 + m_clothVertBegin.at(&cloth1->mesh3d());
		vid[1] = mesh_vid2 + m_clothVertBegin.at(&cloth2->mesh3d());

		m_stitches.push_back(vid);

		m_shouldStitchUpdate = true;
	}

	std::pair<Float3, Float3> ClothManager::getStitchPos(int i)const
	{
		const auto& stp = m_stitches.at(i);

		std::pair<Float3, Float3> vp;
		vp.first = m_X[stp[0]];
		vp.second = m_X[stp[1]];
		return vp;
	}

	void ClothManager::buildStitch()
	{
		if (m_shouldNumericUpdate)
			buildNumerical();
		m_simulationParam.stitch_k = m_simulationParam.stitch_k_raw / m_avgArea;

		// build edges
		auto edges = m_stitches;
		for (const auto& s : m_stitches)
			edges.push_back(Int2(s[1], s[0]));
		std::sort(edges.begin(), edges.end());
		edges.resize(std::unique(edges.begin(), edges.end()) - edges.begin());

		// setup one-ring vertex info
		size_t eIdx = 0;
		m_stitchVV_num.clear();
		m_stitchVV_num.reserve(m_X.size() + 1);
		m_stitchVV.clear();
		for (size_t i = 0; i<m_X.size(); i++)
		{
			m_stitchVV_num.push_back(m_stitchVV.size());
			for (; eIdx<edges.size(); eIdx++)
			{
				const auto& e = edges[eIdx];
				if (e[0] != i)
					break;		// not in the right vertex
				if (eIdx != 0 && e[1] == e[0])
					continue;	// duplicate
				m_stitchVV.push_back(e[1]);
			}
		} // end for i
		m_stitchVV_num.push_back(m_stitchVV.size());

		// compute matrix related values
		m_stitchVL.resize(m_stitchVV.size());
		m_stitchVW.resize(m_stitchVV.size());
		m_stitchVC.resize(m_X.size());
		std::fill(m_stitchVL.begin(), m_stitchVL.end(), ValueType(-1));
		std::fill(m_stitchVW.begin(), m_stitchVW.end(), ValueType(0));
		std::fill(m_stitchVC.begin(), m_stitchVC.end(), ValueType(0));
		for (size_t iv = 0; iv < m_stitches.size(); iv++)
		{
			const auto& v = m_stitches[iv];

			// first, handle spring length			
			ValueType l = (m_X[v[0]] - m_X[v[1]]).length();
			m_stitchVL[findStitchNeighbor(v[0], v[1])] = l;
			m_stitchVL[findStitchNeighbor(v[1], v[0])] = l;
			m_stitchVC[v[0]] += m_simulationParam.stitch_k;
			m_stitchVC[v[1]] += m_simulationParam.stitch_k;
			m_stitchVW[findStitchNeighbor(v[0], v[1])] -= m_simulationParam.stitch_k;
			m_stitchVW[findStitchNeighbor(v[1], v[0])] -= m_simulationParam.stitch_k;
		} // end for all edges

		// copy to GPU
		m_dev_stitch_VV.upload(m_stitchVV);
		m_dev_stitch_VV_num.upload(m_stitchVV_num);
		m_dev_stitch_VC.upload(m_stitchVC);
		m_dev_stitch_VW.upload(m_stitchVW);
		m_dev_stitch_VL.upload(m_stitchVL);
		m_shouldStitchUpdate = false;
	}

	//////////////////////////////////////////////////////////////////////////////////
	void ClothManager::buildTopology()
	{
		if (m_shouldMergePieces)
			mergePieces();
		m_V.resize(m_X.size());
		std::fill(m_V.begin(), m_V.end(), ValueType(0));
		m_fixed.resize(m_X.size());
		std::fill(m_fixed.begin(), m_fixed.end(), ValueType(0));

		// set up all edges + bending edges
		m_edgeWithBendEdge.clear();
		m_allE.clear();
		BMESH_ALL_EDGES(e, eiter, *m_bmesh)
		{
			ldp::Int2 vori(m_bmesh->vofe_first(e)->getIndex(),
				m_bmesh->vofe_last(e)->getIndex());
			m_allE.push_back(vori);
			m_allE.push_back(Int2(vori[1], vori[0]));
			if (m_bmesh->fofe_count(e) > 2)
				throw std::exception("error: non-manifold mesh found!");
			Int2 vbend = -1;
			int vcnt = 0;
			BMESH_F_OF_E(f, e, fiter, *m_bmesh)
			{
				int vsum = 0;
				BMESH_V_OF_F(v, f, viter, *m_bmesh)
				{
					vsum += v->getIndex();
				}
				vsum -= vori[0] + vori[1];
				vbend[vcnt++] = vsum;
			} // end for fiter
			if (vbend[0] >= 0 && vbend[1] >= 0)
			{
				m_allE.push_back(vbend);
				m_allE.push_back(Int2(vbend[1], vbend[0]));
			}
			m_edgeWithBendEdge.push_back(Int4(vori[0], vori[1], vbend[0], vbend[1]));
		} // end for all edges

		// sort edges
		std::sort(m_allE.begin(), m_allE.end());
		m_allE.resize(std::unique(m_allE.begin(), m_allE.end()) - m_allE.begin());

		// setup one-ring vertex info
		size_t eIdx = 0;
		m_allVV_num.clear();
		m_allVV.clear();
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

		// copy to GPU
		m_dev_T.upload((const int*)m_T.data(), m_T.size() * 3);
		m_dev_all_VV.upload(m_allVV);
		m_dev_all_vv_num.upload(m_allVV_num);

		// parameter
		m_simulationParam.spring_k = m_simulationParam.spring_k_raw / m_avgArea;
		m_shouldTopologyUpdate = false;
		m_shouldNumericUpdate = true;
	}

	void ClothManager::buildNumerical()
	{
		if (m_shouldTopologyUpdate)
			buildTopology();
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

		// copy to GPU
		m_dev_X.upload((const ValueType*)m_X.data(), m_X.size() * 3);
		m_dev_old_X.upload((const ValueType*)m_X.data(), m_X.size() * 3);
		m_dev_next_X.upload((const ValueType*)m_X.data(), m_X.size() * 3);
		m_dev_prev_X.upload((const ValueType*)m_X.data(), m_X.size() * 3);
		m_dev_fixed.upload(m_fixed);
		m_dev_more_fixed.upload(m_fixed);
		m_dev_V.upload((const ValueType*)m_V.data(), m_V.size() * 3);
		m_dev_init_B.create(m_X.size()*3);
		cudaMemset(m_dev_init_B.ptr(), 0, m_dev_init_B.sizeBytes());
		m_dev_all_VL.upload(m_allVL);
		m_dev_all_VW.upload(m_allVW);
		m_dev_all_VC.upload(m_allVC);

		// parameter
		m_shouldNumericUpdate = false;
		m_shouldStitchUpdate = true;
	}

	int ClothManager::findNeighbor(int i, int j)const
	{
		for (int index = m_allVV_num[i]; index<m_allVV_num[i + 1]; index++)
		if (m_allVV[index] == j)	
			return index;
		printf("ERROR: failed to find the neighbor in all_VV.\n"); getchar();
		return -1;
	}

	int ClothManager::findStitchNeighbor(int i, int j)const
	{
		for (int index = m_stitchVV_num[i]; index<m_stitchVV_num[i + 1]; index++)
		if (m_stitchVV[index] == j)
			return index;
		printf("ERROR: failed to find the neighbor in stich_VV.\n"); getchar();
		return -1;
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

	//////////////////////////////////////////////////////////////////////////////////
	void ClothManager::get2dBound(ldp::Float2& bmin, ldp::Float2& bmax)const
	{
		bmin = FLT_MAX;
		bmax = FLT_MIN;
		for (const auto& c : m_clothPieces)
		{
			const auto& m = c->mesh2d();
			for (int k = 0; k < 3; k++)
			{
				bmin[k] = std::min(bmin[k], m.boundingBox[0][k]);
				bmax[k] = std::min(bmax[k], m.boundingBox[1][k]);
			}
		}

		for (int k = 0; k < 3; k++)
		{
			if (bmin[k] > bmax[k])
			{
				bmin = 0;
				bmax = 0;
			}
		}
		return;
	}


	//////////////////////////////////////////////////////////////////////////////////
	void ClothManager::loadPiecesFromSvg(std::string filename)
	{
		m_clothPieces.clear();

		svg::SvgManager svgManager;
		svgManager.load(filename.c_str());

		auto polyPaths = svgManager.collectPolyPaths(false);
		auto edgeGroups = svgManager.collectEdgeGroups(false);
		const float pixel2meter = svgManager.getPixelToMeters();

		// 1.1 add closed polygons as loops ----------------------------------------------
		std::vector<ShapeGroup> groups;
		std::vector<ShapeGroup> lines;
		ObjConvertMap objMap;
		for (auto polyPath : polyPaths)
		{
			polyPath->findCorners();
			polyPath->updateEdgeRenderData();
			if (polyPath->isClosed())
			{
				groups.push_back(ShapeGroup());
				polyPathToShape(polyPath, groups.back(), pixel2meter, objMap);
			} // end if closed
			else
			{
				lines.push_back(ShapeGroup());
				polyPathToShape(polyPath, lines.back(), pixel2meter, objMap);
			} // end if not closed
		} // end for polyPath

		// 1.2 mark inside/outside of polygons
		std::vector<int> polyInsideId(groups.size(), -1);
		std::vector<int> lineInsideId(lines.size(), -1);
		std::vector<Vec2> ipts, jpts;
		for (size_t ipoly = 0; ipoly < groups.size(); ipoly++)
		{
			ipts.clear();
			groups[ipoly].collectSamplePoints(ipts, m_clothDesignParam.curveSampleStep);
			for (size_t jpoly = 0; jpoly < groups.size(); jpoly++)
			{
				if (jpoly == ipoly)
					continue;
				jpts.clear();
				groups[jpoly].collectSamplePoints(jpts, m_clothDesignParam.curveSampleStep);
				bool allIn = true;
				for (const auto& pj : jpts)
				{
					if (!this->pointInPolygon((int)ipts.size()-1, ipts.data(), pj))
					{
						allIn = false;
						break;
					}
				} // end for pj
				if (allIn)
					polyInsideId[jpoly] = ipoly;
			} // end for jpoly
			for (size_t jpoly = 0; jpoly < lines.size(); jpoly++)
			{
				jpts.clear();
				lines[jpoly].collectSamplePoints(jpts, m_clothDesignParam.curveSampleStep);
				bool allIn = true;
				for (const auto& pj : jpts)
				{
					if (!this->pointInPolygon((int)ipts.size() - 1, ipts.data(), pj))
					{
						allIn = false;
						break;
					}
				} // end for pj
				if (allIn)
					lineInsideId[jpoly] = ipoly;
			} // end for jpoly
		} // end for ipoly

		// 1.3 create outter panels
		for (size_t ipoly = 0; ipoly < groups.size(); ipoly++)
		{
			if (polyInsideId[ipoly] >= 0)
				continue;
			m_clothPieces.push_back(std::shared_ptr<ClothPiece>(new ClothPiece()));
			const auto& piece = m_clothPieces.back();
			piece->panel().create(groups[ipoly]);

			// add dart
			for (size_t jpoly = 0; jpoly < groups.size(); jpoly++)
			{
				if (polyInsideId[jpoly] != ipoly)
					continue;
				piece->panel().addDart(groups[jpoly]);
			} // end for jpoly

			// add inner lines
			for (size_t jpoly = 0; jpoly < lines.size(); jpoly++)
			{
				if (lineInsideId[jpoly] != ipoly)
					continue;
				piece->panel().addInnerLine(lines[jpoly]);
			} // end for jpoly
		} // end for ipoly

		// 2. triangluation
		triangulate();
	}

	void ClothManager::triangulate()
	{

	}

	void ClothManager::polyPathToShape(const svg::SvgPolyPath* polyPath, 
		std::vector<ShapePtr>& group, float pixel2meter, ObjConvertMap& map)
	{
		for (size_t iCorner = 0; iCorner < polyPath->numCornerEdges(); iCorner++)
		{
			std::vector<Vec2> points;
			const auto& coords = polyPath->getEdgeCoords(iCorner);
			assert(coords.size() >= 4);
			for (size_t i = 0; i < coords.size() - 1; i += 2)
			{
				ldp::Float2 p(coords[i] * pixel2meter, coords[i + 1] * pixel2meter);
				if (points.size())
				if ((p - points.back()).length() < m_clothDesignParam.pointMergeDistThre)
					continue;
				points.push_back(p);
			} // end for i
			if (points.size() >= 2)
			{
				auto key = std::make_pair(polyPath, (int)iCorner);
				map.insert(std::make_pair(key, std::set<AbstractShape*>()));
				auto mapIter = map.find(key);
				size_t lastSize = group.size();
				AbstractShape::create(group, points, m_clothDesignParam.curveFittingThre);
				for (size_t sz = lastSize; sz < group.size(); sz++)
					mapIter->second.insert(group[sz].get());
			}
		}
	}

	bool ClothManager::pointInPolygon(int n, const Vec2* v, Vec2 p)
	{
		float d = -1;
		const float x = p[0], y = p[1];
		float minDist = FLT_MAX;
		for (int i = 0, j = n - 1; i < n; j = i++)
		{
			const float xi = v[i][0], yi = v[i][1], xj = v[j][0], yj = v[j][1];
			const float inv_k = (xj - xi) / (yj - yi);
			if ((yi>y) != (yj > y) && x - xi < inv_k * (y - yi))
				d = -d;
			// project point to line
			ldp::Float2 pvi = p - v[i];
			ldp::Float2 dji = v[j] - v[i];
			float t = pvi.dot(dji) / dji.length();
			t = std::min(1.f, std::max(0.f, t));
			ldp::Float2 p0 = v[i] + t * dji;
			minDist = std::min(minDist, (p-p0).length());
		}
		minDist *= d;
		return minDist >= -m_clothDesignParam.pointInsidePolyThre;
	}


	////sewings/////////////////////////////////////////////////////////////////////////////////
	void ClothManager::addSewing(std::shared_ptr<Sewing> sewing)
	{
		m_sewings.push_back(std::shared_ptr<Sewing>(sewing->clone()));
		buildStitchesFromSewing();
	}

	void ClothManager::addSewings(const std::vector<std::shared_ptr<Sewing>>& sewings)
	{
		for (const auto& s : sewings)
			m_sewings.push_back(std::shared_ptr<Sewing>(s->clone()));
		buildStitchesFromSewing();
	}

	void ClothManager::removeSewing(int arrayPos)
	{
		m_sewings.erase(m_sewings.begin() + arrayPos);
		buildStitchesFromSewing();
	}

	void ClothManager::removeSewingById(int id)
	{
		for (size_t i = 0; i < m_sewings.size(); i++)
		{
			if (m_sewings[i]->getId() == id)
			{
				m_sewings.erase(m_sewings.begin() + i);
				buildStitchesFromSewing();
				break;
			}
		}
	}

	void ClothManager::buildStitchesFromSewing()
	{

		// finally. require stitch rebuilt
		m_shouldStitchUpdate = true;
	}

	////Params//////////////////////////////////////////////////////////////////////////////////
	ClothDesignParam::ClothDesignParam()
	{
		setDefaultParam();
	}

	void ClothDesignParam::setDefaultParam()
	{
		pointMergeDistThre = 1e-4;				// in meters
		curveSampleStep = 1e-2;					// in meters
		pointInsidePolyThre = 1e-2;				// in meters
		curveFittingThre = 1e-3;				// in meters
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
		stitch_k_raw = 150000;
		stitch_k = 0;//will be updated after built topology
		out_iter = 8;
		inner_iter = 40;
		time_step = 1.0 / 240.0;
		stitch_ratio = 10;
		control_mag = 400;
		gravity = ldp::Float3(0, 0, -9.8);
	}
}