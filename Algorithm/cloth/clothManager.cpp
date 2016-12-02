#include "clothManager.h"
#include "LevelSet3D.h"
#include "clothPiece.h"
#include "TransformInfo.h"
#include "panelPolygon.h"
#include "PROGRESSING_BAR.h"
#include "TriangleWrapper.h"
#include "Renderable\ObjMesh.h"
#include "svgpp\SvgManager.h"
#include "svgpp\SvgPolyPath.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <eigen\Dense>
#include <eigen\Sparse>
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
		m_triWrapper.reset(new TriangleWrapper);
		m_simulationMode = SimulationNotInit;
		m_fps = 0;
		m_avgArea = 0;
		m_avgEdgeLength = 0;
		m_curStitchRatio = 0;
		m_shouldMergePieces = false;
		m_shouldTopologyUpdate = false;
		m_shouldNumericUpdate = false;
		m_shouldStitchUpdate = false;
		m_shouldTriangulate = false;
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
		m_shouldTriangulate = true;
		updateDependency();
		triangulate();
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
		updateDependency();
		if (m_shouldTriangulate)
			triangulate();
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

	void ClothManager::setClothDesignParam(ClothDesignParam param)
	{
		auto lastParam = m_clothDesignParam;
		m_clothDesignParam = param;
		if (fabs(lastParam.triangulateThre - m_clothDesignParam.triangulateThre) 
			>= std::numeric_limits<float>::epsilon())
			m_shouldTriangulate = true;
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

	void ClothManager::updateCloths3dMeshBy2d()
	{
		for (auto& cloth : m_clothPieces)
		{
			cloth->mesh3dInit().cloneFrom(&cloth->mesh2d());
			cloth->transformInfo().apply(cloth->mesh3dInit());
			cloth->mesh3d().cloneFrom(&cloth->mesh3dInit());
		}
		m_shouldMergePieces = true;
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
		m_shouldTriangulate = true;
		m_shouldMergePieces = false;
	}

	void ClothManager::addClothPiece(std::shared_ptr<ClothPiece> piece) 
	{ 
		m_clothPieces.push_back(piece); 
		m_shouldMergePieces = true;
		m_shouldTriangulate = true;
	}

	void ClothManager::mergePieces()
	{
		if (m_shouldTriangulate)
			triangulate();
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
		m_stitchEV_num.clear();
		m_stitchEV.clear();
		m_stitchEV_W.clear();
		m_stitchE_length.clear();
		m_stitchVE.clear();
		m_stitchVE_num.clear();
		m_stitchVE_W.clear();
		m_shouldTriangulate = true;
		m_shouldMergePieces = true;
	}

	void ClothManager::addStitchVert(const ClothPiece* cloth1, StitchPoint s1, 
		const ClothPiece* cloth2, StitchPoint s2)
	{
		if (m_shouldMergePieces)
			mergePieces();

		// convert from mesh id to global id
		s1.vids += m_clothVertBegin.at(&cloth1->mesh3d());
		s2.vids += m_clothVertBegin.at(&cloth2->mesh3d());
		s1.w = std::min(1.f, std::max(0.f, s1.w));
		s2.w = std::min(1.f, std::max(0.f, s2.w));

		m_stitches.push_back(StitchPointPair(s1, s2));

		m_shouldStitchUpdate = true;
	}

	int ClothManager::numStitches()
	{
		updateDependency();
		if (m_shouldMergePieces)
			mergePieces();
		return (int)m_stitches.size(); 
	}

	std::pair<Float3, Float3> ClothManager::getStitchPos(int i)
	{
		updateDependency();
		if (m_shouldMergePieces)
			mergePieces();
		StitchPointPair stp = m_stitches.at(i);

		const ObjMesh* fmesh = nullptr, *smesh = nullptr;
		for (auto map : m_clothVertBegin)
		{
			if (stp.first.vids[0] >= map.second 
				&& stp.first.vids[0] < map.second + map.first->vertex_list.size()
				&& fmesh == nullptr)
			{
				fmesh = map.first;
				stp.first.vids -= map.second;
			}

			if (stp.second.vids[0] >= map.second
				&& stp.second.vids[0] < map.second + map.first->vertex_list.size() 
				&& smesh == nullptr)
			{
				smesh = map.first;
				stp.second.vids -= map.second;
			}
		} // end for id

		if (fmesh == nullptr || smesh == nullptr)
			throw std::exception("getStitchPos() error: given stitch not found!\n");

		std::pair<Float3, Float3> vp;
		vp.first = fmesh->vertex_list[stp.first.vids[0]] * (1 - stp.first.w) 
			+ fmesh->vertex_list[stp.first.vids[1]] * stp.first.w;
		vp.second = smesh->vertex_list[stp.second.vids[0]] * (1 - stp.second.w) 
			+ smesh->vertex_list[stp.second.vids[1]] * stp.second.w;
		return vp;
	}

	void ClothManager::buildStitch()
	{
		updateDependency();
		if (m_shouldNumericUpdate)
			buildNumerical();
		m_simulationParam.stitch_k = m_simulationParam.stitch_k_raw / m_avgArea;

		typedef Eigen::SparseMatrix<ValueType> SpMat;
		typedef Eigen::Matrix<ValueType, -1, -1> DMat;
		typedef Eigen::Matrix<ValueType, -1, 1> DVec;
		typedef Eigen::Map<DMat> DMatPtr;
		const float sitchKsqrt = sqrt(m_simulationParam.stitch_k);

		// stitch EV info------------------------------------------------
		std::vector<Eigen::Triplet<float>> cooSys;
		int nRows = 0;
		for (const auto& s : m_stitches)
		{
			cooSys.push_back(Eigen::Triplet<float>(nRows, s.first.vids[0], 1 - s.first.w));
			cooSys.push_back(Eigen::Triplet<float>(nRows, s.first.vids[1], s.first.w));
			cooSys.push_back(Eigen::Triplet<float>(nRows, s.second.vids[0], -1 + s.second.w));
			cooSys.push_back(Eigen::Triplet<float>(nRows, s.second.vids[1], -s.second.w));
			nRows++;
		}
		SpMat A, At;
		A.resize(nRows, m_X.size());
		if (cooSys.size())
			A.setFromTriplets(cooSys.begin(), cooSys.end());
		At = A.transpose();

		m_stitchEV_num.clear();
		m_stitchEV.clear();
		m_stitchEV_W.clear();
		m_stitchEV_num.clear();
		m_stitchEV_num.resize(At.cols() + 1, 0);
		for (int c = 0; c < At.cols(); c++)
		{
			const int bg = At.outerIndexPtr()[c];
			const int ed = At.outerIndexPtr()[c + 1];
			m_stitchEV_num[c] = bg;
			m_stitchEV_num[c + 1] = ed;
			for (int pos = bg; pos<ed; pos++)
			{
				int r = At.innerIndexPtr()[pos];
				ValueType v = At.valuePtr()[pos];
				m_stitchEV.push_back(r);
				m_stitchEV_W.push_back(v);
			} // end for pos
		} // end for c

		// stitch E_lengths-----------------------------------------------
		DMat X(m_X.size(), 3);
		for (int r = 0; r < X.rows(); r++)
		for (int k = 0; k < X.cols(); k++)
			X(r, k) = m_X[r][k];
		DMat Ax = A*X;
		m_stitchE_length.resize(Ax.rows());
		for (int r = 0; r < Ax.rows(); r++)
			m_stitchE_length[r] = Ax.row(r).norm();

		// stich VE info--------------------------------------------------
		m_stitchVE.clear();
		m_stitchVE_W.clear();
		m_stitchVE_num.clear();
		m_stitchVE_num.resize(A.cols() + 1, 0);
		for (int c = 0; c < A.cols(); c++)
		{
			const int bg = A.outerIndexPtr()[c];
			const int ed = A.outerIndexPtr()[c + 1];
			m_stitchVE_num[c] = bg;
			m_stitchVE_num[c + 1] = ed;
			for (int pos = bg; pos<ed; pos++)
			{
				int r = A.innerIndexPtr()[pos];
				ValueType v = A.valuePtr()[pos];
				m_stitchVE.push_back(r);
				m_stitchVE_W.push_back(v);
			} // end for pos
		} // end for c

		// stich VV info--------------------------------------------------
		SpMat AtA = A.transpose()*A;
		m_stitchVV_num.clear();
		m_stitchVV_num.resize(m_X.size() + 1, 0);
		m_stitchVC.clear();
		m_stitchVC.resize(m_X.size(), 0);
		m_stitchVV.clear();
		m_stitchVV.reserve(AtA.nonZeros());
		m_stitchVW.clear();
		m_stitchVW.reserve(AtA.nonZeros());
		m_stitchVL.clear();
		m_stitchVL.reserve(AtA.nonZeros());
		int curOffDiagIdx = 0;
		for(int c = 0; c < AtA.outerSize(); c++)
		{
			const int bg = AtA.outerIndexPtr()[c];
			const int ed = AtA.outerIndexPtr()[c + 1];
			m_stitchVV_num[c] = curOffDiagIdx;
			for(int pos=bg; pos<ed; pos++)
			{
				int r = AtA.innerIndexPtr()[pos];
				ValueType v = AtA.valuePtr()[pos];
				ValueType l = (m_X[r] - m_X[c]).length();
				if(r != c)
				{
					m_stitchVV.push_back(r);
					m_stitchVW.push_back(v * m_simulationParam.stitch_k);
					m_stitchVL.push_back(l);
					curOffDiagIdx++;
				}
				else
					m_stitchVC[c] = v * m_simulationParam.stitch_k;
			} // end for pos
		} // end for c
		m_stitchVV_num.back() = curOffDiagIdx;

		// copy to GPU
		m_dev_stitch_VV.upload(m_stitchVV);
		m_dev_stitch_VV_num.upload(m_stitchVV_num);
		m_dev_stitch_VC.upload(m_stitchVC);
		m_dev_stitch_VW.upload(m_stitchVW);
		m_dev_stitch_VL.upload(m_stitchVL);

		m_dev_stitchEV_num.upload(m_stitchEV_num);		
		m_dev_stitchEV.upload(m_stitchEV);			
		m_dev_stitchEV_W.upload(m_stitchEV_W);		
		m_dev_stitchE_length.upload(m_stitchE_length);	
		m_dev_stitchVE_num.upload(m_stitchVE_num);		
		m_dev_stitchVE.upload(m_stitchVE);			
		m_dev_stitchVE_W.upload(m_stitchVE_W);	
		m_dev_stitchE_curVec.create(m_dev_stitchE_length.size()*3);
		cudaMemset(m_dev_stitchE_curVec.ptr(), 0, m_dev_stitchE_curVec.sizeBytes());
		m_shouldStitchUpdate = false;
	}

	//////////////////////////////////////////////////////////////////////////////////
	void ClothManager::updateDependency()
	{
		if (m_shouldTriangulate)
			m_shouldMergePieces = true;
		if (m_shouldMergePieces)
			m_shouldTopologyUpdate = true;
		if (m_shouldTopologyUpdate)
			m_shouldNumericUpdate = true;
		if (m_shouldNumericUpdate)
			m_shouldStitchUpdate = true;
	}

	void ClothManager::buildTopology()
	{
		updateDependency();
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
		m_curStitchRatio = 1;
		m_shouldTopologyUpdate = false;
		m_shouldNumericUpdate = true;
	}

	void ClothManager::buildNumerical()
	{
		updateDependency();
		if (m_shouldTopologyUpdate)
			buildTopology();
		// compute matrix related values
		m_allVL.resize(m_allVV.size());
		m_allVW.resize(m_allVV.size());
		m_allVC.resize(m_X.size());
		std::fill(m_allVL.begin(), m_allVL.end(), ValueType(0));
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
		std::vector<ShapeGroupPtr> groups;
		std::vector<ShapeGroupPtr> lines;
		std::vector<const svg::SvgPolyPath*> groupSvgPaths;
		ObjConvertMap objMap;
		for (auto polyPath : polyPaths)
		{
			polyPath->findCorners();
			polyPath->updateEdgeRenderData();
			if (polyPath->isClosed())
			{
				groupSvgPaths.push_back(polyPath);
				groups.push_back(ShapeGroupPtr(new ShapeGroup()));
				polyPathToShape(polyPath, groups.back(), pixel2meter, objMap);
			} // end if closed
			else
			{
				lines.push_back(ShapeGroupPtr(new ShapeGroup()));
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
			groups[ipoly]->collectSamplePoints(ipts, m_clothDesignParam.curveSampleStep);
			for (size_t jpoly = 0; jpoly < groups.size(); jpoly++)
			{
				if (jpoly == ipoly)
					continue;
				jpts.clear();
				groups[jpoly]->collectSamplePoints(jpts, m_clothDesignParam.curveSampleStep);
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
				lines[jpoly]->collectSamplePoints(jpts, m_clothDesignParam.curveSampleStep);
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

		// 1.3 create outter panels, darts and inner lines
		for (size_t ipoly = 0; ipoly < groups.size(); ipoly++)
		{
			if (polyInsideId[ipoly] >= 0)
				continue;
			m_clothPieces.push_back(std::shared_ptr<ClothPiece>(new ClothPiece()));
			const auto& piece = m_clothPieces.back();
			piece->panel().create(groups[ipoly]);

			// copy transform:
			// the 2D-to-3D transform defined in the SVG is:
			// (x,y,0)-->R*(0,x-x0,y-y0)+t, where (x0,y0) is the 2d cener and t is the 3d cener
			auto svg = groupSvgPaths[ipoly];
			ldp::Mat4f T = ldp::Mat4f().eye();
			ldp::Mat3f C = ldp::Mat3f().zeros();
			C(0, 2) = C(1, 0) = C(2, 1) = 1;
			auto R = svg->get3dRot().toRotationMatrix3();
			auto t = svg->get3dCenter() * pixel2meter;
			auto t2 = svg->getCenter() * pixel2meter;
			T.setRotationPart(R*C);
			auto tmp = C*ldp::Float3(t2[0], t2[1], 0);
			T.setTranslationPart(t - R*C*ldp::Float3(t2[0], t2[1], 0));
			piece->transformInfo().transform() = T;

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

		// 1.4 make sewing
		for (const auto& eg : edgeGroups)
		{
			const auto& first = objMap[std::make_pair(eg->group.begin()->first, eg->group.begin()->second)];
			std::vector<Sewing::Unit> funits, sunits;
			for (const auto& f : first)
			{
				assert(Sewing::getPtrById(f->getId()));
				funits.push_back(Sewing::Unit(f->getId(), true));
			}
			std::reverse(funits.begin(), funits.end());
			for (auto iter = eg->group.begin(); iter != eg->group.end(); ++iter)
			{
				if (iter == eg->group.begin())
					continue;
				m_sewings.push_back(SewingPtr(new Sewing()));

				m_sewings.back()->addFirsts(funits);

				const auto& second = objMap[std::make_pair(iter->first, iter->second)];
				sunits.clear();
				for (const auto& s : second)
				{
					assert(Sewing::getPtrById(s->getId()));
					sunits.push_back(Sewing::Unit(s->getId(), false));
				}
				m_sewings.back()->addSeconds(sunits);
			}
		} // end for eg

		// 2. triangluation
		triangulate();
		updateDependency();
	}

	void ClothManager::triangulate()
	{
		updateDependency();
		if (m_clothPieces.empty())
			return;
		bool noPanel = true;
		for (const auto& piece : m_clothPieces)
		{
			if (piece->panel().outerPoly())
				noPanel = false;
		}
		if (noPanel)
			return;


		m_triWrapper->triangulate(m_clothPieces, m_sewings,
			m_clothDesignParam.pointMergeDistThre,
			m_clothDesignParam.triangulateThre,
			m_clothDesignParam.pointInsidePolyThre);

		m_stitches = m_triWrapper->sewingVertPairs();

		// params
		m_shouldTriangulate = false;
		m_shouldMergePieces = true;
	}

	void ClothManager::polyPathToShape(const svg::SvgPolyPath* polyPath, 
		std::shared_ptr<ldp::ShapeGroup>& group, float pixel2meter, ObjConvertMap& map)
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
				map.insert(std::make_pair(key, std::vector<AbstractShape*>()));
				auto mapIter = map.find(key);
				size_t lastSize = group->size();
				AbstractShape::create(*group, points, m_clothDesignParam.curveFittingThre);
				for (size_t sz = lastSize; sz < group->size(); sz++)
					mapIter->second.push_back(group->at(sz).get());
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
		m_shouldTriangulate = true;
	}

	void ClothManager::addSewings(const std::vector<std::shared_ptr<Sewing>>& sewings)
	{
		for (const auto& s : sewings)
			m_sewings.push_back(std::shared_ptr<Sewing>(s->clone()));
		m_shouldTriangulate = true;
	}

	/////UI operations///////////////////////////////////////////////////////////////////////////////////////
	bool ClothManager::removeSelected(AbstractPanelObject::Type types)
	{
		auto tmpPieces = m_clothPieces;
		if (types & AbstractPanelObject::TypePanelPolygon)
			m_clothPieces.clear();
		std::set<size_t> removedId;
		std::vector<AbstractPanelObject*> tmpObjs;
		for (auto& piece : tmpPieces)
		{
			auto& panel = piece->panel();
			auto& poly = piece->panel().outerPoly();
			if (poly == nullptr)
				continue;

			// panel
			if (types & AbstractPanelObject::TypePanelPolygon)
			{
				if (panel.isSelected() || poly->isSelected())
				{
					tmpObjs.clear();
					panel.collectObject(tmpObjs);
					for (auto o : tmpObjs)
						removedId.insert(o->getId());
					continue;
				}
				m_clothPieces.push_back(piece);
			}

			// poly
			if (types & AbstractPanelObject::TypeGroup)
			{
				auto tmpPoly = *poly;
				poly->clear();
				for (const auto& shape : tmpPoly)
				{
					if (shape->isSelected())
					{
						tmpObjs.clear();
						shape->collectObject(tmpObjs);
						for (auto o : tmpObjs)
							removedId.insert(o->getId());
						continue;
					} // end for shape
					poly->push_back(shape);
				} // end for tmpPoly
			}

			// darts
			if (types & AbstractPanelObject::TypeGroup)
			{
				auto tmpDarts = panel.darts();
				panel.darts().clear();
				for (const auto& dart : tmpDarts)
				{
					if (dart->isSelected())
					{
						tmpObjs.clear();
						dart->collectObject(tmpObjs);
						for (auto o : tmpObjs)
							removedId.insert(o->getId());
						continue;
					}
					panel.darts().push_back(dart);

					// dart
					auto tmpDart = *dart;
					dart->clear();
					for (const auto& shape : tmpDart)
					{
						if (shape->isSelected())
						{
							tmpObjs.clear();
							shape->collectObject(tmpObjs);
							for (auto o : tmpObjs)
								removedId.insert(o->getId());
							continue;
						} // end for shape
						dart->push_back(shape);
					} // end for tmpDart
				} // end for dart
			}

			// lines
			if (types & AbstractPanelObject::TypeGroup)
			{
				auto tmpLines = panel.innerLines();
				panel.innerLines().clear();
				for (const auto& line : tmpLines)
				{
					if (line->isSelected())
					{
						tmpObjs.clear();
						line->collectObject(tmpObjs);
						for (auto o : tmpObjs)
							removedId.insert(o->getId());
						continue;
					}
					panel.innerLines().push_back(line);

					// line
					auto tmpLine = *line;
					line->clear();
					for (const auto& shape : tmpLine)
					{
						if (shape->isSelected())
						{
							tmpObjs.clear();
							shape->collectObject(tmpObjs);
							for (auto o : tmpObjs)
								removedId.insert(o->getId());
							continue;
						} // end for shape
						line->push_back(shape);
					} // end for tmpDart
				} // end for line
			} // end for piece
		} // end for piece

		// if some shape removed, then we should update the sewings
		if (!removedId.empty())
		{
			auto tempSewings = m_sewings;
			m_sewings.clear();
			for (auto& sew : tempSewings)
			{
				const auto& firsts = sew->firsts();
				const auto& seconds = sew->seconds();
				bool invalid = false;
				for (const auto& f : firsts)
				{
					if (removedId.find(f.id) != removedId.end())
					{
						invalid = true;
						break;
					}
				}
				for (const auto& s : seconds)
				{
					if (removedId.find(s.id) != removedId.end())
					{
						invalid = true;
						break;
					}
				}
				if (!invalid)
					m_sewings.push_back(sew);
			} // end for tempSewings
		} // end if removedId.notEmpty()

		// sewings selected
		if (types & AbstractPanelObject::TypeSewing)
		{
			auto tempSewings = m_sewings;
			m_sewings.clear();
			for (auto& sew : tempSewings)
			{
				if (sew->isSelected())
				{
					tmpObjs.clear();
					sew->collectObject(tmpObjs);
					for (auto o : tmpObjs)
						removedId.insert(o->getId());
					continue;
				} // end for sew
				m_sewings.push_back(sew);
			} // end for tempSewings
		}
		m_stitches.clear();
		m_shouldTriangulate = true;
		return !removedId.empty();
	}

	bool ClothManager::reverseSelectedSewings()
	{
		bool change = false;
		for (auto& sew : m_sewings)
		{
			if (sew->isSelected())
			{
				for (Sewing::Unit &f : sew->firsts())
					f.reverse = !f.reverse;
				std::reverse(sew->firsts().begin(), sew->firsts().end());
				change = true;
			} // end for sew
		} // end for tempSewings
		if (change)
			m_shouldTriangulate = true;
		return change;
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
		triangulateThre = 3e-2;					// in meters
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