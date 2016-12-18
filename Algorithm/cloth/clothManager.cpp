#include "clothManager.h"
#include "LevelSet3D.h"
#include "clothPiece.h"
#include "TransformInfo.h"
#include "graph\Graph.h"
#include "graph\GraphsSewing.h"
#include "graph\GraphPoint.h"
#include "graph\GraphLoop.h"
#include "graph\AbstractGraphCurve.h"
#include "graph\Graph2Mesh.h"
#include "PROGRESSING_BAR.h"
#include "Renderable\ObjMesh.h"
#include "svgpp\SvgManager.h"
#include "svgpp\SvgPolyPath.h"
#include "ldputil.h"
#include <cuda_runtime_api.h>
#include <fstream>
#ifdef ENABLE_EDGE_WISE_STITCH
#include <eigen\Dense>
#include <eigen\Sparse>
#endif
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
		m_bodyMeshInit.reset(new ObjMesh);
		m_bodyTransform.reset(new TransformInfo);
		m_bodyTransform->setIdentity();
		m_bodyLvSet.reset(new LevelSet3D);
		m_graph2mesh.reset(new Graph2Mesh);
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
		m_shouldLevelSetUpdate = false;
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

		m_bodyTransform->setIdentity();
		m_bodyMeshInit->clear();
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
		m_shouldTriangulate = true;
		calcLevelSet();
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
		if (m_shouldLevelSetUpdate)
			calcLevelSet();
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
			updateSewingNormals(mesh);
		} // end for iCloth

		gtime_t t_end = ldp::gtime_now();
		m_fps = 1 / ldp::gtime_seconds(t_begin, t_end);
	}

	void ClothManager::updateSewingNormals(ObjMesh& mesh)
	{
		for (size_t iv = 0; iv < mesh.vertex_normal_list.size(); iv++)
		{
			auto iter = m_sewVofFMap.find(std::make_pair(&mesh, (int)iv));
			if (iter == m_sewVofFMap.end())
				continue;
			Float3 normal = 0.f;
			for (const auto& f : iter->second)
				normal += Float3(m_X[f[1]] - m_X[f[0]]).cross(m_X[f[2]] - m_X[f[0]]);
			mesh.vertex_normal_list[iv] = normal.normalize();
		} // end for iv
		mesh.requireRenderUpdate();
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
		if (fabs(lastParam.stitch_bending_k - m_simulationParam.stitch_bending_k) >= std::numeric_limits<float>::epsilon())
			m_shouldStitchUpdate = true;
	}

	void ClothManager::setClothDesignParam(ClothDesignParam param)
	{
		auto lastParam = g_designParam;
		g_designParam = param;
		if (fabs(lastParam.triangulateThre - g_designParam.triangulateThre)
			>= std::numeric_limits<float>::epsilon())
			m_shouldTriangulate = true;
	}

	void ClothManager::updateCurrentClothsToInitial()
	{
		for (auto& cloth : m_clothPieces)
		{
			cloth->mesh3dInit().cloneFrom(&cloth->mesh3d());
		}
		m_shouldMergePieces = true;
		updateDependency();
	}

	void ClothManager::updateInitialClothsToCurrent()
	{
		for (auto& cloth : m_clothPieces)
		{
			cloth->mesh3d().cloneFrom(&cloth->mesh3dInit());
		}
		m_shouldMergePieces = true;
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

	const TransformInfo& ClothManager::getBodyMeshTransform()const
	{
		return *m_bodyTransform;
	}

	void ClothManager::setBodyMeshTransform(const TransformInfo& info)
	{
		*m_bodyTransform = info;
		m_bodyMesh->cloneFrom(m_bodyMeshInit.get());
		m_bodyTransform->apply(*m_bodyMesh);
		m_shouldLevelSetUpdate = true;
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
		m_sewVofFMap.clear();
		m_graphSewings.clear();
		m_stitches.clear();
		m_stitchVV.clear();
		m_stitchVV_num.clear();
		m_stitchVC.clear();
		m_stitchVW.clear();
		m_stitchVL.clear();
#ifdef ENABLE_EDGE_WISE_STITCH
		m_stitchEV_num.clear();
		m_stitchEV.clear();
		m_stitchEV_W.clear();
		m_stitchE_length.clear();
		m_stitchVE.clear();
		m_stitchVE_num.clear();
		m_stitchVE_W.clear();
#endif
		m_shouldTriangulate = true;
		m_shouldMergePieces = true;
	}

	void ClothManager::addStitchVert(const ClothPiece* cloth1, StitchPoint s1, 
		const ClothPiece* cloth2, StitchPoint s2)
	{
		if (m_shouldMergePieces)
			mergePieces();

#ifndef ENABLE_EDGE_WISE_STITCH
		if (s1.vids[0] != s1.vids[1] || s2.vids[0] != s2.vids[1])
			throw std::exception("edge-wise stiching not supported, pls set StitchPoint.vids[0]=vids[1]");
#endif

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

#ifdef ENABLE_EDGE_WISE_STITCH
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
#else
		
		// build bending edges for boundary sewings
		m_sewVofFMap.clear();
		std::map<Int2, std::pair<Int2, Int2>> edgeBendEdgeMap;
		for (const auto& s1 : m_stitches)
		{
			Int2 sv1(s1.first.vids[0], s1.second.vids[0]);
			for (const auto& s2 : m_stitches)
			{
				Int2 sv2(s2.first.vids[0], s2.second.vids[0]);
				Int2 e1Idx(sv1[0], sv2[0]);
				Int2 e2Idx(sv1[1], sv2[1]);
				if (e1Idx[0] == e1Idx[1] || e2Idx[0] == e2Idx[1])
					continue;
				auto e1 = findEdge(e1Idx[0], e1Idx[1]);
				auto e2 = findEdge(e2Idx[0], e2Idx[1]);
				if (e1 && e2)
				{
					if (m_bmesh->fofe_count(e1) != 1 || m_bmesh->fofe_count(e2) != 1)
						continue;
					Int2 bend = -1;
					BMESH_F_OF_E(f, e1, e1iter, *m_bmesh)
					{
						bend[0] = -e1Idx[0] - e1Idx[1];
						int cnt = 0;
						BMESH_V_OF_F(v, f, viter, *m_bmesh)
						{
							bend[0] += v->getIndex();
						}
						break;
					}
					BMESH_F_OF_E(f, e2, e2iter, *m_bmesh)
					{
						bend[1] = -e2Idx[0] - e2Idx[1];
						int cnt = 0;
						BMESH_V_OF_F(v, f, viter, *m_bmesh)
						{
							bend[1] += v->getIndex();
						}
						break;
					}
					edgeBendEdgeMap[e1Idx] = std::make_pair(e2Idx, bend);
					edgeBendEdgeMap[e2Idx] = std::make_pair(e1Idx, bend);

					// insert boundary face map, but only for 1-to-1 map
					// this is used for better rendering, not related to the simulation
					std::pair<const ObjMesh*, int> localIds[2][2];
					localIds[0][0] = getLocalVertsId(e1Idx[0]);
					localIds[0][1] = getLocalVertsId(e1Idx[1]);
					localIds[1][0] = getLocalVertsId(e2Idx[0]);
					localIds[1][1] = getLocalVertsId(e2Idx[1]);
					BMVert* bv[2][2] = { { m_bmeshVerts[e1Idx[0]], m_bmeshVerts[e1Idx[1]] }, 
					{ m_bmeshVerts[e2Idx[0]], m_bmeshVerts[e2Idx[1]] } };
					BMESH_F_OF_V(f, bv[0][0], viter0, *m_bmesh)
					{
						Int3 idx = m_T[f->getIndex()];
						m_sewVofFMap[localIds[0][0]].insert(idx);
						m_sewVofFMap[localIds[1][0]].insert(idx);
					}
					BMESH_F_OF_V(f, bv[0][1], viter1, *m_bmesh)
					{
						Int3 idx = m_T[f->getIndex()];
						m_sewVofFMap[localIds[0][1]].insert(idx);
						m_sewVofFMap[localIds[1][1]].insert(idx);
					}
					BMESH_F_OF_V(f, bv[1][0], viter2, *m_bmesh)
					{
						Int3 idx = m_T[f->getIndex()];
						m_sewVofFMap[localIds[1][0]].insert(idx);
						m_sewVofFMap[localIds[0][0]].insert(idx);
					}
					BMESH_F_OF_V(f, bv[1][1], viter3, *m_bmesh)
					{
						Int3 idx = m_T[f->getIndex()];
						m_sewVofFMap[localIds[1][1]].insert(idx);
						m_sewVofFMap[localIds[0][1]].insert(idx);
					}
				} // end if e1 and e2
			} // end for s2
		} // end for s1

		// build edges
		std::vector<Int2> edges;
		std::set<Int2> edgeExist;
		std::vector<std::tuple<Int2, Int2, Int2>> edgesWithBend;
		for (const auto& s : m_stitches)
		{
			if (s.first.vids[0] != s.first.vids[1] || s.second.vids[0] != s.second.vids[1])
				throw std::exception("edge-wise stiching not supported, pls set StitchPoint.vids[0]=vids[1]");
			Int2 e(s.first.vids[0], s.second.vids[0]);
			if (e[0] == e[1])
				continue;

			// ignore duplicated edges
			if (edgeExist.find(e) != edgeExist.end())
				continue;
			edgeExist.insert(e);
			edgeExist.insert(Int2(e[1], e[0]));

			// stitcg edges
			edges.push_back(e);
			edges.push_back(Int2(e[1], e[0]));

			// bending edges
			edgesWithBend.push_back(std::make_tuple(e, Int2(-1), Int2(-1)));
		}
		for (auto iter : edgeBendEdgeMap)
		{
			auto e = iter.first; // edge
			auto se = iter.second.first, be = iter.second.second; // stiched edge and bend edge

			if (e[0] >= e[1])
				continue;

			// original edges
			edges.push_back(e);
			edges.push_back(Int2(e[1], e[0]));

			// boundary triangle edges
			edges.push_back(Int2(e[0], be[0]));
			edges.push_back(Int2(be[0], e[0]));
			edges.push_back(Int2(e[1], be[0]));
			edges.push_back(Int2(be[0], e[1]));

			edges.push_back(Int2(be[0], be[1]));
			edges.push_back(Int2(be[1], be[0]));

			edges.push_back(Int2(se[0], be[1]));
			edges.push_back(Int2(be[1], se[0]));
			edges.push_back(Int2(se[1], be[1]));
			edges.push_back(Int2(be[1], se[1]));

			// bend edges, marked as < 0 to distinguish with stitch edges
			edgesWithBend.push_back(std::make_tuple(0-e, se, be));
		}
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
				if (e[1] == e[0])
					continue;	// duplicate
				m_stitchVV.push_back(e[1]);
			}
		} // end for i
		m_stitchVV_num.push_back(m_stitchVV.size());

		// compute matrix related values
		m_stitchVL.resize(m_stitchVV.size());
		m_stitchVW.resize(m_stitchVV.size());
		m_stitchVC.resize(m_X.size());
		std::fill(m_stitchVL.begin(), m_stitchVL.end(), ValueType(0));
		std::fill(m_stitchVW.begin(), m_stitchVW.end(), ValueType(0));
		std::fill(m_stitchVC.begin(), m_stitchVC.end(), ValueType(0));
		for (auto tuple : edgesWithBend)
		{
			Int2 e = std::get<0>(tuple);
			Int2 se = std::get<1>(tuple);
			Int2 be = std::get<2>(tuple);

			// first, handle spring length		
			if (e[0] >= 0 && e[1] >= 0)
			{
				ValueType l = (m_X[e[0]] - m_X[e[1]]).length();
				m_stitchVL[findStitchNeighbor(e[0], e[1])] = l;
				m_stitchVL[findStitchNeighbor(e[1], e[0])] = l;
			}

			// ignore boundary edges for bending
			if (se[0] == -1 || se[1] == -1 || be[0] == -1 || be[1] == -1)
				continue;

			// convert the negative flag back.
			e[0] = -e[0];
			e[1] = -e[1];

			// second, handle bending weights
			ValueType c01 = Cotangent(m_X[e[0]].ptr(), m_X[e[1]].ptr(), m_X[be[0]].ptr());
			ValueType c02 = Cotangent(m_X[se[0]].ptr(), m_X[se[1]].ptr(), m_X[be[1]].ptr());
			ValueType c03 = Cotangent(m_X[e[1]].ptr(), m_X[e[0]].ptr(), m_X[be[0]].ptr());
			ValueType c04 = Cotangent(m_X[se[1]].ptr(), m_X[se[0]].ptr(), m_X[be[1]].ptr());
			ValueType area0 = sqrt(Area_Squared(m_X[e[0]].ptr(), m_X[e[1]].ptr(), m_X[be[0]].ptr()));
			ValueType area1 = sqrt(Area_Squared(m_X[se[0]].ptr(), m_X[se[1]].ptr(), m_X[be[1]].ptr()));
			ValueType weight = 1 / (area0 + area1);
			ValueType k[4];
			k[0] = c03 + c04;
			k[1] = c01 + c02;
			k[2] = -c01 - c03;
			k[3] = -c02 - c04;

			Int6 v;
			v[0] = e[0];
			v[1] = e[1];
			v[2] = se[0];
			v[3] = se[1];
			v[4] = be[0];
			v[5] = be[1];
			const float w = m_simulationParam.stitch_bending_k*weight;
			m_stitchVC[v[0]] += k[0] * k[0] * w / 2;
			m_stitchVC[v[1]] += k[1] * k[1] * w / 2;
			m_stitchVC[v[2]] += k[0] * k[0] * w / 2;
			m_stitchVC[v[3]] += k[1] * k[1] * w / 2;
			m_stitchVC[v[4]] += k[2] * k[2] * w;
			m_stitchVC[v[5]] += k[3] * k[3] * w;
			m_stitchVW[findStitchNeighbor(v[0], v[1])] += k[0] * k[1] * w / 2;
			m_stitchVW[findStitchNeighbor(v[2], v[3])] += k[0] * k[1] * w / 2;
			m_stitchVW[findStitchNeighbor(v[0], v[4])] += k[0] * k[2] * w;
			m_stitchVW[findStitchNeighbor(v[2], v[5])] += k[0] * k[3] * w;

			m_stitchVW[findStitchNeighbor(v[1], v[0])] += k[1] * k[0] * w / 2;
			m_stitchVW[findStitchNeighbor(v[3], v[2])] += k[1] * k[0] * w / 2;
			m_stitchVW[findStitchNeighbor(v[1], v[4])] += k[1] * k[2] * w;
			m_stitchVW[findStitchNeighbor(v[3], v[5])] += k[1] * k[3] * w;

			m_stitchVW[findStitchNeighbor(v[4], v[0])] += k[2] * k[0] * w;
			m_stitchVW[findStitchNeighbor(v[4], v[1])] += k[2] * k[1] * w;
			m_stitchVW[findStitchNeighbor(v[4], v[5])] += k[2] * k[3] * w;

			m_stitchVW[findStitchNeighbor(v[5], v[2])] += k[3] * k[0] * w;
			m_stitchVW[findStitchNeighbor(v[5], v[3])] += k[3] * k[1] * w;
			m_stitchVW[findStitchNeighbor(v[5], v[4])] += k[3] * k[2] * w;
		} // end for all edges
		// copy to GPU
		m_dev_stitch_VV.upload(m_stitchVV);
		m_dev_stitch_VV_num.upload(m_stitchVV_num);
		m_dev_stitch_VC.upload(m_stitchVC);
		m_dev_stitch_VW.upload(m_stitchVW);
		m_dev_stitch_VL.upload(m_stitchVL);
#endif
		m_shouldStitchUpdate = false;
	}

	void ClothManager::calcLevelSet()
	{
		if (!m_shouldLevelSetUpdate)
			return;
		const float step = 0.01;
		m_bodyMesh->updateBoundingBox();
		auto bmin = m_bodyMesh->boundingBox[0];
		auto bmax = m_bodyMesh->boundingBox[1];
		auto brag = bmax - bmin;
		bmin -= 0.1f * brag;
		bmax += 0.1f * brag;
		ldp::Int3 res = (bmax - bmin) / step;
		ldp::Float3 start = bmin;
		m_bodyLvSet->create(res, start, step);
		m_bodyLvSet->fromMesh(*m_bodyMesh);
		m_dev_phi.upload(m_bodyLvSet->value(), m_bodyLvSet->sizeXYZ());
		m_shouldLevelSetUpdate = false;
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

	BMEdge* ClothManager::findEdge(int v1, int v2)
	{
		BMVert* bv1 = m_bmeshVerts[v1];
		BMVert* bv2 = m_bmeshVerts[v2];
		BMESH_E_OF_V(e, bv1, v1iter, *m_bmesh)
		{
			if (m_bmesh->vofe_first(e) == bv2)
				return e;
			if (m_bmesh->vofe_last(e) == bv2)
				return e;
		}
		return nullptr;
	}

	Int3 ClothManager::getLocalFaceVertsId(Int3 globalVertId)const
	{
		for (auto map : m_clothVertBegin)
		{
			if (globalVertId[0] >= map.second && globalVertId[0] < map.second + map.first->vertex_list.size())
			{
				assert(globalVertId[1] >= map.second && globalVertId[1] < map.second + map.first->vertex_list.size());
				assert(globalVertId[2] >= map.second && globalVertId[2] < map.second + map.first->vertex_list.size());
				return globalVertId - map.second;
			}
		} // end for id
		return -1;
	}
	
	std::pair<const ObjMesh*, int> ClothManager::getLocalVertsId(int globalVertId)const
	{
		for (auto map : m_clothVertBegin)
		{
			if (globalVertId >= map.second && globalVertId < map.second + map.first->vertex_list.size())
			{
				return std::make_pair(map.first, globalVertId - map.second);
			}
		} // end for id
		return std::make_pair((const ObjMesh*)nullptr, -1);
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
	struct TmpSvgLine
	{
		std::vector<ldp::Float2> pts;
		int id = 0;
		TmpSvgLine(int i) :id(i) {}
	};
	struct TmpSvgLineGroup
	{
		std::vector<TmpSvgLine> lines;
		std::vector<ldp::Float2> samples;
		int id = 0;
		bool isClosed = false;
		int insideOtherPolyId = -1;
		ldp::Float3 C3;
		ldp::Float2 C2;
		ldp::Mat3f R;
		TmpSvgLineGroup(int i) :id(i) {}
		void samplePoints(float step)
		{
			samples.clear();
			for (const auto& line : lines)
			{
				for (const auto& p : line.pts)
				{
					if (samples.size())
					if ((p - samples.back()).length() < step || (p-samples[0]).length() < step)
						continue;
					samples.push_back(p);
				}
			} // end for line
		}
	};
	void ClothManager::loadPiecesFromSvg(std::string filename)
	{
		m_clothPieces.clear();
		clearSewings();

		svg::SvgManager svgManager;
		svgManager.load(filename.c_str());

		auto polyPaths = svgManager.collectPolyPaths(false);
		auto edgeGroups = svgManager.collectEdgeGroups(false);
		const float pixel2meter = svgManager.getPixelToMeters();

		// 1.0 collect all svg paths
		std::map<int, TmpSvgLineGroup> svgGroups;
		for (auto polyPath : polyPaths)
		{
			polyPath->updateEdgeRenderData();
			TmpSvgLineGroup svgGroup(polyPath->getId());
			svgGroup.isClosed = polyPath->isClosed();
			svgGroup.C2 = polyPath->getCenter() * pixel2meter;
			svgGroup.C3 = polyPath->get3dCenter() * pixel2meter;
			svgGroup.R = polyPath->get3dRot().toRotationMatrix3();
			for (int iCorner = 0; iCorner < polyPath->numCornerEdges(); iCorner++)
			{
				std::vector<Vec2> points;
				const auto& coords = polyPath->getEdgeCoords(iCorner);
				assert(coords.size() >= 4);
				for (size_t i = 0; i < coords.size() - 1; i += 2)
				{
					Float2 p(coords[i] * pixel2meter, coords[i + 1] * pixel2meter);
					if (points.size())
					{
						Float2 last_p = points.back();
						if ((p - last_p).length() < getClothDesignParam().pointMergeDistThre)
							continue;
					}
					points.push_back(p);
				} // end for i
				if (points.size() < 2)
					throw std::exception("loadPiecesFromSvg error: an edge in poly %d is invalid!\n", polyPath->getId());
				svgGroup.lines.push_back(TmpSvgLine(iCorner));
				svgGroup.lines.back().pts = points;
			} // end for iCorner
			svgGroup.samplePoints(getClothDesignParam().curveSampleStep);
			svgGroups.insert(std::make_pair(svgGroup.id, svgGroup));
		} // end for polyPath

		// 1.1 decide inside/outside relations
		for (auto& iter1 : svgGroups)
		{
			auto& group1 = iter1.second;
			for (auto& iter2 : svgGroups)
			{
				if (iter2.first == iter1.first)
					continue;
				auto& group2 = iter2.second;
				if (!group2.isClosed || group2.insideOtherPolyId>=0)
					continue;
				bool allIn = true;
				for (const auto& p1 : group1.samples)
				{
					if (!pointInPolygon((int)group2.samples.size() - 1, group2.samples.data(), p1))
					{
						allIn = false;
						break;
					}
				} // end for pj
				if (allIn)
					group1.insideOtherPolyId = group2.id;
			} // end for iter2
		} // end for iter1

		// 1.2 for all outside polygons, create a new graph panel, and add others that inside it into it
		std::map<Int2, std::vector<AbstractGraphCurve*>> svgLine2GraphCurves;
		for (auto& group_iter : svgGroups)
		{
			const auto& group = group_iter.second;
			if (group.insideOtherPolyId >= 0)
				continue;

			// add piece
			m_clothPieces.push_back(std::shared_ptr<ClothPiece>(new ClothPiece()));
			const auto& piece = m_clothPieces.back();

			std::vector<AbstractGraphCurve*> fittedCurves;
			for (const auto& line : group.lines)
			{
				std::vector<std::vector<GraphPointPtr>> fittedPtsGroups;
				AbstractGraphCurve::fittingCurves(fittedPtsGroups, line.pts, g_designParam.curveFittingThre);
				for (auto& pts : fittedPtsGroups)
				{
					fittedCurves.push_back(piece->graphPanel().addCurve(pts));
					svgLine2GraphCurves[Int2(group.id, line.id)].push_back(fittedCurves.back());
				}
			} // end for line

			// add outer loop
			piece->graphPanel().addLoop(fittedCurves, true);

			// copy transform:
			// the 2D-to-3D transform defined in the SVG is:
			// (x,y,0)-->R*(0,x-x0,y-y0)+t, where (x0,y0) is the 2d cener and t is the 3d cener
			ldp::Mat4f T = ldp::Mat4f().eye();
			ldp::Mat3f C = ldp::Mat3f().zeros();
			C(0, 2) = C(1, 0) = C(2, 1) = 1;
			auto R = group.R;
			auto t = group.C3;
			auto t2 = group.C2;
			T.setRotationPart(R*C);
			auto tmp = C*ldp::Float3(t2[0], t2[1], 0);
			T.setTranslationPart(t - R*C*ldp::Float3(t2[0], t2[1], 0));
			piece->transformInfo().transform() = T;

			// add other loops
			for (auto& inner_iter : svgGroups)
			{
				const auto& inner = inner_iter.second;
				if (inner.insideOtherPolyId != group.id)
					continue;

				std::vector<AbstractGraphCurve*> fittedCurves;
				for (const auto& line : inner.lines)
				{
					std::vector<std::vector<GraphPointPtr>> fittedPtsGroups;
					AbstractGraphCurve::fittingCurves(fittedPtsGroups, line.pts, g_designParam.curveFittingThre);
					for (auto& pts : fittedPtsGroups)
					{
						fittedCurves.push_back(piece->graphPanel().addCurve(pts));
						svgLine2GraphCurves[Int2(inner.id, line.id)].push_back(fittedCurves.back());
					}
				} // end for line
				piece->graphPanel().addLoop(fittedCurves, false);
			} // end for inner_iter
		} // end for group_iter

		// 2. make sewing
		for (const auto& eg : edgeGroups)
		{
			const auto& first = svgLine2GraphCurves[Int2(eg->group.begin()->first->getId(), eg->group.begin()->second)];
			std::vector<GraphsSewing::Unit> funits, sunits;
			for (const auto& f : first)
				funits.push_back(GraphsSewing::Unit(f, true));
			std::reverse(funits.begin(), funits.end());
			for (auto iter = eg->group.begin(); iter != eg->group.end(); ++iter)
			{
				if (iter == eg->group.begin())
					continue;
				m_graphSewings.push_back(GraphsSewingPtr(new GraphsSewing()));
				m_graphSewings.back()->addFirsts(funits);
				const auto& second = svgLine2GraphCurves[Int2(iter->first->getId(), iter->second)];
				sunits.clear();
				for (const auto& s : second)
					sunits.push_back(GraphsSewing::Unit(s, false));
				m_graphSewings.back()->addSeconds(sunits);
			}
		} // end for eg

		// 4. validate all graphs, the corresponding sewings will be updated
		for (auto& piece : m_clothPieces)
			piece->graphPanel().makeGraphValid(m_graphSewings);

		// 3. triangluation
		triangulate();
		updateDependency();
		m_shouldStitchUpdate = true;
		m_shouldLevelSetUpdate = true;
	}

	void ClothManager::triangulate()
	{
		updateDependency();
		if (m_clothPieces.empty())
			return;

		m_graph2mesh->triangulate(m_clothPieces, m_graphSewings,
			g_designParam.pointMergeDistThre,
			g_designParam.triangulateThre,
			g_designParam.pointInsidePolyThre);

		m_stitches = m_graph2mesh->sewingVertPairs();

		// params
		m_shouldTriangulate = false;
		m_shouldMergePieces = true;
	}

	////sewings/////////////////////////////////////////////////////////////////////////////////
	void ClothManager::addGraphSewing(std::shared_ptr<GraphsSewing> sewing)
	{
		m_graphSewings.push_back(sewing);
		m_shouldTriangulate = true;
	}

	void ClothManager::addGraphSewings(const std::vector<std::shared_ptr<GraphsSewing>>& sewings)
	{
		for (const auto& s : sewings)
			addGraphSewing(s);
		m_shouldTriangulate = true;
	}

	/////UI operations///////////////////////////////////////////////////////////////////////////////////////
	bool ClothManager::removeSelectedSewings()
	{
		bool removed = false;
		// sewings selected
		auto tempSewings = m_graphSewings;
		m_graphSewings.clear();
		for (auto& sew : tempSewings)
		{
			if (sew->isSelected())
				removed = true;
			else
				m_graphSewings.push_back(sew);
		} // end for tempSewings
		if (removed)
		{
			m_stitches.clear();
			m_shouldTriangulate = true;
		}
		return removed;
	}

	bool ClothManager::reverseSelectedSewings()
	{
		bool change = false;
		for (auto& sew : m_graphSewings)
		{
			if (sew->isSelected())
			{
				for (GraphsSewing::Unit &f : sew->firsts())
					f.reverse = !f.reverse;
				std::reverse(sew->firsts().begin(), sew->firsts().end());
				change = true;
			} // end for sew
		} // end for tempSewings
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	void ClothManager::fromXml(std::string filename)
	{
		TiXmlDocument doc;
		if (!doc.LoadFile(filename.c_str()))
			throw std::exception(("IOError" + filename).c_str());
		clear();

		auto root = doc.FirstChildElement();
		if (!root)
			throw std::exception("Xml format error, root check failed");
		if (root->Value() != std::string("ClothManager"))
			throw std::exception("Xml format error, root check failed");

		GraphsSewing tmpGraphSewing;
		for (auto pele = root->FirstChildElement(); pele; pele = pele->NextSiblingElement())
		{
			if (pele->Value() == std::string("BodyMesh"))
			{
				for (auto child = pele->FirstChildElement(); child; child = child->NextSiblingElement())
				{
					if (child->Value() == m_bodyTransform->getTypeString())
						m_bodyTransform->fromXML(child);
				}
				std::string objfile = pele->Attribute("ObjFile");
				if (!objfile.empty())
				{
					m_bodyMeshInit->loadObj(objfile.c_str(), true, false);
					setBodyMeshTransform(*m_bodyTransform);
					std::string path, name, ext;
					ldp::fileparts(objfile, path, name, ext);
					std::string setFile = ldp::fullfile(path, name + ".set");
					try
					{
						m_bodyLvSet->load(setFile.c_str());
						m_shouldLevelSetUpdate = false;
					} catch (std::exception e)
					{
						m_shouldLevelSetUpdate = true;
						calcLevelSet();
						//m_bodyLvSet->save(setFile.c_str());
					}
				} // end if not obj empty
			} // end for BodyMesh
			else if (pele->Value() == std::string("Piece"))
			{
				m_clothPieces.push_back(std::shared_ptr<ClothPiece>(new ClothPiece()));
				for (auto child = pele->FirstChildElement(); child; child = child->NextSiblingElement())
				{
					if (child->Value() == m_clothPieces.back()->transformInfo().getTypeString())
						m_clothPieces.back()->transformInfo().fromXML(child);
					else if (child->Value() == m_clothPieces.back()->graphPanel().getTypeString())
						m_clothPieces.back()->graphPanel().fromXML(child);
				}
			} // end for piece
			else if (pele->Value() == tmpGraphSewing.getTypeString())
			{
				m_graphSewings.push_back(std::shared_ptr<GraphsSewing>(new GraphsSewing));
				m_graphSewings.back()->fromXML(pele);
			} // end for sewing
		} // end for pele

		// . validate all graphs, the corresponding sewings will be updated
		for (auto& piece : m_clothPieces)
			piece->graphPanel().makeGraphValid(m_graphSewings);

		// finally initilaize simulation
		simulationInit();
	}

	void ClothManager::toXml(std::string filename)const
	{
		TiXmlDocument doc;
		TiXmlElement* root = new TiXmlElement("ClothManager");
		doc.LinkEndChild(root);

		if (m_bodyMesh->scene_filename)
		{
			TiXmlElement* pele = new TiXmlElement("BodyMesh");
			root->LinkEndChild(pele);
			pele->SetAttribute("ObjFile", m_bodyMesh->scene_filename);
			m_bodyTransform->toXML(pele);
		}

		for (const auto& piece : m_clothPieces)
		{
			TiXmlElement* pele = new TiXmlElement("Piece");
			root->LinkEndChild(pele);
			piece->transformInfo().toXML(pele);
			piece->graphPanel().toXML(pele);
		} // end for piece

		for (const auto& sew : m_graphSewings)
		{
			sew->toXML(root);
		} // end for sew

		doc.SaveFile(filename.c_str());
	}
}