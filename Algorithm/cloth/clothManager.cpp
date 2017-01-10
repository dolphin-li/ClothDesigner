#include "clothManager.h"
#include "LevelSet3D.h"
#include "clothPiece.h"
#include "TransformInfo.h"
#include "SmplManager.h"
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
#include "kdtree\PointTree.h"
#include <cuda_runtime_api.h>
#include <fstream>
#ifdef ENABLE_EDGE_WISE_STITCH
#include <eigen\Dense>
#include <eigen\Sparse>
#endif
#define ENABLE_DEBUG_DUMPING

namespace ldp
{
	std::shared_ptr<SmplManager> ClothManager::m_smplMale;
	std::shared_ptr<SmplManager> ClothManager::m_smplFemale;

	inline float cot_constrained(const float* a, const float* b, const float* c)
	{
		float val = Cotangent(a, b, c);
		//return val;
		return std::min(5.f, std::max(-5.f, val));
	}
	inline float areaWeight_constrained(float area1, float area2, float avgArea)
	{
		float b = 1.f / avgArea;
		//return b;
		return std::min(b * 10.f, std::max(b * 0.1f, 1.f / (area1 + area2)));
	}

	void ClothManager::initSmplDatabase()
	{
		if (m_smplMale.get() == nullptr)
			m_smplMale.reset(new SmplManager);
		if (m_smplFemale.get() == nullptr)
			m_smplFemale.reset(new SmplManager);
		if (!m_smplMale->isInitialized())
			m_smplMale->loadFromMat("data/smpl/basicModel_m_lbs_10_207_0_v1.0.0_wrap.mat");
		if (!m_smplFemale->isInitialized())
			m_smplFemale->loadFromMat("data/smpl/basicModel_f_lbs_10_207_0_v1.0.0_wrap.mat");
	}

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
		initSmplDatabase();
		initCollisionHandler();
	}

	ClothManager::~ClothManager()
	{
		clear();
	}

	void ClothManager::clear()
	{
		simulationDestroy();

		m_V.clear();
		m_V_bending_k_mult.clear();
		m_V_outgo_dist.clear();
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
		m_smplBody = nullptr;

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
		m_curDragInfo.piece_id_end = 0;
		m_curDragInfo.piece_id_start = 0;
		if (m_clothVertBegin.find(info.selected_cloth) != m_clothVertBegin.end())
		{
			m_curDragInfo.vert_id = info.selected_vert_id + m_clothVertBegin.at(info.selected_cloth);
			m_curDragInfo.target = info.target;
			m_curDragInfo.piece_id_start = m_clothVertBegin[info.selected_cloth];
			m_curDragInfo.piece_id_end = m_curDragInfo.piece_id_start + (int)info.selected_cloth->vertex_list.size();
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

			constrain_selfCollision();

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

	void ClothManager::setPieceParam(const ClothPiece* piece, PieceParam param)
	{
		for (auto& pc : m_clothPieces)
		{
			if (pc.get() != piece)
				continue;
			if (fabs(param.bending_k_mult - pc->param().bending_k_mult) < std::numeric_limits<float>::epsilon()
				&& fabs(param.piece_outgo_dist - pc->param().piece_outgo_dist) < std::numeric_limits<float>::epsilon())
				break;
			pc->param() = param;
			m_shouldNumericUpdate = true;
		}
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

	void ClothManager::resetCloths3dMeshBy2d()
	{
		for (auto& cloth : m_clothPieces)
		{
			cloth->mesh3dInit().cloneFrom(&cloth->mesh2d());
			cloth->transformInfo().transform().setRotationPart(ldp::QuaternionF().
				fromAngles(ldp::Float3(ldp::PI_S/2, -ldp::PI_S, 0)).toRotationMatrix3());
			cloth->transformInfo().transform().setTranslationPart(ldp::Float3(0, 0.3, 0));
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

	void ClothManager::exportClothsMerged(ObjMesh& mesh, bool mergeStitchedVertex)const
	{
		// find idx map that remove all stitched vertices
		std::vector<int> idxMap(m_X.size(), -1);
		for (size_t i = 0; i < idxMap.size(); i++)
			idxMap[i] = i;

		if (mergeStitchedVertex)
		{
			for (const auto& stp : m_stitches)
			{
				int sm = std::min(stp.first, stp.second);
				int lg = std::max(stp.first, stp.second);
				while (idxMap[lg] != lg)
					lg = idxMap[lg];
				if (lg < sm) std::swap(sm, lg);
				idxMap[lg] = sm;
			}
			for (size_t i = 0; i < idxMap.size(); i++)
			{
				int m = i;
				while (idxMap[m] != m)
					m = idxMap[m];
				idxMap[i] = m;
			}
		} // end if mergeStitchedVertex

		// create the mesh
		mesh.clear();
		for (const auto& piece : m_clothPieces)
		{
			const auto& m3d = piece->mesh3d();
			const auto& m2d = piece->mesh2d();
			int vs = m_clothVertBegin.at(&m3d);
			for (size_t i = 0; i < m2d.vertex_list.size(); i++)
			{
				int vid = (int)i + vs;
				if (idxMap[vid] == vid)
				{
					idxMap[vid] = (int)mesh.vertex_list.size();
					mesh.vertex_list.push_back(m3d.vertex_list[i]);
				}
				else
					idxMap[vid] = idxMap[idxMap[vid]];
				mesh.vertex_texture_list.push_back(Float2(m2d.vertex_list[i][0], -m2d.vertex_list[i][1]));
			} // end for i
		} // end for piece
		for (const auto& t : m_T)
		{
			ObjMesh::obj_face f;
			f.vertex_count = 3;
			f.material_index = -1;
			for (int k = 0; k < t.size(); k++)
			{
				f.vertex_index[k] = idxMap[t[k]];
				f.texture_index[k] = t[k];
			}
			if (f.vertex_index[0] == f.vertex_index[1] || f.vertex_index[0] == f.vertex_index[2])
			{
				printf("warning, illegal face found, possibly due to an unproper sewing: %d %d %d -> %d %d %d\n", 
					t[0], t[1], t[2], f.vertex_index[0], f.vertex_index[1],
					f.vertex_index[2]);
			}
			else
				mesh.face_list.push_back(f);
		}
	}
	//////////////////////////////////////////////////////////////////////////////////
	void ClothManager::clearClothPieces() 
	{
		m_bmesh.reset((BMesh*)nullptr);
		m_bmeshVerts.clear();
		m_clothVertBegin.clear();
		m_vertex_smplJointBind.reset((SpMat*)nullptr);
		m_vertex_smpl_defaultPosition.clear();
		m_X.clear();
		m_T.clear();
		m_avgArea = 0;
		m_avgEdgeLength = 0;

		clearSewings();
		auto tmp = m_clothPieces;
		for (auto& t : tmp)
			removeClothPiece(t->graphPanel().getId());
		m_shouldTriangulate = true;
	}

	void ClothManager::addClothPiece(std::shared_ptr<ClothPiece> piece) 
	{ 
		m_clothPieces.push_back(piece); 
		m_shouldTriangulate = true;
	}

	void ClothManager::removeClothPiece(size_t graphPanelId)
	{
		for (auto iter = m_clothPieces.begin(); iter != m_clothPieces.end(); ++iter)
		{
			auto& panel = (*iter)->graphPanel();
			if (panel.getId() != graphPanelId)
				continue;
			m_clothPieces.erase(iter);
			m_shouldTriangulate = true;
			break;
		} // end for iter
		auto tmp = m_graphSewings;
		for (auto& sew : tmp)
		if (sew->empty())
			removeGraphSewing(sew->getId());
	}

	void ClothManager::removeClothPiece(ClothPiece* piece)
	{
		if (piece == nullptr)
			return;
		removeClothPiece(piece->graphPanel().getId());
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
		for (auto& piece : m_clothPieces)
		{
			const auto& mesh = piece->mesh3d();
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
		m_stitches.clear();
		m_stitchVV.clear();
		m_stitchVV_num.clear();
		m_stitchVC.clear();
		m_stitchVW.clear();
		m_stitchVL.clear();
		auto tmp = m_graphSewings;
		for (auto& s : tmp)
			removeGraphSewing(s.get());
		m_shouldTriangulate = true;
		m_shouldMergePieces = true;
	}

	void ClothManager::addStitchVert(const ClothPiece* cloth1, StitchPoint s1, 
		const ClothPiece* cloth2, StitchPoint s2)
	{
		if (m_shouldMergePieces)
			mergePieces();

		// convert from mesh id to global id
		s1 += m_clothVertBegin.at(&cloth1->mesh3d());
		s2 += m_clothVertBegin.at(&cloth2->mesh3d());

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
			if (stp.first >= map.second && stp.first < map.second + map.first->vertex_list.size()
				&& fmesh == nullptr)
			{
				fmesh = map.first;
				stp.first -= map.second;
			}

			if (stp.second >= map.second && stp.second < map.second + map.first->vertex_list.size() 
				&& smesh == nullptr)
			{
				smesh = map.first;
				stp.second -= map.second;
			}
		} // end for id

		if (fmesh == nullptr || smesh == nullptr)
		{
			printf("getStitchPos() error: given stitch not found!\n");
			return std::pair<Float3, Float3>();
		}

		std::pair<Float3, Float3> vp;
		vp.first = fmesh->vertex_list[stp.first];
		vp.second = smesh->vertex_list[stp.second];
		return vp;
	}

	void ClothManager::buildStitch()
	{
		updateDependency();
		if (m_shouldNumericUpdate)
			buildNumerical();
		m_simulationParam.stitch_k = m_simulationParam.stitch_k_raw / m_avgArea;
		
		// build bending edges for boundary sewings
		m_sewVofFMap.clear();
		std::map<Int2, std::pair<Int2, Int2>> edgeBendEdgeMap;
		for (const auto& s1 : m_stitches)
		{
			Int2 sv1(s1.first, s1.second);
			for (const auto& s2 : m_stitches)
			{
				Int2 sv2(s2.first, s2.second);
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
			Int2 e(s.first, s.second);
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
			ValueType c01 = cot_constrained(m_X[e[0]].ptr(), m_X[e[1]].ptr(), m_X[be[0]].ptr());
			ValueType c02 = cot_constrained(m_X[se[0]].ptr(), m_X[se[1]].ptr(), m_X[be[1]].ptr());
			ValueType c03 = cot_constrained(m_X[e[1]].ptr(), m_X[e[0]].ptr(), m_X[be[0]].ptr());
			ValueType c04 = cot_constrained(m_X[se[1]].ptr(), m_X[se[0]].ptr(), m_X[be[1]].ptr());
			ValueType area0 = sqrt(Area_Squared(m_X[e[0]].ptr(), m_X[e[1]].ptr(), m_X[be[0]].ptr()));
			ValueType area1 = sqrt(Area_Squared(m_X[se[0]].ptr(), m_X[se[1]].ptr(), m_X[be[1]].ptr()));
			ValueType weight = areaWeight_constrained(area0, area1, m_avgArea);
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

			const float w = weight * m_simulationParam.stitch_bending_k;
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

		m_shouldStitchUpdate = false;
	}

	void ClothManager::updateSmplBody()
	{
		if (m_smplBody == nullptr)
			return;
		m_shouldLevelSetUpdate = true;
		m_smplBody->toObjMesh(*m_bodyMeshInit);
		setBodyMeshTransform(*m_bodyTransform);
		updateClothBySmplJoints();
	}

	void ClothManager::calcLevelSet()
	{
		const float step = 0.005;
		m_bodyMesh->updateBoundingBox();
		auto bmin = m_bodyMesh->boundingBox[0];
		auto bmax = m_bodyMesh->boundingBox[1];
		auto brag = bmax - bmin;
		bmin -= 0.2f * brag;
		bmax += 0.2f * brag;
		ldp::Int3 res = (bmax - bmin) / step;
		ldp::Float3 start = bmin;
		m_bodyLvSet->create(res, start, step);
		m_bodyLvSet->fromMesh(*m_bodyMesh);
		m_dev_phi.upload(m_bodyLvSet->value(), m_bodyLvSet->sizeXYZ());
		m_shouldLevelSetUpdate = false;
	}

	void ClothManager::bindClothesToSmplJoints()
	{
		m_vertex_smplJointBind.reset((SpMat*)nullptr);
		if (m_smplBody == nullptr)
			return;
		m_vertex_smplJointBind.reset(new SpMat);
		m_vertex_smpl_defaultPosition = m_X;

		const int nVerts = (int)m_X.size();
		const int nJoints = m_smplBody->numPoses();
		const static int K = 4;

		// for each vertex, find the k-nearest-neighbor joints and calculate weights
		std::vector<Eigen::Triplet<ValueType>> cooSys;
		std::vector<std::pair<ValueType, int>> distMap;
		ValueType avgMinDist = ValueType(0);
		for (int iVert = 0; iVert < nJoints; iVert++)
		{
			Vec3 v(m_X[iVert]);
			
			// compute the distance to each joint **bone**.
			distMap.clear();
			for (int iJoint = 0; iJoint < m_smplBody->numPoses(); iJoint++)
			{
				int iParent = m_smplBody->getNodeParent(iJoint);
				if (iParent < 0)
					continue;
				Vec3 jb = m_smplBody->getCurNodeCenter(iParent);
				Vec3 je = m_smplBody->getCurNodeCenter(iJoint);
				ValueType val = ldp::pointSegDistance(v, jb, je);
				distMap.push_back(std::make_pair(val, iParent));
			} // end for iJoint

			// sort to make nearest first
			std::sort(distMap.begin(), distMap.end());

			// gather
			const int nnNum = std::min(K, int(distMap.size()));
			ValueType wsum = ValueType(0);
			for (int k = 0; k < nnNum; k++)
			{
				ValueType dist = distMap[k].first;
				int jointIdx = distMap[k].second;
				cooSys.push_back(Eigen::Triplet<ValueType>(jointIdx, iVert, dist));
				if (k == 0)
					avgMinDist += dist;
			} // end for k
		} // end for iVert

		avgMinDist /= m_X.size();
		m_vertex_smplJointBind->resize(nJoints, nVerts);
		if (cooSys.size())
			m_vertex_smplJointBind->setFromTriplets(cooSys.begin(), cooSys.end());

		// convert distance to weights
		for (int iVert = 0; iVert < m_vertex_smplJointBind->outerSize(); iVert++)
		{
			int jb = m_vertex_smplJointBind->outerIndexPtr()[iVert];
			int je = m_vertex_smplJointBind->outerIndexPtr()[iVert + 1];
			ValueType wsum = ValueType(0);
			for (int j = jb; j < je; j++)
			{
				int iJoint = m_vertex_smplJointBind->innerIndexPtr()[j];
				ValueType dist = m_vertex_smplJointBind->valuePtr()[j];
				ValueType w = exp(-sqr(dist)/sqr(avgMinDist));
				wsum += w;
			} // end for j
			for (int j = jb; j < je; j++)
			{
				int iJoint = m_vertex_smplJointBind->innerIndexPtr()[j];
				ValueType dist = m_vertex_smplJointBind->valuePtr()[j];
				ValueType w = exp(-sqr(dist) / sqr(avgMinDist));
				m_vertex_smplJointBind->valuePtr()[j] = w / wsum;
			} // end for j
		} // end for iVert
	}

	void ClothManager::updateClothBySmplJoints()
	{
		if (m_smplBody == nullptr || m_vertex_smplJointBind == nullptr)
			return;
		for (int iVert = 0; iVert < m_vertex_smplJointBind->outerSize(); iVert++)
		{
			int jb = m_vertex_smplJointBind->outerIndexPtr()[iVert];
			int je = m_vertex_smplJointBind->outerIndexPtr()[iVert + 1];
			ValueType wsum = ValueType(0);
			Vec3 vsum = Vec3(0);
			const Vec3 v = m_vertex_smpl_defaultPosition[iVert];
			for (int j = jb; j < je; j++)
			{
				int iJoint = m_vertex_smplJointBind->innerIndexPtr()[j];
				ValueType w = m_vertex_smplJointBind->valuePtr()[j];
				Vec3 c = m_smplBody->getCurNodeCenter(iJoint);
				vsum += w * (m_smplBody->getNodeGlobalRotation(iJoint) * (v - c) 
					+ c + m_smplBody->getNodeGlobalTranslation(iJoint));
				wsum += w;
			} // end for j
			vsum /= wsum;
			m_X[iVert] = vsum;
		} // end for iVert
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

		initCollisionHandler();

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

		// update per-vertex bending and outgo dist
		m_V_bending_k_mult.clear();
		m_V_outgo_dist.clear();
		for (auto& piece : m_clothPieces)
		{
			const auto& mesh = piece->mesh3d();
			for (const auto& v : mesh.vertex_list)
			{
				m_V_bending_k_mult.push_back(piece->param().bending_k_mult);
				m_V_outgo_dist.push_back(piece->param().piece_outgo_dist);
			}
		} // end for iCloth
		assert(m_V_bending_k_mult.size() == m_V.size());

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
			ValueType c01 = cot_constrained(m_X[v[0]].ptr(), m_X[v[1]].ptr(), m_X[v[2]].ptr());
			ValueType c02 = cot_constrained(m_X[v[0]].ptr(), m_X[v[1]].ptr(), m_X[v[3]].ptr());
			ValueType c03 = cot_constrained(m_X[v[1]].ptr(), m_X[v[0]].ptr(), m_X[v[2]].ptr());
			ValueType c04 = cot_constrained(m_X[v[1]].ptr(), m_X[v[0]].ptr(), m_X[v[3]].ptr());
			ValueType area0 = sqrt(Area_Squared(m_X[v[0]].ptr(), m_X[v[1]].ptr(), m_X[v[2]].ptr()));
			ValueType area1 = sqrt(Area_Squared(m_X[v[0]].ptr(), m_X[v[1]].ptr(), m_X[v[3]].ptr()));
			ValueType weight = areaWeight_constrained(area0, area1, m_avgArea);
			ValueType k[4];
			k[0] = c03 + c04;
			k[1] = c01 + c02;
			k[2] = -c01 - c03;
			k[3] = -c02 - c04;

			// gather the per-vertex bending
			Float4 vbendk;
			for (int k = 0; k < 4; k++)
				vbendk[k] = m_V_bending_k_mult[v[k]] * m_simulationParam.bending_k * weight;

			for (int i = 0; i<4; i++)
			for (int j = 0; j<4; j++)
			{
				if (i == j)
					m_allVC[v[i]] += k[i] * k[j] * sqrt(vbendk[i] * vbendk[j]);
				else
					m_allVW[findNeighbor(v[i], v[j])] += k[i] * k[j] * sqrt(vbendk[i] * vbendk[j]);
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
		m_dev_V_outgo_dist.upload(m_V_outgo_dist);
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
		throw std::exception("ERROR: failed to find the neighbor in all_VV.\n");
		return -1;
	}

	int ClothManager::findStitchNeighbor(int i, int j)const
	{
		for (int index = m_stitchVV_num[i]; index<m_stitchVV_num[i + 1]; index++)
		if (m_stitchVV[index] == j)
			return index;
		throw std::exception("ERROR: failed to find the neighbor in stich_VV.\n");
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
			piece->graphPanel().addLoop(fittedCurves, group.isClosed);

			if (!group.isClosed)
			{
				printf("warning: line %d not inside any closed region!\n", group.id);
				piece->graphPanel().setSelected(true);
			}

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
				GraphsSewingPtr gptr(new GraphsSewing());
				gptr->addFirsts(funits);
				const auto& second = svgLine2GraphCurves[Int2(iter->first->getId(), iter->second)];
				sunits.clear();
				for (const auto& s : second)
					sunits.push_back(GraphsSewing::Unit(s, false));
				gptr->addSeconds(sunits);
				addGraphSewing(gptr);
			}
		} // end for eg

		// 4. validate all graphs, the corresponding sewings will be updated
		for (auto& piece : m_clothPieces)
			piece->graphPanel().makeGraphValid();

		// 3. triangluation
		triangulate();
		updateDependency();
		m_shouldStitchUpdate = true;
		m_shouldLevelSetUpdate = true;

		// 5. load body
		m_smplBody = m_smplFemale.get();
		m_smplBody->toObjMesh(*m_bodyMeshInit);
		ldp::TransformInfo info;
		info.setIdentity();
		ldp::Mat4f T = ldp::Mat4f().zeros();
		T(0, 0) = -1;
		T(1, 2) = T(2, 1) = 1;
		T(2, 3) = 0.365427;
		T(3, 3) = 1;
		info.transform() = T;
		setBodyMeshTransform(info);
	}

	void ClothManager::triangulate()
	{
		updateDependency();
		if (m_clothPieces.empty())
			return;

		// make each panel valid
		for (auto& piece : m_clothPieces)
			piece->graphPanel().makeGraphValid();

		// merge all isolated curves into its closed panel
		std::map<Graph*, std::vector<Float2>> closedGraphs, openGraphs;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			auto loop = panel.getBoundingLoop();
			panel.updateBound();
			std::vector<Float2> samples;
			if (loop)
			{
				Float2 lastP = std::numeric_limits<float>::quiet_NaN();
				for (auto iter = loop->samplePoint_begin(g_designParam.curveSampleStep); !iter.isEnd(); ++iter)
				{
					if (samples.size())
					if ((*iter - samples.back()).length() < g_designParam.pointMergeDistThre
						|| (*iter - samples[0]).length() < g_designParam.pointMergeDistThre)
						continue;
					samples.push_back(*iter);
				}
				closedGraphs.insert(std::make_pair(&panel, samples));
			} // end if loop != nullptr
			else
			{
				for (auto iter = panel.point_begin(); iter != panel.point_end(); ++iter)
				{
					bool isEndPoint = false;
					for (auto edge = iter->edge_begin(); !edge.isEnd(); ++edge)
					if (edge->getStartPoint() == iter || edge->getEndPoint() == iter)
					{
						isEndPoint = true;
						break;
					}
					if (isEndPoint)
						samples.push_back(iter->getPosition());
				}
				openGraphs.insert(std::make_pair(&panel, samples));
			} // end if loop == nullptr
		} // end for piece

		// merge opengraphs to closed graphs if possible
		std::set<Graph*> mergedGraph;
		for (auto& openGraph : openGraphs)
		{
			const ldp::Float2 *obox = openGraph.first->bound();
			const auto& opts = openGraph.second;
			for (auto& closedGraph : closedGraphs)
			{
				const ldp::Float2 *cbox = closedGraph.first->bound();
				const auto& cpts = closedGraph.second;
				bool overlap = true;
				for (int k = 0; k<2; k++)
				{
					if (obox[0][k] > cbox[1][k] || obox[1][k] < cbox[0][k])
						overlap = false;
				}
				if (!overlap)
					continue;

				bool allIn = true;
				for (const auto& op : opts)
				{
					if (!pointInPolygon((int)cpts.size() - 1, cpts.data(), op))
					{
						allIn = false;
						break;
					}
				}

				if (allIn)
				{
					closedGraph.first->merge(*openGraph.first);
					mergedGraph.insert(openGraph.first);
					break;
				}
			} // end for closeGraphs
		} // end for openGraph

		// erase those merged
		auto tmpPieces = m_clothPieces;
		m_clothPieces.clear();
		for (auto& piece : tmpPieces)
		{
			if (mergedGraph.find(&piece->graphPanel()) == mergedGraph.end())
				m_clothPieces.push_back(piece);
		}

		// do triangulation
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
	bool ClothManager::addGraphSewing(std::shared_ptr<GraphsSewing> sewing)
	{
		m_graphSewings.push_back(sewing);
		m_shouldTriangulate = true;
		return true;
	}

	void ClothManager::addGraphSewings(const std::vector<std::shared_ptr<GraphsSewing>>& sewings)
	{
		for (const auto& s : sewings)
			addGraphSewing(s);
		m_shouldTriangulate = true;
	}

	void ClothManager::removeGraphSewing(size_t id)
	{
		for (auto iter = m_graphSewings.begin(); iter != m_graphSewings.end(); ++iter)
		{
			auto sew = *iter;
			if (sew->getId() != id)
				continue;
			m_graphSewings.erase(iter);
			m_shouldTriangulate = true;
			break;
		}
	}

	void ClothManager::removeGraphSewing(GraphsSewing* sewing)
	{
		if (sewing == nullptr)
			return;
		removeGraphSewing(sewing->getId());
	}

	/////UI operations///////////////////////////////////////////////////////////////////////////////////////
	bool ClothManager::removeSelectedSewings()
	{
		bool removed = false;

		// sewings selected
		auto tempSewings = m_graphSewings;
		for (auto& sew : tempSewings)
		{
			if (!sew->isSelected())
				continue;
			removeGraphSewing(sew.get());
			removed = true;
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
				sew->reverseFirsts();
				change = true;
			} // end for sew
		} // end for tempSewings
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	bool ClothManager::removeSelectedShapes()
	{
		bool removed = false;

		// 1. remove panels
		auto tmpPieces = m_clothPieces;
		for (auto& piece : tmpPieces)
		{
			auto& panel = piece->graphPanel();
			if (panel.isSelected() || (panel.getBoundingLoop() && panel.getBoundingLoop()->isSelected()))
			{
				removeClothPiece(panel.getId());
				removed = true;
			}
		}

		for (auto& piece : m_clothPieces)
		{
			// 2. remove loops
			auto& panel = piece->graphPanel();
			std::vector<GraphLoop*> tmpLoops;
			for (auto iter = panel.loop_begin(); iter != panel.loop_end(); ++iter)
			{
				if (iter->isSelected())
					tmpLoops.push_back(iter);
			}
			for (auto loop : tmpLoops)
				panel.removeLoop(loop);
			removed |= !tmpLoops.empty();

			// 3. remove curves
			std::vector<AbstractGraphCurve*> tmpCurves;
			for (auto iter = panel.curve_begin(); iter != panel.curve_end(); ++iter)
			{
				if (iter->isSelected())
					tmpCurves.push_back(iter);
			}
			for (auto curve : tmpCurves)
				panel.removeCurve(curve);
			removed |= !tmpCurves.empty();

			// 4. remove key points
			std::vector<GraphPoint*> tmpPts;
			for (auto iter = panel.point_begin(); iter != panel.point_end(); ++iter)
			{
				if (iter->isSelected())
					tmpPts.push_back(iter);
			}
			for (auto pt : tmpPts)
				panel.removeKeyPoints(pt);
			removed |= !tmpPts.empty();
		} // end for piece

		m_shouldTriangulate = true;
		return removed;
	}

	bool ClothManager::removeSelectedLoops()
	{
		bool removed = false;

		for (auto& piece : m_clothPieces)
		{
			// 2. remove loops
			auto& panel = piece->graphPanel();
			std::vector<GraphLoop*> tmpLoops;
			for (auto iter = panel.loop_begin(); iter != panel.loop_end(); ++iter)
			{
				if (iter->isSelected())
					tmpLoops.push_back(iter);
			}
			for (auto loop : tmpLoops)
				removed |= panel.removeLoop(loop);
		} // end for piece

		m_shouldTriangulate = removed;
		return removed;
	}
	
	bool ClothManager::makeSelectedCurvesToLoop()
	{
		bool change = false;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			auto suc = panel.selectedCurvesToLoop();
			change |= suc;
		} // piece
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	bool ClothManager::removeLoopsOfSelectedCurves()
	{
		bool change = false;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			auto suc = panel.removeLoopsOfSelectedCurves();
			change |= suc;
		} // piece
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	bool ClothManager::mergeSelectedCurves()
	{
		bool change = false;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			auto suc = panel.mergeSelectedCurves();
			change |= suc;
		} // piece
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	bool ClothManager::splitSelectedCurve(Float2 position)
	{
		bool change = false;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			auto suc = panel.splitTheSelectedCurve(position);
			change |= suc;
		} // piece
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	bool ClothManager::mergeSelectedKeyPoints()
	{
		bool change = false;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			auto suc = panel.mergeSelectedKeyPoints();
			change |= suc;
		} // piece
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	bool ClothManager::mergeTheSelectedKeyPointToCurve()
	{
		bool change = false;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			auto suc = panel.mergeTheSelectedKeyPointToCurve();
			change |= suc;
		} // piece
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	void ClothManager::clearHighLights()
	{
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			for (auto iter = panel.point_begin(); iter != panel.point_end(); ++iter)
				iter->setHighlighted(false);
			for (auto iter = panel.curve_begin(); iter != panel.curve_end(); ++iter)
				iter->setHighlighted(false);
			for (auto iter = panel.loop_begin(); iter != panel.loop_end(); ++iter)
				iter->setHighlighted(false);
			panel.setHighlighted(false);
		}

		for (auto& sew : m_graphSewings)
			sew->setHighlighted(false);
	}

	bool ClothManager::mirrorSelectedPanel()
	{
		bool change = false;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			if (!panel.isSelected())
				continue;
			for (auto iter = panel.point_begin(); iter != panel.point_end(); ++iter)
			{
				auto p = iter->getPosition();
				iter->setPosition(Float2(-p[0], p[1]));
			}
			change = true;
		} // piece
		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	bool ClothManager::copySelectedPanel()
	{
		bool change = false;
		std::vector<std::shared_ptr<ClothPiece>> copied;

		Graph::PtrMap ptrMap;
		for (auto& piece : m_clothPieces)
		{
			auto& panel = piece->graphPanel();
			if (!panel.isSelected())
				continue;
			auto piece_copy = piece->lightClone();
			auto& panel_copy = piece_copy->graphPanel();
			for (auto iter : panel.getPtrMapAfterClone())
				ptrMap.insert(iter);
			for (auto iter = panel_copy.point_begin(); iter != panel_copy.point_end(); ++iter)
			{
				auto p = iter->getPosition();
				iter->setPosition(Float2(-p[0], p[1]));
			}
			//for (auto iter = panel_copy.curve_begin(); iter != panel_copy.curve_end(); ++iter)
			//	iter->graphSewings().clear();
			copied.push_back(std::shared_ptr<ClothPiece>(piece_copy));
			change = true;
		} // piece

		// clone sew relations
		const size_t nSew = m_graphSewings.size();
		for (size_t iSew = 0; iSew < nSew; iSew++)
		{
			auto oldSew = m_graphSewings[iSew].get();

			// check if the sew should be cloned
			bool shouldClone = true;
			for (auto& u : oldSew->firsts())
			if (ptrMap.find(u.curve) == ptrMap.end())
			{
				shouldClone = false;
				break;
			}
			for (auto& u : oldSew->seconds())
			if (ptrMap.find(u.curve) == ptrMap.end())
			{
				shouldClone = false;
				break;
			}
			if (!shouldClone)
				continue;

			auto sew = oldSew->clone();
			for (auto& u : sew->m_firsts)
			{
				u.curve = (AbstractGraphCurve*)ptrMap[(AbstractGraphObject*)u.curve];
				if (u.curve->graphSewings().find(oldSew) == u.curve->graphSewings().end())
					printf("copySelectedPanel warning: curve %d does not relate to sew %d\n",
					u.curve->getId(), oldSew->getId());
				else
					u.curve->graphSewings().erase(oldSew);
				u.curve->graphSewings().insert(sew);
			}
			for (auto& u : sew->m_seconds)
			{
				u.curve = (AbstractGraphCurve*)ptrMap[(AbstractGraphObject*)u.curve];
				if (u.curve->graphSewings().find(oldSew) == u.curve->graphSewings().end())
					printf("copySelectedPanel warning: curve %d does not relate to sew %d\n",
					u.curve->getId(), oldSew->getId());
				else
					u.curve->graphSewings().erase(oldSew);
				u.curve->graphSewings().insert(sew);
			}

			m_graphSewings.push_back(std::shared_ptr<GraphsSewing>(sew));
		} // end for iSew

		// insert the generated pieces
		m_clothPieces.insert(m_clothPieces.end(), copied.begin(), copied.end());

		if (change)
			m_shouldTriangulate = true;
		return change;
	}

	bool ClothManager::addCurveOnAPanel(const std::vector<std::shared_ptr<ldp::GraphPoint>>& keyPts,
		const std::vector<size_t>& renderIds)
	{
		if (keyPts.size() != renderIds.size())
			return false;

		// at most we accept cubic curves
		if (renderIds.size() > AbstractGraphCurve::maxKeyPointsNum())
			return false;

		// make sure there is exactly one graph contains the points
		Graph* graph = nullptr;
		for (auto& id : renderIds)
		{
			for (auto& piece : m_clothPieces)
			{
				auto& g = piece->graphPanel();
				if (g.contains(id))
				{
					if (graph == nullptr)
						graph = &g;
					else if (graph != &g)
						return false;
				} // end if g contains id
			} // end for piece
		} // end for id

		if (graph == nullptr)
			return false;

		// add the points to the graph
		auto curve = graph->addCurve(keyPts);

		return curve != nullptr;
	}
	///////////////////////////////////////////////////////////////////////////////////////////
	void ClothManager::fromXml(std::string filename)
	{
		TiXmlDocument doc;
		if (!doc.LoadFile(filename.c_str()))
			throw std::exception(("IOError" + filename + "]: " + doc.ErrorDesc()).c_str());
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
					m_shouldLevelSetUpdate = true;
				} // end if not obj empty
			} // end for BodyMesh
			else if (pele->Value() == std::string("SmplBody"))
			{
				m_smplBody = m_smplFemale.get(); // by default, use the female model
				if (pele->Attribute("Gender"))
				{
					if (std::string(pele->Attribute("Gender")) == "female")
						m_smplBody = m_smplFemale.get();
					else if (std::string(pele->Attribute("Gender")) == "male")
						m_smplBody = m_smplMale.get();
				}
				for (auto child = pele->FirstChildElement(); child; child = child->NextSiblingElement())
				{
					if (child->Value() == m_bodyTransform->getTypeString())
						m_bodyTransform->fromXML(child);
					else if (child->Value() == std::string("joints"))
					{
						for (auto jele = child->FirstChildElement(); jele; jele = jele->NextSiblingElement())
						if (jele->Value() == std::string("joint"))
						{
							int id = 0;
							auto idatt = jele->Attribute("id", &id);
							std::vector<float> vars(m_smplBody->numVarEachPose(), 0);
							if (auto ratt = jele->Attribute("rotation"))
							{
								std::stringstream stm(ratt);
								for (auto& v : vars)
									stm >> v;
							}
							if (idatt)
							{
								for (size_t i_axis = 0; i_axis < vars.size(); i_axis++)
									m_smplBody->setCurPoseCoef(id, i_axis, vars[i_axis]);
							}
						} // end for jele of all joint
					} // end if joints
					else if (child->Value() == std::string("shapes"))
					{
						for (auto jele = child->FirstChildElement(); jele; jele = jele->NextSiblingElement())
						if (jele->Value() == std::string("shape"))
						{
							int id = 0;
							auto idatt = jele->Attribute("id", &id);
							double var = 0;
							auto vatt = jele->Attribute("pca", &var);
							if (idatt && vatt)
								m_smplBody->setCurShapeCoef(id, var);
						} // end for jele of all shape
					} // end if shapes
				} // end if child of smplBody
				m_smplBody->updateCurMesh();
				m_smplBody->toObjMesh(*m_bodyMeshInit);
				setBodyMeshTransform(*m_bodyTransform);
				m_shouldLevelSetUpdate = true;
			} // end for SmplBody
			else if (pele->Value() == std::string("Piece"))
			{
				m_clothPieces.push_back(std::shared_ptr<ClothPiece>(new ClothPiece()));
				for (auto child = pele->FirstChildElement(); child; child = child->NextSiblingElement())
				{
					if (child->Value() == m_clothPieces.back()->transformInfo().getTypeString())
						m_clothPieces.back()->transformInfo().fromXML(child);
					else if (child->Value() == m_clothPieces.back()->graphPanel().getTypeString())
						m_clothPieces.back()->graphPanel().fromXML(child);
					else if (child->Value() == std::string("Param"))
					{
						double tmp = 0;
						if (child->Attribute("bend_k_mult", &tmp))
							m_clothPieces.back()->param().bending_k_mult = tmp;
					}
				}
			} // end for piece
			else if (pele->Value() == tmpGraphSewing.getTypeString())
			{
				GraphsSewingPtr gptr(new GraphsSewing);
				gptr->fromXML(pele);
				addGraphSewing(gptr);
			} // end for sewing
		} // end for pele

		// . validate all graphs, the corresponding sewings will be updated
		for (auto& piece : m_clothPieces)
			piece->graphPanel().makeGraphValid();

		// finally initilaize simulation
		simulationInit();
	}

	void ClothManager::toXml(std::string filename)const
	{
		TiXmlDocument doc;
		TiXmlElement* root = new TiXmlElement("ClothManager");
		doc.LinkEndChild(root);

		// body mesh as obj file
		if (std::string(m_bodyMesh->scene_filename) != "")
		{
			TiXmlElement* pele = new TiXmlElement("BodyMesh");
			root->LinkEndChild(pele);
			pele->SetAttribute("ObjFile", m_bodyMesh->scene_filename);
			m_bodyTransform->toXML(pele);
		}
		// body mesh as smpl model
		else if (m_smplBody)
		{
			TiXmlElement* smpl_ele = new TiXmlElement("SmplBody");
			root->LinkEndChild(smpl_ele);
			if (m_smplBody == m_smplFemale.get())
				smpl_ele->SetAttribute("Gender", "female");
			else if (m_smplBody == m_smplMale.get())
				smpl_ele->SetAttribute("Gender", "male");

			// joints
			m_bodyTransform->toXML(smpl_ele);
			TiXmlElement* joints_ele = new TiXmlElement("joints");
			smpl_ele->LinkEndChild(joints_ele);
			for (int i_pose = 0; i_pose < m_smplBody->numPoses(); i_pose++)
			{
				TiXmlElement* ele = new TiXmlElement("joint");
				joints_ele->LinkEndChild(ele);
				std::string s;
				for (int i_axis = 0; i_axis < m_smplBody->numVarEachPose(); i_axis++)
					s += std::to_string(m_smplBody->getCurPoseCoef(i_pose, i_axis)) + " ";
				if (s.size())
					s.erase(s.begin() + s.size() - 1);
				ele->SetAttribute("id", i_pose);
				ele->SetAttribute("rotation", s.c_str());
			} // end for i_pose

			// shapes
			TiXmlElement* shapes_ele = new TiXmlElement("shapes");
			smpl_ele->LinkEndChild(shapes_ele);
			for (int i_shape = 0; i_shape < m_smplBody->numShapes(); i_shape++)
			{
				TiXmlElement* ele = new TiXmlElement("shape");
				shapes_ele->LinkEndChild(ele);
				ele->SetAttribute("id", i_shape);
				ele->SetAttribute("pca", m_smplBody->getCurShapeCoef(i_shape));
			} // end for i_shape
		} // end smplBody

		// close pieces
		for (const auto& piece : m_clothPieces)
		{
			TiXmlElement* pele = new TiXmlElement("Piece");
			root->LinkEndChild(pele);

			// param
			TiXmlElement* param_ele = new TiXmlElement("Param");
			pele->LinkEndChild(param_ele);
			param_ele->SetAttribute("bend_k_mult", piece->param().bending_k_mult);

			// transform info
			piece->transformInfo().toXML(pele);

			// panel
			piece->graphPanel().toXML(pele);
		} // end for piece

		for (const auto& sew : m_graphSewings)
		{
			sew->toXML(root);
		} // end for sew

		doc.SaveFile(filename.c_str());
	}
}