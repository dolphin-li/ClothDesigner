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
#include "Renderable\LoopSubdiv.h"
#include "svgpp\SvgManager.h"
#include "svgpp\SvgPolyPath.h"
#include "ldputil.h"
#include "kdtree\PointTree.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <QString>
#include "GpuSim.h"

namespace ldp
{
	enum{ LEVEL_SET_RESOLUTION = 128 };
	std::shared_ptr<SmplManager> ClothManager::m_smplMale;
	std::shared_ptr<SmplManager> ClothManager::m_smplFemale;

	inline SmplManager::Mat3 convert(ldp::Mat3d A)
	{
		SmplManager::Mat3 B;
		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			B(i, j) = A(i, j);
		return B;
	}

	inline ldp::Mat3d convert(SmplManager::Mat3 A)
	{
		ldp::Mat3d B;
		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			B(i, j) = A(i, j);
		return B;
	}

	inline SmplManager::Vec3 convert(ldp::Double3 v)
	{
		return SmplManager::Vec3(v[0], v[1], v[2]);
	}

	inline ldp::Double3 convert(SmplManager::Vec3 v)
	{
		return ldp::Double3(v[0], v[1], v[2]);
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

	ClothManager::ClothManager()
	{
		m_bodyMesh.reset(new ObjMesh);
		m_bodyMeshInit.reset(new ObjMesh);
		m_bodyTransform.reset(new TransformInfo);
		m_bodyTransform->setIdentity();
		m_bodyLvSet.reset(new LevelSet3D);
		m_graph2mesh.reset(new Graph2Mesh);
		m_gpuSim.reset(new GpuSim);
		m_fullClothSubdiv.reset(new LoopSubdiv);
		initSmplDatabase();
	}

	ClothManager::~ClothManager()
	{
		clear();
	}

	void ClothManager::clear()
	{
		simulationDestroy();

		m_bodyTransform->setIdentity();
		m_bodyMeshInit->clear();
		m_bodyMesh->clear();
		m_bodyLvSet->clear();
		m_gpuSim->clear();

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
	}

	void ClothManager::dragMove(ldp::Float3 target)
	{
		if (m_simulationMode != SimulationOn)
			return;
		m_curDragInfo.target = target;
		if (m_curDragInfo.vert_id >= 0 && m_gpuSim.get())
		{
			auto x0 = m_gpuSim->getCurrentVertPositions()[m_curDragInfo.vert_id];
			m_curDragInfo.dir = target - x0;
			m_curDragInfo.dir.normalizeLocal();
			m_curDragInfo.dir *= (target - x0).length() * 0.1f;
		}
	}

	void ClothManager::dragEnd()
	{
		if (m_simulationMode != SimulationOn)
			return;
		m_curDragInfo.vert_id = -1;
		m_curDragInfo.dir = 0;
		m_curDragInfo.target = 0;
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
		m_gpuSim->init(this);
		mergePieces();
		buildTopology();
		buildStitch();
		buildSubdiv();
		updateSubdiv();
		m_simulationMode = SimulationPause;
		m_simulationInfo = "simulation initialized";
	}

	void ClothManager::simulationUpdate()
	{
		if (m_simulationMode != SimulationOn)
			return;

		m_simulationInfo = "";

		gtime_t tbegin = gtime_now();

		updateDependency();
		if (m_shouldLevelSetUpdate)
			calcLevelSet();
		if (m_shouldTriangulate)
			triangulate();
		if (m_shouldMergePieces)
			mergePieces();
		if (m_shouldTopologyUpdate)
			buildTopology();
		if (m_shouldStitchUpdate)
			buildStitch();

		// handle fix verts
		std::vector<int> fixIds;
		std::vector<ldp::Float3> fixTars;
		if (m_curDragInfo.vert_id >= 0)
		for (size_t i = 0; i < m_gpuSim->getCurrentVertPositions().size(); i++)
		{
			auto x0 = m_gpuSim->getCurrentVertPositions()[m_curDragInfo.vert_id];
			auto x = m_gpuSim->getCurrentVertPositions()[i];
			if ((x0 - x).length() < 0.001)
			{
				fixIds.push_back(i);
				fixTars.push_back(x + m_curDragInfo.dir);
			}
		}
		m_gpuSim->setFixPositions(fixIds.size(), fixIds.data(), fixTars.data());

		// perform simulation for one step
		m_gpuSim->run_one_step();
		m_gpuSim->getResultClothPieces();

		if (m_shouldSubdivBuild)
			buildSubdiv();
		updateSubdiv();

		gtime_t tend = gtime_now();
		m_fps = 1.f / gtime_seconds(tbegin, tend);

		char fps_ary[10];
		sprintf(fps_ary, "%.1f", m_fps);
		m_simulationInfo = std::string("[fps ") + fps_ary + "]" + m_gpuSim->getSolverInfo();
	}

	void ClothManager::simulationDestroy()
	{
		m_simulationMode = SimulationNotInit;
		updateInitialClothsToCurrent();
		m_simulationInfo = "simulation destroyed";
	}

	void ClothManager::setSimulationMode(SimulationMode mode)
	{
		m_simulationMode = mode;
		if (m_simulationMode == SimulationMode::SimulationPause)
			m_simulationInfo = "simulation paused";
	}

	void ClothManager::setSimulationParam(SimulationParam param)
	{
		auto lastParam = m_simulationParam;
		m_simulationParam = param;
		
		if (m_gpuSim.get())
			m_gpuSim->updateParam();
	}

	void ClothManager::setClothDesignParam(ClothDesignParam param)
	{
		auto lastParam = g_designParam;
		g_designParam = param;
		if (fabs(lastParam.triangulateThre - g_designParam.triangulateThre)
			>= std::numeric_limits<float>::epsilon())
			m_shouldTriangulate = true;
		if (m_gpuSim.get())
			m_gpuSim->updateParam();
	}

	void ClothManager::setPieceParam(const ClothPiece* piece, PieceParam param)
	{
		bool changed = false;
		for (auto& pc : m_clothPieces)
		{
			if (pc.get() != piece)
				continue;
			if (fabs(param.bending_k_mult - pc->param().bending_k_mult) < std::numeric_limits<float>::epsilon()
				&& fabs(param.spring_k_mult - pc->param().spring_k_mult) < std::numeric_limits<float>::epsilon()
				&& param.material_name == pc->param().material_name)
				break;
			pc->param() = param;
			changed = true;
		}
		if (changed && m_gpuSim.get())
			m_gpuSim->updateParam();
	}

	ldp::Float3 ClothManager::getVertexByGlobalId(int id)const 
	{ 
		return m_gpuSim->getCurrentVertPositions()[id]; 
	}

	int ClothManager::pieceVertId2GlobalVertId(const ObjMesh* piece, int pieceVertId)const
	{
		auto siter = m_clothVertBegin.find(piece);
		if (siter == m_clothVertBegin.end())
			return pieceVertId;
		else
			return siter->second + pieceVertId;
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

	bool ClothManager::checkTopologyChangedOfMesh2d()const
	{
		bool topChanged = false;
		for (auto& cloth : m_clothPieces)
		{
			// check whether topology changed of each piece
			topChanged |= (cloth->mesh3dInit().face_list.size() != cloth->mesh2d().face_list.size());
			if (!topChanged)
			{
				for (size_t iFace = 0; iFace < cloth->mesh3dInit().face_list.size(); iFace++)
				{
					const auto& f2 = cloth->mesh2d().face_list[iFace];
					const auto& f3 = cloth->mesh3dInit().face_list[iFace];
					topChanged |= (f2.vertex_count != f3.vertex_count);
					// for each face, the verts may be inversed, so we check this case
					bool faceChanged[2] = { false, false };
					for (int k = 0; k < f2.vertex_count; k++)
						faceChanged[0] |= (f2.vertex_index[k] != f3.vertex_index[k]);
					for (int k = 0; k < f2.vertex_count; k++)
						faceChanged[1] |= (f2.vertex_index[k] != f3.vertex_index[f3.vertex_count - 1 - k]);
					topChanged |= (faceChanged[0] && faceChanged[1]);
				}
			}
		}
		return topChanged;
	}

	void ClothManager::updateCloths3dMeshBy2d()
	{
		bool topChanged = checkTopologyChangedOfMesh2d();
		for (auto& cloth : m_clothPieces)
		{
			cloth->mesh3dInit().cloneFrom(&cloth->mesh2d());
			cloth->transformInfo().apply(cloth->mesh3dInit());
			cloth->mesh3d().cloneFrom(&cloth->mesh3dInit());
		}

		// if topology changed, we want to handle it here
		if (topChanged)
		{
			mergePieces();
			buildTopology();
			buildStitch();
			buildSubdiv();
		}

		// else, we just update the subdivision mesh and handle others later
		updateSubdiv();
		m_shouldMergePieces = true;
	}

	void ClothManager::resetCloths3dMeshBy2d()
	{
		bool topChanged = checkTopologyChangedOfMesh2d();
		for (auto& cloth : m_clothPieces)
		{
			cloth->mesh3dInit().cloneFrom(&cloth->mesh2d());
			cloth->transformInfo().transform().setRotationPart(ldp::QuaternionF().
				fromAngles(ldp::Float3(ldp::PI_S/2, -ldp::PI_S, 0)).toRotationMatrix3());
			cloth->transformInfo().transform().setTranslationPart(ldp::Float3(0, 0.3, 0));
			cloth->transformInfo().apply(cloth->mesh3dInit());
			cloth->mesh3d().cloneFrom(&cloth->mesh3dInit());
		}				
		
		// if topology changed, we want to handle it here
		if (topChanged)
		{
			mergePieces();
			buildTopology();
			buildStitch();
			buildSubdiv();
		}

		// else, we just update the subdivision mesh and handle others later
		updateSubdiv();
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
		mesh.cloneFrom(&m_gpuSim->getResultClothMesh());
	}

	void ClothManager::exportClothsSeparated(std::vector<ObjMesh>& meshes)const
	{
		std::vector<ldp::Float3> vmerged = m_gpuSim->getCurrentVertPositions();
		std::vector<float> wmerged(m_gpuSim->getCurrentVertPositions().size(), 1.f);

		for (const auto& stp : m_stitches)
		{
			int sm = std::min(stp.first, stp.second);
			int lg = std::max(stp.first, stp.second);
			vmerged[sm] += m_gpuSim->getCurrentVertPositions()[lg];
			wmerged[sm]++;
			vmerged[lg] += m_gpuSim->getCurrentVertPositions()[sm];
			wmerged[lg]++;
		}
		for (size_t i = 0; i < vmerged.size(); i++)
			vmerged[i] /= wmerged[i];

		// create the mesh
		meshes.clear();
		for (const auto& piece : m_clothPieces)
		{
			meshes.push_back(ObjMesh());
			ObjMesh& mesh = meshes.back();
			const auto& m3d = piece->mesh3d();
			const auto& m2d = piece->mesh2d();
			int vs = m_clothVertBegin.at(&m3d);
			for (size_t i = 0; i < m2d.vertex_list.size(); i++)
			{
				mesh.vertex_list.push_back(vmerged[i+vs]);
				mesh.vertex_texture_list.push_back(Float2(m2d.vertex_list[i][0], -m2d.vertex_list[i][1]));
			} // end for i
			for (const auto& t : m3d.face_list)
			{
				ObjMesh::obj_face f = t;
				for (int k = 0; k < f.vertex_count; k++)
					f.texture_index[k] = f.vertex_index[k];
				mesh.face_list.push_back(f);
			}
		} // end for piece
	}
	//////////////////////////////////////////////////////////////////////////////////
	void ClothManager::clearClothPieces() 
	{
		m_clothVertBegin.clear();
		m_vertex_smplJointBind.reset((SpMat*)nullptr);
		m_vertex_smpl_defaultPosition.clear();
		m_fullClothSubdiv->clear();

		clearSewings();
		auto tmp = m_clothPieces;
		for (auto& t : tmp)
			removeClothPiece(t->graphPanel().getId());
		m_shouldTriangulate = true;
	}

	void ClothManager::addClothPiece(std::shared_ptr<ClothPiece> piece) 
	{ 
		m_clothPieces.push_back(piece);
		m_piecesSubdiv.push_back(std::shared_ptr<LoopSubdiv>(new LoopSubdiv));
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
			m_piecesSubdiv.erase(m_piecesSubdiv.begin() + (iter - m_clothPieces.begin()));
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
		m_clothVertBegin.clear();
		int fcnt = 0;
		int ecnt = 0;
		int vid_s = 0;
		for (auto& piece : m_clothPieces)
		{
			const auto& mesh = piece->mesh3d();
			m_clothVertBegin.insert(std::make_pair(&mesh, vid_s));
			vid_s += mesh.vertex_list.size();
		} // end for iCloth

		m_shouldMergePieces = false;
		m_shouldTopologyUpdate = true;
	}

	void ClothManager::clearSewings()
	{
		m_stitches.clear();
		auto tmp = m_graphSewings;
		for (auto& s : tmp)
			removeGraphSewing(s.get());
		m_shouldTriangulate = true;
		m_shouldMergePieces = true;
	}

	void ClothManager::addStitchVert(const ClothPiece* cloth1, StitchPoint s1, 
		const ClothPiece* cloth2, StitchPoint s2, size_t type)
	{
		if (m_shouldMergePieces)
			mergePieces();

		// convert from mesh id to global id
		s1 += m_clothVertBegin.at(&cloth1->mesh3d());
		s2 += m_clothVertBegin.at(&cloth2->mesh3d());

		m_stitches.push_back(StitchPointPair(s1, s2, type, 0.f));

		m_shouldStitchUpdate = true;
	}

	int ClothManager::numStitches()
	{
		updateDependency();
		if (m_shouldMergePieces)
			mergePieces();
		return (int)m_stitches.size(); 
	}

	size_t ClothManager::getStitchType(int i)
	{
		updateDependency();
		if (m_shouldMergePieces)
			mergePieces();
		return m_stitches.at(i).type;
	}

	StitchPointPair ClothManager::getStitchPointPair(int i)
	{
		updateDependency();
		if (m_shouldMergePieces)
			mergePieces();
		return m_stitches.at(i);
	}

	void ClothManager::updateStitchAngle()
	{
		triangulate();
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

	void ClothManager::updateSmplBody()
	{
		if (m_smplBody == nullptr)
			return;
		m_shouldLevelSetUpdate = true;
		m_smplBody->toObjMesh(*m_bodyMeshInit);
		setBodyMeshTransform(*m_bodyTransform);
		updateClothBySmplJoints();
	}

	void ClothManager::clearBindClothesToSmplJoints()
	{
		m_vertex_smplJointBind.reset((SpMat*)nullptr);
	}

	void ClothManager::bindClothesToSmplJoints()
	{
		m_vertex_smplJointBind.reset((SpMat*)nullptr);
		if (m_smplBody == nullptr)
			return;
		m_vertex_smplJointBind.reset(new SpMat);
		m_vertex_smpl_defaultPosition = m_gpuSim->getCurrentVertPositions();

		const int nVerts = (int)m_gpuSim->getCurrentVertPositions().size();
		const int nJoints = m_smplBody->numPoses();
		const static int K = 4;

		auto bR = m_bodyTransform->transform().getRotationPart();
		auto bT = m_bodyTransform->transform().getTranslationPart();

		typedef kdtree::PointTree<ValueType, 3> KdTree;
		std::vector<KdTree::Point> kdpoints;
		for (size_t i = 0; i < m_bodyMesh->vertex_list.size(); i++)
			kdpoints.push_back(KdTree::Point(m_bodyMesh->vertex_list[i], i));
		KdTree tree;
		tree.build(kdpoints);

		std::vector<Eigen::Triplet<ValueType>> cooSys;

		for (int iVert = 0; iVert < nVerts; iVert++)
		{
			KdTree::Point v(m_gpuSim->getCurrentVertPositions()[iVert]);
			ValueType dist = 0;
			auto nv = tree.nearestPoint(v, dist);

			int jb = m_smplBody->weights().outerIndexPtr()[nv.idx];
			int je = m_smplBody->weights().outerIndexPtr()[nv.idx + 1];
			for(int j=jb; j < je; j++)
			{
				int jointIdx = m_smplBody->weights().innerIndexPtr()[j];
				ValueType w = m_smplBody->weights().valuePtr()[j];
				cooSys.push_back(Eigen::Triplet<ValueType>(jointIdx, iVert, w));
			} // end for j
		} // end for iVert

		m_vertex_smplJointBind->resize(nJoints, nVerts);
		if (cooSys.size())
			m_vertex_smplJointBind->setFromTriplets(cooSys.begin(), cooSys.end());
	}

	void ClothManager::updateClothBySmplJoints()
	{
		if (m_smplBody == nullptr || m_vertex_smplJointBind == nullptr)
			return;
		std::vector<Float3> mX = m_gpuSim->getCurrentVertPositions();
		auto bR = m_bodyTransform->transform().getRotationPart();
		auto bT = m_bodyTransform->transform().getTranslationPart();
		m_smplBody->calcGlobalTrans();
		for (int iVert = 0; iVert < m_vertex_smplJointBind->outerSize(); iVert++)
		{
			int jb = m_vertex_smplJointBind->outerIndexPtr()[iVert];
			int je = m_vertex_smplJointBind->outerIndexPtr()[iVert + 1];
			ValueType wsum = ValueType(0);
			Double3 Tsum = Vec3(0);
			Mat3d Rsum = Mat3d().zeros();
			const Vec3 v = m_vertex_smpl_defaultPosition[iVert];
			for (int j = jb; j < je; j++)
			{
				int iJoint = m_vertex_smplJointBind->innerIndexPtr()[j];
				ValueType w = m_vertex_smplJointBind->valuePtr()[j];
				Rsum += convert(m_smplBody->getCurNodeRots(iJoint)) * w;
				Tsum += convert(m_smplBody->getCurNodeTrans(iJoint)) * w;
				wsum += w;
			} // end for j
			mX[iVert] = bR * (Rsum * bR.inv() * (v - bT) + Tsum) / wsum + bT;
		} // end for iVert
		m_gpuSim->setCurrentVertPositions(mX);
		m_gpuSim->getResultClothPieces();
	}

	bool ClothManager::setClothColorAsBoneWeights()
	{
		if (m_smplBody == nullptr || m_vertex_smplJointBind == nullptr)
			return false;
		int jSelect = m_smplBody->selectedJointId();
		for (size_t iCloth = 0; iCloth < m_clothPieces.size(); iCloth++)
		{
			auto& mesh = m_clothPieces[iCloth]->mesh3d();
			int vb = m_clothVertBegin.at(&mesh);
			mesh.vertex_color_list.clear();
			mesh.vertex_color_list.resize(mesh.vertex_list.size(), Float3(0));
			for (int iVert = 0; iVert < (int)mesh.vertex_color_list.size(); iVert++)
			{
				int vid = vb + iVert;
				int jb = m_vertex_smplJointBind->outerIndexPtr()[vid];
				int je = m_vertex_smplJointBind->outerIndexPtr()[vid + 1];
				for (int j = jb; j < je; j++)
				{
					int iJoint = m_vertex_smplJointBind->innerIndexPtr()[j];
					ValueType w = m_vertex_smplJointBind->valuePtr()[j];
					if (iJoint == jSelect)
					{
						mesh.vertex_color_list[iVert] = w;
						break;
					} // end if iJoint
				} // end for j
			} // end for iVert
			mesh.requireRenderUpdate();
		} // end for iCloth
		return true;
	}
	//////////////////////////////////////////////////////////////////////////////////
	void ClothManager::updateDependency()
	{
		if (m_shouldTriangulate)
			m_shouldMergePieces = true;
		if (m_shouldMergePieces)
			m_shouldTopologyUpdate = true;
		if (m_shouldTopologyUpdate)
			m_shouldStitchUpdate = true;
		if (m_shouldStitchUpdate)
			m_shouldSubdivBuild = true;
	}

	void ClothManager::buildTopology()
	{
		updateDependency();
		if (m_shouldMergePieces)
			mergePieces();

		m_gpuSim->updateTopology();

		// parameter
		m_shouldTopologyUpdate = false;
	}

	void ClothManager::buildStitch()
	{
		updateDependency();
		if (m_shouldTopologyUpdate)
			buildTopology();
		m_gpuSim->updateStitch();
		m_shouldStitchUpdate = false;
	}

	void ClothManager::calcLevelSet()
	{
		m_bodyMesh->updateBoundingBox();
		auto bmin = m_bodyMesh->boundingBox[0];
		auto bmax = m_bodyMesh->boundingBox[1];
		auto brag = bmax - bmin;
		bmin -= 0.2f * brag;
		bmax += 0.2f * brag;
		const float step = powf(brag[0] * brag[1] * brag[2], 1.f / 3.f) / float(LEVEL_SET_RESOLUTION);
		ldp::Int3 res = (bmax - bmin) / step;
		ldp::Float3 start = bmin;
		m_bodyLvSet->create(res, start, step);
		m_bodyLvSet->fromMesh(*m_bodyMesh);
		auto sz = m_bodyLvSet->size();
		std::vector<float> transposeLv(m_bodyLvSet->sizeXYZ(), 0.f);
		for (int z = 0; z < sz[2]; z++)
		for (int y = 0; y < sz[1]; y++)
		for (int x = 0; x < sz[0]; x++)
			transposeLv[x + y*sz[0] + z*sz[0] * sz[1]] = m_bodyLvSet->value(x, y, z)[0];
		m_bodyLvSet_d.fromHost(transposeLv.data(), make_int3(sz[0], sz[1], sz[2]));
		m_shouldLevelSetUpdate = false;
	}

	void ClothManager::buildSubdiv()
	{
		updateDependency();
		if (m_shouldStitchUpdate)
			buildStitch();
		m_piecesSubdiv.resize(m_clothPieces.size());
		for (size_t iPiece = 0; iPiece < m_piecesSubdiv.size(); iPiece++)
		{
			auto& mesh = m_clothPieces[iPiece]->mesh3d();
			auto& subPiece = m_piecesSubdiv[iPiece];
			if (subPiece.get() == nullptr)
				subPiece.reset(new LoopSubdiv);
			subPiece->init(&mesh);
		} // end for ipiece

		m_fullClothSubdiv->init(&m_gpuSim->getResultClothMesh());

		m_shouldSubdivBuild = false;
	}

	void ClothManager::updateSubdiv()
	{
		if (m_shouldSubdivBuild)
			buildSubdiv();

#pragma omp parallel for num_threads(2)  
		for (int thread = 0; thread < 2; thread++)
		{
			if (thread == 0)
			{
				for (size_t iPiece = 0; iPiece < m_piecesSubdiv.size(); iPiece++)
				{
					auto& subPiece = m_piecesSubdiv[iPiece];
					if (subPiece)
						subPiece->run();
				} // end for ipiece
			}
			if (thread == 1 && m_fullClothSubdiv)
			{
				m_fullClothSubdiv->run();
			}
		} // end for thread
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

	const ObjMesh& ClothManager::currentPieceMeshSubdiv(int i)const
	{
		return *m_piecesSubdiv.at(i)->getResultMesh();
	}

	ObjMesh& ClothManager::currentPieceMeshSubdiv(int i)
	{
		return *m_piecesSubdiv.at(i)->getResultMesh();
	}

	const ObjMesh& ClothManager::currentFullMeshSubdiv()const
	{
		return *m_fullClothSubdiv->getResultMesh();
	}

	ObjMesh& ClothManager::currentFullMeshSubdiv()
	{
		return *m_fullClothSubdiv->getResultMesh();
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

	bool ClothManager::toggleSelectedSewingsType()
	{
		bool change = false;
		for (auto& sew : m_graphSewings)
		{
			if (sew->isSelected())
			{
				if (sew->getSewingType() == ldp::GraphsSewing::SewingTypeStitch)
					sew->setSewingType(ldp::GraphsSewing::SewingTypePosition);
				else if (sew->getSewingType() == ldp::GraphsSewing::SewingTypePosition)
					sew->setSewingType(ldp::GraphsSewing::SewingTypeStitch);
				else
					printf("warning: toggleSelectedSewingsType, unhandled type %s\n", sew->getSewingTypeStr());
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
			if (pele->Value() == std::string("SimulationPara"))
			{
				auto para = getSimulationParam();
				TiXmlElement* child = pele->FirstChildElement();
				double tmp_double = 0;
				int tmp_int = 0;

				child->Attribute("spring_stiff", &tmp_double);
				para.spring_k = tmp_double;
				child = child->NextSiblingElement();

				child->Attribute("bend_stiff", &tmp_double);
				para.bending_k = tmp_double;
				child = child->NextSiblingElement();

				child->Attribute("gravityX", &tmp_double);
				para.gravity[0] = tmp_double;
				child = child->NextSiblingElement();

				child->Attribute("gravityY", &tmp_double);
				para.gravity[1] = tmp_double;
				child = child->NextSiblingElement();

				child->Attribute("gravityZ", &tmp_double);
				para.gravity[2] = tmp_double;
				child = child->NextSiblingElement();
				setSimulationParam(para);
			}
			else if (pele->Value() == std::string("BodyMesh"))
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
						const char* tmpc = nullptr;
						if (child->Attribute("bend_k_mult", &tmp))
							m_clothPieces.back()->param().bending_k_mult = tmp;
						if (child->Attribute("spring_k_mult", &tmp))
							m_clothPieces.back()->param().spring_k_mult = tmp;
						if (tmpc = child->Attribute("material_name"))
							m_clothPieces.back()->param().material_name = tmpc;
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

		//simulation para
		TiXmlElement* pele = new TiXmlElement("SimulationPara");
		root->LinkEndChild(pele);
		TiXmlElement* ele = new TiXmlElement("para");
		pele->LinkEndChild(ele);
		auto para = getSimulationParam();

		ele->SetAttribute("spring_stiff", QString::number(para.spring_k).toStdString().c_str());
		ele = new TiXmlElement("para");
		pele->LinkEndChild(ele);
		ele->SetAttribute("bend_stiff", QString::number(para.bending_k).toStdString().c_str());
		ele = new TiXmlElement("para");
		pele->LinkEndChild(ele);
		ele->SetAttribute("gravityX", QString::number(para.gravity[0]).toStdString().c_str());
		ele = new TiXmlElement("para");
		pele->LinkEndChild(ele);
		ele->SetAttribute("gravityY", QString::number(para.gravity[1]).toStdString().c_str());
		ele = new TiXmlElement("para");
		pele->LinkEndChild(ele);
		ele->SetAttribute("gravityZ", QString::number(para.gravity[2]).toStdString().c_str());

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

		// cloth pieces
		for (const auto& piece : m_clothPieces)
		{
			TiXmlElement* pele = new TiXmlElement("Piece");
			root->LinkEndChild(pele);

			// param
			TiXmlElement* param_ele = new TiXmlElement("Param");
			pele->LinkEndChild(param_ele);
			param_ele->SetAttribute("bend_k_mult", piece->param().bending_k_mult);
			param_ele->SetAttribute("spring_k_mult", piece->param().spring_k_mult);
			param_ele->SetAttribute("material_name", piece->param().material_name.c_str());

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