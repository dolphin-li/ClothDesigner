#include "GpuSim.h"

#include "clothManager.h"
#include "Renderable\ObjMesh.h"
#include "cloth\LevelSet3D.h"
#include "cloth\clothPiece.h"
#include "cloth\graph\Graph.h"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\CachedDeviceBuffer.h"
#include "arcsim\adaptiveCloth\dde.hpp"
#include "arcsim\ArcSimManager.h"
#include "arcsim\adaptiveCloth\conf.hpp"

#include <eigen\Dense>
#include <eigen\Sparse>

//#define SOLVE_USE_EIGEN
const static char* g_default_arcsim_material = "data/arcsim/materials/gray-interlock.json";

namespace ldp
{
#pragma region -- utils

	typedef double egreal;
	typedef Eigen::Triplet<egreal> Triplet;
	typedef Eigen::SparseMatrix<egreal> SpMat;
	typedef Eigen::Matrix<egreal, -1, 1> EgVec;
#define DEBUG_POINT(i) {cudaThreadSynchronize(); printf("%d\n", i); cudaSafeCall(cudaGetLastError());}

	static void dump(std::string name, const SpMat& A)
	{
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int col = 0; col < A.cols(); col++)
			{
				int cb = A.outerIndexPtr()[col];
				int ce = A.outerIndexPtr()[col + 1];
				for (int c = cb; c < ce; c++)
				{
					int row = A.innerIndexPtr()[c];
					egreal val = A.valuePtr()[c];
					fprintf(pFile, "%d %d %ef\n", row, col, val);
				}
			}
		}
		fclose(pFile);
	}

	static void cudaSpMat_to_EigenMat(const CudaBsrMatrix& cA, SpMat& A)
	{
		DeviceArray<int> csrRows_d, csrCols_d;
		DeviceArray<float> csrVals_d;
		cA.toCsr(csrRows_d, csrCols_d, csrVals_d);
		std::vector<int> csrRows, csrCols;
		std::vector<float> csrVals;
		csrRows_d.download(csrRows);
		csrCols_d.download(csrCols);
		csrVals_d.download(csrVals);

		std::vector<Triplet> cooSys;
		for (int r = 0; r + 1 < csrRows.size(); r++)
		{
			for (int pos = csrRows[r]; pos < csrRows[r + 1]; pos++)
			{
				int c = csrCols[pos];
				float val = csrVals[pos];
				cooSys.push_back(Triplet(r, c, val));
			}
		}
		A.resize(cA.rows(), cA.cols());
		A.setFromTriplets(cooSys.begin(), cooSys.end());	
	}

	template<class T, int n>
	inline ldp::ldp_basic_vec<T, n> convert(arcsim::Vec<n, T> v)
	{
		ldp::ldp_basic_vec<T, n> r;
		for (int k = 0; k < n; k++)
			r[k] = v[k];
		return r;
	}

	static void cusparseCheck(cusparseStatus_t st, const char* msg = nullptr)
	{
		if (CUSPARSE_STATUS_SUCCESS != st)
		{
			printf("cusparse error[%d]: %s", st, msg);
			throw std::exception(msg);
		}
	}
	static void cublasCheck(cublasStatus_t st, const char* msg = nullptr)
	{
		if (CUBLAS_STATUS_SUCCESS != st)
		{
			printf("cublas error[%d]: %s", st, msg);
			throw std::exception(msg);
		}
	}

#define NOT_IMPLEMENTED throw std::exception((std::string("NotImplemented[")+__FILE__+"]["\
	+std::to_string(__LINE__)+"]").c_str())

	inline Float2 texCoordFromClothManagerMesh2DCoords(Float3 v)
	{
		// arcsim requires this conversion
		return Float2(v[0], -v[1]);
	}
#pragma endregion

	GpuSim::GpuSim()
	{
		cusparseCheck(cusparseCreate(&m_cusparseHandle), "GpuSim: create cusparse handel");
		cublasCheck(cublasCreate_v2(&m_cublasHandle));
		m_A_d.reset(new CudaBsrMatrix(m_cusparseHandle));
		m_A_diag_d.reset(new CudaDiagBlockMatrix());
		m_vert_FaceList_d.reset(new CudaBsrMatrix(m_cusparseHandle, true));
		m_stitch_vertPairs_d.reset(new CudaBsrMatrix(m_cusparseHandle, true));
		m_bmesh.reset(new BMesh());
		m_resultClothMesh.reset(new ObjMesh);
	}

	GpuSim::~GpuSim()
	{
		clear();
		cusparseCheck(cusparseDestroy(m_cusparseHandle));
		cublasCheck(cublasDestroy_v2(m_cublasHandle));
	}

	void GpuSim::init(ClothManager* clothManager)
	{
		clear();
		release_assert(ldp::is_float<ClothManager::ValueType>::value);
		m_arcSimManager = nullptr;
		m_clothManager = clothManager; 
		initParam();
		resetDependency(true);
		initFaceEdgeVertArray();
		m_curStitchRatio = 1.f;
		m_solverInfo = "solver intialized from cloth manager";
	}

	void GpuSim::init(arcsim::ArcSimManager* arcSimManager)
	{
		clear();
		release_assert(ldp::is_float<ClothManager::ValueType>::value);
		m_clothManager = nullptr;
		m_arcSimManager = arcSimManager;
		initParam();
		resetDependency(true);
		initFaceEdgeVertArray();
		m_curStitchRatio = 1.f;
		m_solverInfo = "solver intialized from arcsim";
	}

	void GpuSim::run_one_step()
	{
		if (m_clothManager == nullptr && m_arcSimManager == nullptr)
			return;

		m_solverInfo = "[";

		gtime_t t_start = gtime_now();

		updateSystem();

		// stitching: during stitching, we do not want the speed to high
		m_curStitchRatio = std::max(0.f, 1.f - m_curSimulationTime * m_simParam.stitch_ratio);
		if (m_curStitchRatio > 0.f)
		{
			cudaSafeCall(cudaMemset(m_v_d.ptr(), 0, m_v_d.sizeBytes()));
			m_v_d.copyTo(m_last_v_d);
		}

		// build m_A and m_b
		bindTextures();
		updateNumeric();

		// solve the linear system
		m_x_d.copyTo(m_last_x_d);
		m_v_d.copyTo(m_last_v_d);
		m_dv_d.create(m_dv_d.size());
		linearSolve();

		// post-process collisions
		collisionSolve();

		// process user controls
		userControlSolve();

		// finish, get the result back to cpu and prepare for next.
		m_x_d.download(m_x_h);
		m_curSimulationTime += m_simParam.dt;

		m_shouldExportMesh = true;
		gtime_t t_end = gtime_now();
		m_fps = 1.f / gtime_seconds(t_start, t_end);

		char fps_ary[10];
		sprintf(fps_ary, "%.1f", m_fps);
		m_solverInfo += std::string(", fps ") + fps_ary + "]";
	}

	void GpuSim::restart()
	{
		m_shouldRestart = true;
		updateDependency();
		m_x_h = m_x_init_h;
		m_x_init_d.copyTo(m_x_d);
		m_x_d.copyTo(m_last_x_d);
		m_v_d.create(m_x_init_d.size());
		m_last_v_d.create(m_x_init_d.size());
		m_dv_d.create(m_x_init_d.size());
		m_b_d.create(m_x_init_d.size());
		m_curSimulationTime = 0.f;
		m_curStitchRatio = 1.f;
		m_shouldRestart = false;
	}

	void GpuSim::updateParam()
	{
		if (m_arcSimManager)
		{
			const auto* sim = m_arcSimManager->getSimulator();
			m_simParam.dt = sim->step_time;
			m_simParam.gravity = convert(sim->gravity);
			m_simParam.stitch_ratio = 99999.f; // we donot have stitch in this mode, thus we set max to skip it.
			m_simParam.enable_selfCollision = true;
		} // end if arc
		else if (m_clothManager)
		{
			auto par = m_clothManager->getSimulationParam();
			m_simParam.gravity = par.gravity;
			m_simParam.strecth_mult = par.spring_k;
			m_simParam.bend_mult = par.bending_k;
			m_simParam.enable_selfCollision = par.enable_self_collistion;
		} // end else clothManager
	}

	void GpuSim::updateTopology()
	{
		initFaceEdgeVertArray();
	}

	void GpuSim::updateStitch()
	{
		buildStitch();
	}

	void GpuSim::updateMaterial()
	{
		buildMaterial();
	}

	ObjMesh& GpuSim::getResultClothMesh()
	{
		exportResultClothToObjMesh();
		return *m_resultClothMesh;
	}

	void GpuSim::setCurrentVertPositions(const std::vector<Float3>& X)
	{
		if (X.size() != m_x_h.size())
			throw std::exception("GpuSim::setCurrentVertPosition, size not matched!");
		m_x_h = X;
		m_x_d.upload(m_x_h);
	}

	void GpuSim::setInitVertPositions(const std::vector<Float3>& X)
	{
		if (X.size() != m_x_h.size())
			throw std::exception("GpuSim::setCurrentVertPosition, size not matched!");
		m_x_init_h = X;
		m_x_init_d.upload(m_x_init_h);
		m_shouldRestart = true;
	}

	void GpuSim::getResultClothPieces()
	{
		if (m_clothManager == nullptr)
			return;
		exportResultClothToObjMesh();

		int vbegin = 0;
		int fbegin = 0;
		for (int iCloth = 0; iCloth < m_clothManager->numClothPieces(); iCloth++)
		{
			ObjMesh& mesh = m_clothManager->clothPiece(iCloth)->mesh3d();
			for (size_t iVert = 0; iVert < mesh.vertex_list.size(); iVert++)
			{
				const int oVert = m_vertMerge_in_out_idxMap_h[vbegin + iVert];
				mesh.vertex_list[iVert] = m_resultClothMesh->vertex_list[oVert];
				mesh.vertex_normal_list[iVert] = m_resultClothMesh->vertex_normal_list[oVert];
			} // end for iVert
			for (size_t iFace = 0; iFace < mesh.face_normal_list.size(); iFace++)
			{
				mesh.face_normal_list[iFace] = m_resultClothMesh->face_normal_list[fbegin + iFace];
			} // end for iFace
			mesh.updateBoundingBox();
			mesh.requireRenderUpdate();
			vbegin += mesh.vertex_list.size();
			fbegin += mesh.face_list.size();
		} // end for iCloth
	}

	void GpuSim::clear()
	{
		resetDependency(false);
		m_clothManager = nullptr;
		m_arcSimManager = nullptr;
		m_simParam.setDefault();
		m_fps = 0.f;
		m_curSimulationTime = 0.f;
		m_curStitchRatio = 0.f;
		m_solverInfo = "";
		m_bodyLvSet_h = nullptr;
		m_bodyLvSet_d.release();
		m_resultClothMesh->clear();

		m_bmesh->clear();
		m_bmVerts.clear();
		m_faces_idxWorld_h.clear();
		m_faces_idxWorld_d.release();
		m_faces_idxTex_h.clear();
		m_faces_idxTex_d.release();
		m_faces_idxMat_h.clear();
		m_faces_texStretch_d.release();
		m_faces_texBend_d.release();
		m_faces_materialSpace_d.release();
		m_nodes_materialSpace_d.release();
		m_edgeData_h.clear();
		m_edgeData_d.release();
		m_vert_FaceList_d->clear();
		m_stitch_vertPairs_h.clear();
		m_stitch_vertPairs_d->clear();
		m_stitch_vertMerge_idxMap_h.clear();
		m_vertMerge_in_out_idxMap_h.clear();
		m_stitch_edgeData_h.clear();
		m_stitch_edgeData_d.release();

		m_A_Ids_d.release();
		m_A_Ids_d_unique.release();
		m_A_Ids_d_unique_pos.release();
		m_A_Ids_start_d.release();
		m_A_order_d.release();
		m_A_invOrder_d.release();
		m_beforScan_A.release();
		m_b_Ids_d.release();
		m_b_Ids_d_unique.release();
		m_b_Ids_d_unique_pos.release();
		m_b_Ids_start_d.release();
		m_b_order_d.release();
		m_b_invOrder_d.release();
		m_beforScan_b.release();

		m_A_d->clear();
		m_A_diag_d->clear();
		m_b_d.release();
		m_texCoord_init_h.clear();
		m_texCoord_init_d.release();
		m_x_init_h.clear();
		m_x_init_d.release();
		m_x_h.clear();
		m_x_d.release();
		m_last_x_d.release();
		m_v_d.release();
		m_last_v_d.release();
		m_dv_d.release();

		m_debug_flag = 0;
		m_selfColli_nBuckets = 0;
		m_selfColli_vertIds.release();
		m_selfColli_bucketIds.release();
		m_selfColli_bucketRanges.release();
		m_selfColli_tri_vertCnt.release();
		m_selfColli_tri_vertPair_tId.release();
		m_selfColli_tri_vertPair_vId.release();
		m_nPairs = 0;
		m_stretchSamples_h.clear();
		m_bendingData_h.clear();
		m_densityData_h.clear();
	}

#pragma region -- level set
	void GpuSim::initLevelSet()
	{
		m_shouldLevelsetUpdate = true;
		updateDependency();
		if (m_arcSimManager)
		{
			m_bodyLvSet_h = nullptr;
			if (m_arcSimManager->getSimulator()->obstacles.size() >= 1)
			{
				m_bodyLvSet_h = m_arcSimManager->getSimulator()->obstacles[0].base_objLevelSet.get();
				auto sz = m_bodyLvSet_h->size();
				std::vector<float> transposeLv(m_bodyLvSet_h->sizeXYZ(), 0.f);
				for (int z = 0; z < sz[2]; z++)
				for (int y = 0; y < sz[1]; y++)
				for (int x = 0; x < sz[0]; x++)
					transposeLv[x + y*sz[0] + z*sz[0] * sz[1]] = m_bodyLvSet_h->value(x, y, z)[0];
				m_bodyLvSet_d.fromHost(transposeLv.data(), make_int3(sz[0], sz[1], sz[2]));
				if (m_arcSimManager->getSimulator()->obstacles.size() > 1)
					printf("warning: more than one obstacles given, only 1 used!\n");
			}
		} // end for arcSim
		else if (m_clothManager)
		{
			if (m_clothManager->m_shouldLevelSetUpdate)
				m_clothManager->calcLevelSet();
			m_bodyLvSet_h = m_clothManager->bodyLevelSet();
			m_bodyLvSet_d = m_clothManager->bodyLevelSetDevice();
		} // end for clothManager
		m_shouldLevelsetUpdate = false;
	}
#pragma endregion

#pragma region -- param
	void GpuSim::SimParam::setDefault()
	{
		dt = 1.f / 150.f;
		gravity = Float3(0.f, 0.f, -9.8f);
		pcg_tol = 1e-2f;
		pcg_iter = 400;

		strecth_mult = 1.f;
		bend_mult = 1.f;
		handle_stiffness = 1e3f;
		collision_stiffness = 1e6f;
		friction_stiffness = 1;
		stitch_stiffness = 1e3f;
		repulsion_thickness = 5e-3f;
		projection_thickness = 1e-4f;
		enable_selfCollision = false;
		selfCollision_maxGridSize = 64;
	}

	void GpuSim::initParam()
	{
		m_simParam.setDefault();
		if (m_arcSimManager)
		{
			const auto* sim = m_arcSimManager->getSimulator();
			m_simParam.dt = sim->step_time;
			m_simParam.gravity = convert(sim->gravity);
			m_simParam.stitch_ratio = 99999.f; // we donot have stitch in this mode, thus we set max to skip it.
			m_simParam.enable_selfCollision = true;
		} // end if arc
		else
		{
			auto par = m_clothManager->getSimulationParam();
			m_simParam.gravity = par.gravity;
			m_simParam.stitch_ratio = 4.f;
			m_simParam.strecth_mult = par.spring_k;
			m_simParam.bend_mult = par.bending_k;
			m_simParam.enable_selfCollision = par.enable_self_collistion;
		} // end else clothManager
	}
#pragma endregion

#pragma region -- init topology
	void GpuSim::initFaceEdgeVertArray()
	{
		m_shouldTopologyUpdate = true;
		updateDependency();

		if (m_arcSimManager)
			initFaceEdgeVertArray_arcSim();
		else if (m_clothManager)
			initFaceEdgeVertArray_clothManager();
		else
			throw std::exception("GpuSim, not initialized!");
		initBMesh();
		m_texCoord_init_d.upload(m_texCoord_init_h);
		m_faces_idxWorld_d.upload(m_faces_idxWorld_h);
		m_faces_idxTex_d.upload(m_faces_idxTex_h);
		m_edgeData_d.upload(m_edgeData_h);
		m_x_init_d.upload(m_x_init_h);
		m_x_h = m_x_init_h;
		m_x_init_d.copyTo(m_x_d);
		m_last_x_d.create(m_x_d.size());
		m_v_d.create(m_x_d.size());
		m_last_v_d.create(m_v_d.size());
		m_dv_d.create(m_x_d.size());
		m_b_d.create(m_x_d.size());
		m_project_vw_d.create(m_x_d.size());
		m_stitch_vertMerge_idxMap_h.resize(m_x_init_h.size());
		for (size_t i = 0; i < m_stitch_vertMerge_idxMap_h.size(); i++)
			m_stitch_vertMerge_idxMap_h[i] = i;
		m_vertMerge_in_out_idxMap_h = m_stitch_vertMerge_idxMap_h;

		m_shouldTopologyUpdate = false;
	}
	
	void GpuSim::initFaceEdgeVertArray_arcSim()
	{
		m_faces_idxWorld_h.clear();
		m_faces_idxTex_h.clear();
		m_faces_idxMat_h.clear();
		m_edgeData_h.clear();
		m_x_init_h.clear();
		m_texCoord_init_h.clear();
		auto sim = m_arcSimManager->getSimulator();
		int node_index_begin = 0;
		int tex_index_begin = 0;
		int mat_index_begin = 0;
		int face_index_begin = 0;
		for (size_t iCloth = 0; iCloth < sim->cloths.size(); iCloth++)
		{
			const auto& cloth = sim->cloths[iCloth];
			for (const auto& f : cloth.mesh.faces)
			{
				m_faces_idxWorld_h.push_back(ldp::Int4(f->v[0]->node->index,
					f->v[1]->node->index, f->v[2]->node->index, 0) + node_index_begin);
				m_faces_idxTex_h.push_back(ldp::Int4(f->v[0]->index,
					f->v[1]->index, f->v[2]->index, 0) + tex_index_begin);
				m_faces_idxMat_h.push_back(mat_index_begin + f->label);
			} // end for f
			for (const auto& e : cloth.mesh.edges)
			{
				EdgeData ed;
				ed.edge_idxWorld = ldp::Int4(e->n[0]->index + node_index_begin,
					e->n[1]->index + node_index_begin, -1, -1);
				ed.faceIdx[0] = ed.faceIdx[1] = -1;
				for (int k = 0; k < 2; k++)
				if (e->adjf[k])
				{
					ed.faceIdx[k] = e->adjf[k]->index + face_index_begin;
					ed.edge_idxTex[k] = ldp::Int2(arcsim::edge_vert(e, k, 0)->index,
						arcsim::edge_vert(e, k, 1)->index) + tex_index_begin;
					ed.edge_idxWorld[k + 2] = arcsim::edge_opp_vert(e, k)->node->index + node_index_begin;
					auto t0 = arcsim::edge_vert(e, k, 0)->u;
					auto t1 = arcsim::edge_vert(e, k, 1)->u;
					ed.length_sqr[k] = arcsim::norm2(t1 - t0);
					ed.theta_uv[k] = atan2f(t1[1] - t0[1], t1[0] - t0[0]);
				}
				m_edgeData_h.push_back(ed);
			} // end for e
			for (const auto& n : cloth.mesh.nodes)
				m_x_init_h.push_back(convert(n->x0));
			for (const auto& v : cloth.mesh.verts)
				m_texCoord_init_h.push_back(convert(v->u));
			node_index_begin += cloth.mesh.nodes.size();
			tex_index_begin += cloth.mesh.verts.size();
			mat_index_begin += cloth.materials.size();
			face_index_begin += cloth.mesh.faces.size();
		} // end for iCloth
	}

	inline bool face_edge_same_order(BMesh& bmesh, BMEdge* e, BMFace* f)
	{
		BMVert* v[3] = { nullptr };
		int cnt = 0;
		BMESH_V_OF_F(vtmp, f, v_of_f_iter, bmesh)
		{
			v[cnt++] = vtmp;
			if (cnt >= 3)
				break;
		}
		BMVert* bv[2] = { bmesh.vofe_first(e), bmesh.vofe_last(e) };
		if (v[0] == bv[0] && v[1] == bv[1]
			|| v[1] == bv[0] && v[2] == bv[1]
			|| v[2] == bv[0] && v[0] == bv[1])
			return true;
		else if (v[0] == bv[1] && v[1] == bv[0]
			|| v[1] == bv[1] && v[2] == bv[0]
			|| v[2] == bv[1] && v[0] == bv[0])
			return false;
		throw std::exception("edge not on face!");
	}

	void GpuSim::initFaceEdgeVertArray_clothManager()
	{		
		m_faces_idxWorld_h.clear();
		m_faces_idxTex_h.clear();
		m_faces_idxMat_h.clear();
		m_edgeData_h.clear();
		m_x_init_h.clear();
		m_texCoord_init_h.clear();
		int node_index_begin = 0;
		int tex_index_begin = 0;
		int mat_index_begin = 0;
		int face_index_begin = 0;
		for (int iCloth = 0; iCloth < m_clothManager->numClothPieces(); iCloth++)
		{
			auto cloth = m_clothManager->clothPiece(iCloth);
			for (const auto& f : cloth->mesh3d().face_list)
			{
				m_faces_idxWorld_h.push_back(ldp::Int4(
					f.vertex_index[0] + node_index_begin,
					f.vertex_index[1] + node_index_begin, 
					f.vertex_index[2] + node_index_begin, iCloth));
				m_faces_idxTex_h.push_back(ldp::Int4(f.vertex_index[0],
					f.vertex_index[1], f.vertex_index[2], 0) + tex_index_begin);
				m_faces_idxMat_h.push_back(mat_index_begin); // material set here..
			} // end for f

			BMesh bmesh = *cloth->mesh3d().get_bmesh(false);
			BMESH_ALL_EDGES(e, e_of_m_iter, bmesh)
			{
				const int id0 = bmesh.vofe_first(e)->getIndex();
				const int id1 = bmesh.vofe_last(e)->getIndex();
				EdgeData ed;
				ed.edge_idxWorld = ldp::Int4(id0 + node_index_begin, id1 + node_index_begin, -1, -1);
				ed.faceIdx[0] = ed.faceIdx[1] = -1;
				auto t0 = texCoordFromClothManagerMesh2DCoords(cloth->mesh2d().vertex_list[id0]);
				auto t1 = texCoordFromClothManagerMesh2DCoords(cloth->mesh2d().vertex_list[id1]);
				const float lenSqr = (t0 - t1).sqrLength();
				const float theta = atan2f(t1[1] - t0[1], t1[0] - t0[0]);
				int fcnt = 0;
				BMFace* faces[2] = { nullptr, nullptr };
				BMESH_F_OF_E(f, e, f_of_e_iter, bmesh)
				{
					faces[fcnt++] = f;
					if (fcnt >= 2)
						break;
				}

				// the order is important for the simulation
				if (!face_edge_same_order(bmesh, e, faces[0]))
					std::swap(faces[0], faces[1]);
				for (int k = 0; k < 2; k++)
				if (faces[k])
				{
					ed.faceIdx[k] = faces[k]->getIndex() + face_index_begin;
					ed.edge_idxTex[k] = ldp::Int2(id0, id1) + tex_index_begin;
					BMESH_V_OF_F(v, faces[k], v_of_f_iter, bmesh)
					{
						if (v->getIndex() != id0 && v->getIndex() != id1)
							ed.edge_idxWorld[k + 2] = v->getIndex() + node_index_begin;
					}
					ed.length_sqr[k] = lenSqr;
					ed.theta_uv[k] = theta;
				} // end for k
				m_edgeData_h.push_back(ed);
			} // end for e

			for (const auto& v : cloth->mesh3d().vertex_list)
				m_x_init_h.push_back(v);
			for (const auto& v : cloth->mesh2d().vertex_list)
				m_texCoord_init_h.push_back(Float2(v[0], -v[1]));	// arcsim requires this conversion
			node_index_begin += cloth->mesh3d().vertex_list.size();
			tex_index_begin += cloth->mesh2d().vertex_list.size();
			mat_index_begin += 0;// cloth.materials.size();
			face_index_begin += cloth->mesh3d().face_list.size();
		} // end for iCloth
	}

	void GpuSim::initBMesh()
	{
		std::vector<Int3> flist(m_faces_idxWorld_h.size());
		for (size_t i = 0; i < flist.size(); i++)
		for (int k = 0; k < 3; k++)
			flist[i][k] = m_faces_idxWorld_h[i][k];
		m_bmesh->init_triangles((int)m_x_init_h.size(), (float*)m_x_init_h.data(),
			(int)flist.size(), (int*)flist.data());
		m_bmVerts.clear();
		BMESH_ALL_VERTS(v, v_of_m_iter, *m_bmesh)
		{
			m_bmVerts.push_back(v);
		}
	}
#pragma endregion

#pragma region -- material
	void GpuSim::buildMaterial()
	{
		m_shouldMaterialUpdate = true;
		updateDependency();
		initializeMaterialMemory();
		updateMaterialDataToFaceNode();
		m_shouldMaterialUpdate = false;
	}

	void GpuSim::initializeMaterialMemory()
	{
		m_bendingData_h.clear();
		m_stretchSamples_h.clear();
		m_densityData_h.clear();

		// copy to self cpu
		if (m_arcSimManager)
		{
			for (const auto& cloth : m_arcSimManager->getSimulator()->cloths)
			for (const auto& mat : cloth.materials)
			{
				m_densityData_h.push_back(mat->density);
				m_stretchSamples_h.push_back(StretchingSamples());
				for (int x = 0; x < StretchingSamples::SAMPLES; x++)
				for (int y = 0; y < StretchingSamples::SAMPLES; y++)
				for (int z = 0; z < StretchingSamples::SAMPLES; z++)
					m_stretchSamples_h.back()(x, y, z) = convert(mat->stretching.s[x][y][z]);
				m_bendingData_h.push_back(BendingData());
				auto& bdata = m_bendingData_h.back();
				for (int x = 0; x < bdata.cols(); x++)
				{
					int wrap_x = x;
					if (wrap_x>4)
						wrap_x = 8 - wrap_x;
					if (wrap_x > 2)
						wrap_x = 4 - wrap_x;
					for (int y = 0; y < bdata.rows(); y++)
						bdata(x, y) = mat->bending.d[wrap_x][y];
				}
			} // end for mat, cloth

		} // end if arcSim
		else if (m_clothManager)
		{
			// TO DO: accomplish the importing from cloth manager
			std::shared_ptr<arcsim::Cloth::Material> mat(new arcsim::Cloth::Material);
			arcsim::load_material_data(*mat, g_default_arcsim_material);
			printf("default material: %s\n", g_default_arcsim_material);

			m_densityData_h.push_back(mat->density);
			m_stretchSamples_h.push_back(StretchingSamples());
			for (int x = 0; x < StretchingSamples::SAMPLES; x++)
			for (int y = 0; y < StretchingSamples::SAMPLES; y++)
			for (int z = 0; z < StretchingSamples::SAMPLES; z++)
				m_stretchSamples_h.back()(x, y, z) = convert(mat->stretching.s[x][y][z]);
			m_bendingData_h.push_back(BendingData());

			auto& bdata = m_bendingData_h.back();
			for (int x = 0; x < bdata.cols(); x++)
			{
				int wrap_x = x;
				if (wrap_x>4)
					wrap_x = 8 - wrap_x;
				if (wrap_x > 2)
					wrap_x = 4 - wrap_x;
				for (int y = 0; y < bdata.rows(); y++)
					bdata(x, y) = mat->bending.d[wrap_x][y];
			}
		} // end if clothManager

		// copy to gpu
		for (auto& bd : m_bendingData_h)
		{
			bd.updateHostToDevice();
		} // end for m_bendingData_h

		for (auto& sp : m_stretchSamples_h)
		{
			sp.updateHostToDevice();
		} // end for m_stretchSamples_h

		m_shouldMaterialUpdate = false;
	}

	void GpuSim::updateMaterialDataToFaceNode()
	{
		// stretching forces
		std::vector<cudaTextureObject_t> tmp;
		for (const auto& idx : m_faces_idxMat_h)
			tmp.push_back(m_stretchSamples_h[idx].getCudaArray().getCudaTexture());
		m_faces_texStretch_d.upload(tmp);

		// bending forces
		tmp.clear();
		for (const auto& idx : m_faces_idxMat_h)
			tmp.push_back(m_bendingData_h[idx].getCudaArray().getCudaTexture());
		m_faces_texBend_d.upload(tmp);
		tmp.clear();

		// face material related
		std::vector<FaceMaterailSpaceData> faceData(m_faces_idxTex_h.size());
		for (size_t iFace = 0; iFace < m_faces_idxTex_h.size(); iFace++)
		{
			const auto& f = m_faces_idxTex_h[iFace];
			const auto& t = m_texCoord_init_h;
			const auto& label = m_faces_idxMat_h[iFace];
			FaceMaterailSpaceData& fData = faceData[iFace];
			fData.area = fabs(Float2(t[f[1]] - t[f[0]]).cross(t[f[2]] - t[f[0]])) / 2;
			fData.mass = fData.area * m_densityData_h[label];
		} // end for iFace
		m_faces_materialSpace_d.upload(faceData);

		// node material related
		std::vector<NodeMaterailSpaceData> nodeData(m_x_init_h.size());
		for (size_t iFace = 0; iFace < m_faces_idxTex_h.size(); iFace++)
		{
			const FaceMaterailSpaceData& fData = faceData[iFace];
			const auto& f = m_faces_idxWorld_h[iFace];
			for (int k = 0; k < 3; k++)
			{
				NodeMaterailSpaceData& nData = nodeData[f[k]];
				nData.area += fData.area / 3.f;
				nData.mass += fData.mass / 3.f;
			}
		} // end for iFace
		m_nodes_materialSpace_d.upload(nodeData);
	}
#pragma endregion

#pragma region -- stitch
	void GpuSim::buildStitch()
	{
		m_shouldStitchUpdate = true;
		updateDependency();
		buildStitchVertPairs();
		buildStitchEdges();
		m_shouldStitchUpdate = false;
	}

	void GpuSim::buildStitchVertPairs()
	{
		m_stitch_vertPairs_h.clear();
		m_stitch_vertPairs_d->resize(m_x_init_h.size(), m_x_init_h.size(), 1);
		m_stitch_vertMerge_idxMap_h.clear();
		if (m_clothManager == nullptr)
			return;
		for (int i_stp = 0; i_stp < m_clothManager->numStitches(); i_stp++)
		{
			const auto stp = m_clothManager->getStitchPointPair(i_stp);
			m_stitch_vertPairs_h.push_back(ldp::Int2(stp.first, stp.second));
			m_stitch_vertPairs_h.push_back(ldp::Int2(stp.second, stp.first));
		} // i_stp
		std::sort(m_stitch_vertPairs_h.begin(), m_stitch_vertPairs_h.end());
		m_stitch_vertPairs_h.resize(std::unique(m_stitch_vertPairs_h.begin(),
			m_stitch_vertPairs_h.end()) - m_stitch_vertPairs_h.begin());
		std::vector<int> tmpRow_h, tmpCol_h;
		for (const auto& stp : m_stitch_vertPairs_h)
		{
			tmpRow_h.push_back(stp[0]);
			tmpCol_h.push_back(stp[1]);
		}
		CachedDeviceArray<int> tmpRow_d, tmpCol_d;
		tmpRow_d.fromHost(tmpRow_h);
		tmpCol_d.fromHost(tmpCol_h);
		m_stitch_vertPairs_d->resize(m_x_init_h.size(), m_x_init_h.size(), 1);
		m_stitch_vertPairs_d->setRowFromBooRowPtr(tmpRow_d.data(), tmpRow_d.size());
		cudaSafeCall(cudaMemcpy(m_stitch_vertPairs_d->bsrColIdx(), tmpCol_d.data(),
			tmpCol_d.bytes(), cudaMemcpyDeviceToDevice));

		// ------------------------------------------------------------------------------
		// find idx map that remove all stitched vertices
		m_stitch_vertMerge_idxMap_h.resize(m_x_init_h.size());
		for (size_t i = 0; i < m_stitch_vertMerge_idxMap_h.size(); i++)
			m_stitch_vertMerge_idxMap_h[i] = i;
		for (const auto& stp : m_stitch_vertPairs_h)
		{
			int sm = std::min(stp[0], stp[1]);
			int lg = std::max(stp[0], stp[1]);
			while (m_stitch_vertMerge_idxMap_h[lg] != lg)
				lg = m_stitch_vertMerge_idxMap_h[lg];
			if (lg < sm) std::swap(sm, lg);
			m_stitch_vertMerge_idxMap_h[lg] = sm;
		}
		for (size_t i = 0; i < m_stitch_vertMerge_idxMap_h.size(); i++)
		{
			int m = i;
			while (m_stitch_vertMerge_idxMap_h[m] != m)
				m = m_stitch_vertMerge_idxMap_h[m];
			m_stitch_vertMerge_idxMap_h[i] = m;
		}
	}

	inline bool overlap(Int2 edges[2])
	{
		Int4 idx(edges[0][0], edges[0][1], edges[1][0], edges[1][1]);
		for (int r = 0; r < 4; r++)
		for (int c = r + 1; c < 4; c++)
		if (idx[r] == idx[c])
			return true;
		return false;
	}

	void GpuSim::buildStitchEdges()
	{
		m_stitch_edgeData_h.clear();

		for (size_t idx_stp_i = 0; idx_stp_i < m_stitch_vertPairs_h.size(); idx_stp_i++)
		{
			const Int2 stp_i = m_stitch_vertPairs_h[idx_stp_i];
			for (size_t idx_stp_j = idx_stp_i + 1; idx_stp_j < m_stitch_vertPairs_h.size(); idx_stp_j++)
			{
				const Int2 stp_j = m_stitch_vertPairs_h[idx_stp_j];
				Int2 eIdx[2] = { Int2(stp_i[0], stp_j[0]), Int2(stp_i[1], stp_j[1]) };
				BMEdge* e[2] = { findEdge(eIdx[0][0], eIdx[0][1]), findEdge(eIdx[1][0], eIdx[1][1]) };
				if (e[0] == nullptr || e[1] == nullptr || overlap(eIdx))
					continue;
				if (m_bmesh->fofe_count(e[0]) != 1 || m_bmesh->fofe_count(e[1]) != 1)
					continue;
				BMFace* f[2] = { nullptr, nullptr };
				int op_vid[2] = { 0, 0 };
				for (int k = 0; k < 2; k++)
				{
					eIdx[k][0] = m_bmesh->vofe_first(e[k])->getIndex();
					eIdx[k][1] = m_bmesh->vofe_last(e[k])->getIndex();
					BMIter iter;
					iter.init(e[k]);
					f[k] = m_bmesh->fofe_begin(iter);
					BMESH_V_OF_F(v, f[k], v_of_f_iter, *m_bmesh)
					if (v != m_bmesh->vofe_first(e[k]) && v != m_bmesh->vofe_last(e[k]))
						op_vid[k] = v->getIndex();
				} // end for k
				
				// the order is important for simulation
				if (!face_edge_same_order(*m_bmesh, e[0], f[0]))
				{
					std::swap(f[0], f[1]);
					std::swap(op_vid[0], op_vid[1]);
					std::swap(e[0], e[1]);
					std::swap(eIdx[0], eIdx[1]);
				}

				EdgeData eData;
				eData.edge_idxWorld = Int4(eIdx[0][0], eIdx[0][1], op_vid[0], op_vid[1]);
				for (int k = 0; k < 2; k++)
				{
					eData.faceIdx[k] = f[k]->getIndex();
					eData.edge_idxTex[k] = Int2(eIdx[k][0], eIdx[k][1]);
					const Float2 uv = m_texCoord_init_h[eIdx[k][1]] - m_texCoord_init_h[eIdx[k][0]];
					eData.length_sqr[k] = uv.sqrLength();
					eData.theta_uv[k] = atan2f(uv[1], uv[0]);
				}
				m_stitch_edgeData_h.push_back(eData);
			} // end for idx_stp_j
		} // end for idx_stp_i
		m_stitch_edgeData_d.upload(m_stitch_edgeData_h);
	}
#pragma endregion

#pragma region -- sparse structure
	void GpuSim::setup_sparse_structure()
	{
		m_shouldSparseStructureUpdate = true;
		updateDependency();
		// compute sparse structure via sorting---------------------------------
		const int nVerts = m_x_init_h.size();
		const int nFaces = m_faces_idxWorld_h.size();

		// collect one-ring face list of each vertex
		if (nVerts > 0 && nFaces > 0)
		{
			std::vector<Int2> vert_face_pair_h;
			for (size_t i = 0; i < m_faces_idxWorld_h.size(); i++)
			for (int k = 0; k < 3; k++)
				vert_face_pair_h.push_back(Int2(m_faces_idxWorld_h[i][k], i));
			std::sort(vert_face_pair_h.begin(), vert_face_pair_h.end());
			vert_face_pair_h.resize(std::unique(vert_face_pair_h.begin(),
				vert_face_pair_h.end()) - vert_face_pair_h.begin());
			std::vector<int> vert_face_pair_v_h(vert_face_pair_h.size(), 0);
			std::vector<int> vert_face_pair_f_h(vert_face_pair_h.size(), 0);
			for (size_t i = 0; i < vert_face_pair_h.size(); i++)
			{
				vert_face_pair_v_h[i] = vert_face_pair_h[i][0];
				vert_face_pair_f_h[i] = vert_face_pair_h[i][1];
			}
			CachedDeviceArray<int> vert_face_pair_v_d;
			vert_face_pair_v_d.fromHost(vert_face_pair_v_h);
			m_vert_FaceList_d->resize(nVerts, nFaces, 1);
			m_vert_FaceList_d->setRowFromBooRowPtr(vert_face_pair_v_d.data(), vert_face_pair_v_d.size());
			cudaSafeCall(cudaMemcpy(m_vert_FaceList_d->bsrColIdx(), (const int*)vert_face_pair_f_h.data(),
				vert_face_pair_f_h.size()*sizeof(int), cudaMemcpyHostToDevice));
			cudaSafeCall(cudaThreadSynchronize());
		} // end collect one-ring face list of each vertex

		// ---------------------------------------------------------------------------------------------
		// collect face adjacents
		std::vector<size_t> A_Ids_h;
		std::vector<int> A_Ids_start_h;
		std::vector<int> b_Ids_h;
		std::vector<int> b_Ids_start_h;
		for (const auto& f : m_faces_idxWorld_h)
		{
			A_Ids_start_h.push_back(A_Ids_h.size());
			b_Ids_start_h.push_back(b_Ids_h.size());
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 3; c++)
					A_Ids_h.push_back(ldp::vertPair_to_idx(ldp::Int2(f[r], f[c]), nVerts));
				b_Ids_h.push_back(f[r]);
			}
		}

		// ---------------------------------------------------------------------------------------------
		// collect edge adjacents
		for (const auto& ed : m_edgeData_h)
		{
			A_Ids_start_h.push_back(A_Ids_h.size());
			b_Ids_start_h.push_back(b_Ids_h.size());
			if (ed.faceIdx[0] >= 0 && ed.faceIdx[1] >= 0)
			{
				for (int r = 0; r < 4; r++)
				{
					for (int c = 0; c < 4; c++)
						A_Ids_h.push_back(ldp::vertPair_to_idx(ldp::Int2(
						ed.edge_idxWorld[r], ed.edge_idxWorld[c]), nVerts));
					b_Ids_h.push_back(ed.edge_idxWorld[r]);
				}
			}
		} // end for edgeData

		// ---------------------------------------------------------------------------------------------
		// compute stitch vert pair info
		for (const auto& stp : m_stitch_vertPairs_h)
		{
			A_Ids_start_h.push_back(A_Ids_h.size());
			b_Ids_start_h.push_back(b_Ids_h.size());
			A_Ids_h.push_back(ldp::vertPair_to_idx(Int2(stp[0], stp[0]), nVerts));
			A_Ids_h.push_back(ldp::vertPair_to_idx(Int2(stp[0], stp[1]), nVerts));
			b_Ids_h.push_back(stp[0]);
		} // i_stp

		// ---------------------------------------------------------------------------------------------
		// compute stitch edge info
		for (const auto& ed : m_stitch_edgeData_h)
		{
			A_Ids_start_h.push_back(A_Ids_h.size());
			b_Ids_start_h.push_back(b_Ids_h.size());
			if (ed.faceIdx[0] >= 0 && ed.faceIdx[1] >= 0)
			{
				for (int r = 0; r < 4; r++)
				{
					for (int c = 0; c < 4; c++)
						A_Ids_h.push_back(ldp::vertPair_to_idx(ldp::Int2(
						ed.edge_idxWorld[r], ed.edge_idxWorld[c]), nVerts));
					b_Ids_h.push_back(ed.edge_idxWorld[r]);
				}
			}
		} // end for edgeData

		// ---------------------------------------------------------------------------------------------
		// upload to GPU; make order array, then sort the orders by vertex-pair idx, then unique
		// matrix A
		A_Ids_start_h.push_back(A_Ids_h.size());
		b_Ids_start_h.push_back(b_Ids_h.size());

		m_A_Ids_d.upload(A_Ids_h);
		m_A_Ids_start_d.upload(A_Ids_start_h);
		m_A_order_d.create(A_Ids_h.size());
		m_A_invOrder_d.create(A_Ids_h.size());
		m_A_Ids_d_unique_pos.create(A_Ids_h.size());
		thrust_wrapper::make_counting_array(m_A_invOrder_d.ptr(), m_A_invOrder_d.size());
		thrust_wrapper::make_counting_array(m_A_order_d.ptr(), m_A_order_d.size());
		thrust_wrapper::make_counting_array(m_A_Ids_d_unique_pos.ptr(), m_A_Ids_d_unique_pos.size());
		thrust_wrapper::sort_by_key(m_A_Ids_d.ptr(), m_A_invOrder_d.ptr(), A_Ids_h.size());
		thrust_wrapper::sort_by_key(m_A_invOrder_d.ptr(), m_A_order_d.ptr(), A_Ids_h.size());

		m_A_Ids_d.copyTo(m_A_Ids_d_unique);
		auto nUniqueNnz = thrust_wrapper::unique_by_key(m_A_Ids_d_unique.ptr(), 
			m_A_Ids_d_unique_pos.ptr(), m_A_Ids_d_unique.size());
		m_beforScan_A.create(m_A_order_d.size());

		// rhs b
		m_b_Ids_d.upload(b_Ids_h);
		m_b_Ids_start_d.upload(b_Ids_start_h);
		m_b_order_d.create(b_Ids_h.size());
		m_b_invOrder_d.create(b_Ids_h.size());
		m_b_Ids_d_unique_pos.create(b_Ids_h.size());
		thrust_wrapper::make_counting_array(m_b_invOrder_d.ptr(), m_b_invOrder_d.size());
		thrust_wrapper::make_counting_array(m_b_order_d.ptr(), m_b_order_d.size());
		thrust_wrapper::make_counting_array(m_b_Ids_d_unique_pos.ptr(), m_b_Ids_d_unique_pos.size());
		thrust_wrapper::sort_by_key(m_b_Ids_d.ptr(), m_b_invOrder_d.ptr(), b_Ids_h.size());
		thrust_wrapper::sort_by_key(m_b_invOrder_d.ptr(), m_b_order_d.ptr(), b_Ids_h.size());

		m_b_Ids_d.copyTo(m_b_Ids_d_unique);
		auto nUniqueb = thrust_wrapper::unique_by_key(m_b_Ids_d_unique.ptr(),
			m_b_Ids_d_unique_pos.ptr(), m_b_Ids_d_unique.size());
		m_beforScan_b.create(m_b_order_d.size());
		release_assert(nUniqueb == nVerts);

		// ---------------------------------------------------------------------------------------------
		// convert vertex-pair idx to coo array
		CachedDeviceArray<int> booRow(nUniqueNnz), booCol(nUniqueNnz);
		vertPair_from_idx(booRow.data(), booCol.data(), m_A_Ids_d_unique.ptr(), nVerts, nUniqueNnz);

		// build the sparse matrix via coo
		m_A_d->resize(nVerts, nVerts, 3);
		m_A_d->setRowFromBooRowPtr(booRow.data(), nUniqueNnz);
		cudaSafeCall(cudaMemcpy(m_A_d->bsrColIdx(), booCol.data(), nUniqueNnz*sizeof(int), cudaMemcpyDeviceToDevice));
		cudaSafeCall(cudaMemset(m_A_d->value(), 0, m_A_d->nnz()*sizeof(float)));
		m_A_diag_d->resize(m_A_d->blocksInRow(), m_A_d->rowsPerBlock());

		m_shouldSparseStructureUpdate = false;
	}
#pragma endregion

#pragma region -- dependency
	void GpuSim::updateDependency()
	{
		if (m_shouldLevelsetUpdate)
			m_shouldRestart = true;
		if (m_shouldTopologyUpdate)
		{
			m_shouldStitchUpdate = true;
			m_shouldMaterialUpdate = true;
		}
		if (m_shouldStitchUpdate)
			m_shouldSparseStructureUpdate = true;
		if (m_shouldSparseStructureUpdate)
			m_shouldRestart = true;
		if (m_shouldRestart)
			m_shouldExportMesh = true;
	}

	void GpuSim::resetDependency(bool on)
	{
		m_shouldTopologyUpdate = on;
		m_shouldLevelsetUpdate = on;
		m_shouldMaterialUpdate = on;
		m_shouldStitchUpdate = on;
		m_shouldSparseStructureUpdate = on;
		m_shouldRestart = on;
		m_shouldExportMesh = on;
	}

	void GpuSim::updateSystem()
	{
		updateDependency();
		if (m_shouldTopologyUpdate)
			initFaceEdgeVertArray();
		if (m_shouldLevelsetUpdate)
			initLevelSet();
		if (m_shouldMaterialUpdate)
			buildMaterial();
		if (m_shouldStitchUpdate)
			buildStitch();
		if (m_shouldSparseStructureUpdate)
			setup_sparse_structure();
		if (m_shouldRestart)
			restart();
	}
#pragma endregion

#pragma region --solving
	void GpuSim::linearSolve()
	{
#ifndef SOLVE_USE_EIGEN
		const int nPoint = m_x_d.size();
		const int nVal = nPoint * 3;
		const float const_one = 1.f;
		const float const_one_neg = -1.f;
		const float const_zero = 0.f;

		// all these variables are initialized with 0
		CachedDeviceArray<float> r(nVal);
		CachedDeviceArray<float> z(nVal);
		CachedDeviceArray<float> p(nVal);
		CachedDeviceArray<float> Ap(nVal);
		CachedDeviceArray<float> invD(nVal);
		CachedDeviceArray<float> pcg_orz_rz_pAp(3);

		// norm b
		float norm_b = 0.f;
		cublasCheck(cublasSetPointerMode_v2(m_cublasHandle, CUBLAS_POINTER_MODE_HOST));
		cublasSnrm2_v2(m_cublasHandle, nVal, (float*)m_b_d.ptr(), 1, &norm_b);
		if (norm_b == 0.f)
			return;

		// invD = inv(diag(A))
		pcg_extractInvDiagBlock(*m_A_d, *m_A_diag_d);

		// r = b-Ax
		cudaSafeCall(cudaMemcpy(r.data(), m_b_d.ptr(), r.bytes(), cudaMemcpyDeviceToDevice));
		m_A_d->Mv((float*)m_dv_d.ptr(), r.data(), -1.f, 1.f);

		int iter = 0;
		float err = 0.f;
		for (iter = 0; iter<m_simParam.pcg_iter; iter++)
		{
			// z = invD * r
			m_A_diag_d->Mv(r.data(), z.data());

			// rz = r'*z
			pcg_dot_rz(nVal, r.data(), z.data(), pcg_orz_rz_pAp.data());

			// p = z+beta*p, beta = rz/old_rz
			pcg_update_p(nVal, z.data(), p.data(), pcg_orz_rz_pAp.data());

			// Ap = A*p, pAp = p'*Ap, alpha = rz / pAp
			m_A_d->Mv(p.data(), Ap.data());
			pcg_dot_pAp(nVal, p.data(), Ap.data(), pcg_orz_rz_pAp.data());

			// x = x + alpha*p, r = r - alpha*Ap
			pcg_update_x_r(nVal, p.data(), Ap.data(), (float*)m_dv_d.ptr(), r.data(), pcg_orz_rz_pAp.data());

			// each several iterations, we check the convergence
			if (iter % 10 == 0)
			{
				// Ap = b - A*x
				cudaSafeCall(cudaMemcpy(Ap.data(), m_b_d.ptr(), Ap.bytes(), cudaMemcpyDeviceToDevice));
				m_A_d->Mv((const float*)m_dv_d.ptr(), Ap.data(), -1.f, 1.f);

				float norm_bAx = 0.f;
				cublasCheck(cublasSetPointerMode_v2(m_cublasHandle, CUBLAS_POINTER_MODE_HOST));
				cublasSnrm2_v2(m_cublasHandle, nVal, Ap.data(), 1, &norm_bAx);

				err = norm_bAx / (norm_b + 1e-15f);
				if (err < m_simParam.pcg_tol)
					break;
			}
		} // end for iter
		m_solverInfo += std::string("pcg, iter ") + std::to_string(iter) + ", err " + std::to_string(err);
#else
		SpMat A;
		cudaSpMat_to_EigenMat(*m_A_d, A);
		Eigen::SimplicialCholesky<SpMat> solver(A);
		
		std::vector<ldp::Float3> bvec;
		m_b_d.download(bvec);

		EgVec b(bvec.size() * 3);
		for (int i = 0; i < bvec.size(); i++)
		for (int k = 0; k < 3; k++)
			b[i * 3 + k] = bvec[i][k];
		
		EgVec dv = solver.solve(b);

		std::vector<ldp::Float3> dvvec(dv.size() / 3);
		for (int i = 0; i < dvvec.size(); i++)
		for (int k = 0; k < 3; k++)
			dvvec[i][k] = dv[i * 3 + k];

		m_dv_d.upload(dvvec);
#endif
		update_x_v_by_dv();
		cudaSafeCall(cudaThreadSynchronize());
	}
#pragma endregion

#pragma region -- exportmesh
	void GpuSim::exportResultClothToObjMesh()
	{
		m_shouldExportMesh = true;
		updateDependency();

		ObjMesh& mesh = *m_resultClothMesh;
		mesh.clear();

		if (m_arcSimManager || m_curStitchRatio > 0.f)
		{
			ObjMesh::obj_material mat;
			mesh.material_list.push_back(mat);

			m_vertMerge_in_out_idxMap_h.resize(m_x_init_h.size());
			for (size_t i = 0; i < m_vertMerge_in_out_idxMap_h.size(); i++)
				m_vertMerge_in_out_idxMap_h[i] = i;
			mesh.vertex_list = m_x_h;
			mesh.vertex_texture_list = m_texCoord_init_h;
			for (size_t iFace = 0; iFace < m_faces_idxWorld_h.size(); iFace++)
			{
				ObjMesh::obj_face f;
				f.vertex_count = 3;
				f.material_index = 0;
				for (int k = 0; k < 3; k++)
				{
					f.vertex_index[k] = m_faces_idxWorld_h[iFace][k];
					f.texture_index[k] = m_faces_idxTex_h[iFace][k];
				}
				f.material_index = -1;
				mesh.face_list.push_back(f);
			} // end for iFace
			mesh.updateNormals();
			mesh.updateBoundingBox();
		} // end if arcsim
		else // we merge pices if they have been stiched together
		{
			ObjMesh::obj_material mat_default, mat_sel, mat_high;
			mat_default.diff = Float3(1.f, 1.f, 1.f);
			mat_sel.diff = Float3(0.8f, 0.6f, 0.0f);
			mat_high.diff = Float3(0.0f, 0.6f, 0.8f);
			mesh.material_list.push_back(mat_default);
			mesh.material_list.push_back(mat_sel);
			mesh.material_list.push_back(mat_high);

			m_vertMerge_in_out_idxMap_h = m_stitch_vertMerge_idxMap_h;
			for (size_t id = 0; id < m_x_h.size(); id++)
			{
				if (m_vertMerge_in_out_idxMap_h[id] == id)
				{
					m_vertMerge_in_out_idxMap_h[id] = (int)mesh.vertex_list.size();
					mesh.vertex_list.push_back(m_x_h[id]);
				}
				else
					m_vertMerge_in_out_idxMap_h[id] = m_vertMerge_in_out_idxMap_h[m_vertMerge_in_out_idxMap_h[id]];
				mesh.vertex_texture_list.push_back(m_texCoord_init_h[id]);
			} // end for x

			// write faces
			for (const auto& t : m_faces_idxWorld_h)
			{		
				ObjMesh::obj_face f;
				f.vertex_count = 3;
				f.material_index = -1;
				const auto piece = m_clothManager->clothPiece(t[3]);
				if (piece)
				{
					f.material_index = std::min(0, (int)mesh.material_list.size() - 1);
					if (piece->graphPanel().isHighlighted())
						f.material_index = std::min(2, (int)mesh.material_list.size() - 1);
					else if (piece->graphPanel().isSelected())
						f.material_index = std::min(1, (int)mesh.material_list.size() - 1);
				}
				for (int k = 0; k < t.size(); k++)
					f.vertex_index[k] = m_vertMerge_in_out_idxMap_h[t[k]];
				if (f.vertex_index[0] != f.vertex_index[1] && f.vertex_index[0] != f.vertex_index[2]
					&& f.vertex_index[1] != f.vertex_index[2])
				{
					for (int k = 0; k < t.size(); k++)
						f.texture_index[k] = t[k];
					mesh.face_list.push_back(f);
				}
			} // end for t
			mesh.updateBoundingBox();
			mesh.updateNormals();
		} // end else clothManger

		m_shouldExportMesh = false;
	}
#pragma endregion

#pragma region --helper functions
	BMEdge* GpuSim::findEdge(int v1, int v2)
	{
		if (m_bmVerts.size() == 0)
			throw std::exception("bmesh not initialzed");
		BMVert* bv1 = m_bmVerts[v1];
		BMVert* bv2 = m_bmVerts[v2];
		BMESH_E_OF_V(e, bv1, v1iter, *m_bmesh)
		{
			if (m_bmesh->vofe_first(e) == bv2 || m_bmesh->vofe_last(e) == bv2)
				return e;
		}
		return nullptr;
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray<float>& A, int nTotal)
	{
		std::vector<float> hA;
		A.download(hA);

		int n = A.size();
		if (nTotal >= 0)
			n = nTotal;

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < n; y++)
				fprintf(pFile, "%ef\n", hA[y]);
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray<ldp::Float2>& A, int nTotal)
	{
		std::vector<ldp::Float2> hA;
		A.download(hA);
		int n = A.size();
		if (nTotal >= 0)
			n = nTotal;
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < n; y++)
				fprintf(pFile, "%ef %ef\n", hA[y][0], hA[y][1]);
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray<ldp::Int2>& A, int nTotal)
	{
		std::vector<ldp::Int2> hA;
		A.download(hA);
		int n = A.size();
		if (nTotal >= 0)
			n = nTotal;
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < n; y++)
				fprintf(pFile, "%d %d\n", hA[y][0], hA[y][1]);
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray<int>& A, int nTotal)
	{
		std::vector<int> hA;
		A.download(hA);
		int n = A.size();
		if (nTotal >= 0)
			n = nTotal;
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < n; y++)
				fprintf(pFile, "%d\n", hA[y]);
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec_pair(std::string name, const DeviceArray<size_t>& A, int nVerts, int nTotal)
	{
		std::vector<size_t> hA;
		A.download(hA);
		int n = A.size();
		if (nTotal >= 0)
			n = nTotal;
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < n; y++)
			{
				Int2 v = ldp::vertPair_from_idx(hA[y], nVerts);
				fprintf(pFile, "%d %d\n", v[0], v[1]);
			}
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray<ldp::Float3>& A, int nTotal)
	{
		std::vector<ldp::Float3> hA;
		A.download(hA);
		int n = A.size();
		if (nTotal >= 0)
			n = nTotal;
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < n; y++)
				fprintf(pFile, "%ef %ef %ef\n", hA[y][0], hA[y][1], hA[y][2]);
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray<ldp::Mat3f>& A, int nTotal)
	{
		std::vector<ldp::Mat3f> hA;
		A.download(hA);
		int n = A.size();
		if (nTotal >= 0)
			n = nTotal;
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < n; y++)
			{
				for (int r = 0; r < 3; r++)
				{
					for (int c = 0; c < 3; c++)
						fprintf(pFile, "%ef ", hA[y](r, c));
					fprintf(pFile, "\n");
				}
				fprintf(pFile, "\n");
			}
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray2D<float>& A)
	{
		std::vector<float> hA(A.rows()*A.cols());
		A.download(hA.data(), A.cols()*sizeof(float));

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < A.rows(); y++)
			{
				for (int x = 0; x < A.cols(); x++)
					fprintf(pFile, "%ef ", hA[y*A.cols()+x]);
				fprintf(pFile, "\n");
			}
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpBendDataArray(std::string name, const BendingData& samples)
	{
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < samples.rows(); y++)
			{
				for (int x = 0; x < samples.cols(); x++)
					fprintf(pFile, "%ef ", samples(x, y));
				fprintf(pFile, "\n");
			}
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpStretchSampleArray(std::string name, const StretchingSamples& samples)
	{	
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int z = 0; z < StretchingSamples::SAMPLES; z++)
			{
				for (int y = 0; y < StretchingSamples::SAMPLES; y++)
				{
					for (int x = 0; x < StretchingSamples::SAMPLES; x++)
						fprintf(pFile, "%ef/%ef/%ef/%ef ", samples(x, y, z)[0],
						samples(x, y, z)[1], samples(x, y, z)[2], samples(x, y, z)[3]);
					fprintf(pFile, "\n");
				}
				fprintf(pFile, "\n");
			}
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpEdgeData(std::string name, std::vector<EdgeData>& eDatas)
	{
		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (const auto& eData : eDatas)
			{
				fprintf(pFile, "iw[%d,%d,%d,%d], it[%d,%d;%d,%d], f[%d,%d], L=%ef %ef, theta=%ef %ef\n",
					eData.edge_idxWorld[0], eData.edge_idxWorld[1],
					eData.edge_idxWorld[2], eData.edge_idxWorld[3],
					eData.edge_idxTex[0][0], eData.edge_idxTex[0][1],
					eData.edge_idxTex[1][0], eData.edge_idxTex[1][1],
					eData.faceIdx[0], eData.faceIdx[1],
					eData.length_sqr[0], eData.length_sqr[1],
					eData.theta_uv[0], eData.theta_uv[1]);
			}
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
#pragma endregion
}