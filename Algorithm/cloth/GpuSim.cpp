#include "GpuSim.h"

#include "clothManager.h"
#include "arcsim\ArcSimManager.h"
#include "Renderable\ObjMesh.h"
#include "cloth\LevelSet3D.h"
#include "arcsim\adaptiveCloth\dde.hpp"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\CachedDeviceBuffer.h"

#include <eigen\Dense>
#include <eigen\Sparse>

//#define SOLVE_USE_EIGEN

//#define DEBUG_DUMP
//#define LDP_DEBUG
namespace ldp
{
#pragma region -- utils

	typedef double egreal;
	typedef Eigen::Triplet<egreal> Triplet;
	typedef Eigen::SparseMatrix<egreal> SpMat;
	typedef Eigen::Matrix<egreal, -1, 1> EgVec;

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

	
#pragma endregion

	GpuSim::GpuSim()
	{
		cusparseCheck(cusparseCreate(&m_cusparseHandle), "GpuSim: create cusparse handel");
		cublasCheck(cublasCreate_v2(&m_cublasHandle));
		m_A_d.reset(new CudaBsrMatrix(m_cusparseHandle));
		m_A_diag_d.reset(new CudaDiagBlockMatrix());
		m_vert_FaceList_d.reset(new CudaBsrMatrix(m_cusparseHandle, true));
	}

	GpuSim::~GpuSim()
	{
		releaseMaterialMemory();
		cusparseCheck(cusparseDestroy(m_cusparseHandle));
		cublasCheck(cublasDestroy_v2(m_cublasHandle));
	}

	void GpuSim::SimParam::setDefault()
	{
		dt = 1.f / 200.f;
		gravity = Float3(0.f, 0.f, -9.8f);
		pcg_tol = 1e-6f;

		handle_stiffness = 1e3f;
		collision_stiffness = 1e6f;
		friction_stiffness = 1e0f;
		repulsion_thickness = 5e-3f;
		projection_thickness = 1e-4f;
	}

	void GpuSim::init(ClothManager* clothManager)
	{
		release_assert(ldp::is_float<ClothManager::ValueType>::value);
		m_arcSimManager = nullptr;
		m_clothManager = clothManager;
		NOT_IMPLEMENTED;
	}

	void GpuSim::init(arcsim::ArcSimManager* arcSimManager)
	{
		release_assert(ldp::is_float<ClothManager::ValueType>::value);
		m_clothManager = nullptr;
		m_arcSimManager = arcSimManager;
		m_simParam.setDefault();
		initParam();
		initializeMaterialMemory();
		updateTopology();
		updateNumeric();
#ifdef DEBUG_DUMP
		dumpVec("D:/tmp/m_beforScan_A.txt", m_beforScan_A);
		dumpVec("D:/tmp/m_beforScan_b.txt", m_beforScan_b);
		m_A_d->dump("D:/tmp/m_A.txt");
		dumpVec("D:/tmp/m_b.txt", m_b_d);
		SpMat A;
		cudaSpMat_to_EigenMat(*m_A_d, A);
		ldp::dump("D:/tmp/m_A_eigen.txt", A);
#endif
		restart();
		m_solverInfo = "solver intialized from arcsim";
	}

	void GpuSim::initParam()
	{
		if (m_arcSimManager)
		{
			const auto* sim = m_arcSimManager->getSimulator();
			m_simParam.dt = sim->step_time;
			m_simParam.gravity = convert(sim->gravity);
			m_simParam.pcg_iter = 400;
			m_simParam.pcg_tol = 1e-2f;
			m_simParam.control_mag = 400.f;
			m_simParam.stitch_ratio = 5.f;
			m_simParam.lap_damping_iter = 4;
			m_simParam.air_damping = 0.999f;
			m_simParam.strecth_mult = 1.f;
			m_simParam.bend_mult = 1.f;
		} // end if arc
		else
		{
			NOT_IMPLEMENTED;
		} // end else clothManager
	}

	void GpuSim::run_one_step()
	{
		gtime_t t_start = gtime_now();

		bindTextures();
		updateNumeric();

		m_x_d.copyTo(m_last_x_d);
		m_v_d.copyTo(m_last_v_d);
		cudaSafeCall(cudaMemset(m_dv_d.ptr(), 0, m_dv_d.sizeBytes()));
		// laplacian damping?
		linearSolve();
		collisionSolve();
		userControlSolve();
		m_x_d.download(m_x_h);

		gtime_t t_end = gtime_now();
		m_fps = 1.f / gtime_seconds(t_start, t_end);
	}

	void GpuSim::clothToObjMesh(ObjMesh& mesh)
	{
		mesh.clear();
		mesh.vertex_list = m_x_h;
		mesh.vertex_texture_list = m_texCoord_init_h;
		for (size_t iFace = 0; iFace < m_faces_idxWorld_h.size(); iFace++)
		{
			ObjMesh::obj_face f;
			f.vertex_count = 3;
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
	}

	void GpuSim::restart()
	{
		m_x_h = m_x_init_h;
		m_x_init_d.copyTo(m_x_d);
		m_x_d.copyTo(m_last_x_d);
		cudaSafeCall(cudaMemset(m_v_d.ptr(), 0, m_v_d.sizeBytes()));
		m_v_d.copyTo(m_last_v_d);
		cudaSafeCall(cudaMemset(m_dv_d.ptr(), 0, m_dv_d.sizeBytes()));
		cudaSafeCall(cudaMemset(m_b_d.ptr(), 0, m_b_d.sizeBytes()));
	}

	void GpuSim::updateMaterial()
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

	void GpuSim::updateTopology()
	{
		if (m_arcSimManager)
			updateTopology_arcSim();
		else if (m_clothManager)
			updateTopology_clothManager();
		else
			throw std::exception("GpuSim, not initialized!");

		m_A_diag_d->resize(m_A_d->blocksInRow(), m_A_d->rowsPerBlock());
		m_x_h = m_x_init_h;
		m_x_init_d.copyTo(m_x_d);
		m_last_x_d.create(m_x_d.size());
		m_v_d.create(m_x_d.size());
		m_last_v_d.create(m_v_d.size());
		m_dv_d.create(m_x_d.size());
		m_b_d.create(m_x_d.size());
		cudaSafeCall(cudaMemset(m_v_d.ptr(), 0, m_v_d.sizeBytes()));
		cudaSafeCall(cudaMemset(m_dv_d.ptr(), 0, m_dv_d.sizeBytes()));
		cudaSafeCall(cudaMemset(m_b_d.ptr(), 0, m_b_d.sizeBytes()));
		updateMaterial();
		bindTextures();
	}

	void GpuSim::updateTopology_arcSim()
	{
		// prepare body collision
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

		// prepare triangle faces and edges
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
					ed.length_sqr[k] = arcsim::norm2(t1-t0);
					ed.theta_uv[k] = atan2f(t1[1]-t0[1], t1[0]-t0[0]);
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

		m_x_init_d.upload(m_x_init_h); 
		m_texCoord_init_d.upload(m_texCoord_init_h);
		m_faces_idxWorld_d.upload(m_faces_idxWorld_h);
		m_faces_idxTex_d.upload(m_faces_idxTex_h);
		m_edgeData_d.upload(m_edgeData_h.data(), m_edgeData_h.size());

		// build sparse matrix topology
		setup_sparse_structure_from_cpu();
	}

	void GpuSim::updateTopology_clothManager()
	{		

		NOT_IMPLEMENTED;
	}

	void GpuSim::setup_sparse_structure_from_cpu()
	{
#ifdef DEBUG_DUMP
		// compute sparse structure via eigen
		std::vector<Eigen::Triplet<float>> cooSys;
		for (const auto& f : m_faces_idxWorld_h)
		for (int k = 0; k < 3; k++)
		{
			cooSys.push_back(Eigen::Triplet<float>(f[k], f[k], 0));
			cooSys.push_back(Eigen::Triplet<float>(f[k], f[(k + 1) % 3], 0));
			cooSys.push_back(Eigen::Triplet<float>(f[(k + 1) % 3], f[k], 0));
		}
		for (const auto& ed : m_edgeData_h)
		if (ed.faceIdx[0] >= 0 && ed.faceIdx[1] >= 0)
		{
			const ldp::Int4& f1 = m_faces_idxWorld_h[ed.faceIdx[0]];
			const ldp::Int4& f2 = m_faces_idxWorld_h[ed.faceIdx[1]];
			int v1 = f1[0] + f1[1] + f1[2] - ed.edge_idxWorld[0] - ed.edge_idxWorld[1];
			int v2 = f2[0] + f2[1] + f2[2] - ed.edge_idxWorld[0] - ed.edge_idxWorld[1];
			cooSys.push_back(Eigen::Triplet<float>(v1, v2, 0));
			cooSys.push_back(Eigen::Triplet<float>(v2, v1, 0));
		}

		Eigen::SparseMatrix<float> A;
		A.resize(m_x_init_h.size(), m_x_init_h.size());
		if (!cooSys.empty())
			A.setFromTriplets(cooSys.begin(), cooSys.end());

		m_A_d->resize(A.rows(), A.cols(), 3);
		m_A_d->beginConstructRowPtr();
		cudaSafeCall(cudaMemcpy(m_A_d->bsrRowPtr(), A.outerIndexPtr(),
			(1+A.outerSize())*sizeof(int), cudaMemcpyHostToDevice));
		m_A_d->endConstructRowPtr();
		cudaSafeCall(cudaMemcpy(m_A_d->bsrColIdx(), A.innerIndexPtr(),
			A.nonZeros()*sizeof(float), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemset(m_A_d->value(), 0, m_A_d->nnz()*sizeof(float)));

		m_A_d->dump("D:/tmp/eigen_A.txt");
#endif
		// compute sparse structure via sorting---------------------------------
		const int nVerts = m_x_init_h.size();
		const int nFaces = m_faces_idxWorld_h.size();

		// 0. collect one-ring face list of each vertex
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
#ifdef DEBUG_DUMP
			m_vert_FaceList_d->dump("D:/tmp/vertFace.txt");
#endif
		} // end collect one-ring face list of each vertex

		// 1. collect face adjacents
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

		// 2. collect edge adjacents
		for (const auto& ed : m_edgeData_h)
		{
			A_Ids_start_h.push_back(A_Ids_h.size());
			b_Ids_start_h.push_back(b_Ids_h.size());
			if (ed.faceIdx[0] >= 0 && ed.faceIdx[1] >= 0)
			{
				const ldp::Int4& f1 = m_faces_idxWorld_h[ed.faceIdx[0]];
				const ldp::Int4& f2 = m_faces_idxWorld_h[ed.faceIdx[1]];
				for (int r = 0; r < 4; r++)
				{
					for (int c = 0; c < 4; c++)
						A_Ids_h.push_back(ldp::vertPair_to_idx(ldp::Int2(
						ed.edge_idxWorld[r], ed.edge_idxWorld[c]), nVerts));
					b_Ids_h.push_back(ed.edge_idxWorld[r]);
				}
			}
		} // end for edgeData
		A_Ids_start_h.push_back(A_Ids_h.size());
		b_Ids_start_h.push_back(b_Ids_h.size());

		// 3. upload to GPU; make order array, then sort the orders by vertex-pair idx, then unique

		// matrix A
		m_A_Ids_d.upload(A_Ids_h);
#ifdef DEBUG_DUMP
		dumpVec_pair("D:/tmp/m_A_Ids_d_inOrder.txt", m_A_Ids_d, nVerts);
#endif
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
#ifdef DEBUG_DUMP
		dumpVec("D:/tmp/m_b_Ids_d_inOrder.txt", m_b_Ids_d);
#endif
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
		
		// 4. convert vertex-pair idx to coo array
		CachedDeviceArray<int> booRow(nUniqueNnz), booCol(nUniqueNnz);
		vertPair_from_idx(booRow.data(), booCol.data(), m_A_Ids_d_unique.ptr(), nVerts, nUniqueNnz);

		// 5. build the sparse matrix via coo
		m_A_d->resize(nVerts, nVerts, 3);
		m_A_d->setRowFromBooRowPtr(booRow.data(), nUniqueNnz);
		cudaSafeCall(cudaMemcpy(m_A_d->bsrColIdx(), booCol.data(), nUniqueNnz*sizeof(int), cudaMemcpyDeviceToDevice));
		cudaSafeCall(cudaMemset(m_A_d->value(), 0, m_A_d->nnz()*sizeof(float)));

#ifdef DEBUG_DUMP
		m_A_d->dump("D:/tmp/eigen_A_1.txt");
		dumpVec_pair("D:/tmp/m_A_Ids_d.txt", m_A_Ids_d, nVerts);
		dumpVec_pair("D:/tmp/m_A_Ids_d_unique.txt", m_A_Ids_d_unique, nVerts, nUniqueNnz);
		dumpVec("D:/tmp/m_A_Ids_d_unique_pos.txt", m_A_Ids_d_unique_pos, nUniqueNnz);
		dumpVec("D:/tmp/m_A_order_d.txt", m_A_order_d);
		dumpVec("D:/tmp/m_A_invOrder_d.txt", m_A_invOrder_d);
		dumpVec("D:/tmp/m_b_Ids_d.txt", m_b_Ids_d);
		dumpVec("D:/tmp/m_b_Ids_d_unique.txt", m_b_Ids_d_unique, nUniqueb);
		dumpVec("D:/tmp/m_b_Ids_d_unique_pos.txt", m_b_Ids_d_unique_pos, nUniqueb);
		dumpVec("D:/tmp/m_b_order_d.txt", m_b_order_d);
		dumpVec("D:/tmp/m_b_invOrder_d.txt", m_b_invOrder_d);
#endif
	}

	void GpuSim::initializeMaterialMemory()
	{
		releaseMaterialMemory();

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
		} // end if clothManager
		
		// copy to gpu
		for (auto& bd : m_bendingData_h)
		{
			bd.updateHostToDevice();

#ifdef DEBUG_DUMP
			dumpBendDataArray("D:/tmp/bendData_h.txt", bd);
			bd.updateDeviceToHost();
			dumpBendDataArray("D:/tmp/bendData_d.txt", bd);
			BendingData tmp;
			bd.getCudaArray().copyTo(tmp.getCudaArray());
			tmp.updateDeviceToHost();
			dumpBendDataArray("D:/tmp/bendData_d1.txt", tmp);
#endif

		} // end for m_bendingData_h
		
		for (auto& sp : m_stretchSamples_h)
		{
			sp.updateHostToDevice();

#ifdef DEBUG_DUMP
			dumpStretchSampleArray("D:/tmp/stretchSample_h.txt", sp);
			sp.updateDeviceToHost();
			dumpStretchSampleArray("D:/tmp/stretchSample_d.txt", sp);
			StretchingSamples tmp;
			sp.getCudaArray().copyTo(tmp.getCudaArray());
			tmp.updateDeviceToHost();
			dumpStretchSampleArray("D:/tmp/stretchSample_d1.txt", tmp);
#endif
		} // end for m_stretchSamples_h
	}

	void GpuSim::releaseMaterialMemory()
	{
		m_bendingData_h.clear();
		m_stretchSamples_h.clear();
		m_densityData_h.clear();
	}

	void GpuSim::linearSolve()
	{
#ifndef SOLVE_USE_EIGEN
		const int nPoint = m_x_d.size();
		const int nVal = nPoint * 3;
		const float const_one = 1.f;
		const float const_one_neg = -1.f;
		const float const_zero = 0.f;
		CachedDeviceArray<float> r(nVal);
		CachedDeviceArray<float> z(nVal);
		CachedDeviceArray<float> p(nVal);
		CachedDeviceArray<float> Ap(nVal);
		CachedDeviceArray<float> invD(nVal);
		std::vector<float> tmp1, tmp2;
		tmp1.resize(nVal);
		tmp2.resize(nVal);

		// x = 0
		cudaSafeCall(cudaMemset(m_dv_d.ptr(), 0, m_dv_d.sizeBytes()));
		cudaSafeCall(cudaMemset(p.data(), 0, p.bytes()));

		// norm b
		float norm_b = 0.f;
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
		float rz = 0.f, rz_old = 0.f, pAp = 0.f, alpha = 0.f, beta = 0.f;
		for (iter = 0; iter<m_simParam.pcg_iter; iter++)
		{
			// z = invD * r
			m_A_diag_d->Mv(r.data(), z.data());

			// rz = r'*z
			rz_old = rz;
			rz = pcg_dot(nVal, r.data(), z.data());
			beta = iter == 0 ? 0.f : rz / rz_old;
			if (isinf(beta) || rz == 0.f)
				break;

			// p = z+beta*p
			pcg_update_p(nVal, z.data(), p.data(), beta);

			// Ap = A*p, pAp = p'*Ap, alpha = rz / pAp
			m_A_d->Mv(p.data(), Ap.data());
			pAp = pcg_dot(nVal, p.data(), Ap.data());
			alpha = rz / pAp;
			if (isinf(alpha) || alpha == 0.f)
				break;

			// x = x + alpha*p, r = r - alpha*Ap
			pcg_update_x_r(nVal, p.data(), Ap.data(), (float*)m_dv_d.ptr(), r.data(), alpha);

			// each several iterations, we check the convergence
			if (iter % 10 == 0)
			{
				// Ap = b - A*x
				cudaSafeCall(cudaMemcpy(Ap.data(), m_b_d.ptr(), Ap.bytes(), cudaMemcpyDeviceToDevice));
				m_A_d->Mv((const float*)m_dv_d.ptr(), Ap.data(), -1.f, 1.f);

				float norm_bAx = 0.f;
				cublasSnrm2_v2(m_cublasHandle, nVal, Ap.data(), 1, &norm_bAx);

				err = norm_bAx / (norm_b + 1e-15f);
				if (err < m_simParam.pcg_tol)
					break;
			}
		} // end for iter
		m_solverInfo = std::string("pcg, iter ") + std::to_string(iter) + ", err " + std::to_string(err);
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
}