#include "GpuSim.h"

#include "clothManager.h"
#include "arcsim\ArcSimManager.h"
#include "Renderable\ObjMesh.h"
#include "cloth\LevelSet3D.h"
#include "arcsim\adaptiveCloth\dde.hpp"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\CachedDeviceBuffer.h"

#define DEBUG_DUMP
#define LDP_DEBUG
namespace ldp
{
#pragma region -- utils
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

#define NOT_IMPLEMENTED throw std::exception((std::string("NotImplemented[")+__FILE__+"]["\
	+std::to_string(__LINE__)+"]").c_str())
#pragma endregion

	GpuSim::GpuSim()
	{
		cusparseCheck(cusparseCreate(&m_cusparseHandle), "GpuSim: create cusparse handel");
		m_A.reset(new CudaBsrMatrix(m_cusparseHandle));
	}

	GpuSim::~GpuSim()
	{
		releaseMaterialMemory();
		cusparseCheck(cusparseDestroy(m_cusparseHandle));
	}

	void GpuSim::SimParam::setDefault()
	{
		dt = 1.f / 200.f;
		gravity = Float3(0.f, 0.f, -9.8f);
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
		m_A->dump("D:/tmp/m_A.txt");
#endif
		restart();
	}

	void GpuSim::initParam()
	{
		if (m_arcSimManager)
		{
			const auto* sim = m_arcSimManager->getSimulator();
			m_simParam.dt = sim->step_time;
		} // end if arc
		else
		{
			NOT_IMPLEMENTED;
		} // end else clothManager
	}

	void GpuSim::run_one_step()
	{

	}

	void GpuSim::restart()
	{
		m_x_init_d.copyTo(m_x);
		cudaSafeCall(cudaMemset(m_v.ptr(), 0, m_v.sizeBytes()));
		cudaSafeCall(cudaMemset(m_dv.ptr(), 0, m_dv.sizeBytes()));
		cudaSafeCall(cudaMemset(m_b.ptr(), 0, m_b.sizeBytes()));
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
			fData.area = fabs(Float2(t[f[1]] - t[f[0]]).cross(t[f[2]] - t[f[0]])) * 0.5f;
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

		m_x_init_d.copyTo(m_x);
		m_v.create(m_x.size());
		m_dv.create(m_x.size());
		m_b.create(m_x.size());
		updateMaterial();
		bindTextures();
	}

	void GpuSim::linearSolve()
	{

	}

	void GpuSim::updateTopology_arcSim()
	{
		// prepare body collision
		m_bodyLvSet_h = nullptr;
		if (m_arcSimManager->getSimulator()->obstacles.size() >= 1)
		{
			m_bodyLvSet_h = m_arcSimManager->getSimulator()->obstacles[0].base_objLevelSet.get();
			m_bodyLvSet_d.upload(m_bodyLvSet_h->value(), m_bodyLvSet_h->sizeXYZ());
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
					ed.length_sqr = arcsim::norm2(t1-t0);
					ed.theta_uv = atan2f(t1[1]-t0[1], t1[0]-t0[0]);
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

		m_A->resize(A.rows(), A.cols(), 3);
		m_A->beginConstructRowPtr();
		cudaSafeCall(cudaMemcpy(m_A->bsrRowPtr(), A.outerIndexPtr(),
			(1+A.outerSize())*sizeof(int), cudaMemcpyHostToDevice));
		m_A->endConstructRowPtr();
		cudaSafeCall(cudaMemcpy(m_A->bsrColIdx(), A.innerIndexPtr(),
			A.nonZeros()*sizeof(float), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemset(m_A->value(), 0, m_A->nnz()*sizeof(float)));

		m_A->dump("D:/tmp/eigen_A.txt");
#endif
		// compute sparse structure via sorting---------------------------------
		const int nVerts = m_x_init_h.size();

		// 1. collect face adjacents
		std::vector<size_t> A_Ids_h;
		std::vector<int> A_Ids_start_h;
		std::vector<int> b_Ids_h;
		std::vector<int> b_Ids_start_h;
		for (const auto& f : m_faces_idxWorld_h)
		{
			A_Ids_start_h.push_back(A_Ids_h.size());
			b_Ids_start_h.push_back(b_Ids_h.size());
			for (int k = 0; k < 3; k++)
			{
				A_Ids_h.push_back(ldp::vertPair_to_idx(ldp::Int2(f[k], f[k]), nVerts));
				A_Ids_h.push_back(ldp::vertPair_to_idx(ldp::Int2(f[k], f[(k + 1) % 3]), nVerts));
				A_Ids_h.push_back(ldp::vertPair_to_idx(ldp::Int2(f[(k + 1) % 3], f[k]), nVerts));
				b_Ids_h.push_back(f[k]);
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
				int v[4] = { 0 };
				v[0] = ed.edge_idxWorld[0];
				v[1] = ed.edge_idxWorld[1];
				v[2] = f1[0] + f1[1] + f1[2] - ed.edge_idxWorld[0] - ed.edge_idxWorld[1];
				v[3] = f2[0] + f2[1] + f2[2] - ed.edge_idxWorld[0] - ed.edge_idxWorld[1];
				for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
						A_Ids_h.push_back(ldp::vertPair_to_idx(ldp::Int2(v[i], v[j]), nVerts));
					b_Ids_h.push_back(v[i]);
				}
			} // end for edgeData
		}
		A_Ids_start_h.push_back(A_Ids_h.size());
		b_Ids_start_h.push_back(b_Ids_h.size());

		// 3. upload to GPU; make order array, then sort the orders by vertex-pair idx, then unique

		// matrix A
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
		
		// 4. convert vertex-pair idx to coo array
		CachedDeviceBuffer booRow(nUniqueNnz*sizeof(int));
		CachedDeviceBuffer booCol(nUniqueNnz*sizeof(int));
		vertPair_from_idx((int*)booRow.data(), (int*)booCol.data(), m_A_Ids_d_unique.ptr(), nVerts, nUniqueNnz);

		// 5. build the sparse matrix via coo
		m_A->resize(nVerts, nVerts, 3);
		m_A->setRowFromBooRowPtr((const int*)booRow.data(), nUniqueNnz);
		cudaSafeCall(cudaMemcpy(m_A->bsrColIdx(), (const int*)booCol.data(),
			nUniqueNnz*sizeof(int), cudaMemcpyDeviceToDevice));
		cudaSafeCall(cudaMemset(m_A->value(), 0, m_A->nnz()*sizeof(float)));

#ifdef DEBUG_DUMP
		m_A->dump("D:/tmp/eigen_A_1.txt");
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
#ifdef BEND_USE_LINEAR_TEX
				{
					bdata(x, 0) = mat->bending.d[x][0];
					bdata(x, 1) = mat->bending.d[x][1];
					bdata(x, 2) = mat->bending.d[x][2];
					bdata(x, 3) = mat->bending.d[x][1];
					bdata(x, 4) = mat->bending.d[x][0];
					bdata(x, 5) = mat->bending.d[x][1];
					bdata(x, 6) = mat->bending.d[x][2];
					bdata(x, 7) = mat->bending.d[x][1];
					bdata(x, 8) = mat->bending.d[x][0];
				}
#else
				for (int y = 0; y < bdata.rows(); y++)
					bdata(x, y) = mat->bending.d[x][y];
#endif
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