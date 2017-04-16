#include "GpuSim.h"

#include "clothManager.h"
#include "arcsim\ArcSimManager.h"
#include "Renderable\ObjMesh.h"
#include "cloth\LevelSet3D.h"
#include "arcsim\adaptiveCloth\dde.hpp"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\CachedDeviceBuffer.h"
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

	void GpuSim::init(ClothManager* clothManager)
	{
		release_assert(ldp::is_float<ClothManager::ValueType>::value);
		m_arcSimManager = nullptr;
		m_clothManager = clothManager;
	}

	void GpuSim::init(arcsim::ArcSimManager* arcSimManager)
	{
		m_clothManager = nullptr;
		m_arcSimManager = arcSimManager;
		initializeMaterialMemory();
		updateMaterial();
		updateTopology();
		updateNumeric();
		restart();
	}

	void GpuSim::run_one_step()
	{

	}

	void GpuSim::restart()
	{
		m_x_init.copyTo(m_x);
		m_v.create(m_x.size());
		m_dv.create(m_x.size());
		m_b.create(m_x.size());
		cudaSafeCall(cudaMemset(m_v.ptr(), 0, m_v.sizeBytes()));
		cudaSafeCall(cudaMemset(m_dv.ptr(), 0, m_dv.sizeBytes()));
		cudaSafeCall(cudaMemset(m_b.ptr(), 0, m_b.sizeBytes()));
	}

	void GpuSim::updateMaterial()
	{
		std::vector<cudaTextureObject_t> tmp;
		for (const auto& idx : m_faces_idxMat_h)
			tmp.push_back(m_stretchSamples_h[idx].getCudaArray().getCudaTexture());
		m_faces_texStretch_d.upload(tmp);

		tmp.clear();
		for (const auto& idx : m_faces_idxMat_h)
			tmp.push_back(m_bendingData_h[idx].getCudaArray().getCudaTexture());
		m_faces_texBend_d.upload(tmp);
	}

	void GpuSim::updateTopology()
	{
		if (m_arcSimManager)
			updateTopology_arcSim();
		else if (m_clothManager)
			updateTopology_clothManager();
		else
			throw std::exception("GpuSim, not initialized!");
	}

	void GpuSim::updateNumeric()
	{
		
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
		std::vector<ldp::Float3> tmp_x_init;
		std::vector<ldp::Float2> tmp_texCoord_init;
		auto sim = m_arcSimManager->getSimulator();
		int node_index_begin = 0;
		int tex_index_begin = 0;
		int mat_index_begin = 0;
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
				ed.edge_idxWorld[0] = e->n[0]->index + node_index_begin;
				ed.edge_idxWorld[1] = e->n[1]->index + node_index_begin;
				ed.faceIdx[0] = ed.faceIdx[1] = -1;
				if (e->adjf[0])
					ed.faceIdx[0] = e->adjf[0]->index + tex_index_begin;
				if (e->adjf[1])
					ed.faceIdx[1] = e->adjf[1]->index + tex_index_begin;
				m_edgeData_h.push_back(ed);
			} // end for e
			for (const auto& n : cloth.mesh.nodes)
				tmp_x_init.push_back(convert(n->x0));
			for (const auto& v : cloth.mesh.verts)
				tmp_texCoord_init.push_back(convert(v->u));
			node_index_begin += cloth.mesh.nodes.size();
			tex_index_begin += cloth.mesh.verts.size();
			mat_index_begin += cloth.materials.size();
		} // end for iCloth
		m_x_init.upload(tmp_x_init); 
		m_texCoord_init.upload(tmp_texCoord_init);
		m_faces_idxTex_d.upload(m_faces_idxTex_h);
		m_edgeData_d.upload(m_edgeData_h.data(), m_edgeData_h.size());
		m_edgeThetaIdeals_h.clear();
		m_edgeThetaIdeals_h.resize(m_edgeData_h.size(), 0);
		m_edgeThetaIdeals_d.upload(m_edgeThetaIdeals_h);

		// build sparse matrix topology
		setup_sparse_structure_from_cpu();
	}

	void GpuSim::updateTopology_clothManager()
	{		

	}

	void GpuSim::setup_sparse_structure_from_cpu()
	{
#if 0
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
		A.resize(m_x_init.size(), m_x_init.size());
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
#else
		// compute sparse structure via sorting---------------------------------
		const int nVerts = m_x_init.size();

		// 1. collect face adjacents
		m_faceEdge_vertIds_h.clear();
		for (const auto& f : m_faces_idxWorld_h)
		for (int k = 0; k < 3; k++)
		{
			m_faceEdge_vertIds_h.push_back(ldp::vertPair_to_idx(ldp::Int2(f[k], f[k]), nVerts));
			m_faceEdge_vertIds_h.push_back(ldp::vertPair_to_idx(ldp::Int2(f[k], f[(k + 1) % 3]), nVerts));
			m_faceEdge_vertIds_h.push_back(ldp::vertPair_to_idx(ldp::Int2(f[(k + 1) % 3], f[k]), nVerts));
		}

		// 2. collect edge adjacents
		for (const auto& ed : m_edgeData_h)
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
			for (int j = 0; j < 4; j++)
				m_faceEdge_vertIds_h.push_back(ldp::vertPair_to_idx(ldp::Int2(v[i], v[j]), nVerts));
		} // end for edgeData

		// 3. upload to GPU
		m_faceEdge_vertIds_d.upload(m_faceEdge_vertIds_h);

		// 4. make order array
		m_faceEdge_order_d.create(m_faceEdge_vertIds_h.size());
		thrust_wrapper::make_counting_array(m_faceEdge_order_d.ptr(), m_faceEdge_order_d.size());

		// 5. sort the order array by vertex-pair idx, then unique
		thrust_wrapper::sort_by_key(m_faceEdge_vertIds_d.ptr(), m_faceEdge_order_d.ptr(), m_faceEdge_vertIds_d.size());
		auto nUniqueNnz = thrust_wrapper::unique(m_faceEdge_vertIds_d.ptr(), m_faceEdge_vertIds_d.size());
		
		// 6. convert vertex-pair idx to coo array
		CachedDeviceBuffer booRow(nUniqueNnz*sizeof(int));
		CachedDeviceBuffer booCol(nUniqueNnz*sizeof(int));
		vertPair_from_idx((int*)booRow.data(), (int*)booCol.data(), 
			m_faceEdge_vertIds_d.ptr(), nVerts, nUniqueNnz);

		// 7. build the sparse matrix via coo
		m_A->resize(nVerts, nVerts, 3);
		m_A->setRowFromBooRowPtr((const int*)booRow.data(), nUniqueNnz);
		cudaSafeCall(cudaMemcpy(m_A->bsrColIdx(), (const int*)booCol.data(),
			nUniqueNnz*sizeof(int), cudaMemcpyDeviceToDevice));
		cudaSafeCall(cudaMemset(m_A->value(), 0, m_A->nnz()*sizeof(float)));
		//m_A->dump("D:/tmp/eigen_A_1.txt");
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
				m_stretchSamples_h.push_back(StretchingSamples());
				for (int x = 0; x < StretchingSamples::SAMPLES; x++)
				for (int y = 0; y < StretchingSamples::SAMPLES; y++)
				for (int z = 0; z < StretchingSamples::SAMPLES; z++)
					m_stretchSamples_h.back()(x, y, z) = convert(mat->stretching.s[x][y][z]);
				m_bendingData_h.push_back(BendingData());
				auto& bdata = m_bendingData_h.back();
				for (int x = 0; x < bdata.cols(); x++)
				for (int y = 0; y < bdata.rows(); y++)
					bdata(x, y) = mat->bending.d[x][y];
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

			//dumpBendDataArray("D:/tmp/bendData_h.txt", bd);
			//bd.updateDeviceToHost();
			//dumpBendDataArray("D:/tmp/bendData_d.txt", bd);
			//BendingData tmp;
			//bd.getCudaArray().copyTo(tmp.getCudaArray());
			//tmp.updateDeviceToHost();
			//dumpBendDataArray("D:/tmp/bendData_d1.txt", tmp);

		} // end for m_bendingData_h
		
		for (auto& sp : m_stretchSamples_h)
		{
			sp.updateHostToDevice();

			//dumpStretchSampleArray("D:/tmp/stretchSample_h.txt", sp);
			//sp.updateDeviceToHost();
			//dumpStretchSampleArray("D:/tmp/stretchSample_d.txt", sp);
			//StretchingSamples tmp;
			//sp.getCudaArray().copyTo(tmp.getCudaArray());
			//tmp.updateDeviceToHost();
			//dumpStretchSampleArray("D:/tmp/stretchSample_d1.txt", tmp);
		} // end for m_stretchSamples_h
	}

	void GpuSim::releaseMaterialMemory()
	{
		m_bendingData_h.clear();
		m_stretchSamples_h.clear();
	}

	void GpuSim::dumpVec(std::string name, const DeviceArray<float>& A)
	{
		std::vector<float> hA;
		A.download(hA);

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < A.size(); y++)
				fprintf(pFile, "%ef\n", hA[y]);
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray<ldp::Float2>& A)
	{
		std::vector<ldp::Float2> hA;
		A.download(hA);

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < A.size(); y++)
				fprintf(pFile, "%ef %ef\n", hA[y][0], hA[y][1]);
			fclose(pFile);
			printf("saved: %s\n", name.c_str());
		}
	}
	void GpuSim::dumpVec(std::string name, const DeviceArray<ldp::Float3>& A)
	{
		std::vector<ldp::Float3> hA;
		A.download(hA);

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < A.size(); y++)
				fprintf(pFile, "%ef %ef %ef\n", hA[y][0], hA[y][1], hA[y][2]);
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