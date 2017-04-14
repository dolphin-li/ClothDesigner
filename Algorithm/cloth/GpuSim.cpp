#include "GpuSim.h"

#include "clothManager.h"
#include "arcsim\ArcSimManager.h"
#include "Renderable\ObjMesh.h"
#include "cloth\LevelSet3D.h"
namespace ldp
{
	static void cusparseCheck(cusparseStatus_t st, const char* msg = nullptr)
	{
		if (CUSPARSE_STATUS_SUCCESS != st)
		{
			printf("cusparse error[%d]: %s", st, msg);
			throw std::exception(msg);
		}
	}

	GpuSim::GpuSim()
	{
		cusparseCheck(cusparseCreate(&m_cusparseHandle), "GpuSim: create cusparse handel");
		m_A.reset(new CudaBsrMatrix(m_cusparseHandle));
	}

	GpuSim::~GpuSim()
	{
	}

	void GpuSim::init(ClothManager* clothManager)
	{
		release_assert(ldp::is_float<ClothManager::ValueType>::value);
		m_arcSimManager = nullptr;
		m_clothManager = clothManager;
		m_x_init.upload((const float3*)m_clothManager->m_X.data(), m_clothManager->m_X.size());
		updateTopology();
		updateNumeric();
		restart();
	}

	void GpuSim::init(arcsim::ArcSimManager* arcSimManager)
	{
		m_clothManager = nullptr;
		m_arcSimManager = arcSimManager;
		m_x_init.upload((const float3*)m_arcSimManager->getClothMesh()->vertex_list.data(), 
			m_arcSimManager->getClothMesh()->vertex_list.size());
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
		cudaSafeCall(cudaMemset(m_v.ptr(), 0, m_v.sizeBytes()));
		cudaSafeCall(cudaMemset(m_dv.ptr(), 0, m_dv.sizeBytes()));
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
		m_bodyLvSet_h = m_arcSimManager->getSimulator()->obstacles[0].base_objLevelSet.get();
		m_bodyLvSet_d.upload(m_bodyLvSet_h->value(), m_bodyLvSet_h->sizeXYZ());

		// prepare triangle faces
		m_faces_h.clear();
		for (const auto& f : m_arcSimManager->getClothMesh()->face_list)
		for (int k = 0; k < f.vertex_count - 2; k++)
			m_faces_h.push_back(ldp::Int3(f.vertex_index[0], f.vertex_index[k + 1], f.vertex_index[k + 2]));
		m_faces_d.upload((const int3*)m_faces_h.data(), m_faces_h.size());

		// prepare edges with bending edges
		edges_bend_from_objmesh(m_arcSimManager->getClothMesh(), m_edges_bend_h);
		m_edges_bend_d.upload((const int4*)m_edges_bend_h.data(), m_edges_bend_h.size());
		m_edgeTheta_ideals_h.clear();
		m_edgeTheta_ideals_h.resize(m_edges_bend_h.size(), 0);
		m_edgeTheta_ideals_d.upload(m_edgeTheta_ideals_h);

		// build sparse matrix topology
		setup_sparse_structure_from_cpu();
	}

	void GpuSim::updateTopology_clothManager()
	{		
		// prepare body collision
		m_bodyLvSet_h = m_clothManager->bodyLevelSet();
		m_bodyLvSet_d.upload(m_bodyLvSet_h->value(), m_bodyLvSet_h->sizeXYZ());

		// prepare triangle faces
		m_faces_h = m_clothManager->m_T;
		m_faces_d.upload((const int3*)m_faces_h.data(), m_faces_h.size());

		// prepare edges with bending edges
		m_edges_bend_h = m_clothManager->m_edgeWithBendEdge;
		m_edges_bend_d.upload((const int4*)m_edges_bend_h.data(), m_edges_bend_h.size());
		m_edgeTheta_ideals_h.clear();
		m_edgeTheta_ideals_h.resize(m_edges_bend_h.size(), 0);
		m_edgeTheta_ideals_d.upload(m_edgeTheta_ideals_h);

		// build sparse matrix topology
		setup_sparse_structure_from_cpu();
	}

	void GpuSim::edges_bend_from_objmesh(ObjMesh* mesh, std::vector<ldp::Int4>& edges_bend)const
	{
		edges_bend.clear();
		auto cloth_bmesh = mesh->get_bmesh(false);
		BMESH_ALL_EDGES(e, eiter, *cloth_bmesh)
		{
			ldp::Int2 vori(cloth_bmesh->vofe_first(e)->getIndex(),
				cloth_bmesh->vofe_last(e)->getIndex());
			Int2 vbend = -1;
			int vcnt = 0;
			BMESH_F_OF_E(f, e, fiter, *cloth_bmesh)
			{
				int vsum = 0;
				BMESH_V_OF_F(v, f, viter, *cloth_bmesh)
				{
					vsum += v->getIndex();
				}
				vsum -= vori[0] + vori[1];
				vbend[vcnt++] = vsum;
				if (vcnt >= vbend.size())
					break;
			} // end for fiter
			edges_bend.push_back(Int4(vori[0], vori[1], vbend[0], vbend[1]));
		} // end for all edges
	}

	void GpuSim::setup_sparse_structure_from_cpu()
	{
		std::vector<Eigen::Triplet<float>> cooSys;
		for (const auto& f : m_faces_h)
		for (int k = 0; k < 3; k++)
		{
			cooSys.push_back(Eigen::Triplet<float>(f[k], f[(k + 1) % 3], 0));
			cooSys.push_back(Eigen::Triplet<float>(f[(k + 1) % 3], f[k], 0));
		}
		for (const auto& f : m_edges_bend_h)
		if (f[2] >= 0 && f[3] >= 0)
		{
			cooSys.push_back(Eigen::Triplet<float>(f[2], f[3], 0));
			cooSys.push_back(Eigen::Triplet<float>(f[3], f[2], 0));
		}

		Eigen::SparseMatrix<float> A;
		A.resize(m_x_init.size(), m_x_init.size());
		if (!cooSys.empty())
			A.setFromTriplets(cooSys.begin(), cooSys.end());

		m_A->resize(A.rows(), A.cols(), 3);
		m_A->beginConstructRowPtr();
		cudaSafeCall(cudaMemcpy(m_A->bsrRowPtr(), A.outerIndexPtr(),
			A.outerSize()*sizeof(float), cudaMemcpyHostToDevice));
		m_A->endConstructRowPtr();
		cudaSafeCall(cudaMemcpy(m_A->bsrColIdx(), A.innerIndexPtr(),
			A.innerSize()*sizeof(float), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemset(m_A->value(), 0, m_A->nnz()*sizeof(float)));
	}
}