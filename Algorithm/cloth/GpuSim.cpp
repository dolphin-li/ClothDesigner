#include "GpuSim.h"

#include "clothManager.h"

namespace ldp
{
	GpuSim::GpuSim()
	{
	}

	GpuSim::~GpuSim()
	{
	}

	void GpuSim::init(ClothManager* clothManager)
	{
		release_assert(ldp::is_float<ClothManager::ValueType>::value);
		m_clothManager = clothManager;
		m_x_init.upload((const float3*)m_clothManager->m_X.data(), m_clothManager->m_X.size());
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
		m_faces.upload((const int3*)m_clothManager->m_T.data(), m_clothManager->m_T.size());
		m_edges_bendedges.upload((const int4*)m_clothManager->m_edgeWithBendEdge.data(),
			m_clothManager->m_edgeWithBendEdge.size());
		m_edgeTheta_ideals.create(m_edges_bendedges.size());
		cudaSafeCall(cudaMemset(m_edgeTheta_ideals.ptr(), 0, m_edgeTheta_ideals.sizeBytes()));

		std::vector<Eigen::Triplet<float>> cooSys;
	}

	void GpuSim::updateNumeric()
	{
		
	}

	void GpuSim::linearSolve()
	{

	}
}