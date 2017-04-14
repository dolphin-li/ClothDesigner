#pragma once

#include <vector>
#include <memory>
#include "device_array.h"
#include "ldpMat\ldp_basic_vec.h"
#include <map>
#include <set>
#include "definations.h"
#include "graph\AbstractGraphObject.h"
#include "CudaBsrMatrix.h"
#ifndef __CUDACC__
#include <eigen\Dense>
#include <eigen\Sparse>
#endif

namespace arcsim
{
	class ArcSimManager;
}

namespace ldp
{
	class ClothManager;
	class LevelSet3D;
	class GpuSim
	{
	public:
		GpuSim();
		~GpuSim();

		// init from cloth manager
		void init(ClothManager* clothManager);

		// init from arcsim
		void init(arcsim::ArcSimManager* arcSimManager);

		// perform simulation for one time-step
		void run_one_step();

		// reset cloths to the initial state
		void restart();

		// update cloth topologies
		void updateTopology();

		// simulatio parameter update; resulting in numerical updates
		void updateNumeric();

	protected:
		void updateTopology_arcSim();
		void updateTopology_clothManager();
		void linearSolve();

		void edges_bend_from_objmesh(ObjMesh* mesh, std::vector<ldp::Int4>& edges_bend)const;
		void setup_sparse_structure_from_cpu();
	private:
		ClothManager* m_clothManager = nullptr;
		arcsim::ArcSimManager* m_arcSimManager = nullptr;
		cusparseHandle_t m_cusparseHandle = nullptr;

		ldp::LevelSet3D* m_bodyLvSet_h = nullptr;
		DeviceArray<float> m_bodyLvSet_d;

		std::vector<ldp::Int3> m_faces_h;
		DeviceArray<int3> m_faces_d;
		std::vector<ldp::Int4> m_edges_bend_h;
		DeviceArray<int4> m_edges_bend_d;
		std::vector<float> m_edgeTheta_ideals_h;
		DeviceArray<float> m_edgeTheta_ideals_d;

		// solve for the simulation linear system: A*dv=b
		std::shared_ptr<CudaBsrMatrix> m_A;
		DeviceArray<float3> m_b;
		DeviceArray<float2> m_u_init;			// material space vertex texCoord		
		DeviceArray<float3> m_x_init;			// world space vertex position
		DeviceArray<float3> m_x;				// position of current step	
		DeviceArray<float3> m_v;				// velocity of current step
		DeviceArray<float3> m_dv;				// velocity changed in this step
	};
}