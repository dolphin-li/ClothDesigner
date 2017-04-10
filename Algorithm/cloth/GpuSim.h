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

namespace ldp
{
	class ClothManager;

	class GpuSim
	{
	public:
		GpuSim();
		~GpuSim();

		// 1. build topology
		// 2. get body level set
		void init(ClothManager* clothManager);

		// perform simulation for one time-step
		void run_one_step();

		// reset cloths to the initial state
		void restart();

		// update cloth topologies
		void updateTopology();

		// simulatio parameter update; resulting in numerical updates
		void updateNumeric();

	protected:
		void linearSolve();
	private:
		ClothManager* m_clothManager = nullptr;

		DeviceArray<int3> m_faces;
		DeviceArray<int4> m_edges_bendedges;

		// ideal dihedral_angle for bending, 0 for planar
		DeviceArray<float> m_edgeTheta_ideals;

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