#include "clothManager.h"
#include "LevelSet3D.h"
#include "LEVEL_SET_COLLISION.h"
#include "cudpp\cuda_utils.h"
#include "cudpp\helper_math.h"

//#include "MY_MATH.h"
//#include "COLLISION_HANDLER.h"
#include "SelfCollider.h"

namespace ldp
{
	enum
	{
		threadsPerBlock = 256
	};

	__constant__ float g_gravity[3] = { 0, -9.8, 0 };

#pragma region --laplacian damping
	__global__ void Laplacian_Damping_Kernel(const float* V, float* next_V, const float* fixed, 
		const float* more_fixed, const int* all_VV, const int* all_vv_num, const int number, const float rate)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		float3 nvi = make_float3(0, 0, 0);
		float3 vi = make_float3(V[i * 3 + 0], V[i * 3 + 1], V[i * 3 + 2]);
		float r = (more_fixed[i] == 0) * (fixed[i] == 0);
		const int ed = all_vv_num[i + 1];
		for (int index = all_vv_num[i]; index<ed; index++)
		{
			int j = all_VV[index];
			float3 vj = make_float3(V[j * 3 + 0], V[j * 3 + 1], V[j * 3 + 2]);
			nvi += (vj - vi) * r;
		}
		nvi = (vi + nvi * rate) * r;
		next_V[i * 3 + 0] = nvi.x;
		next_V[i * 3 + 1] = nvi.y;
		next_V[i * 3 + 2] = nvi.z;

	}
	void ClothManager::laplaceDamping()
	{
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		for (int l = 0; l<m_simulationParam.lap_damping; l++)
		{
			Laplacian_Damping_Kernel << <blocksPerGrid, threadsPerBlock >> >(
				m_dev_V.ptr(), m_dev_next_X.ptr(), m_dev_fixed.ptr(), 
				m_dev_more_fixed.ptr(), m_dev_all_VV.ptr(), m_dev_all_vv_num.ptr(), 
				m_X.size(), 0.1);
			m_dev_next_X.copyTo(m_dev_V);
		}
	}
#pragma endregion

#pragma region --update
	__global__ void Update_Kernel(float* X, float* V, float* fixed, const float* more_fixed, 
		const float damping, const float t, const int number, const float dir_x, 
		const float dir_y, const float dir_z)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		if (more_fixed[i] != 0)
		{
			X[i * 3 + 0] += dir_x;
			X[i * 3 + 1] += dir_y;
			X[i * 3 + 2] += dir_z;
			V[i * 3 + 0] = 0;
			V[i * 3 + 1] = 0;
			V[i * 3 + 2] = 0;
			return;
		}

		if (fixed[i] != 0)
		{
			V[i * 3 + 0] = 0;
			V[i * 3 + 1] = 0;
			V[i * 3 + 2] = 0;
			return;
		}

		//Apply damping
		V[i * 3 + 0] *= damping;
		V[i * 3 + 1] *= damping;
		V[i * 3 + 2] *= damping;

		//Apply gravity
		V[i * 3 + 0] += g_gravity[0] * t;
		V[i * 3 + 1] += g_gravity[1] * t;
		V[i * 3 + 2] += g_gravity[2] * t;

		//Position update
		X[i * 3 + 0] = X[i * 3 + 0] + V[i * 3 + 0] * t;
		X[i * 3 + 1] = X[i * 3 + 1] + V[i * 3 + 1] * t;
		X[i * 3 + 2] = X[i * 3 + 2] + V[i * 3 + 2] * t;
	}

	void ClothManager::updateAfterLap()
	{
		// ldp hack here: make the gravity not important when we are stitching.
		Float3 gravity = m_simulationParam.gravity * powf(1 - std::max(0.f, std::min(1.f, m_curStitchRatio)), 2);

		cudaMemcpyToSymbol(g_gravity, gravity.ptr(), 3 * sizeof(float));
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		Update_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_X.ptr(), m_dev_V.ptr(), m_dev_fixed.ptr(), m_dev_more_fixed.ptr(), 
			m_simulationParam.air_damping, m_simulationParam.time_step, m_X.size(), 
			m_curDragInfo.dir[0], m_curDragInfo.dir[1], m_curDragInfo.dir[2]);
	}
#pragma endregion

#pragma region --constrain0
	__global__ void Constraint_0_Kernel(const float* X, float* init_B, float* new_VC, 
		const float *fixed, const float* more_fixed, const float inv_t, const int number)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		float c = (1 + fixed[i] + more_fixed[i])*inv_t*inv_t;
		init_B[i * 3 + 0] = c*X[i * 3 + 0];
		init_B[i * 3 + 1] = c*X[i * 3 + 1];
		init_B[i * 3 + 2] = c*X[i * 3 + 2];
		new_VC[i] += c;
	}

	void ClothManager::constrain0()
	{
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		m_dev_all_VC.copyTo(m_dev_new_VC);
		Constraint_0_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_X.ptr(), m_dev_init_B.ptr(), m_dev_new_VC.ptr(), m_dev_fixed.ptr(), 
			m_dev_more_fixed.ptr(), 1 / m_simulationParam.time_step, m_X.size());
		cudaSafeCall(cudaGetLastError(), "constrain0");
		m_dev_X.copyTo(m_dev_prev_X);
	}
#pragma endregion
	
#pragma region --constrain1
__global__ void Constraint_1_Kernel(const float* X, const float* init_B,
		float* next_X, const int* all_VV, const float* all_VL, const float* all_VW, 
		const float* new_VC, const int* all_vv_num, const float spring_k, const int number,
		const int* stitch_VV, const float* stitch_VW, const float* stitch_VC, const int* stitch_vv_num,
		const float* stitch_VL, float stitch_k, float stitch_ratio)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		float3 b = make_float3(init_B[i * 3 + 0], init_B[i * 3 + 1], init_B[i * 3 + 2]);
		float3 k = make_float3(0, 0, 0);
		const float3 xi = make_float3(X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]);
		float diag = new_VC[i];

		const int bg = all_vv_num[i], ed = all_vv_num[i + 1];
		for (int index = bg; index<ed; index++)
		{
			const int j = all_VV[index];
			const float jl = all_VL[index];
			const float3 xj = make_float3(X[j * 3 + 0], X[j * 3 + 1], X[j * 3 + 2]);

			// Remove the off-diagonal (Jacobi method)
			b -= all_VW[index] * xj;

			// Add the other part of b: spring-length constraint
			const float3 d = normalize(xi - xj);
			b += d * spring_k* jl;
			k += (d * d + max(0., 1.-jl) * (1 - d * d) - 1) * spring_k; // ldp: what is this? cannot understand
		}

		// handel stitch
		if (stitch_VV)
		{
			const int bg = stitch_vv_num[i], ed = stitch_vv_num[i + 1];
			float sumStitchW = 0;
			const float bend_stitch_w = powf(1 - stitch_ratio, 10.f);
			for (int index = bg; index<ed; index++)
			{
				const int j = stitch_VV[index];
				const float jl = stitch_VL[index];
				const float3 xj = make_float3(X[j * 3 + 0], X[j * 3 + 1], X[j * 3 + 2]);

				// Remove the off-diagonal (Jacobi method)
				b += (jl != 0)*stitch_k*xj - bend_stitch_w*stitch_VW[index] * xj;;
				sumStitchW += (jl != 0)*stitch_k;

				// Add the other part of b: spring-length constraint
				float3 d = (xi - xj) / (length(xi - xj) + 1e-16);
				b += d * stitch_k* jl * stitch_ratio;
				k += (d * d + max(0., 1. - jl * stitch_ratio) * (1 - d * d) - 1) * stitch_k; // ldp: what is this? cannot understand
			}
			diag += sumStitchW + bend_stitch_w*stitch_VC[i];
		} // end if stitch_VV

		const float3 nxi = xi + (b - diag * xi) / (diag + k);

		next_X[i * 3 + 0] = nxi.x;
		next_X[i * 3 + 1] = nxi.y;
		next_X[i * 3 + 2] = nxi.z;
	}

	void ClothManager::constrain1()
	{
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		Constraint_1_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_X.ptr(), m_dev_init_B.ptr(), m_dev_next_X.ptr(), 
			m_dev_all_VV.ptr(), m_dev_all_VL.ptr(), m_dev_all_VW.ptr(), m_dev_new_VC.ptr(), 
			m_dev_all_vv_num.ptr(), m_simulationParam.spring_k, m_X.size(),
			m_dev_stitch_VV.ptr(), m_dev_stitch_VW.ptr(), m_dev_stitch_VC.ptr(), m_dev_stitch_VV_num.ptr(),
			m_dev_stitch_VL.ptr(), m_simulationParam.stitch_k, m_curStitchRatio);
		cudaSafeCall(cudaGetLastError(), "constrain1");
	}
#pragma endregion

#pragma region --constrain2
	__global__ void Constraint_2_Kernel(const float* prev_X, const float* X, 
		float* next_X, float omega, int number, float under_relax)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		float3 xi = make_float3(X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]);
		float3 nxi = make_float3(next_X[i * 3 + 0], next_X[i * 3 + 1], next_X[i * 3 + 2]);
		float3 pxi = make_float3(prev_X[i * 3 + 0], prev_X[i * 3 + 1], prev_X[i * 3 + 2]);
		nxi = (nxi - xi) * under_relax + xi;
		nxi = omega * (nxi - pxi) + pxi;
		next_X[i * 3 + 0] = nxi.x;
		next_X[i * 3 + 1] = nxi.y;
		next_X[i * 3 + 2] = nxi.z;
	}

	void ClothManager::constrain2(float omega)
	{
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		Constraint_2_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_prev_X.ptr(), m_dev_X.ptr(), m_dev_next_X.ptr(),
			omega, m_X.size(), m_simulationParam.under_relax);
		cudaSafeCall(cudaGetLastError(), "constrain2");
	}
#pragma endregion

#pragma region --constrain3
	__global__ void Constraint_3_Kernel(float* X, const float *old_X, int number,
		const float *phi, const float* outgo_dist, const float3 start, 
		const float h, const float inv_h, const int size_x, const int size_y, 
		const int size_z)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	
			return;

		if (phi)
		{
			const float level_set_goal = 1.0f + outgo_dist[i] * inv_h;
			float3 xi = make_float3(X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]);
			float3 oxi = make_float3(old_X[i * 3 + 0], old_X[i * 3 + 1], old_X[i * 3 + 2]);
			float3 t = (xi - start) * inv_h;
			float depth = Level_Set_Depth(phi, t.x, t.y, t.z, level_set_goal, size_x, size_y, size_z, size_y*size_z)*h;
			if (depth<0)
			{
				t = xi - oxi;
				float t_length = length(t);
				if (t_length>1e-16f)
				{
					t /= t_length;
					t_length = t_length - fabsf(depth)*1.2;
					if (t_length<0)	t_length = 0;
				}
				t = (oxi + t*t_length - start)*inv_h;
				Level_Set_Projection(phi, t.x, t.y, t.z, level_set_goal, size_x, size_y, size_z, size_y*size_z);
				X[i * 3 + 0] = t.x*h + start.x;
				X[i * 3 + 1] = t.y*h + start.y;
				X[i * 3 + 2] = t.z*h + start.z;
			} // end if depth
		} // end if phi
	}

	void ClothManager::constrain3()
	{
		const auto start = m_bodyLvSet->getStartPos();
		const auto h = m_bodyLvSet->getStep();
		const auto inv_h = 1 / h;
		const auto size = m_bodyLvSet->size();
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		Constraint_3_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_X.ptr(), m_dev_old_X.ptr(), m_X.size(),
			m_dev_phi.ptr(), m_dev_V_outgo_dist.ptr(), make_float3(start[0], start[1], start[2]), h, inv_h, 
			size[0], size[1], size[2]);
		cudaSafeCall(cudaGetLastError(), "constrain3");
	}
#pragma endregion

#pragma region --constrain4
	__global__ void Constraint_4_Kernel(float* X, float* init_B, float* V, const float* fixed, 
		const float* more_fixed, float inv_t, int number)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		float c = (1 + fixed[i] + more_fixed[i])*inv_t*inv_t;
		V[i * 3 + 0] += (X[i * 3 + 0] - init_B[i * 3 + 0] / c)*inv_t;
		V[i * 3 + 1] += (X[i * 3 + 1] - init_B[i * 3 + 1] / c)*inv_t;
		V[i * 3 + 2] += (X[i * 3 + 2] - init_B[i * 3 + 2] / c)*inv_t;
	}


	void ClothManager::constrain4()
	{
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		Constraint_4_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_X.ptr(), m_dev_init_B.ptr(), m_dev_V.ptr(), m_dev_fixed.ptr(), 
			m_dev_more_fixed.ptr(), 1 / m_simulationParam.time_step, m_X.size());
		cudaSafeCall(cudaGetLastError(), "constrain4");
	}
#pragma endregion

#pragma region --drag control
#define RADIUS_SQUARED	0.000625
	__global__ void Control_Kernel(float* X, float *more_fixed, float control_mag, const int number, 
		const int select_v, const int start_id, const int end_id)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= end_id || i < start_id)	return;

		more_fixed[i] = 0;
		if (select_v != -1)
		{
			float3 xi = make_float3(X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]);
			float3 xs = make_float3(X[select_v * 3 + 0], X[select_v * 3 + 1], X[select_v * 3 + 2]);
			if (dot(xi-xs, xi-xs) < RADIUS_SQUARED)	
				more_fixed[i] = control_mag;
		}
	}

	void ClothManager::resetMoreFixed()
	{
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		Control_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_X.ptr(), m_dev_more_fixed.ptr(), m_simulationParam.control_mag, 
			m_X.size(), m_curDragInfo.vert_id, m_curDragInfo.piece_id_start,
			m_curDragInfo.piece_id_end);
		cudaSafeCall(cudaGetLastError(), "resetMoreFixed");
	}
#pragma endregion

#pragma region --collision handler
	//std::shared_ptr<COLLISION_HANDLER> g_chl;
	void ClothManager::initCollisionHandler()
	{
		//g_chl.reset(new COLLISION_HANDLER());
		m_collider.reset(new SelfCollider());
	}

	void ClothManager::constrain_selfCollision()
	{
		if (m_simulationParam.enable_self_collistion)
		{
			m_collider->run((float3*)m_dev_old_X.ptr(), (float3*)m_dev_X.ptr(), (float3*)m_dev_V.ptr(), m_X.size(), 
				(const int3*)m_dev_T.ptr(), m_T.size(), (const float3*)m_X.data(), 1.f / m_simulationParam.time_step,
				m_dev_stitchPair_num.ptr(), m_dev_stitchPair.ptr(), m_dev_stitchPair.size());
			//g_chl->Run(m_dev_old_X.ptr(), m_dev_X.ptr(), m_dev_V.ptr(), m_X.size(),
			//	m_dev_T.ptr(), m_T.size(), (float*)m_X.data(), 1.f / m_simulationParam.time_step);
		}
	}
#pragma endregion
}