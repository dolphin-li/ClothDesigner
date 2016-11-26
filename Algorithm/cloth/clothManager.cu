#include "clothManager.h"
#include "cuda_utils.h"
#include "LevelSet3D.h"
#include "LEVEL_SET_COLLISION.h"
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

		next_V[i * 3 + 0] = 0;
		next_V[i * 3 + 1] = 0;
		next_V[i * 3 + 2] = 0;
		if (more_fixed[i] != 0)	return;
		if (fixed[i] != 0)			return;


		for (int index = all_vv_num[i]; index<all_vv_num[i + 1]; index++)
		{
			int j = all_VV[index];
			next_V[i * 3 + 0] += V[j * 3 + 0] - V[i * 3 + 0];
			next_V[i * 3 + 1] += V[j * 3 + 1] - V[i * 3 + 1];
			next_V[i * 3 + 2] += V[j * 3 + 2] - V[i * 3 + 2];
		}
		next_V[i * 3 + 0] = V[i * 3 + 0] + next_V[i * 3 + 0] * rate;
		next_V[i * 3 + 1] = V[i * 3 + 1] + next_V[i * 3 + 1] * rate;
		next_V[i * 3 + 2] = V[i * 3 + 2] + next_V[i * 3 + 2] * rate;

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
		V[i * 3 + 1] += g_gravity[0] * t;
		V[i * 3 + 1] += g_gravity[1] * t;
		V[i * 3 + 1] += g_gravity[2] * t;

		//Position update
		X[i * 3 + 0] = X[i * 3 + 0] + V[i * 3 + 0] * t;
		X[i * 3 + 1] = X[i * 3 + 1] + V[i * 3 + 1] * t;
		X[i * 3 + 2] = X[i * 3 + 2] + V[i * 3 + 2] * t;
	}

	void ClothManager::updateAfterLap()
	{
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
		m_dev_X.copyTo(m_dev_prev_X);
	}
#pragma endregion
	
#pragma region --constrain1
	__global__ void Constraint_1_Kernel(const float* X, const float* init_B, float* F, 
		float* next_X, const int* all_VV, const float* all_VL, const float* all_VW, 
		const float* new_VC, const int* all_vv_num, const float spring_k, const int number)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		float b[3];
		b[0] = init_B[i * 3 + 0];
		b[1] = init_B[i * 3 + 1];
		b[2] = init_B[i * 3 + 2];

		float k[3];
		k[0] = 0;
		k[1] = 0;
		k[2] = 0;

		for (int index = all_vv_num[i]; index<all_vv_num[i + 1]; index++)
		{
			int j = all_VV[index];

			// Remove the off-diagonal (Jacobi method)
			b[0] -= all_VW[index] * X[j * 3 + 0];
			b[1] -= all_VW[index] * X[j * 3 + 1];
			b[2] -= all_VW[index] * X[j * 3 + 2];

			// Add the other part of b
			if (all_VL[index] == -1)	continue;
			float d[3];
			d[0] = X[i * 3 + 0] - X[j * 3 + 0];
			d[1] = X[i * 3 + 1] - X[j * 3 + 1];
			d[2] = X[i * 3 + 2] - X[j * 3 + 2];

			float inv_d2 = 1.0f / DOT(d, d);
			float new_L = spring_k*all_VL[index] * sqrtf(inv_d2);
			b[0] += d[0] * new_L;
			b[1] += d[1] * new_L;
			b[2] += d[2] * new_L;

			k[0] += d[0] * d[0] * inv_d2*spring_k;
			k[1] += d[1] * d[1] * inv_d2*spring_k;
			k[2] += d[2] * d[2] * inv_d2*spring_k;
			if (spring_k>new_L)
			{
				k[0] += (spring_k - new_L)*(1 - d[0] * d[0] * inv_d2);
				k[1] += (spring_k - new_L)*(1 - d[1] * d[1] * inv_d2);
				k[2] += (spring_k - new_L)*(1 - d[2] * d[2] * inv_d2);
			}
			k[0] -= spring_k;
			k[1] -= spring_k;
			k[2] -= spring_k;
		}

		next_X[i * 3 + 0] = X[i * 3 + 0] + (b[0] - new_VC[i] * X[i * 3 + 0]) / (new_VC[i] + k[0]);
		next_X[i * 3 + 1] = X[i * 3 + 1] + (b[1] - new_VC[i] * X[i * 3 + 1]) / (new_VC[i] + k[1]);
		next_X[i * 3 + 2] = X[i * 3 + 2] + (b[2] - new_VC[i] * X[i * 3 + 2]) / (new_VC[i] + k[2]);

		//F[i*3+0]=b[0]-new_VC[i]*X[i*3+0];
		//F[i*3+1]=b[1]-new_VC[i]*X[i*3+1];
		//F[i*3+2]=b[2]-new_VC[i]*X[i*3+2];
	}

	void ClothManager::constrain1()
	{
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		Constraint_1_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_X.ptr(), m_dev_init_B.ptr(), m_dev_F.ptr(), m_dev_next_X.ptr(), 
			m_dev_all_VV.ptr(), m_dev_all_VL.ptr(), m_dev_all_VW.ptr(), m_dev_new_VC.ptr(), 
			m_dev_all_vv_num.ptr(), m_simulationParam.spring_k, m_X.size());
	}
#pragma endregion

#pragma region --constrain2
	__global__ void Constraint_2_Kernel(const float* prev_X, const float* X, 
		float* next_X, float omega, int number, float under_relax)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		float displace[3];
		displace[0] = next_X[i * 3 + 0] - X[i * 3 + 0];
		displace[1] = next_X[i * 3 + 1] - X[i * 3 + 1];
		displace[2] = next_X[i * 3 + 2] - X[i * 3 + 2];

		next_X[i * 3 + 0] = displace[0] * under_relax + X[i * 3 + 0];
		next_X[i * 3 + 1] = displace[1] * under_relax + X[i * 3 + 1];
		next_X[i * 3 + 2] = displace[2] * under_relax + X[i * 3 + 2];

		next_X[i * 3 + 0] = omega*(next_X[i * 3 + 0] - prev_X[i * 3 + 0]) + prev_X[i * 3 + 0];
		next_X[i * 3 + 1] = omega*(next_X[i * 3 + 1] - prev_X[i * 3 + 1]) + prev_X[i * 3 + 1];
		next_X[i * 3 + 2] = omega*(next_X[i * 3 + 2] - prev_X[i * 3 + 2]) + prev_X[i * 3 + 2];
	}

	void ClothManager::constrain2(float omega)
	{
		const int blocksPerGrid = divUp(m_X.size(), threadsPerBlock);
		Constraint_2_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_prev_X.ptr(), m_dev_X.ptr(), m_dev_next_X.ptr(),
			omega, m_X.size(), m_simulationParam.under_relax);
	}
#pragma endregion

#pragma region --constrain3
	__global__ void Constraint_3_Kernel(float* X, const float *old_X, int number,
		const float *phi, const float start_x, const float start_y, const float start_z, 
		const float h, const float inv_h, const int size_x, const int size_y, 
		const int size_z, const int size_yz, const float velocity_cap)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	
			return;

		//float c=(1+fixed[i]+more_fixed[i])*inv_t*inv_t;
		if (phi)
		{
			float tx = (X[i * 3 + 0] - start_x)*inv_h;
			float ty = (X[i * 3 + 1] - start_y)*inv_h;
			float tz = (X[i * 3 + 2] - start_z)*inv_h;
			float depth = Level_Set_Depth(phi, tx, ty, tz, 1.0f, size_x, size_y, size_z, size_yz)*h;
			if (depth<0)
			{
				tx = X[i * 3 + 0] - old_X[i * 3 + 0];
				ty = X[i * 3 + 1] - old_X[i * 3 + 1];
				tz = X[i * 3 + 2] - old_X[i * 3 + 2];
				float t_length = sqrtf(tx*tx + ty*ty + tz*tz);
				if (t_length>1e-16f)
				{
					tx /= t_length;
					ty /= t_length;
					tz /= t_length;
					t_length = t_length - fabsf(depth)*1.2; //1.2
					if (t_length<0)	t_length = 0;
				}

				X[i * 3 + 0] = old_X[i * 3 + 0] + tx*t_length;
				X[i * 3 + 1] = old_X[i * 3 + 1] + ty*t_length;
				X[i * 3 + 2] = old_X[i * 3 + 2] + tz*t_length;

				tx = (X[i * 3 + 0] - start_x)*inv_h;
				ty = (X[i * 3 + 1] - start_y)*inv_h;
				tz = (X[i * 3 + 2] - start_z)*inv_h;
				Level_Set_Projection(phi, tx, ty, tz, 1.0f, size_x, size_y, size_z, size_yz);
				X[i * 3 + 0] = tx*h + start_x;
				X[i * 3 + 1] = ty*h + start_y;
				X[i * 3 + 2] = tz*h + start_z;
			} // end if depth
		} // end if phi

		/*{
		//Clamping
		float dis[3];
		dis[0]=X[i*3+0]-old_X[i*3+0];
		dis[1]=X[i*3+1]-old_X[i*3+1];
		dis[2]=X[i*3+2]-old_X[i*3+2];
		float dis_mag=sqrt(DOT(dis, dis));
		float rate=MIN(1, velocity_cap/dis_mag);

		X[i*3+0]=old_X[i*3+0]+dis[0]*rate;
		X[i*3+1]=old_X[i*3+1]+dis[1]*rate;
		X[i*3+2]=old_X[i*3+2]+dis[2]*rate;
		}*/


		//V[i*3+0]+=(X[i*3+0]-init_B[i*3+0]/c)*inv_t;
		//V[i*3+1]+=(X[i*3+1]-init_B[i*3+1]/c)*inv_t;
		//V[i*3+2]+=(X[i*3+2]-init_B[i*3+2]/c)*inv_t;
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
			m_dev_phi.ptr(), start[0], start[1], start[2], h, inv_h, 
			size[0], size[1], size[2], size[1]*size[2], 
			m_simulationParam.velocity_cap);
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
	}
#pragma endregion
}