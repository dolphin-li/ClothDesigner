#include "clothManager.h"

namespace ldp
{
	void ClothManager::laplaceDamping()
	{
		m_dev_X.copyTo(m_dev_old_X);
		for (int l = 0; l<m_simulationParam.lap_damping; l++)
		{
			//Laplacian_Damping_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			//	dev_V, dev_next_X, dev_fixed, dev_more_fixed, dev_all_VV, dev_all_vv_num, number, 0.1);
			m_dev_next_X.copyTo(m_dev_V);
		}
	}

	void ClothManager::updateAfterLap()
	{
		//Update_Kernel << <blocksPerGrid, threadsPerBlock >> >(
		//	dev_X, dev_V, dev_fixed, dev_more_fixed, air_damping, t, number, dir[0], dir[1], dir[2],
		//	dev_phi, start_x, start_y, start_z, h, inv_h, size_x, size_y, size_z, size_yz);
	}

	void ClothManager::constrain0()
	{
		m_dev_all_VC.copyTo(m_dev_new_VC);
		//Constraint_0_Kernel << <blocksPerGrid, threadsPerBlock >> >(
		//	dev_X, dev_init_B, dev_new_VC, dev_fixed, dev_more_fixed, 1 / t, spring_k, number);
		m_dev_X.copyTo(m_dev_prev_X);
	}	
	
	void ClothManager::constrain1()
	{
		//Constraint_1_Kernel << <blocksPerGrid, threadsPerBlock >> >(
		//	dev_X, dev_init_B, dev_F, dev_next_X, dev_all_VV, dev_all_VL,
		//	dev_all_VW, dev_new_VC, dev_all_vv_num, spring_k, number);
	}

	void ClothManager::constrain2()
	{
		//Constraint_2_Kernel << <blocksPerGrid, threadsPerBlock >> >(
		//	dev_prev_X, dev_X, dev_next_X, omega, number, under_relax);
	}

	void ClothManager::constrain3()
	{
		//Constraint_3_Kernel << <blocksPerGrid, threadsPerBlock >> >(dev_X, dev_old_X, number,
		//	dev_phi, start_x, start_y, start_z, h, inv_h, size_x, size_y, size_z, size_yz, velocity_cap);
	}

	void ClothManager::constrain4()
	{
		//Constraint_4_Kernel << <blocksPerGrid, threadsPerBlock >> >(
		//	dev_X, dev_init_B, dev_V, dev_fixed, dev_more_fixed, 1 / t, number);
	}
}