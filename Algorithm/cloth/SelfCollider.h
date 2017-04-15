#pragma once

#include "cudpp\helper_math.h"
#include "cudpp\device_array.h"
namespace ldp
{

	class SelfCollider
	{
	public:
		SelfCollider();
		~SelfCollider();

		void run(float3* dev_old_X, float3* dev_new_X, float3* dev_V, int number, 
			const int3* dev_T, int t_number, const float3* host_X, float inv_t,
			const int* dev_stitchVV_csrRowInfo, const int* dev_stitchVV_csrIndex, 
			int nnzStitch);
	protected:
		void allocate(int number, int t_number, int bucket_size);
		void deallocate();
	private:
		float3* m_dev_I = nullptr;				// impulse
		float3* m_dev_R = nullptr;				// repulse
		float*	m_dev_W = nullptr;				// weights of repulse
		int*	m_dev_vertex_id = nullptr;
		int*	m_dev_vertex_bucket = nullptr;
		int*	m_dev_bucket_ranges = nullptr;
		int*	m_dev_c_status = nullptr;
		int*	m_dev_t_key = nullptr;
		int*	m_dev_t_idx = nullptr;
	};
}

