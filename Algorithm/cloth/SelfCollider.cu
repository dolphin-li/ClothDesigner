#include "SelfCollider.h"

#include "cuda_utils.h"
#include "thrust_wrapper.h"
#include <thrust\device_ptr.h>
#include <thrust\device_vector.h>

//#define ENABLE_TIMER

#ifdef ENABLE_TIMER
#include "TIMER.h"
#endif


namespace ldp
{
	enum
	{
		threadsPerBlock = 256,
		t_threadsPerBlock = 256
	};

#pragma region --kernels

	texture<float, cudaTextureType1D, cudaReadModeElementType> g_oldX_tex, g_newX_tex;
	texture<int, cudaTextureType1D, cudaReadModeElementType> g_cstatus_tex, g_bucket_range_tex, g_vid_tex;

	__device__ __forceinline__ float3 get_oldX(int i)
	{
		return make_float3(tex1Dfetch(g_oldX_tex, i*3),
			tex1Dfetch(g_oldX_tex, i*3+1), tex1Dfetch(g_oldX_tex, i*3+2));
	}
	__device__ __forceinline__ float3 get_newX(int i)
	{
		return make_float3(tex1Dfetch(g_newX_tex, i * 3),
			tex1Dfetch(g_newX_tex, i * 3 + 1), tex1Dfetch(g_newX_tex, i * 3 + 2));
	}
	__device__ __forceinline__ int get_cstatus(int i)
	{
		return tex1Dfetch(g_cstatus_tex, i);
	}
	__device__ __forceinline__ int get_vid(int i)
	{
		return tex1Dfetch(g_vid_tex, i);
	}
	__device__ __forceinline__ int2 get_bucket_range(int i)
	{
		return make_int2(tex1Dfetch(g_bucket_range_tex, 2 * i), tex1Dfetch(g_bucket_range_tex, 2 * i + 1));
	}

	__device__ __forceinline__ int v2id_x(float3 v, float3 start, float inv_h)
	{
		return int((v.x - start.x)*inv_h);
	}
	__device__ __forceinline__ int v2id_y(float3 v, float3 start, float inv_h)
	{
		return int((v.y - start.y)*inv_h);
	}
	__device__ __forceinline__ int v2id_z(float3 v, float3 start, float inv_h)
	{
		return int((v.z - start.z)*inv_h);
	}
	__device__ __forceinline__ int xyz2id(int x, int y, int z, int3 size)
	{
		return (x * size.y + y) * size.z + z;
	}
	__device__ __forceinline__ int v2id(float3 v, float3 start, float inv_h, int3 size)
	{
		const int x = v2id_x(v, start, inv_h);
		const int y = v2id_y(v, start, inv_h);
		const int z = v2id_z(v, start, inv_h);
		return xyz2id(x, y, z, size);
	}
	template<class T>
	void debug_print_device(T* dev_ptr, int pos, std::string str)
	{
		thrust::device_ptr<T> d(dev_ptr);
		T v = d[pos];
		std::cout << str << v << std::endl;
	}
	///////////////////////////////////////////////////////////////////////////////////////////
	//  Vertex Grid kernel
	///////////////////////////////////////////////////////////////////////////////////////////
	__global__ void Grid_0_Kernel(const float3* X, int* vertex_id, int* vertex_bucket, 
		int number, float3 start, float inv_h, int3 size)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;
		vertex_id[i] = i;
		vertex_bucket[i] = v2id(X[i], start, inv_h, size);
	}

	__global__ void Grid_1_Kernel(int* vertex_bucket, int* bucket_ranges, int number)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		const int vi = vertex_bucket[i];
		if (i == 0 || vi != vertex_bucket[i - 1])	
			bucket_ranges[vi * 2 + 0] = i;		//begin at i
		if (i == number - 1 || vi != vertex_bucket[i + 1])	
			bucket_ranges[vi * 2 + 1] = i + 1;	//  end at i+1
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	//  Triangle Test kernel
	///////////////////////////////////////////////////////////////////////////////////////////
	template<typename T>
	__device__ __forceinline__ T cudaMin(const T a, const T b, const T c)
	{
		return min(a, min(b, c));
	}
	template<typename T>
	__device__ __forceinline__ T cudaMax(const T a, const T b, const T c)
	{
		return max(a, max(b, c));
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	////  Squared vertex-edge distance
	////	r is the barycentric weight of the closest point: 1-r, r.
	///////////////////////////////////////////////////////////////////////////////////////////
	__device__ __forceinline__ float cuda_Squared_VE_Distance(float3 xi, float3 xa, float3 xb, float &r, float3 &N)
	{
		const float3 xia = xi - xa;
		const float3 xba = xb - xa;
		const float xia_xba = dot(xia, xba);
		const float xba_xba = dot(xba, xba);
		if (xia_xba<0)				
			r = 0;
		else if (xia_xba>xba_xba)	
			r = 1;
		else						
			r = xia_xba / xba_xba;
		N = xi - xa * (1.f - r) - xb * r;
		return dot(N, N);
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	////  Squared vertex-triangle distance
	////	bb and bc are the barycentric weights of the closest point: 1-bb-bc, bb, bc.
	///////////////////////////////////////////////////////////////////////////////////////////
	__device__ __forceinline__ float cuda_Squared_VT_Distance(float gap, float3 xi, float3 xa, float3 xb,
		float3 xc, float &ba, float &bb, float &bc, float3& N)
	{
		const float3 xba = xb - xa;
		const float3 xca = xc - xa;
		const float3 xia = xi - xa;
		N = cross(xba, xca);
		const float nn = dot(N, N);
		const float weight_iaca = dot(N, cross(xia, xca));
		const float weight_baia = dot(N, cross(xba, xia));

		if (nn>1e-16f && weight_iaca >= 0 && weight_baia >= 0 && nn - weight_iaca - weight_baia >= 0)
		{
			bb = weight_iaca / nn;
			bc = weight_baia / nn;
			ba = 1 - bb - bc;
		}
		else
		{
			return 999999;
		}

		N = xi - xa * ba - xb * bb - xc * bc;
		return dot(N, N);
	}

	__global__ void Triangle_Test_Kernel0(const float3* old_X, const float3* new_X, const int number, 
		const int3* T, const int t_number, const int* c_status, const int l,
		const float3 start, const float inv_h, const int3 size, int* t_key, int* t_idx)
	{
		int t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= t_number)	return;

		int3 vabc = T[t];
		const int va_status = c_status[v2id(old_X[vabc.x], start, inv_h, size)];
		const int vb_status = c_status[v2id(old_X[vabc.y], start, inv_h, size)];
		const int vc_status = c_status[v2id(old_X[vabc.z], start, inv_h, size)];
		const float3 oa = old_X[vabc.x], ob = old_X[vabc.y], oc = old_X[vabc.z];
		const float3 na = new_X[vabc.x], nb = new_X[vabc.y], nc = new_X[vabc.z];
		int min_i = cudaMin(min(v2id_x(oa, start, inv_h), v2id_x(na, start, inv_h)), min(v2id_x(ob, start, inv_h), v2id_x(nb, start, inv_h)), min(v2id_x(oc, start, inv_h), v2id_x(nc, start, inv_h))) - 1;
		int min_j = cudaMin(min(v2id_y(oa, start, inv_h), v2id_y(na, start, inv_h)), min(v2id_y(ob, start, inv_h), v2id_y(nb, start, inv_h)), min(v2id_y(oc, start, inv_h), v2id_y(nc, start, inv_h))) - 1;
		int min_k = cudaMin(min(v2id_z(oa, start, inv_h), v2id_z(na, start, inv_h)), min(v2id_z(ob, start, inv_h), v2id_z(nb, start, inv_h)), min(v2id_z(oc, start, inv_h), v2id_z(nc, start, inv_h))) - 1;
		int max_i = cudaMax(max(v2id_x(oa, start, inv_h), v2id_x(na, start, inv_h)), max(v2id_x(ob, start, inv_h), v2id_x(nb, start, inv_h)), max(v2id_x(oc, start, inv_h), v2id_x(nc, start, inv_h))) + 1;
		int max_j = cudaMax(max(v2id_y(oa, start, inv_h), v2id_y(na, start, inv_h)), max(v2id_y(ob, start, inv_h), v2id_y(nb, start, inv_h)), max(v2id_y(oc, start, inv_h), v2id_y(nc, start, inv_h))) + 1;
		int max_k = cudaMax(max(v2id_z(oa, start, inv_h), v2id_z(na, start, inv_h)), max(v2id_z(ob, start, inv_h), v2id_z(nb, start, inv_h)), max(v2id_z(oc, start, inv_h), v2id_z(nc, start, inv_h))) + 1;
		min_i = max(min(min_i, size.x - 1), 0);
		min_j = max(min(min_j, size.y - 1), 0);
		min_k = max(min(min_k, size.z - 1), 0);
		max_i = max(min(max_i, size.x - 1), 0);
		max_j = max(min(max_j, size.y - 1), 0);
		max_k = max(min(max_k, size.z - 1), 0);

		t_idx[t] = t;
		t_key[t] = 1;
		for (int i = min_i; i <= max_i; i++)
		for (int j = min_j; j <= max_j; j++)
		for (int k = min_k; k <= max_k; k++)
		{
			int vid = xyz2id(i, j, k, size);
			if (c_status[vid] < l && va_status < l && vb_status < l && vc_status < l)
				continue;
			t_key[t] = 0;
		} // end for i, j, k
	}

	__global__ void Triangle_Test_Kernel(const int number, 
		const int3* T, const int t_number, float3* I, float3* R, float* W, int* c_status, 
		const int l, const float gap,
		const float3 start, const float inv_h, const int3 size, const int* t_key, const int* t_idx,
		int* not_converged)
	{
		int t = blockDim.x * blockIdx.x + threadIdx.x;
		if (t >= t_number)	return;

		//Convert
		if (t_key && t_idx)
			t = t_idx[t];

		const int3 vabc = T[t];
		const float3 oa = get_oldX(vabc.x), ob = get_oldX(vabc.y), oc = get_oldX(vabc.z);
		const float3 na = get_newX(vabc.x), nb = get_newX(vabc.y), nc = get_newX(vabc.z);
		const int va_status = get_cstatus(v2id(oa, start, inv_h, size));
		const int vb_status = get_cstatus(v2id(ob, start, inv_h, size));
		const int vc_status = get_cstatus(v2id(oc, start, inv_h, size));
		const int min_i = max(0, min(size.x - 1, cudaMin(min(v2id_x(oa, start, inv_h), v2id_x(na, start, inv_h)), min(v2id_x(ob, start, inv_h), v2id_x(nb, start, inv_h)), min(v2id_x(oc, start, inv_h), v2id_x(nc, start, inv_h))) - 1));
		const int min_j = max(0, min(size.y - 1, cudaMin(min(v2id_y(oa, start, inv_h), v2id_y(na, start, inv_h)), min(v2id_y(ob, start, inv_h), v2id_y(nb, start, inv_h)), min(v2id_y(oc, start, inv_h), v2id_y(nc, start, inv_h))) - 1));
		const int min_k = max(0, min(size.z - 1, cudaMin(min(v2id_z(oa, start, inv_h), v2id_z(na, start, inv_h)), min(v2id_z(ob, start, inv_h), v2id_z(nb, start, inv_h)), min(v2id_z(oc, start, inv_h), v2id_z(nc, start, inv_h))) - 1));
		const int max_i = max(0, min(size.x - 1, cudaMax(max(v2id_x(oa, start, inv_h), v2id_x(na, start, inv_h)), max(v2id_x(ob, start, inv_h), v2id_x(nb, start, inv_h)), max(v2id_x(oc, start, inv_h), v2id_x(nc, start, inv_h))) + 1));
		const int max_j = max(0, min(size.y - 1, cudaMax(max(v2id_y(oa, start, inv_h), v2id_y(na, start, inv_h)), max(v2id_y(ob, start, inv_h), v2id_y(nb, start, inv_h)), max(v2id_y(oc, start, inv_h), v2id_y(nc, start, inv_h))) + 1));
		const int max_k = max(0, min(size.z - 1, cudaMax(max(v2id_z(oa, start, inv_h), v2id_z(na, start, inv_h)), max(v2id_z(ob, start, inv_h), v2id_z(nb, start, inv_h)), max(v2id_z(oc, start, inv_h), v2id_z(nc, start, inv_h))) + 1));

		for (int i = min_i; i <= max_i; i++)
		for (int j = min_j; j <= max_j; j++)
		for (int k = min_k; k <= max_k; k++)
		{
			const int vid = xyz2id(i, j, k, size);
			if (get_cstatus(vid) < l && va_status < l && vb_status < l && vc_status < l)
				continue;
			for (int ptr = get_bucket_range(vid).x; ptr<get_bucket_range(vid).y; ptr++)
			{
				const int vi = get_vid(ptr);
				if (vi == vabc.x || vi == vabc.y || vi == vabc.z
					|| vabc.x == vabc.y || vabc.y == vabc.z || vabc.z == vabc.x)
					continue;

				const float3 x0 = get_newX(vi);
				const float3 x1 = get_newX(vabc.x);
				const float3 x2 = get_newX(vabc.y);
				const float3 x3 = get_newX(vabc.z);
				if (   x0.x + gap * 2 < cudaMin(x1.x, x2.x, x3.x)
					|| x0.y + gap * 2 < cudaMin(x1.y, x2.y, x3.y)
					|| x0.z + gap * 2 < cudaMin(x1.z, x2.z, x3.z)
					|| x0.x - gap * 2 > cudaMax(x1.x, x2.x, x3.x)
					|| x0.y - gap * 2 > cudaMax(x1.y, x2.y, x3.y)
					|| x0.z - gap * 2 > cudaMax(x1.z, x2.z, x3.z)
					|| x0.x + x0.y + gap * 3 < cudaMin(x1.x + x1.y, x2.x + x2.y, x3.x + x3.y)
					|| x0.y + x0.z + gap * 3 < cudaMin(x1.y + x1.z, x2.y + x2.z, x3.y + x3.z)
					|| x0.z + x0.x + gap * 3 < cudaMin(x1.z + x1.x, x2.z + x2.x, x3.z + x3.x)
					|| x0.x - x0.y + gap * 3 < cudaMin(x1.x - x1.y, x2.x - x2.y, x3.x - x3.y)
					|| x0.y - x0.z + gap * 3 < cudaMin(x1.y - x1.z, x2.y - x2.z, x3.y - x3.z)
					|| x0.z - x0.x + gap * 3 < cudaMin(x1.z - x1.x, x2.z - x2.x, x3.z - x3.x)
					|| x0.x + x0.y - gap * 3 > cudaMax(x1.x + x1.y, x2.x + x2.y, x3.x + x3.y)
					|| x0.y + x0.z - gap * 3 > cudaMax(x1.y + x1.z, x2.y + x2.z, x3.y + x3.z)
					|| x0.z + x0.x - gap * 3 > cudaMax(x1.z + x1.x, x2.z + x2.x, x3.z + x3.x)
					|| x0.x - x0.y - gap * 3 > cudaMax(x1.x - x1.y, x2.x - x2.y, x3.x - x3.y)
					|| x0.y - x0.z - gap * 3 > cudaMax(x1.y - x1.z, x2.y - x2.z, x3.y - x3.z)
					|| x0.z - x0.x - gap * 3 > cudaMax(x1.z - x1.x, x2.z - x2.x, x3.z - x3.x))
					continue;

				float ba = 0.f, bb = 0.f, bc = 0.f;
				float3 N = make_float3(0, 0, 0);
				float d = cuda_Squared_VT_Distance(gap, x0, x1, x2, x3, ba, bb, bc, N);
				if (d > gap*gap * 4)
					continue;		//proximity found!

				//Take normal into consideration
				d = sqrt(d);
				float3 old_N = get_oldX(vi) - ba * oa - bb * ob - bc * oc;
				if (dot(old_N, N) < 0)
					d = -d;

				if (d<gap * 2 && l == 0)	//apply repulsion force
				{
					const float k = (gap * 2 - d) * 5 / d;
	
					atomicAdd(&R[vi].x, k*N.x);
					atomicAdd(&R[vi].y, k*N.y);
					atomicAdd(&R[vi].z, k*N.z);
					atomicAdd(&R[vabc.x].x, -k*N.x * ba);
					atomicAdd(&R[vabc.x].y, -k*N.y * ba);
					atomicAdd(&R[vabc.x].z, -k*N.z * ba);
					atomicAdd(&R[vabc.y].x, -k*N.x * bb);
					atomicAdd(&R[vabc.y].y, -k*N.y * bb);
					atomicAdd(&R[vabc.y].z, -k*N.z * bb);
					atomicAdd(&R[vabc.z].x, -k*N.x * bc);
					atomicAdd(&R[vabc.z].y, -k*N.y * bc);
					atomicAdd(&R[vabc.z].z, -k*N.z * bc);
	
					atomicAdd(&W[vi], 1);
					atomicAdd(&W[vabc.x], 1);
					atomicAdd(&W[vabc.y], 1);
					atomicAdd(&W[vabc.z], 1);
				}

				if (d <= gap)
				{
					const float k = (gap*1.1 - d) / (d*(1 + ba*ba + bb*bb + bc*bc));
					atomicAdd(&I[vi].x, k*N.x);
					atomicAdd(&I[vi].y, k*N.y);
					atomicAdd(&I[vi].z, k*N.z);
					atomicAdd(&I[vabc.x].x, -k*N.x * ba);
					atomicAdd(&I[vabc.x].y, -k*N.y * ba);
					atomicAdd(&I[vabc.x].z, -k*N.z * ba);
					atomicAdd(&I[vabc.y].x, -k*N.x * bb);
					atomicAdd(&I[vabc.y].y, -k*N.y * bb);
					atomicAdd(&I[vabc.y].z, -k*N.z * bb);
					atomicAdd(&I[vabc.z].x, -k*N.x * bc);
					atomicAdd(&I[vabc.z].y, -k*N.y * bc);
					atomicAdd(&I[vabc.z].z, -k*N.z * bc);
					c_status[vid] = l + 1;
					not_converged[0] = 1;
				}
			} // end for ptr
		} // end for i, j, k
	}

	__global__ void Triangle_Test_2_Kernel(float3* new_X, float3* V, const float3* I, 
		const float3* R, const float* W, const int number, const int l, const float inv_t)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= number)	return;

		const float wi = W[i];
		if (l == 0 && wi != 0)
			V[i] += R[i] * inv_t / wi;
		new_X[i] += I[i] * 0.4;
	}
#pragma endregion

	SelfCollider::SelfCollider()
	{}

	SelfCollider::~SelfCollider()
	{
		deallocate();
	}

	void SelfCollider::allocate(int number, int t_number, int bucket_size)
	{
		deallocate();
		m_dev_I = (float3*)thrust_wrapper::cached_allocate(sizeof(float3)*number);
		m_dev_R = (float3*)thrust_wrapper::cached_allocate(sizeof(float3)*number);
		m_dev_W = (float*)thrust_wrapper::cached_allocate(sizeof(float)*number);
		m_dev_vertex_id = (int*)thrust_wrapper::cached_allocate(sizeof(int)*number);
		m_dev_vertex_bucket = (int*)thrust_wrapper::cached_allocate(sizeof(int)*number);
		m_dev_bucket_ranges = (int*)thrust_wrapper::cached_allocate(sizeof(int)*bucket_size);
		m_dev_c_status = (int*)thrust_wrapper::cached_allocate(sizeof(int)*bucket_size / 2);
		m_dev_t_key = (int*)thrust_wrapper::cached_allocate(sizeof(int)*t_number);
		m_dev_t_idx = (int*)thrust_wrapper::cached_allocate(sizeof(int)*t_number);
	}

	void SelfCollider::deallocate()
	{
		thrust_wrapper::cached_free((char*)m_dev_I);
		thrust_wrapper::cached_free((char*)m_dev_R);
		thrust_wrapper::cached_free((char*)m_dev_W);
		thrust_wrapper::cached_free((char*)m_dev_vertex_id);
		thrust_wrapper::cached_free((char*)m_dev_vertex_bucket);
		thrust_wrapper::cached_free((char*)m_dev_bucket_ranges);
		thrust_wrapper::cached_free((char*)m_dev_c_status);
		thrust_wrapper::cached_free((char*)m_dev_t_key);
		thrust_wrapper::cached_free((char*)m_dev_t_idx);
	}

	void SelfCollider::run(const float3* dev_old_X, float3* dev_new_X, float3* dev_V, int number,
		const int3* dev_T, int t_number, const float3* host_X, float inv_t)
	{
#ifdef ENABLE_TIMER
		TIMER timer;
		cudaThreadSynchronize();
		timer.Start();
#endif

		float	gap = 0.003; //0.0015
		float	h = 0.006; //twice the max vertex velocity
		float	inv_h = 1.0 / h;

		///////////////////////////////////////////////////////////////////////////
		//	Step 1:Compute bounding box to initialize grid
		///////////////////////////////////////////////////////////////////////////
		float3 bmin = make_float3(1e15f, 1e15f, 1e15f);
		float3 bmax = -bmin;
		for (int i = 0; i<number; i++)
		{
			bmin.x = min(bmin.x, host_X[i].x);
			bmax.x = max(bmax.x, host_X[i].x);
			bmin.y = min(bmin.y, host_X[i].y);
			bmax.y = max(bmax.y, host_X[i].y);
			bmin.z = min(bmin.z, host_X[i].z);
			bmax.z = max(bmax.z, host_X[i].z);
		}

		// Initialize the culling grid sizes
		const float3 start = bmin - 1.5f * h;
		int3 size;
		size.x = (int)((bmax.x - start.x)*inv_h + 2);
		size.y = (int)((bmax.y - start.y)*inv_h + 2);
		size.z = (int)((bmax.z - start.z)*inv_h + 2);
		const int	bucket_size = size.x*size.y*size.z * 2;
		const int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;
		const int t_blocksPerGrid = (t_number + t_threadsPerBlock - 1) / t_threadsPerBlock;
		allocate(number, t_number, bucket_size);

		// bind some frequent-queried arraies to textures, to speed up.
		size_t offset = 0;
		cudaChannelFormatDesc desc_float = cudaCreateChannelDesc<float>();
		cudaChannelFormatDesc desc_int = cudaCreateChannelDesc<int>();
		cudaSafeCall(cudaBindTexture(&offset, &g_oldX_tex, dev_old_X, &desc_float,
			number * sizeof(float3)));
		cudaSafeCall(cudaBindTexture(&offset, &g_newX_tex, dev_new_X, &desc_float,
			number * sizeof(float3)));
		cudaSafeCall(cudaBindTexture(&offset, &g_cstatus_tex, m_dev_c_status, &desc_int,
			bucket_size / 2 * sizeof(int)));
		cudaSafeCall(cudaBindTexture(&offset, &g_vid_tex, m_dev_vertex_id, &desc_int,
			number * sizeof(int)));
		cudaSafeCall(cudaBindTexture(&offset, &g_bucket_range_tex, m_dev_bucket_ranges, &desc_int,
			bucket_size * sizeof(int)));

		///////////////////////////////////////////////////////////////////////////
		//	Step 2: Vertex Grid Construction (for VT tests)
		///////////////////////////////////////////////////////////////////////////

		//assign vertex_id and vertex_bucket (let us use the new X for grid test)
		Grid_0_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			dev_old_X, m_dev_vertex_id, m_dev_vertex_bucket, number,
			start, inv_h, size);
		thrust_wrapper::sort_by_key(m_dev_vertex_bucket, m_dev_vertex_id, number);

		cudaMemset(m_dev_bucket_ranges, 0, sizeof(int)*bucket_size);
		Grid_1_Kernel << <blocksPerGrid, threadsPerBlock >> >(
			m_dev_vertex_bucket, m_dev_bucket_ranges, number);
		cudaMemset(m_dev_c_status, 0, sizeof(int)*bucket_size / 2);

		static thrust::device_vector<int> not_converged(1);
		int l = 0;
		for (l = 0; l<64; l++)
		{
			cudaMemset(m_dev_I, 0, sizeof(float3)*number);
			cudaMemset(m_dev_R, 0, sizeof(float3)*number);
			cudaMemset(m_dev_W, 0, sizeof(float)*number);

			// reset converge flag
			not_converged[0] = 0;

			Triangle_Test_Kernel << <t_blocksPerGrid, t_threadsPerBlock >> >(
				number, dev_T, t_number, m_dev_I, m_dev_R, m_dev_W, m_dev_c_status, l, gap,
				start, inv_h, size, 0, 0, not_converged.data().get());

			// check converge flag, this is faster than max_element check
			if (!not_converged[0])
				break;
			//int res = thrust_wrapper::max_element(m_dev_c_status, bucket_size / 2);
			//if (res != l + 1)	
			//	break;

			Triangle_Test_2_Kernel << <blocksPerGrid, threadsPerBlock >> >(
				dev_new_X, dev_V, m_dev_I, m_dev_R, m_dev_W, number, l, inv_t);
		}

#ifdef ENABLE_TIMER
		cudaThreadSynchronize();
		float t = timer.Get_Time();
		printf("L%d %f\n", l, t / max(l,1));
#endif
		if (l == 64)
			printf("ERROR: collision still not converge.\n"); 
	}
}