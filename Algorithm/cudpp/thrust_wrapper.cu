#include <device_functions.h>
#include <driver_functions.h>
#include <vector_functions.h>
#include <cuda_fp16.h>
#include <channel_descriptor.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include <surface_types.h>
#include <surface_functions.h>
#include "helper_math.h"

#include <thrust\device_ptr.h>
#include <thrust\sort.h>
#include <thrust\scan.h>
#include <thrust\unique.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <map>
#include <algorithm>
#include "thrust_wrapper.h"

namespace thrust_wrapper
{
	char* cached_allocate(size_t bytes)
	{
		return g_allocator.allocate(bytes);
	}
	void cached_free(char* ptr)
	{
		if (ptr != nullptr)
			g_allocator.deallocate(ptr, 0);
	}

	int max_element(const int* data, int n)
	{
		thrust::device_ptr<int> data_begin((int*)data);
		thrust::device_ptr<int> data_end((int*)data + n);
		auto ret = thrust::max_element(thrust::cuda::par(g_allocator), data_begin, data_end);
		return ret[0];
	}

	void stable_sort_by_key(int* key_d, float4* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<float4> points_begin(value_d);
		thrust::stable_sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void stable_sort_by_key(float* key_d, int* value_d, int n)
	{
		thrust::device_ptr<float> key_begin(key_d);
		thrust::device_ptr<float> key_end(key_d + n);
		thrust::device_ptr<int> points_begin(value_d);
		thrust::stable_sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void sort_by_key(int* key_d, float4* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<float4> points_begin(value_d);
		thrust::sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void sort_by_key(float* key_d, int* value_d, int n)
	{
		thrust::device_ptr<float> key_begin(key_d);
		thrust::device_ptr<float> key_end(key_d + n);
		thrust::device_ptr<int> points_begin(value_d);
		thrust::sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void sort_by_key(int* key_d, int* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<int> points_begin(value_d);
		thrust::sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void sort_by_key(size_t* key_d, int* value_d, int n)
	{
		thrust::device_ptr<size_t> key_begin(key_d);
		thrust::device_ptr<size_t> key_end(key_d + n);
		thrust::device_ptr<int> points_begin(value_d);
		thrust::sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void sort_by_key(int* key_d, float* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<float> points_begin(value_d);
		thrust::sort_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
	}

	void exclusive_scan(const int* in, int* out, int n)
	{
		thrust::device_ptr<int> in_begin((int*)in);
		thrust::device_ptr<int> in_end((int*)in + n);
		thrust::device_ptr<int> out_begin(out);
		thrust::exclusive_scan(thrust::cuda::par(g_allocator), in_begin, in_end, out_begin);
	}

	void exclusive_scan(const unsigned int* in, unsigned int* out, int n)
	{
		thrust::device_ptr<unsigned int> in_begin((unsigned int*)in);
		thrust::device_ptr<unsigned int> in_end((unsigned int*)in + n);
		thrust::device_ptr<unsigned int> out_begin(out);
		thrust::exclusive_scan(thrust::cuda::par(g_allocator), in_begin, in_end, out_begin);
	}

	void inclusive_scan(const int* in, int* out, int n)
	{
		thrust::device_ptr<int> in_begin((int*)in);
		thrust::device_ptr<int> in_end((int*)in + n);
		thrust::device_ptr<int> out_begin(out);
		thrust::inclusive_scan(thrust::cuda::par(g_allocator), in_begin, in_end, out_begin);
	}
	void inclusive_scan_by_key(const int* key_d, const float4* value_d, float4* dst_d, int n)
	{
		thrust::device_ptr<int> key_begin((int*)key_d);
		thrust::device_ptr<int> key_end((int*)key_d + n);
		thrust::device_ptr<float4> points_begin((float4*)value_d);
		thrust::device_ptr<float4> dst_begin(dst_d);
		thrust::inclusive_scan_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin, dst_begin);
	}
	void inclusive_scan_by_key(const int* key_d, const float3* value_d, float3* dst_d, int n)
	{
		thrust::device_ptr<int> key_begin((int*)key_d);
		thrust::device_ptr<int> key_end((int*)key_d + n);
		thrust::device_ptr<float3> points_begin((float3*)value_d);
		thrust::device_ptr<float3> dst_begin(dst_d);
		thrust::inclusive_scan_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin, dst_begin);
	}

	size_t unique_by_key(int* key_d, float4* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<float4> points_begin(value_d);
		auto ptr = thrust::unique_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
		return ptr.first - key_begin;
	}

	size_t unique_by_key(int* key_d, int* value_d, int n)
	{
		thrust::device_ptr<int> key_begin(key_d);
		thrust::device_ptr<int> key_end(key_d + n);
		thrust::device_ptr<int> points_begin(value_d);
		auto ptr = thrust::unique_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
		return ptr.first - key_begin;
	}

	size_t unique_by_key(size_t* key_d, int* value_d, int n)
	{
		thrust::device_ptr<size_t> key_begin(key_d);
		thrust::device_ptr<size_t> key_end(key_d + n);
		thrust::device_ptr<int> points_begin(value_d);
		auto ptr = thrust::unique_by_key(thrust::cuda::par(g_allocator), key_begin, key_end, points_begin);
		return ptr.first - key_begin;
	}

	int reduce(const int* in, int n)
	{
		thrust::device_ptr<int> key_begin((int*)in);
		thrust::device_ptr<int> key_end((int*)in + n);
		return thrust::reduce(thrust::cuda::par(g_allocator), key_begin, key_end, 0);
	}

	size_t unique(int* data, int n)
	{
		thrust::device_ptr<int> key_begin(data);
		thrust::device_ptr<int> key_end(data + n);
		auto ptr = thrust::unique(thrust::cuda::par(g_allocator), key_begin, key_end);
		return ptr - key_begin;
	}

	size_t unique(size_t* data, int n)
	{
		thrust::device_ptr<size_t> key_begin(data);
		thrust::device_ptr<size_t> key_end(data + n);
		auto ptr = thrust::unique(thrust::cuda::par(g_allocator), key_begin, key_end);
		return ptr - key_begin;
	}

	size_t unique_counting_by_key_copy(const int* key_d, int input_count_begin,
		int* out_key_d, int* out_value_d, int n)
	{
		thrust::device_ptr<int> key_begin((int*)key_d);
		thrust::device_ptr<int> key_end((int*)key_d + n);
		thrust::counting_iterator<int> input_value_begin(input_count_begin);
		thrust::device_ptr<int> out_key_begin(out_key_d);
		thrust::device_ptr<int> out_value_begin(out_value_d);
		auto ptr = thrust::unique_by_key_copy(thrust::cuda::par(g_allocator), key_begin, key_end, input_value_begin,
			out_key_begin, out_value_begin);
		return ptr.first - out_key_begin;
	}

	void scatter(const int* in_d, const int* map_d, int* out_d, int n)
	{
		thrust::device_ptr<int> first((int*)in_d);
		thrust::device_ptr<int> last((int*)in_d + n);
		thrust::device_ptr<int> map((int*)map_d);
		thrust::device_ptr<int> output(out_d);
		thrust::scatter(thrust::cuda::par(g_allocator), first, last, map, output);
	}

	void gather(const int* in_d, const int* map_d, int* out_d, int n)
	{
		thrust::device_ptr<int> first((int*)in_d);
		thrust::device_ptr<int> map((int*)map_d);
		thrust::device_ptr<int> map_last((int*)map_d + n);
		thrust::device_ptr<int> output(out_d);
		thrust::gather(thrust::cuda::par(g_allocator), map, map_last, first, output);
	}


	void gather(const float4* in_d, const int* map_d, float4* out_d, int n)
	{
		thrust::device_ptr<float4> first((float4*)in_d);
		thrust::device_ptr<int> map((int*)map_d);
		thrust::device_ptr<int> map_last((int*)map_d + n);
		thrust::device_ptr<float4> output(out_d);
		thrust::gather(thrust::cuda::par(g_allocator), map, map_last, first, output);
	}

	//! converts a float4 point (xyz) to a tuple of three float vals (used to separate the
	//! float4 input buffer into three arrays in the beginning of the tree build)
	struct pointxyz_to_px_py_pz
	{
		__device__
		thrust::tuple<float, float, float> operator()(const float4& val)
		{
			return thrust::make_tuple(val.x, val.y, val.z);
		}
	};

	void seperate_channels(const float4* xyzw, float* x, float* y, float* z, int n)
	{
		thrust::device_ptr<float4> xyzw_begin((float4*)xyzw);
		thrust::device_ptr<float4> xyzw_end((float4*)xyzw + n);
		thrust::device_ptr<float> xs(x);
		thrust::device_ptr<float> ys(y);
		thrust::device_ptr<float> zs(z);
		thrust::transform(thrust::cuda::par(g_allocator), xyzw_begin, xyzw_end,
			thrust::make_zip_iterator(thrust::make_tuple(xs, ys, zs)),
			pointxyz_to_px_py_pz());
	}

	void make_counting_array(int* ptr_d, int n, int begin)
	{
		thrust::device_ptr<int> data(ptr_d);
		thrust::counting_iterator<int> it(begin);
		thrust::copy(thrust::cuda::par(g_allocator), it, it + n, data);
	}

	void copy(int* dst, const int* src, int size)
	{
		thrust::device_ptr<int> src_begin((int*)src);
		thrust::device_ptr<int> src_end((int*)src + size);
		thrust::device_ptr<int> dst_begin(dst);
		thrust::copy(thrust::cuda::par(g_allocator), src_begin, src_end, dst_begin);
	}

	void copy(float* dst, const float* src, int size)
	{
		thrust::device_ptr<float> src_begin((float*)src);
		thrust::device_ptr<float> src_end((float*)src + size);
		thrust::device_ptr<float> dst_begin(dst);
		thrust::copy(thrust::cuda::par(g_allocator), src_begin, src_end, dst_begin);
	}

	void assign(int* ptr_d, int value, int n)
	{
		thrust::device_ptr<int> data_begin(ptr_d);
		thrust::device_ptr<int> data_end(ptr_d + n);
		thrust::fill(thrust::cuda::par(g_allocator), data_begin, data_end, value);
	}

	void assign(float* ptr_d, float value, int n)
	{
		thrust::device_ptr<float> data_begin(ptr_d);
		thrust::device_ptr<float> data_end(ptr_d + n);
		thrust::fill(thrust::cuda::par(g_allocator), data_begin, data_end, value);
	}

	void assign(float4* ptr_d, float4 value, int n)
	{
		thrust::device_ptr<float4> data_begin(ptr_d);
		thrust::device_ptr<float4> data_end(ptr_d + n);
		thrust::fill(thrust::cuda::par(g_allocator), data_begin, data_end, value);
	}


	float dot(int n, const float* a, const float* b)
	{
		thrust::device_ptr<float> a_begin((float*)a);
		thrust::device_ptr<float> a_end((float*)a + n);
		thrust::device_ptr<float> b_begin((float*)b);
		thrust::device_ptr<float> b_end((float*)b + n);
		return thrust::inner_product(thrust::cuda::par(g_allocator), a_begin, a_end, b_begin, 0.f);
	}
}