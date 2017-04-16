#pragma once

#include "Cuda3DArray.h"
#include "CachedDeviceBuffer.h"
namespace Cuda3DArray_Internal
{
	int add_ref(void* addr, int delta)
	{
		return _InterlockedExchangeAdd((long volatile*)addr, delta);
	}

	template<class T>
	__global__ void copyArrayHostToDeviceKernel(const T* data, int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
			{
				int pos = (z*res.y + y)*res.x + x;
				surf3Dwrite(data[pos], surf, x*sizeof(T), y, z);
			}
		}
	}

	template<class T>
	__global__ void copyArrayDeviceToHostKernel(T* data, int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
			{
				int pos = (z*res.y + y)*res.x + x;
				surf3Dread(&data[pos], surf, x*sizeof(T), y, z);
			}
		}
	}

	template<class T> void copy_d2h_t(const cudaSurfaceObject_t ary, int3 size, T* data)
	{
		const int sizeXYZ = size.x * size.y * size.z;
		CachedDeviceBuffer tmp;
		tmp.create(sizeXYZ*sizeof(T));

		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(size.x, block.x);
		grid.y = divUp(size.y, block.y);

		copyArrayDeviceToHostKernel << <grid, block >> >((T*)tmp.data(), size, ary);
		cudaSafeCall(cudaGetLastError());

		tmp.toHost(data);
	}

	void copy_d2h(const cudaSurfaceObject_t ary, int3 size, float* data)
	{
		copy_d2h_t(ary, size, data);
	}
	void copy_d2h(const cudaSurfaceObject_t ary, int3 size, float4* data)
	{
		copy_d2h_t(ary, size, data);
	}
	void copy_d2h(const cudaSurfaceObject_t ary, int3 size, int* data)
	{
		copy_d2h_t(ary, size, data);
	}
	void copy_d2h(const cudaSurfaceObject_t ary, int3 size, int4* data)
	{
		copy_d2h_t(ary, size, data);
	}

	template<class T> void copy_h2d_t(cudaSurfaceObject_t ary, int3 size, const T* data)
	{
		const int sizeXYZ = size.x * size.y * size.z;
		CachedDeviceBuffer tmp;
		tmp.fromHost(data, sizeXYZ*sizeof(T));

		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(size.x, block.x);
		grid.y = divUp(size.y, block.y);
		copyArrayHostToDeviceKernel << <grid, block >> >((const T*)tmp.data(), size, ary);
		cudaSafeCall(cudaGetLastError());
	}

	void copy_h2d(cudaSurfaceObject_t ary, int3 size, const float* data)
	{
		copy_h2d_t(ary, size, data);
	}
	void copy_h2d(cudaSurfaceObject_t ary, int3 size, const float4* data)
	{
		copy_h2d_t(ary, size, data);
	}
	void copy_h2d(cudaSurfaceObject_t ary, int3 size, const int* data)
	{
		copy_h2d_t(ary, size, data);
	}
	void copy_h2d(cudaSurfaceObject_t ary, int3 size, const int4* data)
	{
		copy_h2d_t(ary, size, data);
	}
}