#pragma once

#include <device_functions.h>
#include <driver_functions.h>
#include <vector_functions.h>
#include <surface_types.h>
#include "cuda_utils.h"
#include <intrin.h>

namespace Cuda3DArray_Internal
{
	int add_ref(void* addr, int delta);
	void copy_d2h(const cudaSurfaceObject_t ary, int3 size, int* data);
	void copy_d2h(const cudaSurfaceObject_t ary, int3 size, int4* data);
	void copy_d2h(const cudaSurfaceObject_t ary, int3 size, float* data);
	void copy_d2h(const cudaSurfaceObject_t ary, int3 size, float4* data);
	void copy_h2d(cudaSurfaceObject_t ary, int3 size, const int* data);
	void copy_h2d(cudaSurfaceObject_t ary, int3 size, const int4* data);
	void copy_h2d(cudaSurfaceObject_t ary, int3 size, const float* data);
	void copy_h2d(cudaSurfaceObject_t ary, int3 size, const float4* data);
}

template<class T>
class Cuda3DArray
{
public:
	Cuda3DArray(){ m_size = make_int3(0, 0, 0); }
	Cuda3DArray(const Cuda3DArray& rhs) :m_data(rhs.m_data), 
		m_size(rhs.m_size), m_tex(rhs.m_tex), m_surf(rhs.m_surf)
	{
		if (m_refcount)
			Cuda3DArray_Internal::add_ref(m_refcount, 1);
	}
	Cuda3DArray(int3 size){ create(size); }
	~Cuda3DArray(){ release(); }
	void create(int3 size)
	{
		if (m_size.x == size.x && m_size.y == size.y && m_size.z == size.z)
			return;
		release();
		if (size.x == 0 || size.y == 0 || size.z == 0)
			return;
		m_size = size;
		cudaExtent ext = make_cudaExtent(m_size.x, m_size.y, m_size.z);
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaSafeCall(cudaMalloc3DArray(&m_data, &desc, ext));
		createTexture();
		createSurface();
		m_refcount = new int;
		*m_refcount = 1;
	}
	void release()
	{
		if (m_refcount && Cuda3DArray_Internal::add_ref(m_refcount, -1) == 1)
		{
			delete m_refcount;
			if (m_surf)
				cudaSafeCall(cudaDestroySurfaceObject(m_surf));		
			if (m_tex)
				cudaSafeCall(cudaDestroyTextureObject(m_tex));
			if (m_data)
				cudaSafeCall(cudaFreeArray(m_data));
		}

		m_size = make_int3(0, 0, 0);
		m_tex = 0;
		m_surf = 0;
		m_data = nullptr;
		m_refcount = nullptr;
	}
	cudaArray_t data(){ return m_data; }
	cudaArray_t data()const{ return m_data; }
	int3 size()const{ return m_size; }
	size_t sizeXYZ()const{ return size_t(m_size.x)*size_t(m_size.y)*size_t(m_size.z); }
	bool empty()const{ return !m_data; }
	cudaTextureObject_t getCudaTexture()const{ return m_tex; }
	cudaSurfaceObject_t getSurface()const{ return m_surf; }
	void fromHost(const T* host, int3 size)
	{
		create(size);
		Cuda3DArray_Internal::copy_h2d(m_surf, m_size, host);
	}
	void toHost(T* host)const
	{
		Cuda3DArray_Internal::copy_d2h(m_surf, m_size, host);
	}
	Cuda3DArray& operator = (const Cuda3DArray& rhs)
	{
		if (this != &rhs)
		{
			if (rhs.m_refcount)
				Cuda3DArray_Internal::add_ref(rhs.m_refcount, 1);
			release();

			m_data = rhs.m_data;
			m_size = rhs.m_size;
			m_tex = rhs.m_tex;
			m_surf = rhs.m_surf;
			m_refcount = rhs.m_refcount;
		}
		return *this;
	}
	void swap(Cuda3DArray& rhs)
	{
		std::swap(m_data, rhs.m_data);
		std::swap(m_size, rhs.m_size);
		std::swap(m_tex, rhs.m_tex);
		std::swap(m_surf, rhs.m_surf);
		std::swap(m_refcount, rhs.m_refcount);
	}
	void copyTo(Cuda3DArray& rhs)const
	{
		if (empty())
			rhs.release();
		else
		{
			rhs.create(m_size);
			cudaMemcpy3DParms params = { 0 };
			params.dstArray = rhs.m_data;
			params.srcArray = m_data;
			params.extent = make_cudaExtent(m_size.x, m_size.y, m_size.z);
			params.kind = cudaMemcpyDeviceToDevice;
			cudaSafeCall(cudaMemcpy3D(&params));
		}
	}
protected:
	void createTexture()
	{
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = m_data;
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = 0;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		cudaSafeCall(cudaCreateTextureObject(&m_tex, &texRes, &texDescr, NULL));
	}
	void createSurface()
	{
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = m_data;
		cudaSafeCall(cudaCreateSurfaceObject(&m_surf, &texRes));
	}
protected:
	cudaArray_t m_data = nullptr;
	int3 m_size;
	cudaTextureObject_t m_tex = 0;
	cudaSurfaceObject_t m_surf = 0;
	int* m_refcount = nullptr;
};