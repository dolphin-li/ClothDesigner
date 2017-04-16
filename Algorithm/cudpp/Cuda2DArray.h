#pragma once

#include <device_functions.h>
#include <driver_functions.h>
#include <vector_functions.h>
#include <surface_types.h>
#include "cuda_utils.h"
#include <intrin.h>
#include "device_array.h"
namespace Cuda2DArray_Internal
{
	int add_ref(void* addr, int delta);
}

template<class T>
class Cuda2DArray
{
public:
	Cuda2DArray(){ m_size = make_int2(0, 0); }
	Cuda2DArray(const Cuda2DArray& rhs) :m_data(rhs.m_data), 
		m_size(rhs.m_size), m_tex(rhs.m_tex), m_filterMode(rhs.m_filterMode)
	{
		if (m_refcount)
			Cuda2DArray_Internal::add_ref(m_refcount, 1);
	}
	Cuda2DArray(int2 size, cudaTextureFilterMode mode = cudaFilterModeLinear){ create(size, mode); }
	~Cuda2DArray(){ release(); }
	void create(int2 size, cudaTextureFilterMode mode = cudaFilterModeLinear)
	{
		if (m_size.x == size.x && m_size.y == size.y && m_filterMode == mode)
			return;
		release();
		if (size.x == 0 || size.y == 0)
			return;
		m_filterMode = mode;
		m_size = size;
		m_data.create(m_size.y, m_size.x);
		createTexture();
		m_refcount = new int;
		*m_refcount = 1;
	}
	void release()
	{
		if (m_refcount && Cuda2DArray_Internal::add_ref(m_refcount, -1) == 1)
		{
			delete m_refcount;	
			if (m_tex)
				cudaSafeCall(cudaDestroyTextureObject(m_tex));
			if (m_data)
				m_data.release();
		}

		m_size = make_int2(0, 0);
		m_tex = 0;
		m_refcount = nullptr;
		m_filterMode = cudaFilterModeLinear;
	}
	DeviceArray2D<T>& data(){ return m_data; }
	const DeviceArray2D<T>& data()const{ return m_data; }
	int2 size()const{ return m_size; }
	size_t sizeXY()const{ return size_t(m_size.x)*size_t(m_size.y); }
	bool empty()const{ return !m_data; }
	cudaTextureObject_t getCudaTexture()const{ return m_tex; }
	cudaTextureFilterMode getCudaTextureFilterMode()const{ return m_filterMode; }
	void fromHost(const T* host, int2 size, cudaTextureFilterMode mode = cudaFilterModeLinear)
	{
		create(size, mode);
		m_data.upload(host, m_size.x * sizeof(T), m_size.y, m_size.x);
	}
	void toHost(T* host)const
	{
		m_data.download(host, m_size.x * sizeof(T));
	}
	Cuda2DArray& operator = (const Cuda2DArray& rhs)
	{
		if (this != &rhs)
		{
			if (rhs.m_refcount)
				Cuda2DArray_Internal::add_ref(rhs.m_refcount, 1);
			release();

			m_data = rhs.m_data;
			m_size = rhs.m_size;
			m_tex = rhs.m_tex;
			m_refcount = rhs.m_refcount;
			m_filterMode = rhs.m_filterMode;
		}
		return *this;
	}
	void swap(Cuda2DArray& rhs)
	{
		m_data.swap(rhs.m_data);
		std::swap(m_size, rhs.m_size);
		std::swap(m_tex, rhs.m_tex);
		std::swap(m_surf, rhs.m_surf);
		std::swap(m_refcount, rhs.m_refcount);
		std::swap(m_filterMode, rhs.m_filterMode);
	}
	void copyTo(Cuda2DArray& rhs)const
	{
		if (empty())
			rhs.release();
		else
		{
			m_data.copyTo(rhs.data());
			if (rhs.m_filterMode != m_filterMode)
			{
				rhs.m_filterMode = m_filterMode;
				rhs.createTexture();
			}
		}
	}
protected:
	void createTexture()
	{
		if (m_tex)
			cudaSafeCall(cudaDestroyTextureObject(m_tex));
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypePitch2D;
		texRes.res.pitch2D.height = m_data.rows();
		texRes.res.pitch2D.width = m_data.cols();
		texRes.res.pitch2D.pitchInBytes = m_data.step();
		texRes.res.pitch2D.desc = cudaCreateChannelDesc<T>();
		texRes.res.pitch2D.devPtr = m_data.ptr();
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = 0;
		texDescr.filterMode = m_filterMode;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		cudaSafeCall(cudaCreateTextureObject(&m_tex, &texRes, &texDescr, NULL));
	}
protected:
	DeviceArray2D<T> m_data;
	int2 m_size;
	cudaTextureObject_t m_tex = 0;
	int* m_refcount = nullptr;
	cudaTextureFilterMode m_filterMode = cudaFilterModeLinear;
};