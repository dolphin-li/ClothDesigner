#include "CachedDeviceBuffer.h"
#include "thrust_wrapper.h"
#include "cuda_utils.h"
#include <intrin.h>
#include <algorithm>
#define CV_XADD(addr,delta) _InterlockedExchangeAdd((long volatile*)(addr), (delta))

CachedDeviceBuffer::CachedDeviceBuffer()
{

}

CachedDeviceBuffer::CachedDeviceBuffer(const CachedDeviceBuffer& rhs):
m_data(rhs.m_data), m_bytes(rhs.m_bytes), m_refcount(rhs.m_refcount)
{
	if (m_refcount)
		CV_XADD(m_refcount, 1);
}

CachedDeviceBuffer::CachedDeviceBuffer(size_t bytes)
{
	create(bytes);
}

CachedDeviceBuffer::~CachedDeviceBuffer()
{
	release();
}

CachedDeviceBuffer& CachedDeviceBuffer::operator = (const CachedDeviceBuffer& rhs)
{
	if (this != &rhs)
	{
		if (rhs.m_refcount)
			CV_XADD(rhs.m_refcount, 1);
		release();

		m_data = rhs.m_data;
		m_bytes = rhs.m_bytes;
		m_refcount = rhs.m_refcount;
	}
	return *this;
}

void CachedDeviceBuffer::swap(CachedDeviceBuffer& rhs)
{
	std::swap(m_data, rhs.m_data);
	std::swap(m_bytes, rhs.m_bytes);
	std::swap(m_refcount, rhs.m_refcount);
}

void CachedDeviceBuffer::copyTo(CachedDeviceBuffer& rhs)
{
	if (empty())
		rhs.release();
	else
	{
		rhs.create(m_bytes);
		cudaSafeCall(cudaMemcpy(rhs.m_data, m_data, m_bytes, cudaMemcpyDeviceToDevice));
		cudaSafeCall(cudaDeviceSynchronize());
	}
}

void CachedDeviceBuffer::create(size_t bytes)
{
	if (bytes == m_bytes || bytes == 0)
		return;
	release();
	m_data = thrust_wrapper::cached_allocate(bytes);
	m_bytes = bytes;
	m_refcount = new int;
	*m_refcount = 1;
}
void CachedDeviceBuffer::release()
{
	if (m_refcount && CV_XADD(m_refcount, -1) == 1)
	{
		delete m_refcount;
		thrust_wrapper::cached_free(m_data);
	}

	m_bytes = 0;
	m_data = nullptr;
	m_refcount = nullptr;
}

void CachedDeviceBuffer::fromHost(const void* host, size_t bytes)
{
	create(bytes);
	cudaSafeCall(cudaMemcpy(m_data, host, m_bytes, cudaMemcpyHostToDevice));
}
void CachedDeviceBuffer::toHost(void* host)const
{
	cudaSafeCall(cudaMemcpy(host, m_data, m_bytes, cudaMemcpyDeviceToHost));
}