#include "CachedDeviceBuffer.h"
#include "thrust_wrapper.h"
#include "cuda_utils.h"

CachedDeviceBuffer::CachedDeviceBuffer()
{

}

CachedDeviceBuffer::CachedDeviceBuffer(size_t bytes)
{
	create(bytes);
}

CachedDeviceBuffer::~CachedDeviceBuffer()
{
	release();
}

void CachedDeviceBuffer::create(size_t bytes)
{
	if (bytes == m_bytes)
		return;
	release();
	m_data = thrust_wrapper::cached_allocate(bytes);
	m_bytes = bytes;
}
void CachedDeviceBuffer::release()
{
	thrust_wrapper::cached_free(m_data);
	m_bytes = 0;
	m_data = nullptr;
}

void CachedDeviceBuffer::fromHost(const void* host, size_t bytes)
{
	create(bytes);
	cudaSafeCall(cudaMemcpy(m_data, host, m_bytes,
		cudaMemcpyHostToDevice), "CachedDeviceBuffer::fromHost");
}
void CachedDeviceBuffer::toHost(void* host)const
{
	cudaSafeCall(cudaMemcpy(host, m_data, m_bytes,
		cudaMemcpyDeviceToHost), "CachedDeviceBuffer::toHost");
}