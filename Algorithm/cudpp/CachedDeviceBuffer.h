#pragma once

#include "thrust_wrapper.h"

class CachedDeviceBuffer
{
public:
	CachedDeviceBuffer();
	CachedDeviceBuffer(const CachedDeviceBuffer& rhs);
	CachedDeviceBuffer(size_t bytes);
	~CachedDeviceBuffer();
	void create(size_t bytes);
	void release();
	char* data(){ return m_data; }
	const char* data()const{ return m_data; }
	size_t bytes()const{ return m_bytes; }
	bool empty()const{ return !m_data; }
	void fromHost(const void* host, size_t bytes);
	void toHost(void* host)const;
	CachedDeviceBuffer& operator = (const CachedDeviceBuffer& rhs);
	void swap(CachedDeviceBuffer& rhs);
	void copyTo(CachedDeviceBuffer& rhs);
protected:
	char* m_data = nullptr;
	size_t m_bytes = 0;
	int* m_refcount = nullptr;
};