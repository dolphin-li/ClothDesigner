#pragma once

#include "thrust_wrapper.h"

class CachedDeviceBuffer
{
public:
	CachedDeviceBuffer();
	CachedDeviceBuffer(size_t bytes);
	~CachedDeviceBuffer();
	void create(size_t bytes);
	void release();
	char* data(){ return m_data; }
	const char* data()const{ return m_data; }
	size_t bytes()const{ return m_bytes; }
	void fromHost(const void* host, size_t bytes);
	void toHost(void* host)const;
protected:
	char* m_data = nullptr;
	size_t m_bytes = 0;
};