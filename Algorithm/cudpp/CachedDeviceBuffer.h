#pragma once

#include "thrust_wrapper.h"
#include <vector>
class CachedDeviceBuffer
{
public:
	CachedDeviceBuffer();
	CachedDeviceBuffer(const CachedDeviceBuffer& rhs);
	CachedDeviceBuffer(size_t bytes);
	~CachedDeviceBuffer();
	void create(size_t bytes, bool setZero = true);
	void release();
	char* data(){ return m_data; }
	const char* data()const{ return m_data; }
	size_t bytes()const{ return m_bytes; }
	bool empty()const{ return !m_data; }
	void fromHost(const void* host, size_t bytes);
	void toHost(void* host)const;
	CachedDeviceBuffer& operator = (const CachedDeviceBuffer& rhs);
	void swap(CachedDeviceBuffer& rhs);
	void copyTo(CachedDeviceBuffer& rhs)const;
protected:
	char* m_data = nullptr;
	size_t m_bytes = 0;
	int* m_refcount = nullptr;
};

template<class T>
class CachedDeviceArray : public CachedDeviceBuffer
{
public:
	typedef T type;
	enum { elem_size = sizeof(T) };

	CachedDeviceArray() :CachedDeviceBuffer(){}
	CachedDeviceArray(size_t size) :CachedDeviceBuffer(size*elem_size){}
	CachedDeviceArray(const CachedDeviceArray& other) :CachedDeviceBuffer(other){}
	CachedDeviceArray& operator = (const CachedDeviceArray& other)
	{
		CachedDeviceBuffer::operator=(other);
		return *this;
	}
	void create(size_t size, bool setZero = true)
	{
		CachedDeviceBuffer::create(size*elem_size, setZero);
	}
	void release()
	{
		CachedDeviceBuffer::release();
	}
	void copyTo(CachedDeviceArray& other) const
	{
		CachedDeviceBuffer::copyTo(other);
	}
	void fromHost(const T *host_ptr, size_t size)
	{
		CachedDeviceBuffer::fromHost(host_ptr, size*elem_size);
	}
	void toHost(T *host_ptr) const
	{
		CachedDeviceArray::toHost(host_ptr);
	}
	size_t size()const{ return bytes() / elem_size; }
	void fromHost(const std::vector<T>& data)
	{
		fromHost(data.data(), data.size());
	}
	void toHost(std::vector<T>& data) const
	{
		data.resize(size());
		toHost(data.data());
	}
	void swap(CachedDeviceArray& other_arg)
	{
		CachedDeviceBuffer::swap(other_arg);
	}
	T* data(){ return (T*)CachedDeviceBuffer::data(); }
	const T* data()const { return (const T*)CachedDeviceBuffer::data(); }
	operator T*(){ return data(); }
	operator const T*() const{ return data(); }
};