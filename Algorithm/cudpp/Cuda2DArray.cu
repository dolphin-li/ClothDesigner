#pragma once

#include "Cuda2DArray.h"
#include "CachedDeviceBuffer.h"
namespace Cuda2DArray_Internal
{
	int add_ref(void* addr, int delta)
	{
		return _InterlockedExchangeAdd((long volatile*)addr, delta);
	}
}