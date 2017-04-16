/***********************************************************/
/**	\file
	\brief		cuda utils used by Kinect Fusion
	\details	
	\author		Yizhong Zhang
	\date		11/13/2013
*/
/***********************************************************/
#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime_api.h"
#pragma comment(lib, "cudart.lib")
#include <exception>

#define cudaSafeCall(err) ___cudaSafeCall(err, __FILE__, __LINE__)


static inline void ___cudaSafeCall(cudaError_t err, const char* file, int line)
{
	if (cudaSuccess != err)
	{
		printf("[CUDA error][%s][%d]: %s\n", file, line, cudaGetErrorString(err));
		throw std::exception("cuda error");
	}
}

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }



#endif