#include "GpuSim.h"

#include "ldpMat\ldp_basic_mat.h"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\CachedDeviceBuffer.h"
namespace ldp
{
	enum{
		CTA_SIZE = 512
	};

	template<class T>
	static cudaTextureObject_t createTexture(DeviceArray2D<T>& ary, cudaTextureFilterMode filterMode)
	{
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypePitch2D;
		texRes.res.pitch2D.height = ary.rows();
		texRes.res.pitch2D.width = ary.cols();
		texRes.res.pitch2D.pitchInBytes = ary.step();
		texRes.res.pitch2D.desc = cudaCreateChannelDesc<T>();
		texRes.res.pitch2D.devPtr = ary.ptr();
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = 0;
		texDescr.filterMode = filterMode;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		cudaTextureObject_t tex;
		cudaSafeCall(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
		return tex;
	}

	__global__ void vertPair_to_idx_kernel(const int* v1, const int* v2, int* ids, int nVerts, int nPairs)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < nPairs)
			ids[i] = vertPair_to_idx(ldp::Int2(v1[i], v2[i]), nVerts);
	}
	__global__ void vertPair_from_idx_kernel(int* v1, int* v2, const int* ids, int nVerts, int nPairs)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < nPairs)
		{
			ldp::Int2 vert = vertPair_from_idx(ids[i], nVerts);
			v1[i] = vert[0];
			v2[i] = vert[1];
		}
	}

	void GpuSim::vertPair_to_idx(const int* v1, const int* v2, int* ids, int nVerts, int nPairs)
	{
		vertPair_to_idx_kernel << <divUp(nPairs, CTA_SIZE), CTA_SIZE >> >(
			v1, v2, ids, nVerts, nPairs);
		cudaSafeCall(cudaGetLastError());
	}

	void GpuSim::vertPair_from_idx(int* v1, int* v2, const int* ids, int nVerts, int nPairs)
	{
		vertPair_from_idx_kernel << <divUp(nPairs, CTA_SIZE), CTA_SIZE >> >(
			v1, v2, ids, nVerts, nPairs);
		cudaSafeCall(cudaGetLastError());
	}
}