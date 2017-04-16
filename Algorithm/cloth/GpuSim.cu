#include "GpuSim.h"

#include "ldpMat\ldp_basic_mat.h"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\CachedDeviceBuffer.h"
namespace ldp
{
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

#pragma region --StretchingSamples and BendingData
	__global__ void copyArrayHostToDeviceKernel(const float4* data, int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
			{
				int pos = (z*res.y + y)*res.x + x;
				surf3Dwrite(data[pos], surf, x*sizeof(float4), y, z);
			}
		}
	}
	__global__ void copyArrayDeviceToHostKernel(float4* data, int3 res, cudaSurfaceObject_t surf)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < res.x && y < res.y)
		{
			for (int z = 0; z < res.z; ++z)
			{
				int pos = (z*res.y + y)*res.x + x;
				surf3Dread(&data[pos], surf, x*sizeof(float4), y, z);
			}
		}
	}

	void GpuSim::StretchingSamples::initDeviceMemory()
	{
		releaseDeviceMemory();
		cudaExtent ext = make_cudaExtent(SAMPLES, SAMPLES, SAMPLES);
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
		cudaSafeCall(cudaMalloc3DArray(&m_ary, &desc, ext));
		m_tex = GpuSim::createTexture(m_ary, cudaFilterModeLinear);
		m_surf = GpuSim::createSurface(m_ary);
	}

	void GpuSim::StretchingSamples::releaseDeviceMemory()
	{
		if (m_surf)
		{
			cudaSafeCall(cudaDestroySurfaceObject(m_surf));
			m_surf = 0;
		}
		if (m_tex)
		{
			cudaSafeCall(cudaDestroyTextureObject(m_tex));
			m_tex = 0;
		}
		if (m_ary)
		{
			int deviceCount = 0;
			cudaGetDeviceCount(&deviceCount);
			int device = 0;
			cudaGetDevice(&device);
			printf("device count: %d, device: %d\n", deviceCount, device);
			printf("1: %ld\n", m_ary);
			cudaSafeCall(cudaFreeArray(m_ary));
			printf("2: %ld\n", m_ary);
			m_ary = nullptr;
		}
	}

	void GpuSim::StretchingSamples::updateHostToDevice()
	{
		CachedDeviceBuffer tmp;
		tmp.fromHost(data(), size()*sizeof(float4));

		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(SAMPLES, block.x);
		grid.y = divUp(SAMPLES, block.y);

		copyArrayHostToDeviceKernel << <grid, block >> >((const float4*)tmp.data(), 
			make_int3(SAMPLES, SAMPLES, SAMPLES), m_surf);
		cudaSafeCall(cudaGetLastError());
	}

	void GpuSim::StretchingSamples::updateDeviceToHost()
	{
		CachedDeviceBuffer tmp;
		tmp.create(size()*sizeof(float4));

		dim3 block(32, 16);
		dim3 grid(1, 1, 1);
		grid.x = divUp(SAMPLES, block.x);
		grid.y = divUp(SAMPLES, block.y);

		copyArrayDeviceToHostKernel << <grid, block >> >((float4*)tmp.data(), 
			make_int3(SAMPLES, SAMPLES, SAMPLES), m_surf);
		cudaSafeCall(cudaGetLastError());

		tmp.toHost(data());
	}

	void GpuSim::BendingData::initDeviceMemory()
	{
		releaseDeviceMemory();
		m_ary.create(rows(), cols());
		m_tex = ldp::createTexture(m_ary, cudaFilterModeLinear);
	}

	void GpuSim::BendingData::releaseDeviceMemory()
	{
		if (m_tex)
			cudaSafeCall(cudaDestroyTextureObject(m_tex));
		m_tex = 0;
		m_ary.release();
	}

	void GpuSim::BendingData::updateHostToDevice()
	{
		m_ary.upload(data(), cols()*sizeof(float), rows(), cols());
	}

	void GpuSim::BendingData::updateDeviceToHost()
	{
		m_ary.download(data(), cols()*sizeof(float));
	}
#pragma endregion
}