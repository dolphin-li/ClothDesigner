#include "GpuSim.h"

#include "ldpMat\ldp_basic_mat.h"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\CachedDeviceBuffer.h"
#include "LEVEL_SET_COLLISION.h"
#include <math.h>
#include "LevelSet3D.h"
namespace ldp
{

//#define USE_ONE_RING_EDGE_FOR_LEVELSET_COLLISION
//#define DEBUG_DUMP
#define LDP_DEBUG

#pragma region --utils
	enum{
		CTA_SIZE = 256,
		CTA_SIZE_X = 16,
		CTA_SIZE_Y = 16
	};
#define CHECK_ZERO(a){if(a)printf("!!!error: %s=%d\n", #a, a);}

	typedef ldp_basic_vec<float, 1> Float1;
	typedef ldp_basic_vec<float, 9> Float9;
	typedef ldp_basic_vec<float, 12> FloatC;
	typedef ldp_basic_mat_sqr<float, 9> Mat9f;
	typedef ldp_basic_mat_sqr<float, 12> MatCf;
	typedef ldp_basic_mat<float, 3, 2> Mat32f;
	typedef ldp_basic_mat<float, 5, 3> Mat53f;
	typedef ldp_basic_mat<float, 3, 9> Mat39f;
	typedef ldp_basic_mat<float, 2, 3> Mat23f;
	typedef ldp_basic_mat_col<float, 3> Mat31f;
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
	__device__ __host__ void print(float a, const char* msg = nullptr)
	{
		printf("%s: %f\n", msg, a);
	}
	template<int N, int M> __device__ __host__ void print(ldp_basic_mat<float, N, M> A, const char* msg = nullptr);
	template<int N> __device__ __host__  void print(ldp_basic_vec<float, N> x, const char* msg = nullptr);
	template<> __device__ __host__ void print(ldp_basic_mat<float, 2, 2> A, const char* msg)
	{
		printf("%s\n%f %f\n%f %f\n", msg, A(0, 0), A(0, 1), A(1, 0), A(1, 1));
	}
	template<> __device__ __host__ void print(ldp_basic_mat<float, 2, 3> A, const char* msg)
	{
		printf("%s\n%f %f %f\n%f %f %f\n", msg, A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2));
	}
	template<> __device__ __host__ void print(ldp_basic_mat<float, 5, 3> A, const char* msg)
	{
		printf("%s\n%f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n", msg, 
			A(0, 0), A(0, 1), A(0, 2), 
			A(1, 0), A(1, 1), A(1, 2),
			A(2, 0), A(2, 1), A(2, 2),
			A(3, 0), A(3, 1), A(3, 2),
			A(4, 0), A(4, 1), A(4, 2)
			);
	}
	template<> __device__ __host__ void print(ldp_basic_mat<float, 3, 2> A, const char* msg)
	{
		printf("%s\n%f %f\n%f %f\n%f %f\n", msg, A(0, 0), A(0, 1), A(1, 0), A(1, 1), A(2, 0), A(2, 1));
	}
	template<> __device__ __host__ void print(ldp_basic_mat<float, 3, 1> A, const char* msg)
	{
		printf("%s\n%f\n%f\n%f\n", msg, A(0, 0), A(1, 0), A(2, 0));
	}
	template<> __device__ __host__ void print(ldp_basic_mat<float, 3, 3> A, const char* msg)
	{
		printf("%s\n%f %f %f\n%f %f %f\n%f %f %f\n", msg, A(0, 0), A(0, 1), A(0, 2),
			A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2));
	}
	template<> __device__ __host__ void print(ldp_basic_mat<float, 3, 9> A, const char* msg)
	{
		printf("%s\n%f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f\n", 
			msg,
			   A(0, 0), A(0, 1), A(0, 2), A(0, 3), A(0, 4), A(0, 5), A(0, 6), A(0, 7), A(0, 8),
			   A(1, 0), A(1, 1), A(1, 2), A(1, 3), A(1, 4), A(1, 5), A(1, 6), A(1, 7), A(1, 8),
			   A(2, 0), A(2, 1), A(2, 2), A(2, 3), A(2, 4), A(2, 5), A(2, 6), A(2, 7), A(2, 8)
			   );
	}
	template<> __device__ __host__ void print(ldp_basic_mat<float, 9, 9> A, const char* msg)
	{
		printf("%s\n%f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f\n",
			msg,
			A(0, 0), A(0, 1), A(0, 2), A(0, 3), A(0, 4), A(0, 5), A(0, 6), A(0, 7), A(0, 8),
			A(1, 0), A(1, 1), A(1, 2), A(1, 3), A(1, 4), A(1, 5), A(1, 6), A(1, 7), A(1, 8),
			A(2, 0), A(2, 1), A(2, 2), A(2, 3), A(2, 4), A(2, 5), A(2, 6), A(2, 7), A(2, 8)
			);
		printf("%f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f\n",
			A(3, 0), A(3, 1), A(3, 2), A(3, 3), A(3, 4), A(3, 5), A(3, 6), A(3, 7), A(3, 8),
			A(4, 0), A(4, 1), A(4, 2), A(4, 3), A(4, 4), A(4, 5), A(4, 6), A(4, 7), A(4, 8),
			A(5, 0), A(5, 1), A(5, 2), A(5, 3), A(5, 4), A(5, 5), A(5, 6), A(5, 7), A(5, 8)
			);
		printf("%f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f\n",
			A(6, 0), A(6, 1), A(6, 2), A(6, 3), A(6, 4), A(6, 5), A(6, 6), A(6, 7), A(6, 8),
			A(7, 0), A(7, 1), A(7, 2), A(7, 3), A(7, 4), A(7, 5), A(7, 6), A(7, 7), A(7, 8),
			A(8, 0), A(8, 1), A(8, 2), A(8, 3), A(8, 4), A(8, 5), A(8, 6), A(8, 7), A(8, 8)
			);
	}
	template<> __device__ __host__ void print(ldp_basic_mat<float, 12, 12> A, const char* msg)
	{
		printf("%s\n%f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f %f %f\n",
			msg,
			A(0, 0), A(0, 1), A(0, 2), A(0, 3), A(0, 4), A(0, 5), A(0, 6), A(0, 7), A(0, 8), A(0, 9), A(0, 10), A(0, 11),
			A(1, 0), A(1, 1), A(1, 2), A(1, 3), A(1, 4), A(1, 5), A(1, 6), A(1, 7), A(1, 8), A(1, 9), A(1, 10), A(1, 11),
			A(2, 0), A(2, 1), A(2, 2), A(2, 3), A(2, 4), A(2, 5), A(2, 6), A(2, 7), A(2, 8), A(2, 9), A(2, 10), A(2, 11)
			);
		printf("%f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f %f %f\n",
			A(3, 0), A(3, 1), A(3, 2), A(3, 3), A(3, 4), A(3, 5), A(3, 6), A(3, 7), A(3, 8), A(3, 9), A(3, 10), A(3, 11),
			A(4, 0), A(4, 1), A(4, 2), A(4, 3), A(4, 4), A(4, 5), A(4, 6), A(4, 7), A(4, 8), A(4, 9), A(4, 10), A(4, 11),
			A(5, 0), A(5, 1), A(5, 2), A(5, 3), A(5, 4), A(5, 5), A(5, 6), A(5, 7), A(5, 8), A(5, 9), A(5, 10), A(5, 11)
			);
		printf("%f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f %f %f\n",
			A(6, 0), A(6, 1), A(6, 2), A(6, 3), A(6, 4), A(6, 5), A(6, 6), A(6, 7), A(6, 8), A(6, 9), A(6, 10), A(6, 11),
			A(7, 0), A(7, 1), A(7, 2), A(7, 3), A(7, 4), A(7, 5), A(7, 6), A(7, 7), A(7, 8), A(7, 9), A(7, 10), A(7, 11),
			A(8, 0), A(8, 1), A(8, 2), A(8, 3), A(8, 4), A(8, 5), A(8, 6), A(8, 7), A(8, 8), A(8, 9), A(8, 10), A(8, 11)
			);
		printf("%f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f %f %f %f %f %f %f %f %f %f\n",
			A(9, 0), A(9, 1), A(9, 2), A(9, 3), A(9, 4), A(9, 5), A(9, 6), A(9, 7), A(9, 8), A(9, 9), A(9, 10), A(9, 11),
			A(10, 0), A(10, 1), A(10, 2), A(10, 3), A(10, 4), A(10, 5), A(10, 6), A(10, 7), A(10, 8), A(10, 9), A(10, 10), A(10, 11),
			A(11, 0), A(11, 1), A(11, 2), A(11, 3), A(11, 4), A(11, 5), A(11, 6), A(11, 7), A(11, 8), A(11, 9), A(11, 10), A(11, 11)
			);
	}
	template<> __device__ __host__ void print(ldp_basic_vec<float, 1> A, const char* msg)
	{
		printf("%s: %f\n", msg, A[0]);
	}
	template<> __device__ __host__ void print(ldp_basic_vec<float, 2> x, const char* msg)
	{
		printf("%s: %f %f\n", msg, x[0], x[1]);
	}
	template<> __device__ __host__ void print(ldp_basic_vec<float, 3> x, const char* msg)
	{
		printf("%s: %f %f %f\n", msg, x[0], x[1], x[2]);
	}
	template<> __device__ __host__ void print(ldp_basic_vec<float, 4> x, const char* msg)
	{
		printf("%s: %f %f %f %f\n", msg, x[0], x[1], x[2], x[3]);
	}
	template<> __device__ __host__ void print(ldp_basic_vec<float, 9> x, const char* msg)
	{
		printf("%s\n%f %f %f %f %f %f %f %f %f\n", msg,
			x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
	}
	template<> __device__ __host__ void print(ldp_basic_vec<float, 12> x, const char* msg)
	{
		printf("%s\n%f %f %f %f %f %f %f %f %f %f %f %f\n", msg,
			x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]);
	}
#define printVal(A) print(A, #A)

	template<int N, int M>
	__device__ __host__ __forceinline__ ldp_basic_vec<float, M> mat_getRow(ldp_basic_mat<float, N, M> A, int row)
	{
		ldp_basic_vec<float, M> x;
		for (int k = 0; k < M; k++)
			x[k] = A(row, k);
		return x;
	}
	template<int N, int M>
	__device__ __host__ __forceinline__ ldp_basic_vec<float, N> mat_getCol(ldp_basic_mat<float, N, M> A, int col)
	{
		ldp_basic_vec<float, N> x;
		for (int k = 0; k < N; k++)
			x[k] = A(k, col);
		return x;
	}
	template<int N, int M>
	__device__ __host__ __forceinline__ ldp_basic_mat<float, N, M> outer(
		ldp::ldp_basic_vec<float, N> x, ldp::ldp_basic_vec<float, M> y)
	{
		ldp_basic_mat<float, N, M> A;
		for (int row = 0; row < N; row++)
		for (int col = 0; col < M; col++)
			A(row, col) = x[row] * y[col];
		return A;
	}
	template<class T, int N>
	__device__ __host__ __forceinline__ T sum(ldp::ldp_basic_vec<T, N> x)
	{
		T s = T(0);
		for (int i = 0; i < N; i++)
			s += x[i];
		return s;
	}

	__device__ __host__ __forceinline__ Mat23f make_rows(Float3 a, Float3 b)
	{
		Mat23f C;
		for (int k = 0; k < 3; k++)
		{
			C(0, k) = a[k];
			C(1, k) = b[k];
		}
		return C;
	}
	__device__ __host__ __forceinline__ Float9 make_Float9(Float3 a, Float3 b, Float3 c)
	{
		Float9 d;
		for (int k = 0; k < 3; k++)
		{
			d[k] = a[k];
			d[3 + k] = b[k];
			d[6 + k] = c[k];
		}
		return d;
	}
	__device__ __host__ __forceinline__ FloatC make_Float12(Float3 a, Float3 b, Float3 c, Float3 d)
	{
		FloatC v;
		for (int k = 0; k < 3; k++)
		{
			v[k] = a[k];
			v[3 + k] = b[k];
			v[6 + k] = c[k];
			v[9 + k] = d[k];
		}
		return v;
	}
	__device__ __host__ __forceinline__ Mat32f derivative(const Float3 x[3], const Float2 t[3])
	{
		return Mat32f(Mat31f(x[1] - x[0]), Mat31f(x[2] - x[0])) * Mat2f(t[1] - t[0], t[2] - t[0]).inv();
	}
	__device__ __host__ __forceinline__ Mat23f derivative(const Float2 t[3])
	{
		return Mat2f(t[1] - t[0], t[2] - t[0]).inv().trans()*
			make_rows(Float3(-1, 1, 0), Float3(-1, 0, 1));
	}
	__device__ __host__ __forceinline__ Float3 get_subFloat3(Float9 a, int i)
	{
		return Float3(a[i*3], a[i*3+1], a[i*3+2]);
	}
	__device__ __host__ __forceinline__ Mat3f get_subMat3f(Mat9f A, int row, int col)
	{
		Mat3f B;
		for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			B(r, c) = A(r+row*3, c+col*3);
		return B;
	}
	__device__ __host__ __forceinline__ Float3 get_subFloat3(FloatC a, int i)
	{
		return Float3(a[i * 3], a[i * 3 + 1], a[i * 3 + 2]);
	}
	__device__ __host__ __forceinline__ Mat3f get_subMat3f(MatCf A, int row, int col)
	{
		Mat3f B;
		for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			B(r, c) = A(r + row * 3, c + col * 3);
		return B;
	}
	__device__ __host__ __forceinline__ float unwrap_angle(float theta, float theta_ref)
	{
		if (theta - theta_ref > M_PI)
			theta -= 2 * M_PI;
		if (theta - theta_ref < -M_PI)
			theta += 2 * M_PI;
		return theta;
	}

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

	template<class T>
	static cudaTextureObject_t createTexture(DeviceArray<T>& ary, cudaTextureFilterMode filterMode)
	{
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeLinear;
		texRes.res.linear.sizeInBytes = ary.sizeBytes();
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


	__device__ __forceinline__ float fetch_float(cudaTextureObject_t A, int i)
	{
		float val = 0.f;
		tex1Dfetch(&val, A, i);
		return val;
	}
	__device__ __forceinline__ int fetch_int(cudaTextureObject_t A, int i)
	{
		int val = 0;
		tex1Dfetch(&val, A, i);
		return val;
	}
	__device__ __forceinline__ Mat3f fetch_Mat3f(cudaTextureObject_t A, int i)
	{
		Mat3f B;
		for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			tex1Dfetch(&B(r, c), A, i + c + r * 3);
		return B;
	}
#pragma endregion

#pragma region -- vert pair <--> idx
	__global__ void vertPair_to_idx_kernel(const int* v1, const int* v2, size_t* ids, int nVerts, int nPairs)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < nPairs)
			ids[i] = vertPair_to_idx(ldp::Int2(v1[i], v2[i]), nVerts);
	}
	__global__ void vertPair_from_idx_kernel(int* v1, int* v2, const size_t* ids, int nVerts, int nPairs)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < nPairs)
		{
			ldp::Int2 vert = vertPair_from_idx(ids[i], nVerts);
			v1[i] = vert[0];
			v2[i] = vert[1];
		}
	}

	void GpuSim::vertPair_to_idx(const int* v1, const int* v2, size_t* ids, int nVerts, int nPairs)
	{
		vertPair_to_idx_kernel << <divUp(nPairs, CTA_SIZE), CTA_SIZE >> >(
			v1, v2, ids, nVerts, nPairs);
		cudaSafeCall(cudaGetLastError());
	}

	void GpuSim::vertPair_from_idx(int* v1, int* v2, const size_t* ids, int nVerts, int nPairs)
	{
		vertPair_from_idx_kernel << <divUp(nPairs, CTA_SIZE), CTA_SIZE >> >(
			v1, v2, ids, nVerts, nPairs);
		cudaSafeCall(cudaGetLastError());
	}
#pragma endregion

#pragma region --texture bind
	texture<float, cudaTextureType1D, cudaReadModeElementType> gt_x;
	texture<float, cudaTextureType1D, cudaReadModeElementType> gt_x_init;
	texture<float, cudaTextureType1D, cudaReadModeElementType> gt_v;
	texture<float2, cudaTextureType1D, cudaReadModeElementType> gt_texCoord_init;
	texture<int4, cudaTextureType1D, cudaReadModeElementType> gt_faces_idxWorld;
	texture<int4, cudaTextureType1D, cudaReadModeElementType> gt_faces_idxTex;
	texture<int, cudaTextureType1D, cudaReadModeElementType> gt_A_order;
	texture<int, cudaTextureType1D, cudaReadModeElementType> gt_b_order;
	texture<float4, cudaTextureType1D, cudaReadModeElementType> gt_faceMaterialData;
	texture<float4, cudaTextureType1D, cudaReadModeElementType> gt_nodeMaterialData;
	__device__ __forceinline__ Float3 texRead_x(int i)
	{
		return ldp::Float3(tex1Dfetch(gt_x, i * 3),
			tex1Dfetch(gt_x, i * 3 + 1), tex1Dfetch(gt_x, i * 3 + 2));
	}
	__device__ __forceinline__ Float3 texRead_x_init(int i)
	{
		return ldp::Float3(tex1Dfetch(gt_x_init, i * 3),
			tex1Dfetch(gt_x_init, i * 3 + 1), tex1Dfetch(gt_x_init, i * 3 + 2));
	}
	__device__ __forceinline__ Float3 texRead_v(int i)
	{
		return ldp::Float3(tex1Dfetch(gt_v, i * 3),
			tex1Dfetch(gt_v, i * 3 + 1), tex1Dfetch(gt_v, i * 3 + 2));
	}
	__device__ __forceinline__ Float2 texRead_texCoord_init(int i)
	{
		float2 a = tex1Dfetch(gt_texCoord_init, i);
		return ldp::Float2(a.x, a.y);
	}
	__device__ __forceinline__ Int3 texRead_faces_idxWorld(int i)
	{
		int4 a = tex1Dfetch(gt_faces_idxWorld, i);
		return ldp::Int3(a.x, a.y, a.z);
	}
	__device__ __forceinline__ Int3 texRead_faces_idxTex(int i)
	{
		int4 a = tex1Dfetch(gt_faces_idxTex, i);
		return ldp::Int3(a.x, a.y, a.z);
	}
	__device__ __forceinline__ int texRead_A_order(int i)
	{
		return tex1Dfetch(gt_A_order, i);
	}
	__device__ __forceinline__ int texRead_b_order(int i)
	{
		return tex1Dfetch(gt_b_order, i);
	}
	__device__ __forceinline__ GpuSim::FaceMaterailSpaceData texRead_faceMaterialData(int i)
	{
		float4 v = tex1Dfetch(gt_faceMaterialData, i);
		GpuSim::FaceMaterailSpaceData data;
		data.area = v.x;
		data.mass = v.y;
		return data;
	}
	__device__ __forceinline__ GpuSim::NodeMaterailSpaceData texRead_nodeMaterialData(int i)
	{
		float4 v = tex1Dfetch(gt_nodeMaterialData, i);
		GpuSim::NodeMaterailSpaceData data;
		data.area = v.x;
		data.mass = v.y;
		return data;
	}
	__device__ __forceinline__ Float4 texRead_strechSample(cudaTextureObject_t t, float x, float y, float z)
	{
		float4 v = tex3D<float4>(t, x, y, z);
		return ldp::Float4(v.x, v.y, v.z, v.w);
	}
	__device__ __forceinline__ float texRead_bendData(cudaTextureObject_t t, float x, float y)
	{
		return tex2D<float>(t, x, y);
	}

	void GpuSim::bindTextures()
	{
		size_t offset;
		cudaChannelFormatDesc desc_float = cudaCreateChannelDesc<float>();
		cudaChannelFormatDesc desc_float2 = cudaCreateChannelDesc<float2>();
		cudaChannelFormatDesc desc_float4 = cudaCreateChannelDesc<float4>();
		cudaChannelFormatDesc desc_int = cudaCreateChannelDesc<int>();
		cudaChannelFormatDesc desc_int4 = cudaCreateChannelDesc<int4>();

		cudaSafeCall(cudaBindTexture(&offset, &gt_x, m_x_d.ptr(),
			&desc_float, m_x_d.size()*sizeof(float3)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_x_init, m_x_init_d.ptr(),
			&desc_float, m_x_init_d.size()*sizeof(float3)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_v, m_v_d.ptr(),
			&desc_float, m_v_d.size()*sizeof(float3)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_texCoord_init, m_texCoord_init_d.ptr(),
			&desc_float2, m_texCoord_init_d.size()*sizeof(float2)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_faces_idxWorld, m_faces_idxWorld_d.ptr(),
			&desc_int4, m_faces_idxWorld_d.size()*sizeof(int4)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_faces_idxTex, m_faces_idxTex_d.ptr(),
			&desc_int4, m_faces_idxTex_d.size()*sizeof(int4)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_A_order, m_A_order_d.ptr(),
			&desc_int, m_A_order_d.size()*sizeof(int)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_b_order, m_b_order_d.ptr(),
			&desc_int, m_b_order_d.size()*sizeof(int)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_faceMaterialData, m_faces_materialSpace_d.ptr(),
			&desc_float4, m_faces_materialSpace_d.size()*sizeof(float4)));
		CHECK_ZERO(offset);

		cudaSafeCall(cudaBindTexture(&offset, &gt_nodeMaterialData, m_nodes_materialSpace_d.ptr(),
			&desc_float4, m_nodes_materialSpace_d.size()*sizeof(float4)));
		CHECK_ZERO(offset);
	}
#pragma endregion

#pragma region --constraints
	struct NodeCon
	{
		float friction_stiffness = 0.f;
		float collision_stiffness = 0.f;
		float repulsion_thickness = 0.f;
		float projection_thickness = 0.f;
		float lvStep = 0.f;
		Float3 lvStart = 0.f;
		cudaTextureObject_t lvTex = 0;
		__device__ inline float violation(const float value)const { return max(-value, 0.f); }
		__device__ inline float value(Float3 p)const
		{
			const Float3 t = (p - lvStart) / lvStep + 0.5f;
			return Level_Set_Depth(lvTex, t[0], t[1], t[2], repulsion_thickness/lvStep)*lvStep;
		}
		__device__ inline Float3 gradient(Float3 p)const
		{
			const Float3 t = (p - lvStart) / lvStep + 0.5f;
			Float3 g;
			Level_Set_Gradient(lvTex, t[0], t[1], t[2], g[0], g[1], g[2]);
			return g;
		}
		__device__ inline Float3 project(const Float3 p)const
		{
			const float d = value(p) + repulsion_thickness - projection_thickness;
			return gradient(p) * violation(d);
		}
		__device__ inline float energy(const float value, const float area)const
		{
			const float v = violation(value);
			return area * collision_stiffness*v*v*v / repulsion_thickness / 6.f;
		}
		__device__ inline float energy_grad(const float value, const float area)const
		{
			const float v = violation(value);
			return -area * collision_stiffness*v*v / repulsion_thickness / 2.f;
		}
		__device__ inline float energy_hess(const float value, const float area)const
		{
			return area * collision_stiffness*violation(value) / repulsion_thickness;
		}
		__device__ inline Float3 friction(const Float3 p, const Float3 velocity, 
			const float mass, const float area, const float dt, Mat3f &jac)const
		{
			const float fn = abs(energy_grad(value(p), area));
			const Float3 n = -gradient(p);
			const Mat3f T = Mat3f().eye() - outer(n, n);
			const Float3 Tv = T*velocity;
			const float f_by_v = velocity.length() == 0.f ? 0.f : min(friction_stiffness*fn / Tv.length(), mass / dt);
			jac = -f_by_v*T;
			return jac*velocity;
		}
	};
#pragma endregion

#pragma region --update numeric
	__device__ __forceinline__ float distance(const Float3 &x, const Float3 &a, const Float3 &b)
	{
		Float3 e = b - a;
		Float3 xp = e*e.dot(x - a) / e.dot(e);
		return max((x - a - xp).length(), 1e-3f*e.length());
	}
	__device__ __forceinline__ Float2 barycentric_weights(Float3 x, Float3 a, Float3 b)
	{
		double t = (b-a).dot(x - a) / (b-a).sqrLength();
		return Float2(1 - t, t);
	}
	__device__ __forceinline__ Float3 faceNormal(int iFace)
	{
		ldp::Int3 f = texRead_faces_idxWorld(iFace);
		Float3 n = Float3(texRead_x(f[1]) - texRead_x(f[0])).cross(texRead_x(f[2]) - texRead_x(f[0]));
		if (n.length() == 0.f)
			return 0.f;
		return n.normalizeLocal();
	}
	__device__ __forceinline__ float faceArea(int iFace)
	{
		return texRead_faceMaterialData(iFace).area;
	}

	__device__ __forceinline__ Float4 stretching_stiffness(const Mat2f &G, cudaTextureObject_t samples)
	{
		float a = (G(0, 0) + 0.25f)*GpuSim::StretchingSamples::SAMPLES;
		float b = (G(1, 1) + 0.25f)*GpuSim::StretchingSamples::SAMPLES;
		float c = fabsf(G(0, 1))*GpuSim::StretchingSamples::SAMPLES;
		return texRead_strechSample(samples, a, b, c);
	}

	__device__ __forceinline__ float bending_stiffness(GpuSim::EdgeData eData, float dihe_angle, float area,
		const cudaTextureObject_t* t_bendDataTex, int side)
	{
		// because samples are per 0.05 cm^-1 = 5 m^-1
		const float len = 0.5f * (sqrt(eData.length_sqr[0]) + sqrt(eData.length_sqr[1]));
		float value = min(4.f, dihe_angle*len / area * 0.5f*0.2f);
		float bias_angle = fabs((eData.theta_uv[side] + eData.theta_initial) * 4.f / M_PI);

		const cudaTextureObject_t bendDataTex = t_bendDataTex[eData.faceIdx[side]];
		int value_i = min(3, max(0, (int)value));
		value -= value_i;
		const int bias_id = (int)bias_angle;
		bias_angle -= bias_id;
		float actual_ke = texRead_bendData(bendDataTex, bias_id, value_i) * (1 - bias_angle)*(1 - value)
			+ texRead_bendData(bendDataTex, bias_id + 1, value_i) * (bias_angle)*(1 - value)
			+ texRead_bendData(bendDataTex, bias_id, value_i + 1) * (1 - bias_angle)*(value)
			+texRead_bendData(bendDataTex, bias_id + 1, value_i + 1) * (bias_angle)*(value);
		if (actual_ke < 0) actual_ke = 0;
		return actual_ke;
	}

	__device__ __host__ __forceinline__  Mat39f kronecker_eye_row(const Mat23f &A, int iRow)
	{
		Mat39f C;
		C.zeros();
		for (int k = 0; k < 3; k++)
			C(0, k * 3) = C(1, k * 3 + 1) = C(2, k * 3 + 2) = A(iRow, k);
		return C;
	}

	__device__ __host__ __forceinline__  float dihedral_angle(Float3 a, Float3 b, Float3 n0, Float3 n1, float ref_theta)
	{
		if ((a - b).length() == 0.f || n0.length() == 0.f || n1.length() == 0.f)
			return 0.f;
		const Float3 e = (a - b).normalize();
		const float cosine = n0.dot(n1);
		const float sine = e.dot(n0.cross(n1));
		const float theta = atan2(sine, cosine);
		return unwrap_angle(theta, ref_theta);
	}

	__device__ void computeStretchForces(int iFace, int A_start, Mat3f* beforeScan_A,
		int b_start, Float3* beforeScan_b, const Float3* velocity,
		const cudaTextureObject_t* t_stretchSamples, float dt, float stretchMult)
	{
		const ldp::Int3 face_idxWorld = texRead_faces_idxWorld(iFace);
		const ldp::Int3 face_idxTex = texRead_faces_idxTex(iFace);
		const ldp::Float3 x[3] = { texRead_x(face_idxWorld[0]), texRead_x(face_idxWorld[1]),
			texRead_x(face_idxWorld[2]) };
		const ldp::Float2 t[3] = { texRead_texCoord_init(face_idxTex[0]),
			texRead_texCoord_init(face_idxTex[1]), texRead_texCoord_init(face_idxTex[2]) };
		const float area = texRead_faceMaterialData(iFace).area;

		// arcsim::stretching_force()---------------------------
		const Mat32f F = derivative(x, t);
		const Mat2f G = (F.trans()*F - Mat2f().eye()) * 0.5f;
		const Float4 k = stretching_stiffness(G, t_stretchSamples[iFace]) * stretchMult;
		const Mat23f D = derivative(t);
		const Mat39f Du = kronecker_eye_row(D, 0);
		const Mat39f Dv = kronecker_eye_row(D, 1);
		const Float3 xu = mat_getCol(F, 0);
		const Float3 xv = mat_getCol(F, 1); // should equal Du*mat_to_vec(X)
		const Float9 fuu = Du.trans()*xu;
		const Float9 fvv = Dv.trans()*xv;
		const Float9 fuv = (Du.trans()*xv + Dv.trans()*xu) * 0.5f;
		Float9 grad_e = k[0] * G(0, 0)*fuu + k[2] * G(1, 1)*fvv
			+ k[1] * (G(0, 0)*fvv + G(1, 1)*fuu) + 2 * k[3] * G(0, 1)*fuv;
		Mat9f hess_e = k[0] * (outer(fuu, fuu) + max(G(0, 0), 0.f)*Du.trans()*Du)
			+ k[2] * (outer(fvv, fvv) + max(G(1, 1), 0.f)*Dv.trans()*Dv)
			+ k[1] * (outer(fuu, fvv) + max(G(0, 0), 0.f)*Dv.trans()*Dv
			+ outer(fvv, fuu) + max(G(1, 1), 0.f)*Du.trans()*Du)
			+ 2.*k[3] * (outer(fuv, fuv));

		const Float9 vs = make_Float9(velocity[face_idxWorld[0]], 
			velocity[face_idxWorld[1]], velocity[face_idxWorld[2]]);
		hess_e = (dt*dt*area) * hess_e;
		grad_e = -area * dt * grad_e - hess_e*vs;

		//// output to global matrix
		for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
		{
			int pos = texRead_A_order(A_start + row * 3 + col);
			beforeScan_A[pos] = get_subMat3f(hess_e, row, col);
		} // end for row, col

		for (int row = 0; row < 3; row++)
		{
			int pos = texRead_b_order(b_start + row);
			beforeScan_b[pos] = get_subFloat3(grad_e, row);
		} // end for row

	}

	__device__ void computeBendForces(int iEdge, const GpuSim::EdgeData* edgeDatas,
		int A_start, Mat3f* beforeScan_A, int b_start, Float3* beforeScan_b, 
		const Float3* velocity, const cudaTextureObject_t* t_bendDatas, float dt, float bendMult)
	{
		const GpuSim::EdgeData edgeData = edgeDatas[iEdge];
		if (edgeData.faceIdx[0] < 0 || edgeData.faceIdx[1] < 0)
			return;
		const ldp::Float3 ex[4] = { texRead_x(edgeData.edge_idxWorld[0]), texRead_x(edgeData.edge_idxWorld[1]), 
			texRead_x(edgeData.edge_idxWorld[2]), texRead_x(edgeData.edge_idxWorld[3]) };
		Float3 n0 = faceNormal(edgeData.faceIdx[0]), n1 = faceNormal(edgeData.faceIdx[1]);
		const float area = texRead_faceMaterialData(edgeData.faceIdx[0]).area 
			+ texRead_faceMaterialData(edgeData.faceIdx[1]).area;
		const float dihe_theta = dihedral_angle(ex[0], ex[1], n0, n1, edgeData.dihedral_ideal);
		const float h0 = distance(ex[2], ex[0], ex[1]), h1 = distance(ex[3], ex[0], ex[1]);
		const Float2 w_f0 = barycentric_weights(ex[2], ex[0], ex[1]);
		const Float2 w_f1 = barycentric_weights(ex[3], ex[0], ex[1]);
		const FloatC dtheta = make_Float12(-(w_f0[0] * n0 / h0 + w_f1[0] * n1 / h1),
			-(w_f0[1] * n0 / h0 + w_f1[1] * n1 / h1), n0 / h0, n1 / h1);
		const float ke = min(
			bending_stiffness(edgeData, dihe_theta, area, t_bendDatas, 0),
			bending_stiffness(edgeData, dihe_theta, area, t_bendDatas, 1)
			) * bendMult;
		const float len = 0.5f * (sqrt(edgeData.length_sqr[0]) + sqrt(edgeData.length_sqr[1]));
		const float shape = ldp::sqr(len) / (2.f * area);
		const FloatC vs = make_Float12(velocity[edgeData.edge_idxWorld[0]], velocity[edgeData.edge_idxWorld[1]], 
			velocity[edgeData.edge_idxWorld[2]], velocity[edgeData.edge_idxWorld[3]]);
		FloatC F = -dt*0.5f * ke*shape*(dihe_theta - edgeData.dihedral_ideal)*dtheta;
		MatCf J = dt*dt*0.5f*ke*shape*outer(dtheta, dtheta);
		F -= J*vs;
		// output to global matrix
		for (int row = 0; row < 4; row++)
		for (int col = 0; col < 4; col++)
		{
			int pos = texRead_A_order(A_start + row * 4 + col);
			beforeScan_A[pos] = get_subMat3f(J, row, col);
		} // end for row, col

		for (int row = 0; row < 4; row++)
		{
			int pos = texRead_b_order(b_start + row);
			beforeScan_b[pos] = get_subFloat3(F, row);
		} // end for row
	}

	__global__ void computeNumeric_kernel(const GpuSim::EdgeData* edgeData, 
		const cudaTextureObject_t* t_stretchSamples, const cudaTextureObject_t* t_bendDatas,
		const int* A_starts, Mat3f* beforeScan_A, const int* b_starts, Float3* beforeScan_b,
		const Float3* velocity, int nFaces, int nEdges, float dt, float stretchMult, float bendMult)
	{
		int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

		// compute stretching forces here
		if (thread_id < nFaces)
		{
			const int A_start = A_starts[thread_id];
			const int b_start = b_starts[thread_id];
			const ldp::Int3 face_idxWorld = texRead_faces_idxWorld(thread_id);
			computeStretchForces(thread_id, A_start, beforeScan_A, 
				b_start, beforeScan_b, velocity, t_stretchSamples, dt, stretchMult);
		} // end if nFaces_numAb
		// compute bending forces here
		else if (thread_id < nFaces + nEdges)
		{
			const int A_start = A_starts[thread_id];
			const int b_start = b_starts[thread_id];
			computeBendForces(thread_id - nFaces, edgeData, A_start, beforeScan_A,
				b_start, beforeScan_b, velocity, t_bendDatas, dt, bendMult);
		}
	}

	__global__ void compute_A_kernel(const int* A_unique_pos, const Mat3f* A_beforeScan, 
		const int* A_cooRow, const int* A_cooCol, Mat3f* A_values, 
		int nVerts, int nnzBlocks, int nA_beforScan)
	{
		int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
		if (thread_id >= nnzBlocks)
			return;
		const int scan_begin = A_unique_pos[thread_id];
		const int scan_end = (thread_id == nnzBlocks - 1) ? nA_beforScan : A_unique_pos[thread_id + 1];

		// compute A(row, col)
		const int row = A_cooRow[thread_id];
		const int col = A_cooCol[thread_id];
		ldp::Mat3f sum;
		sum.zeros();

		// internal forces scan
		for (int scan_i = scan_begin; scan_i < scan_end; scan_i++)
			sum += A_beforeScan[scan_i];
		
		// external forces and diag term
		if (row == col)
		{
			const float mass = texRead_nodeMaterialData(row).mass;
			sum += ldp::Mat3f().eye() * mass;
		}

		// write into A, note that A is row majored while Mat3f is col majored
		A_values[thread_id] = sum.trans();
	}

	__global__ void compute_b_kernel(const int* b_unique_pos, const Float3* b_beforeScan,
		Float3* b_values, const float dt, const Float3 gravity, const int nVerts, const int nb_beforScan,
		const Float3* velocity, cudaTextureObject_t v_f_rowPtrTex, cudaTextureObject_t v_f_colTex)
	{
		int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
		if (thread_id >= nVerts)
			return;

		Float3 sum = 0.f;

		// internal forces scan
		const int scan_begin = b_unique_pos[thread_id];
		const int scan_end = (thread_id == nVerts - 1) ? nb_beforScan : b_unique_pos[thread_id + 1];
		for (int scan_i = scan_begin; scan_i < scan_end; scan_i++)
			sum += b_beforeScan[scan_i];

		// gravity forces and diag term
		const float mass = texRead_nodeMaterialData(thread_id).mass;
		sum += gravity * mass * dt;

		// wind forces
		const int fpos_begin = fetch_int(v_f_rowPtrTex, thread_id);
		const int fpos_end = fetch_int(v_f_rowPtrTex, thread_id + 1);
		for (int fpos = fpos_begin; fpos < fpos_end; ++fpos)
		{
			const int fid = fetch_int(v_f_colTex, fpos);
			const Int3 f = texRead_faces_idxWorld(fid);
			const Float3 vrel = -(velocity[f[0]] + velocity[f[1]] + velocity[f[2]]) / 3.;
			const float vn = faceNormal(fid).dot(vrel);
			sum += dt * faceArea(fid)*abs(vn)*vn / 3.f * faceNormal(fid);
		} // end for fid

		// write into b
		b_values[thread_id] = sum;
	}

	__global__ void add_LevelSet_constrain_kernel(const NodeCon con,
		const cudaTextureObject_t A_rowTex, const cudaTextureObject_t A_colTex,
		Mat3f* A_values, Float3* b_values, const float dt, const int nVerts)
	{
		int iVert = threadIdx.x + blockIdx.x * blockDim.x;
		if (iVert >= nVerts)
			return;
		const int rb = fetch_int(A_rowTex, iVert);
		const int re = fetch_int(A_rowTex, iVert + 1);

#ifdef USE_ONE_RING_EDGE_FOR_LEVELSET_COLLISION
		enum{SAMPLE_N = 10};
		float sumW = 0.f;
		Mat3f thisA = 0.f;
		Float3 thisb = 0.f;
		const Float3 x = texRead_x(iVert);
		const Float3 v = texRead_v(iVert);
		GpuSim::NodeMaterailSpaceData xData = texRead_nodeMaterialData(iVert);
		for(int pos = rb; pos < re; pos++)
		{
			int iVert1 = fetch_int(A_colTex, pos);
			const Float3 x1 = texRead_x(iVert1);
			const Float3 v1 = texRead_v(iVert1);
			GpuSim::NodeMaterailSpaceData xData1 = texRead_nodeMaterialData(iVert1);

			float minValue = FLT_MAX;
			float minW = 0.f;
			for (int k = 0; k < SAMPLE_N; k++)
			{
				const float w = 1.f - float(k) / (SAMPLE_N - 1);
				float val = con.value(w*x + (1 - w)*x1);
				if (val < minValue)
				{
					minValue = val;
					minW = w;
				}
			}
			if (con.violation(minValue) == 0.f) continue;

			const Float3 xm = minW*x + (1 - minW)*x1;
			const Float3 vm = minW*v + (1 - minW)*v1;
			const float area = minW*xData.area + (1 - minW)*xData1.area;
			const float mass = minW*xData.mass + (1 - minW)*xData1.mass;
			const float g = con.energy_grad(minValue, area);
			const float h = con.energy_hess(minValue, area);
			const Float3 grad = con.gradient(xm);
			const float v_dot_grad = vm.dot(grad);

			Mat3f fric_jac;
			Float3 fric_force = con.friction(xm, vm, mass, area, dt, fric_jac);

			minW *= (x1 - x).length();
			thisA += minW*dt*dt*h*outer(grad, grad) - minW*dt * fric_jac;
			thisb += -minW*dt*(g + dt*h*v_dot_grad)*grad + minW*dt * fric_force;
			sumW += minW;
		} // end for pos

		if(sumW != 0.f)
		{
			thisA *= 1.f / sumW;
			thisb *= 1.f / sumW;
		}
#else
		const Float3 x = texRead_x(iVert);
		const Float3 v = texRead_v(iVert);
		GpuSim::NodeMaterailSpaceData xData = texRead_nodeMaterialData(iVert);
		const float value = con.value(x);
		const float g = con.energy_grad(value, xData.area);
		const float h = con.energy_hess(value, xData.area);
		const Float3 grad = con.gradient(x);
		const float v_dot_grad = v.dot(grad);

		Mat3f fric_jac;
		Float3 fric_force = con.friction(x, v, xData.mass, xData.area, dt, fric_jac);

		const Mat3f thisA = dt*dt*h*outer(grad, grad) - dt * fric_jac;
		const Float3 thisb = -dt*(g + dt*h*v_dot_grad)*grad + dt * fric_force;
#endif

		// write into A
		for (int pos = rb; pos < re; pos++)
		{
			const int col = fetch_int(A_colTex, pos);
			if (iVert == col)
				A_values[pos] += thisA.trans();
		}

		// write into b
		b_values[iVert] += thisb;
	}

	void GpuSim::updateNumeric()
	{
		cudaSafeCall(cudaMemset(m_beforScan_A.ptr(), 0, m_beforScan_A.sizeBytes()));
		cudaSafeCall(cudaMemset(m_beforScan_b.ptr(), 0, m_beforScan_b.sizeBytes()));

		// ldp hack here: make the gravity not important when we are stitching.
		const Float3 gravity = m_simParam.gravity;// *powf(1 - std::max(0.f, std::min(1.f, m_curStitchRatio)), 2);
		const int nFaces = m_faces_idxWorld_d.size();
		const int nEdges = m_edgeData_d.size();
		computeNumeric_kernel << <divUp(nFaces + nEdges, CTA_SIZE), CTA_SIZE >> >(
			m_edgeData_d.ptr(), m_faces_texStretch_d.ptr(), m_faces_texBend_d.ptr(),
			m_A_Ids_start_d.ptr(), m_beforScan_A.ptr(), m_b_Ids_start_d.ptr(), m_beforScan_b.ptr(),
			m_v_d.ptr(), nFaces, nEdges, m_simParam.dt, m_simParam.strecth_mult, m_simParam.bend_mult);
		cudaSafeCall(cudaGetLastError());

		// scanning into the sparse matrix
		// since we have unique positions, we can do scan similar with CSR-vector multiplication.
		const int nVerts = m_x_d.size();
		const int nnzBlocks = m_A_d->nnzBlocks();
		const int nA_beforScan = m_beforScan_A.size();
		const int nb_beforScan = m_beforScan_b.size();
		compute_A_kernel << <divUp(nnzBlocks, CTA_SIZE), CTA_SIZE >> >(
			m_A_Ids_d_unique_pos.ptr(), m_beforScan_A.ptr(), m_A_d->bsrRowPtr_coo(), m_A_d->bsrColIdx(),
			(Mat3f*)m_A_d->value(), nVerts, nnzBlocks, nA_beforScan
			);
		cudaSafeCall(cudaGetLastError());
		compute_b_kernel << <divUp(nVerts, CTA_SIZE), CTA_SIZE >> >(
			m_b_Ids_d_unique_pos.ptr(), m_beforScan_b.ptr(),
			(Float3*)m_b_d.ptr(), m_simParam.dt, gravity, nVerts, nb_beforScan,
			m_v_d.ptr(), m_vert_FaceList_d->bsrRowPtrTexture(), m_vert_FaceList_d->bsrColIdxTexture()
			);
		cudaSafeCall(cudaGetLastError());

		// add body-cloth force term using level set
		NodeCon con;
		con.lvTex = m_bodyLvSet_d.getCudaTexture();
		con.lvStep = m_bodyLvSet_h->getStep();
		con.lvStart = m_bodyLvSet_h->getStartPos();
		con.projection_thickness = m_simParam.projection_thickness;
		con.repulsion_thickness = m_simParam.repulsion_thickness;
		con.collision_stiffness = m_simParam.collision_stiffness;
		con.friction_stiffness = m_simParam.friction_stiffness;
		add_LevelSet_constrain_kernel << <divUp(nVerts, CTA_SIZE), CTA_SIZE >> >(
			con, m_A_d->bsrRowPtrTexture(), m_A_d->bsrColIdxTexture(), (Mat3f*)m_A_d->value(), 
			m_b_d.ptr(), m_simParam.dt, nVerts
			);
		cudaSafeCall(cudaGetLastError());
	}
#pragma endregion

#pragma region --collision solve
	__global__ void project_outside_kernel(const NodeCon con,
		Float3* positions, const int nVerts)
	{
		int iVert = threadIdx.x + blockIdx.x * blockDim.x;
		if (iVert >= nVerts)
			return;

		Float3 x = positions[iVert];
		x += con.project(x);
		positions[iVert] = x;
	}


	void GpuSim::project_outside()
	{
		NodeCon con;
		con.lvTex = m_bodyLvSet_d.getCudaTexture();
		con.lvStep = m_bodyLvSet_h->getStep();
		con.lvStart = m_bodyLvSet_h->getStartPos();
		con.projection_thickness = m_simParam.projection_thickness;
		con.repulsion_thickness = m_simParam.repulsion_thickness;
		con.collision_stiffness = m_simParam.collision_stiffness;
		con.friction_stiffness = m_simParam.friction_stiffness;
		project_outside_kernel << <divUp(m_x_d.size(), CTA_SIZE), CTA_SIZE >> >(
			con, m_x_d.ptr(), m_x_d.size()
			);
		cudaSafeCall(cudaGetLastError());
	}

	void GpuSim::collisionSolve()
	{
		project_outside();
	}
#pragma endregion

#pragma region --user control solve
	void GpuSim::userControlSolve()
	{

	}
#pragma endregion

#pragma region --update x, v by dv
	__global__ void update_x_v_by_dv_kernel(const Float3* dv, Float3* v, Float3* x,
		float dt, int nverts)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= nverts)	return;
		v[i] += dv[i];
		x[i] += dt * v[i];
	}

	void GpuSim::update_x_v_by_dv()
	{
		// v += dv; x += dt*v;
		update_x_v_by_dv_kernel << <divUp(m_x_d.size(), CTA_SIZE), CTA_SIZE >> >
			(m_dv_d.ptr(), m_v_d.ptr(), m_x_d.ptr(), m_simParam.dt, m_dv_d.size());
		cudaSafeCall(cudaGetLastError());
	}
#pragma endregion

#pragma region --pcg
	__global__ void pcg_vecMul_kernel(int n, const float* a_d, const float* b_d, float* c_d,
		float alpha, float beta)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n)	return;
		c_d[i] = alpha * a_d[i] * b_d[i] + beta;
	}
	__global__ void pcg_update_p_kernel(int n, const float* z_d, float* p_d, float beta)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n)	return;
		p_d[i] = beta * p_d[i] + z_d[i];
	}
	__global__ void pcg_update_x_r_kernel(int n, const float* p_d, const float* Ap_d, 
		float* x_d, float* r_d, float alpha)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i >= n)	return;
		x_d[i] += alpha * p_d[i];
		r_d[i] -= alpha * Ap_d[i];
	}

	__global__ void pcg_extractInvDiagBlock_kernel(int n, Mat3f* diag, 
		cudaTextureObject_t A_RowTex, cudaTextureObject_t A_colTex, const Mat3f* A_val)
	{
		const int r = blockDim.x * blockIdx.x + threadIdx.x;
		if (r >= n)	return;
		const int rb = fetch_int(A_RowTex, r);
		const int re = fetch_int(A_RowTex, r + 1);
		Mat3f D;
		for (int pos = rb; pos < re; pos++)
		{
			const int c = fetch_int(A_colTex, pos);
			if (r == c)
				D = A_val[pos];
		}
		diag[r] = D.inv();
	}

	void GpuSim::pcg_vecMul(int n, const float* a_d, const float* b_d, float* c_d, float alpha, float beta)const
	{
		pcg_vecMul_kernel << <divUp(n, CTA_SIZE), CTA_SIZE >> >(n, a_d, b_d, c_d, alpha, beta);
		cudaSafeCall(cudaGetLastError());
	}

	void GpuSim::pcg_update_p(int n, const float* z_d, float* p_d, float beta)const
	{
		pcg_update_p_kernel << <divUp(n, CTA_SIZE), CTA_SIZE >> >(n, z_d, p_d, beta);
		cudaSafeCall(cudaGetLastError());
	}
	
	void GpuSim::pcg_update_x_r(int n, const float* p_d, const float* Ap_d, float* x_d, float* r_d, float alpha)const
	{
		pcg_update_x_r_kernel << <divUp(n, CTA_SIZE), CTA_SIZE >> >(n, p_d, Ap_d, x_d, r_d, alpha);
		cudaSafeCall(cudaGetLastError());
	}

	float GpuSim::pcg_dot(int n, const float* a_d, const float* b_d)const
	{
		float s = 0.f;
		cublasSdot_v2(m_cublasHandle, n, a_d, 1, b_d, 1, &s);
		return s;
	}

	void GpuSim::pcg_extractInvDiagBlock(const CudaBsrMatrix& A, CudaDiagBlockMatrix& invD)
	{
		pcg_extractInvDiagBlock_kernel << <divUp(A.blocksInRow(), CTA_SIZE), CTA_SIZE >> >(
			A.blocksInRow(), (Mat3f*)invD.value(), A.bsrRowPtrTexture(), A.bsrColIdxTexture(), (const Mat3f*)A.value());
		cudaSafeCall(cudaGetLastError());
	}
#pragma endregion
}