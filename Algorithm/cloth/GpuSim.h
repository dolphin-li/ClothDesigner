#pragma once

#include <vector>
#include <memory>
#include "cudpp\CudaBsrMatrix.h"
#include "ldpMat\ldp_basic_mat.h"

namespace arcsim
{
	class ArcSimManager;
}

namespace ldp
{
	class ClothManager;
	class LevelSet3D;		
	__device__ __host__ inline int vertPair_to_idx(ldp::Int2 v, int n)
	{
		return v[0] * n + v[1];
	}
	__device__ __host__ inline ldp::Int2 vertPair_from_idx(int idx, int n)
	{
		return ldp::Int2(idx / n, idx%n);
	}
	class GpuSim
	{
	public:
		//		 c
		//		 /\
		//		/  \
		//	  a ---- b
		//		\  /
		//		 \/
		//		 d
		// edge:	 ab
		// bendEdge: cd
		// faceIdx: abc, bad
		// _idxWorld: index in world space
		// _idxTex: index in tex space
		struct EdgeData
		{
			ldp::Int2 edge_idxWorld;
			ldp::Int2 faceIdx;
		};
	public:
		GpuSim();
		~GpuSim();

		// init from cloth manager
		void init(ClothManager* clothManager);

		// init from arcsim
		void init(arcsim::ArcSimManager* arcSimManager);

		// perform simulation for one time-step
		void run_one_step();

		// reset cloths to the initial state
		void restart();

		// update the material parameters
		void updateMaterial();

		// update cloth topologies
		void updateTopology();

		// simulatio parameter update; resulting in numerical updates
		void updateNumeric();
	protected:
		void releaseMaterialMemory();
		void initializeMaterialMemory();

		void updateTopology_arcSim();
		void updateTopology_clothManager();

		void linearSolve();

	protected:
		void setup_sparse_structure_from_cpu();
	private:
		ClothManager* m_clothManager = nullptr;
		arcsim::ArcSimManager* m_arcSimManager = nullptr;
		cusparseHandle_t m_cusparseHandle = nullptr;

		ldp::LevelSet3D* m_bodyLvSet_h = nullptr;
		DeviceArray<float> m_bodyLvSet_d;

		std::vector<ldp::Int4> m_faces_idxWorld_h;				// triangle face with 1 paded int
		DeviceArray<ldp::Int4> m_faces_idxWorld_d;
		std::vector<ldp::Int4> m_faces_idxTex_h;
		DeviceArray<ldp::Int4> m_faces_idxTex_d;
		std::vector<int> m_faces_idxMat_h;						// material index of each face
		DeviceArray<cudaTextureObject_t> m_faces_texStretch_d;	// cuda texture of stretching
		DeviceArray<cudaTextureObject_t> m_faces_texBend_d;		// cuda texture of bending
		std::vector<EdgeData> m_edgeData_h;
		DeviceArray<EdgeData> m_edgeData_d;

		///////////////// precomputed data /////////////////////////////////////////////////////////////
		std::vector<float> m_edgeThetaIdeals_h;					// the folding angles (in arc) on the edge
		DeviceArray<float> m_edgeThetaIdeals_d;			

		std::vector<int> m_faceEdge_vertIds_h;	// computing internal forces, the vertex index of each face/edge/bend
		DeviceArray<int> m_faceEdge_vertIds_d;
		std::vector<int> m_faceEdge_order_h;	// the face/edge/bend filling order, beforScan_A[order]
		DeviceArray<int> m_faceEdge_order_d;
		DeviceArray<ldp::Mat3f> m_faceEdge_beforScan_A;
		DeviceArray<ldp::Float3> m_faceEdge_beforScan_b;
		///////////////// solve for the simulation linear system: A*dv=b////////////////////////////////
		std::shared_ptr<CudaBsrMatrix> m_A;
		DeviceArray<ldp::Float3> m_b;							
		DeviceArray<ldp::Float2> m_texCoord_init;				// material (tex) space vertex texCoord		
		DeviceArray<ldp::Float3> m_x_init;						// world space vertex position
		DeviceArray<ldp::Float3> m_x;							// position of current step	
		DeviceArray<ldp::Float3> m_v;							// velocity of current step
		DeviceArray<ldp::Float3> m_dv;							// velocity changed in this step
		
		//////////////////////// material related///////////////////////////////////////////////////////
	public:
		class StretchingData { 
		public:
			enum {
				DIMS = 2,
				POINTS = 5
			};
			StretchingData(){
				d.resize(rows()*cols());
			}
			ldp::Float4& operator()(int x, int y){
				return d[y*cols() + x];
			}
			const ldp::Float4& operator()(int x, int y)const{
				return d[y*cols() + x];
			}
			ldp::Float4* data(){ return d.data(); }
			const ldp::Float4* data()const{ return d.data(); }
			int size()const{ return d.size(); }
			int rows()const{ return POINTS; }
			int cols()const{ return DIMS; }
		protected:
			std::vector<ldp::Float4> d;
		};
		class StretchingSamples {
		public:
			enum{
				SAMPLES = 40,
				SAMPLES2 = SAMPLES*SAMPLES,
			};
			StretchingSamples(){
				m_data.resize(SAMPLES*SAMPLES*SAMPLES);
				initDeviceMemory();
			}
			~StretchingSamples(){
				releaseDeviceMemory();
			}
			ldp::Float4& operator()(int x, int y, int z){
				return m_data[z*SAMPLES2 + y*SAMPLES + x];
			}
			const ldp::Float4& operator()(int x, int y, int z)const{
				return m_data[z*SAMPLES2 + y*SAMPLES + x];
			}
			ldp::Float4* data(){ return m_data.data(); }
			const ldp::Float4* data()const{ return m_data.data(); }
			int size()const{ return m_data.size(); }
	
			cudaArray_t getCudaArray()const{ return m_ary; }
			cudaTextureObject_t getCudaTexture()const{ return m_tex; }
			cudaSurfaceObject_t getCudaSurface()const{ return m_surf; }

			void initDeviceMemory();
			void releaseDeviceMemory();
			void updateHostToDevice();
			void updateDeviceToHost();
		protected:
			std::vector<ldp::Float4> m_data;
			cudaArray_t m_ary = nullptr;
			cudaTextureObject_t m_tex = 0;
			cudaSurfaceObject_t m_surf = 0;
		};
		class BendingData {
		public:
			enum {
				DIMS = 3,
				POINTS = 5
			};
			BendingData(){
				m_data.resize(rows()*cols());
				initDeviceMemory();
			}
			~BendingData(){
				releaseDeviceMemory();
			}
			float& operator()(int x, int y){
				return m_data[y*cols() + x];
			}
			const float& operator()(int x, int y)const{
				return m_data[y*cols() + x];
			}
			float* data(){ return m_data.data(); }
			const float* data()const{ return m_data.data(); }
			int size()const{ return m_data.size(); }
			int rows()const{ return POINTS; }
			int cols()const{ return DIMS; }

			const DeviceArray2D<float>& getCudaArray()const{ return m_ary; }
			cudaTextureObject_t getCudaTexture()const{ return m_tex; }

			void initDeviceMemory();
			void releaseDeviceMemory();
			void updateHostToDevice();
			void updateDeviceToHost();
		protected:
			std::vector<float> m_data;
			DeviceArray2D<float> m_ary;
			cudaTextureObject_t m_tex;
		};
		std::vector<StretchingSamples> m_stretchSamples_h;			
		std::vector<BendingData> m_bendingData_h;
	public:
		static cudaTextureObject_t createTexture(cudaArray_t ary, cudaTextureFilterMode filterMode);
		static cudaSurfaceObject_t createSurface(cudaArray_t ary);
		static void dumpVec(std::string name, const DeviceArray<float>& A);
		static void dumpVec(std::string name, const DeviceArray2D<float>& A);
		static void dumpVec(std::string name, const DeviceArray<ldp::Float3>& A);
		static void dumpVec(std::string name, const DeviceArray<ldp::Float2>& A);
		static void dumpStretchSampleArray(std::string name, const StretchingSamples& samples);
	};
}