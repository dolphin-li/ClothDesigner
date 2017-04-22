#pragma once

#include <vector>
#include <memory>
#include "cudpp\CudaDiagBlockMatrix.h"
#include "cudpp\CudaBsrMatrix.h"
#include "ldpMat\ldp_basic_mat.h"
#include "cudpp\Cuda3DArray.h"
#include "cudpp\Cuda2DArray.h"
#include <cublas.h>
namespace arcsim
{
	class ArcSimManager;
}

class ObjMesh;
namespace ldp
{
	class ClothManager;
	class LevelSet3D;		
	__device__ __host__ inline size_t vertPair_to_idx(ldp::Int2 v, int n)
	{
		return size_t(v[0]) * size_t(n) + size_t(v[1]);
	}
	__device__ __host__ inline ldp::Int2 vertPair_from_idx(size_t idx, int n)
	{
		return ldp::Int2(int(idx / n), int(idx%n));
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
			ldp::Int4 edge_idxWorld;
			ldp::Int2 edge_idxTex[2]; // two face side tex space index
			ldp::Int2 faceIdx;
			ldp::Float2 length_sqr;
			ldp::Float2 theta_uv;	// uv direction
			float dihedral_ideal = 0.f;
			float theta_initial = 0.f;
		};
		struct FaceMaterailSpaceData // must be sizeof float4
		{
			float area = 0.f;
			float mass = 0.f;
			float __place_holder_1 = 0.f;
			float __place_holder_2 = 0.f;
		};
		struct NodeMaterailSpaceData // must be sizeof float4
		{
			float area = 0.f;
			float mass = 0.f;
			float __place_holder_1 = 0.f;
			float __place_holder_2 = 0.f;
		};
		struct SimParam
		{
			float dt = 0.f;				// time step
			Float3 gravity;
			int pcg_iter = 0;			// iteration of pcg solvers
			float pcg_tol = 0.f;		// tolerance of pcg method
			float control_mag = 0.f;	// for user control
			float stitch_ratio = 0.f;	// for stitching edges length reduction
			int lap_damping_iter = 0;
			float air_damping = 0.f;
			float strecth_mult = 0.f;
			float bend_mult = 0.f;
			float handle_stiffness = 0.f;
			float collision_stiffness = 0.f;
			float repulsion_thickness = 0.f;
			float projection_thickness = 0.f;

			void setDefault();
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

		float getFps()const{ return m_fps; }
		float getStepTime()const{ return m_simParam.dt; }
		std::string getSolverInfo()const{ return m_solverInfo; }

		void clothToObjMesh(ObjMesh& mesh);

		// reset cloths to the initial state
		void restart();

		// update the material parameters
		void updateMaterial();

		// update cloth topologies
		void updateTopology();

		// simulatio parameter update; resulting in numerical updates
		void updateNumeric();
	protected:
		void initParam();

		void releaseMaterialMemory();
		void initializeMaterialMemory();

		void updateTopology_arcSim();
		void updateTopology_clothManager();

		void linearSolve();
		void collisionSolve();
		void userControlSolve();

		void update_x_v_by_dv();
	protected:
		void setup_sparse_structure_from_cpu();
		void bindTextures();
	private:
		ClothManager* m_clothManager = nullptr;
		arcsim::ArcSimManager* m_arcSimManager = nullptr;
		cusparseHandle_t m_cusparseHandle = nullptr;
		cublasHandle_t m_cublasHandle = nullptr;
		SimParam m_simParam;
		float m_fps = 0.f;
		std::string m_solverInfo;

		ldp::LevelSet3D* m_bodyLvSet_h = nullptr;
		Cuda3DArray<float> m_bodyLvSet_d;

		std::vector<ldp::Int4> m_faces_idxWorld_h;				// triangle face with 1 paded int
		DeviceArray<ldp::Int4> m_faces_idxWorld_d;
		std::vector<ldp::Int4> m_faces_idxTex_h;
		DeviceArray<ldp::Int4> m_faces_idxTex_d;
		std::vector<int> m_faces_idxMat_h;						// material index of each face
		DeviceArray<cudaTextureObject_t> m_faces_texStretch_d;	// cuda texture of stretching
		DeviceArray<cudaTextureObject_t> m_faces_texBend_d;		// cuda texture of bending
		DeviceArray<FaceMaterailSpaceData> m_faces_materialSpace_d;
		DeviceArray<NodeMaterailSpaceData> m_nodes_materialSpace_d;
		std::vector<EdgeData> m_edgeData_h;
		DeviceArray<EdgeData> m_edgeData_d;
		std::shared_ptr<CudaBsrMatrix> m_vert_FaceList_d;

		///////////////// precomputed data /////////////////////////////////////////////////////////////	
		DeviceArray<size_t> m_A_Ids_d;			// for sparse matrix, encode the (row, col) pairs, sorted
		DeviceArray<size_t> m_A_Ids_d_unique;
		DeviceArray<int> m_A_Ids_d_unique_pos;	// the array position kept after unique
		DeviceArray<int> m_A_Ids_start_d;		// the starting position of A_order
		DeviceArray<int> m_A_order_d;			// the face/edge/bend filling order, beforScan_A[order]
		DeviceArray<int> m_A_invOrder_d;
		DeviceArray<ldp::Mat3f> m_beforScan_A;
		DeviceArray<int> m_b_Ids_d;				// for rhs construction
		DeviceArray<int> m_b_Ids_d_unique;
		DeviceArray<int> m_b_Ids_d_unique_pos;
		DeviceArray<int> m_b_Ids_start_d;		// the starting position of b_order
		DeviceArray<int> m_b_order_d;			// the face/edge/bend filling order, beforScan_b[order]
		DeviceArray<int> m_b_invOrder_d;
		DeviceArray<ldp::Float3> m_beforScan_b;
		///////////////// solve for the simulation linear system: A*dv=b////////////////////////////////
		std::shared_ptr<CudaBsrMatrix> m_A_d;
		std::shared_ptr<CudaDiagBlockMatrix> m_A_diag_d;
		DeviceArray<ldp::Float3> m_b_d;
		std::vector<ldp::Float2> m_texCoord_init_h;				// material (tex) space vertex texCoord							
		DeviceArray<ldp::Float2> m_texCoord_init_d;				// material (tex) space vertex texCoord	
		std::vector<ldp::Float3> m_x_init_h;					// world space vertex position	
		DeviceArray<ldp::Float3> m_x_init_d;					// world space vertex position
		std::vector<ldp::Float3> m_x_h;							// position of current step	
		DeviceArray<ldp::Float3> m_x_d;							// position of current step	
		DeviceArray<ldp::Float3> m_last_x_d;					// position of last step	
		DeviceArray<ldp::Float3> m_v_d;							// velocity of current step
		DeviceArray<ldp::Float3> m_last_v_d;					// velocity of last step	
		DeviceArray<ldp::Float3> m_dv_d;						// velocity changed in this step
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
				SAMPLES = 30,
				SAMPLES2 = SAMPLES*SAMPLES,
			};
			StretchingSamples(){
				m_data.resize(SAMPLES*SAMPLES*SAMPLES);
				m_ary.create(make_int3(SAMPLES, SAMPLES, SAMPLES));
			}
			StretchingSamples(const StretchingSamples& rhs){
				m_data = rhs.m_data;
				m_ary = rhs.m_ary;
			}
			~StretchingSamples(){
				m_ary.release();
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
	
			const Cuda3DArray<float4>& getCudaArray()const{ return m_ary; }
			Cuda3DArray<float4>& getCudaArray(){ return m_ary; }

			void updateHostToDevice(){ m_ary.fromHost((const float4*)m_data.data(), m_ary.size()); }
			void updateDeviceToHost(){ m_ary.toHost((float4*)m_data.data()); }
		protected:
			std::vector<ldp::Float4> m_data;
			Cuda3DArray<float4> m_ary;
		};
		
		class BendingData {
		public:
			enum {
				POINTS = 5,
				DIMS = 9,
				FilterMode = cudaFilterModePoint,
			};
			BendingData(){
				m_data.resize(rows()*cols());
				m_ary.create(make_int2(cols(), rows()), (cudaTextureFilterMode)FilterMode);
			}
			~BendingData(){
				m_ary.release();
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

			const Cuda2DArray<float>& getCudaArray()const{ return m_ary; }
			Cuda2DArray<float>& getCudaArray(){ return m_ary; }

			void updateHostToDevice(){ m_ary.fromHost(data(), m_ary.size(), (cudaTextureFilterMode)FilterMode); }
			void updateDeviceToHost(){ m_ary.toHost(data()); }
		protected:
			std::vector<float> m_data;
			Cuda2DArray<float> m_ary;
		};
		std::vector<StretchingSamples> m_stretchSamples_h;			
		std::vector<BendingData> m_bendingData_h;
		std::vector<float> m_densityData_h;
	public:
		void pcg_vecMul(int n, const float* a_d, const float* b_d, 
			float* c_d, float alpha = 1.f, float beta = 0.f)const; // c=alpha * a * b + beta
		void pcg_update_p(int n, const float* z_d, float* p_d, float beta)const;
		void pcg_update_x_r(int n, const float* p_d, const float* Ap_d, float* x_d, float* r_d, float alpha)const;
		float pcg_dot(int n, const float* a_d, const float* b_d)const;
		void pcg_extractInvDiagBlock(const CudaBsrMatrix& A, CudaDiagBlockMatrix& invD);
		static void vertPair_to_idx(const int* ids_v1, const int* ids_v2, size_t* ids, int nVerts, int nPairs);
		static void vertPair_from_idx(int* ids_v1, int* ids_v2, const size_t* ids, int nVerts, int nPairs);
		static void dumpVec(std::string name, const DeviceArray2D<float>& A);
		static void dumpVec(std::string name, const DeviceArray<float>& A, int nTotal=-1);
		static void dumpVec(std::string name, const DeviceArray<ldp::Float3>& A, int nTotal = -1);
		static void dumpVec(std::string name, const DeviceArray<ldp::Mat3f>& A, int nTotal = -1);
		static void dumpVec(std::string name, const DeviceArray<ldp::Float2>& A, int nTotal = -1);
		static void dumpVec(std::string name, const DeviceArray<int>& A, int nTotal = -1);
		static void dumpVec_pair(std::string name, const DeviceArray<size_t>& A, int nVerts, int nTotal = -1);
		static void dumpStretchSampleArray(std::string name, const StretchingSamples& samples);
		static void dumpBendDataArray(std::string name, const BendingData& samples);
	};
}