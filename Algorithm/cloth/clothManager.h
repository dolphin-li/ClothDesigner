#pragma once

#include <vector>
#include <memory>
#include "device_array.h"
#include "ldpMat\ldp_basic_vec.h"
//#define ENABLE_SELF_COLLISION
#ifdef ENABLE_SELF_COLLISION
#include "COLLISION_HANDLER.h"
#endif
class ObjMesh;
namespace ldp
{
	struct SimulationParam
	{
		float rho;				// for chebshev accereration
		float under_relax;		// jacobi relax param
		int lap_damping;		// loops of laplacian damping
		float air_damping;		// damping of the air
		float bending_k;		// related to the thickness of the cloth
		float spring_k;			// related to the elasticity of the cloth
		int out_iter;			// number of iterations
		int inner_iter;			// number of iterations
		float control_mag;		// for dragging, the stiffness of dragged point
		float time_step;		// simulation time step
		SimulationParam();
		void setDefaultParam();
	};

	struct DragInfo
	{
		ObjMesh* selected_cloth;
		int selected_vert_id;
		ldp::Float3 target;
		DragInfo()
		{
			selected_cloth = nullptr;
			selected_vert_id = -1;
			target = 0;
		}
	};

	class ClothPiece;
	class PanelPolygon;
	class LevelSet3D;
	class ClothManager
	{
	public:
		typedef float ValueType;
		typedef ldp::ldp_basic_vec3<float> Vec3;
		enum SimulationMode
		{
			SimulationNotInit,
			SimulationOn,
			SimulationPause,
		};
	protected:
		struct DragInfoInternal
		{
			int vert_id;
			ldp::Float3 dir;
			ldp::Float3 target;
		};
	public:
		ClothManager();
		~ClothManager();

		void clear();

		void simulationInit();
		void simulationUpdate();
		void simulationDestroy();

		void dragBegin(DragInfo info);
		void dragMove(ldp::Float3 target);
		void dragEnd();

		void setSimulationMode(SimulationMode mode);
		void setSimulationParam(SimulationParam param);

		float getFps()const { return m_fps; }
		SimulationMode getSimulationMode()const { return m_simulationMode; }
		const ObjMesh* bodyMesh()const { return m_bodyMesh.get(); }
		ObjMesh* bodyMesh() { return m_bodyMesh.get(); }
		const LevelSet3D* bodyLevelSet()const { return m_bodyLvSet.get(); }
		LevelSet3D* bodyLevelSet() { return m_bodyLvSet.get(); }
		int numClothPieces()const { return (int)m_clothPieces.size(); }
		const ClothPiece* clothPiece(int i)const { return m_clothPieces.at(i).get(); }
		ClothPiece* clothPiece(int i) { return m_clothPieces.at(i).get(); }
		void clearClothPieces() { m_clothPieces.clear(); }
		void addClothPiece(std::shared_ptr<ClothPiece> piece) { m_clothPieces.push_back(piece); }
		void removeClothPiece(int i) { m_clothPieces.erase(m_clothPieces.begin() + i); }
		SimulationParam getSimulationParam()const { return m_simulationParam; }
	private:
		std::vector<std::shared_ptr<ClothPiece>> m_clothPieces;
		std::shared_ptr<ObjMesh> m_bodyMesh;
		std::shared_ptr<LevelSet3D> m_bodyLvSet;
		SimulationMode m_simulationMode;
		SimulationParam m_simulationParam;
		float m_fps;
		DragInfoInternal m_curDragInfo;
		// Topology related--------------------------------------------------------------
	protected:
		void buildTopology();
		int findNeighbor(int i, int j)const;
	private:
		std::vector<int> m_clothVertBegin;			// index begin of each cloth piece
		std::vector<Vec3> m_X;						// vertex position list
		std::vector<Vec3> m_V;						// vertex velocity list
		std::vector<ldp::Int3> m_T;					// triangle list
		std::vector<ldp::Int2> m_allE;				// edges + bending edges, for [0,1,2]+[0,1,3], bend_e=[2,3]
		std::vector<int> m_allVV;					// one-ring vertex of each vertex based an allE, NOT including self
		std::vector<ValueType> m_allVL;				// off-diag values of spring length
		std::vector<ValueType> m_allVW;				// off-diag values of springs
		std::vector<ValueType> m_allVC;				// diag values of springs
		std::vector<int> m_allVV_num;				// num of one-ring vertex of each vertex
		std::vector<float> m_fixed;					// fix constraints of vertices
		// GPU related-------------------------------------------------------------------
	protected:
		void allocateGpuMemory();
		void releaseGpuMemory();
		void copyToGpuMatrix();
		void debug_save_values();

		///// kernel wrappers
		void laplaceDamping();						// apply laplacian damping
		void updateAfterLap();						// X += V(apply air damping, gravity, etc.
		void constrain0();							// compute init_B and new_VC
		void constrain1();							// inner loop, jacobi update
		void constrain2(float omega);				// inner loop, chebshev relax
		void constrain3();							// collision handle using level set.
		void constrain4();							// update velocity
		void resetMoreFixed();						// for draging
	private:
		DeviceArray<ValueType> m_dev_X;				// position
		DeviceArray<ValueType> m_dev_old_X;			// position backup
		DeviceArray<ValueType> m_dev_next_X;		// next X for temporary storage
		DeviceArray<ValueType> m_dev_prev_X;		// prev X for temporary storage
		DeviceArray<ValueType> m_dev_fixed;			// fixed constraint, indicating which vertex should be fixed
		DeviceArray<ValueType> m_dev_more_fixed;	// for dragging
		DeviceArray<ValueType> m_dev_V;				// velocity
		DeviceArray<ValueType> m_dev_init_B;		// Initialized momentum condition in B
		DeviceArray<int> m_dev_T;					// trangle list
		DeviceArray<int> m_dev_all_VV;				// one-ring vertex list, NOT including itself
		DeviceArray<int> m_dev_all_vv_num;			// csr index of allVV
		DeviceArray<ValueType> m_dev_all_VL;		// off-diagnal values
		DeviceArray<ValueType> m_dev_all_VW;		// off-diagnal values
		DeviceArray<ValueType> m_dev_all_VC;		// diagnal values
		DeviceArray<ValueType> m_dev_new_VC;		// diagnal values 
		DeviceArray<ValueType> m_dev_phi;			// level set values
#ifdef ENABLE_SELF_COLLISION
		COLLISION_HANDLER<ValueType> m_collider;
#endif
	};
}
