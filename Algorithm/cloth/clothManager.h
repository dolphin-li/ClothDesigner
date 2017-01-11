#pragma once

#include <vector>
#include <memory>
#include "device_array.h"
#include "ldpMat\ldp_basic_vec.h"
#include <map>
#include <set>
#include "definations.h"
#include "graph\AbstractGraphObject.h"
#ifndef __CUDACC__
#include <eigen\Dense>
#include <eigen\Sparse>
#endif
namespace svg
{
	class SvgManager;
	class SvgPolyPath;
}
class SmplManager;
class COLLISION_HANDLER;
namespace ldp
{
	class GraphsSewing;
	class GraphPoint;
	class Graph2Mesh;
	class ClothPiece;
	class LevelSet3D;
	class BMesh;
	class BMVert;
	class BMEdge;
	class TransformInfo;
	class ClothManager
	{
	public:
		typedef float ValueType;
		typedef ldp::ldp_basic_vec3<ValueType> Vec3;
		typedef ldp::ldp_basic_vec2<ValueType> Vec2;
#ifndef __CUDACC__
		typedef Eigen::SparseMatrix<ValueType> SpMat;
#else
		class SpMat;
#endif
	protected:
		struct DragInfoInternal
		{
			int vert_id = -1;
			ldp::Float3 dir;
			ldp::Float3 target;
			int piece_id_start = 0;
			int piece_id_end = 0;
		};
	public:
		ClothManager();
		~ClothManager();

		void clear();

		// load cloth pieces from svg of my format.
		void loadPiecesFromSvg(std::string filename);
		void fromXml(std::string filename);
		void toXml(std::string filename)const;

		/// simulation main functions
		void simulationInit();							// must be called after the body and all cloths ready.
		void simulationUpdate();
		void simulationDestroy();

		/// dragging related
		void dragBegin(DragInfo info);
		void dragMove(ldp::Float3 target);
		void dragEnd();

		/// parameters related
		void setSimulationMode(SimulationMode mode);
		void setSimulationParam(SimulationParam param);
		void setClothDesignParam(ClothDesignParam param);
		void setPieceParam(const ClothPiece* piece, PieceParam param);
		float getFps()const { return m_fps; }
		SimulationMode getSimulationMode()const { return m_simulationMode; }
		SimulationParam getSimulationParam()const { return m_simulationParam; }
		static ClothDesignParam getClothDesignParam() { return g_designParam; }

		/// mesh backup related
		void updateCurrentClothsToInitial();
		void updateInitialClothsToCurrent();
		void updateCloths3dMeshBy2d();
		void resetCloths3dMeshBy2d();
		void triangulate();

		/// stitch related
		void clearSewings();
		void addStitchVert(const ClothPiece* cloth1, StitchPoint s1, const ClothPiece* cloth2, StitchPoint s2);
		std::pair<Float3, Float3> getStitchPos(int i);
		int numStitches();

		int numGraphSewings()const { return m_graphSewings.size(); }
		const GraphsSewing* graphSewing(int i)const { return m_graphSewings.at(i).get(); }
		GraphsSewing* graphSewing(int i) { return m_graphSewings.at(i).get(); }
		std::shared_ptr<GraphsSewing> graphSewingShared(int i) { return m_graphSewings.at(i); }
		bool addGraphSewing(std::shared_ptr<GraphsSewing> sewing);
		void addGraphSewings(const std::vector<std::shared_ptr<GraphsSewing>>& sewings);
		void removeGraphSewing(size_t id);
		void removeGraphSewing(GraphsSewing* sewing);

		/// body mesh
		const ObjMesh* bodyMesh()const { return m_bodyMesh.get(); }
		ObjMesh* bodyMesh() { return m_bodyMesh.get(); }
		const ObjMesh* bodyMeshInit()const { return m_bodyMeshInit.get(); }
		ObjMesh* bodyMeshInit() { return m_bodyMeshInit.get(); }
		const TransformInfo& getBodyMeshTransform()const;
		void setBodyMeshTransform(const TransformInfo& info);
		const LevelSet3D* bodyLevelSet()const { return m_bodyLvSet.get(); }
		LevelSet3D* bodyLevelSet() { return m_bodyLvSet.get(); }
		SmplManager* bodySmplManager() { return m_smplBody; }
		const SmplManager* bodySmplManager()const { return m_smplBody; }
		void updateSmplBody();
		void bindClothesToSmplJoints();
		void updateClothBySmplJoints();

		/// cloth pieces
		int numClothPieces()const { return (int)m_clothPieces.size(); }
		const ClothPiece* clothPiece(int i)const { return m_clothPieces.at(i).get(); }
		ClothPiece* clothPiece(int i) { return m_clothPieces.at(i).get(); }
		std::shared_ptr<ClothPiece> clothPieceShared(int i) { return m_clothPieces.at(i); }
		void clearClothPieces();
		void addClothPiece(std::shared_ptr<ClothPiece> piece);
		void removeClothPiece(size_t graphPanelId);
		void removeClothPiece(ClothPiece* piece);
		void exportClothsMerged(ObjMesh& mesh, bool mergeStitchedVertex = false)const;

		/// bounding box
		void get2dBound(ldp::Float2& bmin, ldp::Float2& bmax)const;

		/// UI operations///
		bool removeSelectedSewings();
		bool reverseSelectedSewings();
		bool removeSelectedShapes();
		bool removeSelectedLoops();
		bool removeLoopsOfSelectedCurves();
		bool makeSelectedCurvesToLoop();
		bool mergeSelectedCurves();
		bool mergeSelectedKeyPoints();
		bool mergeTheSelectedKeyPointToCurve();
		bool splitSelectedCurve(Float2 position);
		void clearHighLights();
		bool mirrorSelectedPanel();
		bool copySelectedPanel();
		bool addCurveOnAPanel(const std::vector<std::shared_ptr<ldp::GraphPoint>>& keyPts,
			const std::vector<size_t>& renderIds);
		bool setClothColorAsBoneWeights();
	protected:
		static void initSmplDatabase();
		void initCollisionHandler();
	private:
		std::vector<std::shared_ptr<GraphsSewing>> m_graphSewings;
		std::vector<std::shared_ptr<ClothPiece>> m_clothPieces;
		std::shared_ptr<ObjMesh> m_bodyMesh, m_bodyMeshInit;
		static std::shared_ptr<SmplManager> m_smplMale, m_smplFemale;
		SmplManager* m_smplBody = nullptr;
		std::shared_ptr<TransformInfo> m_bodyTransform;
		std::shared_ptr<LevelSet3D> m_bodyLvSet;
		SimulationMode m_simulationMode = SimulationNotInit;
		SimulationParam m_simulationParam;
		ValueType m_avgArea = ValueType(0);
		ValueType m_avgEdgeLength = ValueType(0);
		ValueType m_fps = ValueType(0);
		bool m_shouldTriangulate = false;
		bool m_shouldMergePieces = false;
		bool m_shouldTopologyUpdate = false;
		bool m_shouldNumericUpdate = false;
		bool m_shouldStitchUpdate = false;
		bool m_shouldLevelSetUpdate = false;
		DragInfoInternal m_curDragInfo;
		// 2D-3D triangulation related---------------------------------------------------
		std::shared_ptr<Graph2Mesh> m_graph2mesh;
	protected:
		BMEdge* findEdge(int v1, int v2);
	protected:
		// Topology related--------------------------------------------------------------
	protected:
		void updateDependency();
		void calcLevelSet();
		void mergePieces();
		void buildTopology();
		void buildNumerical();
		void buildStitch();
		void splitClothPiecesFromComputedMereged();
		int findNeighbor(int i, int j)const;
		int findStitchNeighbor(int i, int j)const;
		Int3 getLocalFaceVertsId(Int3 globalVertId)const;
		std::pair<const ObjMesh*, int> getLocalVertsId(int globalVertId)const;
		void updateSewingNormals(ObjMesh& mesh);
	private:
		std::shared_ptr<BMesh> m_bmesh;					// topology mesh
		std::vector<BMVert*> m_bmeshVerts;				// topology mesh
		std::map<const ObjMesh*, int> m_clothVertBegin;	// index begin of each cloth piece
		std::map<std::pair<const ObjMesh*, int>, std::set<Int3>> m_sewVofFMap;// two boundary faces that stitched, for normal calculation
		std::shared_ptr<SpMat> m_vertex_smplJointBind;	// bind each cloth vertex to some smpl joints 
		std::vector<Vec3> m_vertex_smpl_defaultPosition;
		std::vector<Vec3> m_X;							// vertex position list
		std::vector<Vec3> m_V;							// vertex velocity list
		std::vector<ValueType> m_V_bending_k_mult;			// bending param of each vertex
		std::vector<ValueType> m_V_outgo_dist;			// we want some vertices to go outside some distance
		std::vector<Int3> m_T;							// triangle list
		std::vector<Int2> m_allE;						// edges + bending edges, sorted, for [0,1,2]+[0,1,3], bend_e=[2,3]
		std::vector<int> m_allVV;						// one-ring vertex of each vertex based an allE, NOT including self
		std::vector<ValueType> m_allVL;					// off-diag values of spring length * spring_k
		std::vector<ValueType> m_allVW;					// off-diag values of springs
		std::vector<ValueType> m_allVC;					// diag values of springs
		std::vector<int> m_allVV_num;					// num of one-ring vertex of each vertex
		std::vector<ValueType> m_fixed;					// fix constraints of vertices
		std::vector<Int4> m_edgeWithBendEdge;			// original edges + beding edges, before sorted and unique.
		std::vector<StitchPointPair> m_stitches;		// the elements that must be stitched together, for sewing
		std::vector<int> m_stitchVV;
		std::vector<int> m_stitchVV_num;				// csr header of the sparse matrix vv
		std::vector<ValueType> m_stitchVW;
		std::vector<ValueType> m_stitchVC;
		std::vector<ValueType> m_stitchVL;

		ValueType m_curStitchRatio;						// the stitchEdge * ratio is the current stitched length
		// GPU related-------------------------------------------------------------------
	protected:
		void debug_save_values();

		///// kernel wrappers
		void laplaceDamping();						// apply laplacian damping
		void updateAfterLap();						// X += V(apply air damping, gravity, etc.
		void constrain0();							// compute init_B and new_VC
		void constrain1();							// inner loop, jacobi update
		void constrain2(float omega);				// inner loop, chebshev relax
		void constrain3();							// collision handle using level set.
		void constrain4();							// update velocity
		void constrain_selfCollision();
		void resetMoreFixed();						// for draging
	private:
		DeviceArray<ValueType> m_dev_X;				// position
		DeviceArray<ValueType> m_dev_old_X;			// position backup
		DeviceArray<ValueType> m_dev_next_X;		// next X for temporary storage
		DeviceArray<ValueType> m_dev_prev_X;		// prev X for temporary storage
		DeviceArray<ValueType> m_dev_fixed;			// fixed constraint, indicating which vertex should be fixed
		DeviceArray<ValueType> m_dev_more_fixed;	// for dragging
		DeviceArray<ValueType> m_dev_V;				// velocity
		DeviceArray<ValueType> m_dev_V_outgo_dist;	// vertex outgo shifts
		DeviceArray<ValueType> m_dev_init_B;		// Initialized momentum condition in B
		DeviceArray<int> m_dev_T;					// trangle list
		DeviceArray<int> m_dev_all_VV;				// one-ring vertex list, NOT including itself
		DeviceArray<int> m_dev_all_vv_num;			// csr index of allVV
		DeviceArray<ValueType> m_dev_all_VL;		// off-diagnal values * springk
		DeviceArray<ValueType> m_dev_all_VW;		// off-diagnal values
		DeviceArray<ValueType> m_dev_all_VC;		// diagnal values
		DeviceArray<ValueType> m_dev_new_VC;		// diagnal values 
		DeviceArray<ValueType> m_dev_phi;			// level set values
		DeviceArray<int> m_dev_stitch_VV;
		DeviceArray<int> m_dev_stitch_VV_num;
		DeviceArray<ValueType> m_dev_stitch_VW;
		DeviceArray<ValueType> m_dev_stitch_VC;
		DeviceArray<ValueType> m_dev_stitch_VL;

		//// self collision handler--------------------------------------------------------
		std::shared_ptr<COLLISION_HANDLER> m_collider;
	};
}
