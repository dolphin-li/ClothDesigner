#pragma once

#include <vector>
#include <memory>
#include "cudpp\device_array.h"
#include "ldpMat\ldp_basic_vec.h"
#include <map>
#include <set>
#include "definations.h"
#include "graph\AbstractGraphObject.h"
#include "cudpp\Cuda3DArray.h"
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
namespace ldp
{
	class LoopSubdiv;
	class GpuSim;
	class GraphsSewing;
	class GraphPoint;
	class Graph2Mesh;
	class ClothPiece;
	class LevelSet3D;
	class TransformInfo;
	class ClothManager
	{
		friend class GpuSim;
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
		std::string getSimulationInfo()const{ return m_simulationInfo; }
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
		void addStitchVert(const ClothPiece* cloth1, StitchPoint s1, 
			const ClothPiece* cloth2, StitchPoint s2, size_t type);
		std::pair<Float3, Float3> getStitchPos(int i);
		size_t getStitchType(int i);
		StitchPointPair getStitchPointPair(int i);
		void updateStitchAngle();
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
		const Cuda3DArray<ValueType>& bodyLevelSetDevice()const{ return m_bodyLvSet_d; }
		SmplManager* bodySmplManager() { return m_smplBody; }
		const SmplManager* bodySmplManager()const { return m_smplBody; }
		void updateSmplBody();
		void bindClothesToSmplJoints();
		void clearBindClothesToSmplJoints();
		void updateClothBySmplJoints();

		/// cloth pieces
		int numClothPieces()const { return (int)m_clothPieces.size(); }
		const ClothPiece* clothPiece(int i)const { return m_clothPieces.at(i).get(); }
		ClothPiece* clothPiece(int i) { return m_clothPieces.at(i).get(); }
		const ObjMesh& currentPieceMeshSubdiv(int i)const;
		ObjMesh& currentPieceMeshSubdiv(int i);
		const ObjMesh& currentFullMeshSubdiv()const;
		ObjMesh& currentFullMeshSubdiv();
		std::shared_ptr<ClothPiece> clothPieceShared(int i) { return m_clothPieces.at(i); }
		void clearClothPieces();
		void addClothPiece(std::shared_ptr<ClothPiece> piece);
		void removeClothPiece(size_t graphPanelId);
		void removeClothPiece(ClothPiece* piece);
		void exportClothsMerged(ObjMesh& mesh, bool mergeStitchedVertex = false)const;
		void exportClothsSeparated(std::vector<ObjMesh>& mesh)const;
		ldp::Float3 getVertexByGlobalId(int id)const;
		int pieceVertId2GlobalVertId(const ObjMesh* piece, int pieceVertId)const;

		/// bounding box
		void get2dBound(ldp::Float2& bmin, ldp::Float2& bmax)const;

		/// UI operations///
		bool removeSelectedSewings();
		bool reverseSelectedSewings();
		bool toggleSelectedSewingsType();
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
		void updateDependency();
		void calcLevelSet();
		void mergePieces();
		void buildTopology();
		void buildStitch();
		void buildSubdiv();
		void updateSubdiv();
		bool checkTopologyChangedOfMesh2d()const;
	private:
		std::string m_simulationInfo;
		std::vector<std::shared_ptr<GraphsSewing>> m_graphSewings;
		std::vector<std::shared_ptr<ClothPiece>> m_clothPieces;
		std::shared_ptr<LoopSubdiv> m_fullClothSubdiv;
		std::vector<std::shared_ptr<LoopSubdiv>> m_piecesSubdiv;
		std::shared_ptr<ObjMesh> m_bodyMesh, m_bodyMeshInit;
		static std::shared_ptr<SmplManager> m_smplMale, m_smplFemale;
		SmplManager* m_smplBody = nullptr;
		std::shared_ptr<TransformInfo> m_bodyTransform;
		std::shared_ptr<GpuSim> m_gpuSim;				// gpu cloth simulator
		std::shared_ptr<LevelSet3D> m_bodyLvSet;
		Cuda3DArray<ValueType> m_bodyLvSet_d;
		SimulationMode m_simulationMode = SimulationNotInit;
		SimulationParam m_simulationParam;
		ValueType m_fps = ValueType(0);
		bool m_shouldTriangulate = false;
		bool m_shouldMergePieces = false;
		bool m_shouldTopologyUpdate = false;
		bool m_shouldStitchUpdate = false;
		bool m_shouldLevelSetUpdate = false;
		bool m_shouldSubdivBuild = false;
		DragInfoInternal m_curDragInfo;
		std::shared_ptr<Graph2Mesh> m_graph2mesh;		// 2D-3D triangulation related
		std::map<const ObjMesh*, int> m_clothVertBegin;	// index begin of each cloth piece
		std::shared_ptr<SpMat> m_vertex_smplJointBind;	// bind each cloth vertex to some smpl joints 
		std::vector<Vec3> m_vertex_smpl_defaultPosition;
		std::vector<StitchPointPair> m_stitches;		// the elements that must be stitched together, for sewing
	};
}
