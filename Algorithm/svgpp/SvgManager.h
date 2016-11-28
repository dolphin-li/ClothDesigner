#pragma once
#include <memory>
#include <map>
#include <set>
#include "ldpMat\ldp_basic_mat.h"
class Camera;
#undef min
#undef max
namespace svg
{
	class SvgGroup;
	class SvgAbstractObject;
	class SvgEdgeGroup;
	class SvgPolyPath;
	class SvgManager
	{
	public:
		struct Layer
		{
			Layer();
			void updateIndex(SvgAbstractObject* obj, int &idx);

			std::shared_ptr<SvgAbstractObject> root;
			std::map<int, SvgAbstractObject*> idxMap;
			std::set<SvgAbstractObject*> groups_for_selection;
			std::vector<std::shared_ptr<SvgEdgeGroup>> edgeGroups;
			std::string name;
			bool selected;

			// for tmp usage, need not be pushed when layer clone
			std::vector<std::set<std::pair<int, int>>> tmpLoadedEdgeGroups; // only used when loading
			std::vector<ldp::Float3> tmpLoadedEdgeGroupsColor; // only used when loading
		};
		enum SelectOp{
			SelectThis,
			SelectUnion,
			SlectionUnionInverse,
			SelectAll,
			SelectNone,
			SelectInverse,
		};
	public:
		SvgManager(){}
		~SvgManager();

		void load(const char* svg_file, bool clearOld = true);
		void save(const char* svg_file, bool selectionOnly=false);
		void render();
		void renderIndex();
		int width()const;
		int height()const;
		ldp::Float4 getBound()const;
		std::shared_ptr<SvgManager> clone()const;
		Layer* addLayer(std::string name = "");
		Layer* getLayer(std::string name);
		const Layer* getLayer(std::string name)const;
		Layer* getCurrentLayer();
		const Layer* getCurrentLayer()const;
		void setCurrentLayer(std::string name);
		void removeLayer(std::string name);
		void renameLayer(std::string oldname, std::string newname);
		void mergeSelectedLayers();
		Layer* selectedToNewLayer();
		const std::map<std::string, std::shared_ptr<Layer>>& layers()const { return m_layers; }
		SvgAbstractObject* getObjectById(int id);
		const SvgAbstractObject* getObjectById(int id)const;
		
		// extend all polyPaths symmetrically
		// axis: 0=x, 1=y, -1=auto
		void symmetryCopySelectedPoly(int axis = -1);

		// should be called after adding/removing units or groups
		void updateIndex();

		// should be called after geometric changes
		void updateBound();

		// estimate the conversion from pixels to meters
		float estimatePixelToMetersFromSelected()const;
		void setPixelToMeters(float v){ m_one_pixel_is_how_many_meters = v; }
		float getPixelToMeters()const{ return m_one_pixel_is_how_many_meters; }

		// given an index, select shapes
		void selectShapeByIndex(int id, SelectOp op = SelectThis);
		void selectShapeByIndex(const std::set<int>& ids, SelectOp op = SelectThis);
		void selectPathBySimilarSelectedWidth();
		void selectPathByWidths(const std::set<float>& widths);
		void selectGroupByIndex(int id, SelectOp op = SelectThis);
		void selectGroupByIndex(const std::set<int>& ids, SelectOp op = SelectThis);
		void highlightShapeByIndex(int lastId, int thisId);

		///// group may fail when the selected have been in different groups
		bool groupSelected();
		void ungroupSelected();
		void removeSelected();
		void removeSingleNodeAndEmptyNode();
		bool removeSelectedPolyCorners(); // return true if actually remove some corners
		bool splitSelectedPolyEdgeByMidPoint();// return true if actually add some corners

		///// path operations
		void splitSelectedPath(); // split selected, non-closed paths
		bool mergeSelectedPath(); // merge will fail if the paths have been in different groups
		void convertSelectedPathToConnectedGroups();
		void closeSelectedPolygons();
		void selectClosedPolygons();
		std::vector<SvgPolyPath*> collectPolyPaths(bool selectionOnly = true)const;
		std::vector<SvgEdgeGroup*> collectEdgeGroups(bool selectionOnly = true)const;
		void smoothSelectedPoly(double thre);
		void selectedPathsSplitByIntersect();
		void doublePathWidth();

		///// pair operations
		void makeSelectedToPair();
		// if not onlyUseSelectedEdges, then pairs on selected whole poly will be removed
		// else only the selected poly-edge pairs will be removed
		void removeSelectedPairs(bool onlyUseSelectedEdges = true); 
	protected:
		void removeSelected(SvgAbstractObject* obj);
		void removeInvalidPaths(SvgAbstractObject* obj);
		bool groupSelected_findCommonParent(std::shared_ptr<SvgAbstractObject> obj,
			std::shared_ptr<SvgAbstractObject> objParent,
			std::shared_ptr<SvgAbstractObject>& commonParent, 
			int& cnt);
		void ungroupSelected_collect(SvgAbstractObject* obj, std::set<SvgGroup*>& groups);
		void removeSingleNodeAndEmptyNode(std::shared_ptr<SvgAbstractObject>& obj);
		void splitPath(std::shared_ptr<SvgAbstractObject>& obj, bool to_single_segs);
		void cloneEdgeGroup(const Layer* oldlayer, Layer* newlayer, bool selectionOnly)const;
	private:
		std::map<std::string, std::shared_ptr<Layer>> m_layers;
		std::string m_currentLayerName;
		float m_one_pixel_is_how_many_meters; // convert from pixels to meters
	};
}
