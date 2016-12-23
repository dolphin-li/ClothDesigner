#pragma once

#include <string>
#include <memory>
#include <vector>
#include <hash_map>
class Viewer2d;
class UiSewData;
namespace ldp
{
	class ClothManager;
	class GraphsSewing;
	class ClothPiece;
	class ClothDesignParam;
	class AbstractGraphObject;
	class TransformInfo;
	class HistoryStack
	{
	public:
		typedef std::hash_map<AbstractGraphObject*, AbstractGraphObject*> PtrMap;
		enum Type
		{
			Type_CloneFlag, // after this flag, the push/pop oeration needs heavy clones
			TypeUiSewChanged,
			TypePatternSelect,
			Type3dTransform,
			TypePatternTransform,
			TypeGeneral,
		};
	public:
		HistoryStack();
		~HistoryStack();

		void init(ClothManager* manager, Viewer2d* viewer2d);
		void push(std::string name, Type);
		void stepBackward();
		void stepForward();

		int index()const { return convert_array_to_index(m_rollPos); }
		int size()const;
	protected:
		void clear();
		void stepTo(int index);
		GraphsSewing* cloneSew(GraphsSewing* oldSew, PtrMap& ptrMap);
		int convert_array_to_index(int pos)const { return (pos - m_rollHead + MAX_ROLLBACK_STEP) % MAX_ROLLBACK_STEP; }
		int convert_index_to_array(int pos)const { return (m_rollHead + pos) % MAX_ROLLBACK_STEP; }
	private:
		enum
		{
			MAX_ROLLBACK_STEP = 100,
		};
		struct RollBackControl
		{
			Type type;
			std::string name;

			// for TypeGeneral
			std::vector<std::shared_ptr<GraphsSewing>> graphSewings;
			std::vector<std::shared_ptr<ClothPiece>> pieces;
			std::shared_ptr<ClothDesignParam> dparam;
			std::shared_ptr<TransformInfo> bodyTrans;

			// for TypeUiSewChanged
			std::shared_ptr<UiSewData> uiSewData;
		};
		std::vector<RollBackControl> m_rollBackControls;
		int m_rollHead = -1;
		int m_rollPos = -1;
		int m_rollTail = -1;
		ClothManager* m_manager = nullptr;
		Viewer2d* m_viewer2d = nullptr;
		bool m_shouldGeneralUpdate = false;
	};
}