#pragma once

#include <string>
#include <memory>
#include <vector>
#include <hash_map>
namespace ldp
{
	class ClothManager;
	class GraphsSewing;
	class ClothPiece;
	class ClothDesignParam;
	class AbstractGraphObject;
	class HistoryStack
	{
	public:
		typedef std::hash_map<AbstractGraphObject*, AbstractGraphObject*> PtrMap;
		enum Type
		{
			TypePatternSelect,
			TypePatternTransform,
			Type3dTransform,
			TypeGeneral,
		};
	public:
		HistoryStack();
		~HistoryStack();

		void init(ClothManager* manager);
		void push(std::string name, Type);
		void stepBackward();
		void stepForward();

		int pos()const;
		int size()const;
	protected:
		void clear();
		void stepTo(int pos);
		GraphsSewing* cloneSew(GraphsSewing* oldSew, PtrMap& ptrMap);
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
		};
		std::vector<RollBackControl> m_rollBackControls;
		int m_rollHead;
		int m_rollPos;
		int m_rollTail;
		ClothManager* m_manager;
	};
}