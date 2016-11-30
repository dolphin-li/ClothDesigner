#pragma once

#include <string>
#include <memory>
#include <vector>

namespace ldp
{
	class ClothManager;
	class Sewing;
	class ClothPiece;
	class HistoryStack
	{
	public:
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
		void stepTo(int pos);
		void stepBackward();
		void stepForward();

		int pos()const;
		int size()const;
	protected:
		void clear();
	private:
		enum
		{
			MAX_ROLLBACK_STEP = 100,
		};
		struct RollBackControl
		{
			Type type;
			std::string name;
			std::vector<std::shared_ptr<Sewing>> sewings;
			std::vector<std::shared_ptr<ClothPiece>> pieces;
		};
		std::vector<RollBackControl> m_rollBackControls;
		int m_rollHead;
		int m_rollPos;
		int m_rollTail;
		ClothManager* m_manager;
	};
}