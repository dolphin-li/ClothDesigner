#include "HistoryStack.h"
#include "clothManager.h"
#include "graph\GraphsSewing.h"
#include "graph\Graph.h"
#include "graph\AbstractGraphCurve.h"
#include "clothPiece.h"
namespace ldp
{
	HistoryStack::HistoryStack()
	{
	}

	HistoryStack::~HistoryStack()
	{
		clear();
	}

	void HistoryStack::init(ClothManager* manager)
	{
		clear();
		m_manager = manager;
	}

	void HistoryStack::clear()
	{
		m_manager = nullptr;
		m_rollBackControls.clear();
		m_rollBackControls.resize(MAX_ROLLBACK_STEP);
		m_rollHead = 0;
		m_rollTail = 0;
		m_rollPos = m_rollHead - 1;
	}

	void HistoryStack::push(std::string name, Type type)
	{
		if (m_manager == nullptr)
			throw std::exception("HistoryStack: not initialzied!");
		m_rollPos = (m_rollPos + 1) % MAX_ROLLBACK_STEP;
		m_rollTail = (m_rollPos + 1) % MAX_ROLLBACK_STEP;
		if (m_rollTail == m_rollHead)
			m_rollHead = (m_rollHead + 1) % MAX_ROLLBACK_STEP;

		// clone basic info
		auto& myData = m_rollBackControls[m_rollPos];
		myData.type = type;
		myData.name = name;
		myData.dparam.reset(new ClothDesignParam(m_manager->getClothDesignParam()));
		myData.pieces.clear();

		// clone shapes and gather pointer map
		Graph::PtrMap ptrMap;
		for (int i = 0; i < m_manager->numClothPieces(); i++)
		{
			myData.pieces.push_back(std::shared_ptr<ClothPiece>(m_manager->clothPiece(i)->lightClone()));
			for (auto iter : m_manager->clothPiece(i)->graphPanel().getPtrMapAfterClone())
				ptrMap.insert(iter);
		}

		// clone graph sewing and map to new pointers
		myData.graphSewings.clear();
		for (int i = 0; i < m_manager->numGraphSewings(); i++)
			myData.graphSewings.push_back(GraphsSewingPtr(cloneSew(m_manager->graphSewing(i), ptrMap)));
	}

	void HistoryStack::stepTo(int pos)
	{
		if (m_manager == nullptr)
			throw std::exception("HistoryStack: not initialzied!");

		if (pos < 0 || pos >= size())
			return;

		m_rollPos = (m_rollHead + pos) % MAX_ROLLBACK_STEP;

		const auto& myData = m_rollBackControls[m_rollPos];
		m_manager->setClothDesignParam(*myData.dparam);

		// clone ptr related
		Graph::PtrMap ptrMap;
		m_manager->clearClothPieces();
		for (int i = 0; i < myData.pieces.size(); i++)
		{
			m_manager->addClothPiece(std::shared_ptr<ClothPiece>(myData.pieces[i]->lightClone()));
			for (auto iter : myData.pieces[i]->graphPanel().getPtrMapAfterClone())
				ptrMap.insert(iter);
		}

		// clone graph sewing and map to new pointers
		m_manager->clearSewings();
		for (int i = 0; i < myData.graphSewings.size(); i++)
			m_manager->addGraphSewing(GraphsSewingPtr(cloneSew(myData.graphSewings[i].get(), ptrMap)));

		// init simulation
		m_manager->simulationInit();
	}

	int HistoryStack::pos()const
	{
		return (m_rollPos - m_rollHead + MAX_ROLLBACK_STEP) % MAX_ROLLBACK_STEP;
	}

	int HistoryStack::size()const
	{
		return (m_rollTail - m_rollHead + MAX_ROLLBACK_STEP) % MAX_ROLLBACK_STEP;
	}

	void HistoryStack::stepBackward()
	{
		if (m_manager == nullptr)
			throw std::exception("HistoryStack: not initialzied!");
		stepTo(pos() - 1);
	}

	void HistoryStack::stepForward()
	{
		if (m_manager == nullptr)
			throw std::exception("HistoryStack: not initialzied!");
		stepTo(pos() + 1);
	}

	GraphsSewing* HistoryStack::cloneSew(GraphsSewing* oldSew, PtrMap& ptrMap)
	{
		auto sew = oldSew->clone();
		for (auto& u : sew->m_firsts)
		{
			u.curve = (AbstractGraphCurve*)ptrMap[(AbstractGraphObject*)u.curve];
			if (u.curve->graphSewings().find(oldSew) == u.curve->graphSewings().end())
				printf("history stack warning: curve %d does not relate to sew %d\n",
				u.curve->getId(), oldSew->getId());
			else
				u.curve->graphSewings().erase(oldSew);
			u.curve->graphSewings().insert(sew);
		}
		for (auto& u : sew->m_seconds)
		{
			u.curve = (AbstractGraphCurve*)ptrMap[(AbstractGraphObject*)u.curve];
			if (u.curve->graphSewings().find(oldSew) == u.curve->graphSewings().end())
				printf("history stack warning: curve %d does not relate to sew %d\n",
				u.curve->getId(), oldSew->getId());
			else
				u.curve->graphSewings().erase(oldSew);
			u.curve->graphSewings().insert(sew);
		}
		return sew;
	}
}