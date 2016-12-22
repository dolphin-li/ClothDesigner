#include "HistoryStack.h"
#include "clothManager.h"
#include "graph\GraphsSewing.h"
#include "graph\Graph.h"
#include "graph\AbstractGraphCurve.h"
#include "clothPiece.h"
#include "../Viewer2d.h"
namespace ldp
{
	HistoryStack::HistoryStack()
	{
	}

	HistoryStack::~HistoryStack()
	{
		clear();
	}

	void HistoryStack::init(ClothManager* manager, Viewer2d* v2d)
	{
		clear();
		m_manager = manager;
		m_viewer2d = v2d;
	}

	void HistoryStack::clear()
	{
		m_manager = nullptr;
		m_viewer2d = nullptr;
		m_rollBackControls.clear();
		m_rollBackControls.resize(MAX_ROLLBACK_STEP);
		m_rollHead = 0;
		m_rollTail = 0;
		m_rollPos = m_rollHead - 1;
	}

	void HistoryStack::push(std::string name, Type type)
	{
		if (m_manager == nullptr || m_viewer2d == nullptr)
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
		m_shouldGeneralUpdate = (myData.type > Type_CloneFlag);

		// general type operation
		Graph::PtrMap ptrMap;
		if (m_shouldGeneralUpdate)
		{
			// clone shapes and gather pointer map
			myData.pieces.clear();
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
		} // end if general type operation
		else
		{
			throw std::exception("not implemented!");
			myData.pieces.clear();
			for (int i = 0; i < m_manager->numClothPieces(); i++)
				myData.pieces.push_back(m_manager->clothPieceShared(i));
			for (int i = 0; i < m_manager->numGraphSewings(); i++)
				myData.graphSewings.push_back(m_manager->graphSewingShared(i));
		} // do not need to clone, we just share

		// ui changed operation
		myData.uiSewData.reset(new UiSewData(m_viewer2d->getUiSewData()));
		if (!ptrMap.empty())
		{
			for (auto& f : myData.uiSewData->firsts)
				f.curve = (AbstractGraphCurve*)ptrMap[f.curve];
			for (auto& f : myData.uiSewData->seconds)
				f.curve = (AbstractGraphCurve*)ptrMap[f.curve];
			myData.uiSewData->f.curve = nullptr;
			myData.uiSewData->s.curve = nullptr;
		}
	}

	void HistoryStack::stepTo(int index_)
	{
		if (m_manager == nullptr || m_viewer2d == nullptr)
			throw std::exception("HistoryStack: not initialzied!");

		if (index_ < 0 || index_ >= size())
			return;
		
		// transfer update type
		int front_index = std::max(0, std::min(index(), index_));
		int back_index = std::min(size(), std::max(index(), index_));
		for (int i = front_index; i <= back_index; i++)
		{
			int pos = convert_index_to_array(i);
			if (m_rollBackControls[pos].type > Type_CloneFlag)
			{
				m_shouldGeneralUpdate = true;
				break;
			}
		} // end for i

		// update index
		m_rollPos = convert_index_to_array(index_);
		const auto& myData = m_rollBackControls[m_rollPos];

		// perform general type transfer
		Graph::PtrMap ptrMap;
		m_manager->setClothDesignParam(*myData.dparam);
		if (m_shouldGeneralUpdate)
		{
			// clone ptr related
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
		} // end if general type

		// ui changed operation
		auto tmpUiSewData = *myData.uiSewData;
		if (!ptrMap.empty())
		{
			for (auto& f : tmpUiSewData.firsts)
			{
				auto iter = ptrMap.find(f.curve);
				if (iter == ptrMap.end())
					throw std::exception("unknown curve mapping in sewing history!");
				f.curve = (AbstractGraphCurve*)iter->second;
			}
			for (auto& f : tmpUiSewData.seconds)
			{
				auto iter = ptrMap.find(f.curve);
				if (iter == ptrMap.end())
					throw std::exception("unknown curve mapping in sewing history!");
				f.curve = (AbstractGraphCurve*)iter->second;
			}
			myData.uiSewData->f.curve = nullptr;
			myData.uiSewData->s.curve = nullptr;
		}
		m_viewer2d->setUiSewData(tmpUiSewData);

		if (m_viewer2d->getManager())
			m_viewer2d->getManager()->clearHighLights();
	}

	int HistoryStack::size()const
	{
		return (m_rollTail - m_rollHead + MAX_ROLLBACK_STEP) % MAX_ROLLBACK_STEP;
	}

	void HistoryStack::stepBackward()
	{
		if (m_manager == nullptr)
			throw std::exception("HistoryStack: not initialzied!");
		stepTo(index() - 1);
	}

	void HistoryStack::stepForward()
	{
		if (m_manager == nullptr)
			throw std::exception("HistoryStack: not initialzied!");
		stepTo(index() + 1);
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