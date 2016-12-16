#include "HistoryStack.h"
#include "clothManager.h"
#include "PanelObject\panelPolygon.h"
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
		IdxPool::disableIdxIncrement();
		m_rollBackControls.clear();
		IdxPool::enableIdxIncrement();
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

		IdxPool::disableIdxIncrement();
		AbstractPanelObject::disableIdxMapUpdate();

		auto& myData = m_rollBackControls[m_rollPos];
		myData.type = type;
		myData.name = name;
		myData.dparam.reset(new ClothDesignParam(m_manager->getClothDesignParam()));
		myData.pieces.clear();
		for (int i = 0; i < m_manager->numClothPieces(); i++)
			myData.pieces.push_back(std::shared_ptr<ClothPiece>(m_manager->clothPiece(i)->lightClone()));
		myData.sewings.clear();
		for (int i = 0; i < m_manager->numSewings(); i++)
			myData.sewings.push_back(SewingPtr(m_manager->sewing(i)->clone()));
		
		AbstractPanelObject::enableIdxMapUpdate();
		IdxPool::enableIdxIncrement();
	}

	void HistoryStack::stepTo(int pos)
	{
		if (m_manager == nullptr)
			throw std::exception("HistoryStack: not initialzied!");

		if (pos < 0 || pos >= size())
			return;

		m_rollPos = (m_rollHead + pos) % MAX_ROLLBACK_STEP;

		IdxPool::disableIdxIncrement();

		const auto& myData = m_rollBackControls[m_rollPos];
		m_manager->setClothDesignParam(*myData.dparam);
		m_manager->clearClothPieces();
		for (int i = 0; i < myData.pieces.size(); i++)
			m_manager->addClothPiece(std::shared_ptr<ClothPiece>(myData.pieces[i]->lightClone()));
		m_manager->clearSewings();
		for (int i = 0; i < myData.sewings.size(); i++)
			m_manager->addSewing(SewingPtr(myData.sewings[i]->clone())); 
		m_manager->clearHighLights();
		m_manager->simulationInit();

		IdxPool::enableIdxIncrement();
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

}