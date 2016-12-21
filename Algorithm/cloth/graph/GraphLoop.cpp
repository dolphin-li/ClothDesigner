#include "GraphLoop.h"
#include "AbstractGraphCurve.h"
#include "GraphPoint.h"
#include "tinyxml\tinyxml.h"
namespace ldp
{
	GraphLoop::GraphLoop() : AbstractGraphObject()
	{
	
	}

	GraphLoop* GraphLoop::clone()const
	{
		GraphLoop* loop = (GraphLoop*)create(getType());
		loop->m_startEdge = m_startEdge;
		loop->setSelected(isSelected());
		loop->m_isBoundingLoop = m_isBoundingLoop;
		return loop;
	}

	bool GraphLoop::isClosed()const
	{
		auto iter = edge_begin();
		for (; !iter.isEnd(); ++iter) {}
		return iter.isClosed();
	}

	TiXmlElement* GraphLoop::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractGraphObject::toXML(parent);

		ele->SetAttribute("bounding-loop", m_isBoundingLoop);

		for (auto iter = edge_begin(); !iter.isEnd(); ++iter)
		{
			TiXmlElement* child = new TiXmlElement(iter->getTypeString().c_str());
			ele->LinkEndChild(child);
			child->SetAttribute("id", iter->getId());
		} 
		return ele;
	}

	void GraphLoop::fromXML(TiXmlElement* self)
	{
		AbstractGraphObject::fromXML(self);

		int bd = 0;
		if (self->Attribute("bounding-loop", &bd))
			m_isBoundingLoop = !!bd;

		// collect curves
		m_loadedCurves.clear();
		for (auto child = self->FirstChildElement(); child; child = child->NextSiblingElement())
		{
			int id = 0;
			if (!child->Attribute("id", &id))
				throw std::exception(("cannot find id for " + getTypeString()).c_str());
			auto obj = getObjByIdx_loading(id);
			assert(obj);
			if (!obj->isCurve())
				throw std::exception("type error");
			m_loadedCurves.push_back((AbstractGraphCurve*)obj);
		} // end for child
		assert(m_loadedCurves.size());
	}

	//////////////////////////////////////////////////////////////
	GraphLoop::EdgeIter::EdgeIter(GraphLoop* l, AbstractGraphCurve* s) : m_loop(l), m_curEdge(s)
	{
		if (m_curEdge)
		{
			auto nl = m_loop->getNextEdge(m_curEdge);
			if (nl)
			{
				if (m_curEdge->getEndPoint() == nl->getStartPoint() 
					|| m_curEdge->getEndPoint() == nl->getEndPoint())
					m_shouldReverse = false;
				else if (m_curEdge->getStartPoint() == nl->getStartPoint() 
					|| m_curEdge->getStartPoint() == nl->getEndPoint())
					m_shouldReverse = true;
				else
					throw std::exception(("GraphLoop::EdgeIter: invalid loop " 
					+ std::to_string(m_loop->getId())).c_str());
			}
		}
	}
	
	AbstractGraphCurve* GraphLoop::getNextEdge(AbstractGraphCurve* curve)
	{
		for (auto& lk : curve->m_graphLinks)
		{
			if (lk.loop == this)
				return lk.next;
		}
		assert(0);
		return nullptr;
	}
	
	GraphLoop::EdgeIter& GraphLoop::EdgeIter::operator++()
	{
		assert(m_curEdge);
		m_startEdgeVisited = true;
		auto prevEndPt = m_shouldReverse ? m_curEdge->getStartPoint() : m_curEdge->getEndPoint();
		m_curEdge = m_loop->getNextEdge(m_curEdge);
		if (m_curEdge)
		{
			if (m_curEdge->getStartPoint() == prevEndPt)
				m_shouldReverse = false;
			else if (m_curEdge->getEndPoint() == prevEndPt)
				m_shouldReverse = true;
			else
				throw std::exception(("GraphLoop::EdgeIter: invalid loop " + std::to_string(m_loop->getId())).c_str());
		}
		return *this;
	}

	GraphLoop::KeyPointIter::KeyPointIter(GraphLoop* l, AbstractGraphCurve* s)
	{
		m_edgeIter = EdgeIter(l, s);
		m_curPtPos = m_edgeIter.shouldReverse() ? m_edgeIter->numKeyPoints() - 1 : 0;
		m_curPoint = m_edgeIter->keyPoint(m_curPtPos);
	}

	GraphLoop::KeyPointIter& GraphLoop::KeyPointIter::operator++()
	{
		m_curPtPos += m_edgeIter.shouldReverse() ? -1 : 1;
		const int endPos = m_edgeIter.shouldReverse() ? -1 : m_edgeIter->numKeyPoints();
		if (m_curPtPos == endPos)
		{
			++m_edgeIter;
			m_curPtPos = m_edgeIter.shouldReverse() ? m_edgeIter->numKeyPoints() - 1 : 0;
		}
		if (m_edgeIter)
			m_curPoint = m_edgeIter->keyPoint(m_curPtPos);
		else
			m_curPoint = nullptr;
		return *this;
	}

	GraphLoop::SamplePointIter::SamplePointIter(GraphLoop* l, AbstractGraphCurve* s, float step)
	{
		m_edgeIter = EdgeIter(l, s);
		m_step = step;
		const auto& vec = m_edgeIter->samplePointsOnShape(m_step/m_edgeIter->getLength());
		m_curPtPos = m_edgeIter.shouldReverse() ? (int)vec.size() - 1 : 0;
		m_curSample = &vec[m_curPtPos];
	}

	GraphLoop::SamplePointIter& GraphLoop::SamplePointIter::operator++()
	{
		const auto& vec = m_edgeIter->samplePointsOnShape(m_step/m_edgeIter->getLength());
		m_curPtPos += m_edgeIter.shouldReverse() ? -1 : 1;
		const int endPos = m_edgeIter.shouldReverse() ? -1 : (int)vec.size();
		if (m_curPtPos == endPos)
		{
			++m_edgeIter;
			m_curPtPos = m_edgeIter.shouldReverse() ? (int)vec.size() - 1 : 0;
		}
		if (m_edgeIter)
			m_curSample = &m_edgeIter->samplePointsOnShape(m_step/m_edgeIter->getLength())[m_curPtPos];
		else
			m_curSample = nullptr;
		return *this;
	}
}