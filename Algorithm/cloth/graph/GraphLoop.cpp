#include "GraphLoop.h"
#include "AbstractGraphCurve.h"
#include "GraphPoint.h"
#include "tinyxml\tinyxml.h"
namespace ldp
{

	GraphLoop::EdgeIter& GraphLoop::EdgeIter::operator++()
	{
		assert(curEdge);
		startEdgeVisited = true;
		curEdge = loop->getNextEdge(curEdge);
		return *this;
	}
	GraphLoop::EdgeIter GraphLoop::EdgeIter::operator++()const
	{
		EdgeIter iter(loop, curEdge);
		return ++iter;
	}

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

	AbstractGraphCurve* GraphLoop::getNextEdge(AbstractGraphCurve* curve)
	{
		if (curve->m_leftLoop == this)
			return curve->m_leftNextEdge;
		else if (curve->m_rightLoop == this)
			return curve->m_rightNextEdge;
		else
			assert(0);
		return nullptr;
	}

	void GraphLoop::samplePoints(std::vector<Float2>& pts, float step, float angleThreCos)
	{
		pts.clear();
		for (auto iter = edge_begin(); !iter.isEnd(); ++iter)
		{
			const std::vector<Float2>& samples = iter->samplePointsOnShape(step);
			if (pts.size())
			{
				for (size_t i = 0; i < samples.size(); i++)
				{
					Float2 p = samples[i];
					// if too close
					if ((p - pts.back()).length() < step || (p - pts[0]).length() < step)
						continue;
					pts.push_back(p);
				} // end for i
			}
		}
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
}