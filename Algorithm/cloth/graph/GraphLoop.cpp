#include "GraphLoop.h"
#include "AbstractGraphCurve.h"
#include "GraphPoint.h"
#include "tinyxml\tinyxml.h"
namespace ldp
{
	GraphLoop::GraphLoop() : AbstractGraphObject()
	{
	
	}

	GraphLoop::GraphLoop(size_t id) : AbstractGraphObject(id)
	{

	}

	GraphLoop* GraphLoop::clone()const
	{
		GraphLoop* loop = (GraphLoop*)create(getType(), getId());
		loop->m_startEdge = m_startEdge;
		return loop;
	}

	TiXmlElement* GraphLoop::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractGraphObject::toXML(parent);

		auto edge = m_startEdge;
		do
		{
			TiXmlElement* child = new TiXmlElement(edge->getTypeString().c_str());
			ele->LinkEndChild(child);
			child->SetAttribute("id", edge->getId());
		} while (edge && edge != m_startEdge);
		return ele;
	}

	void GraphLoop::fromXML(TiXmlElement* self)
	{
		AbstractGraphObject::fromXML(self);

		// collect curves
		std::vector<AbstractGraphCurve*> curves;
		for (auto child = self->FirstChildElement(); child; child = child->NextSiblingElement())
		{
			int id = 0;
			if (!child->Attribute("id", &id))
				throw std::exception(("cannot find id for " + getTypeString()).c_str());
			auto obj = getObjByIdx(id);
			assert(obj);
			if (!obj->isCurve())
				throw std::exception("type error");
			curves.push_back((AbstractGraphCurve*)obj);
		} // end for child
		assert(curves.size());

		// connect to next edge
		m_startEdge = curves[0];
		for (size_t i = 0; i+1 < curves.size(); i++)
			curves[i]->nextEdge() = curves[i+1];
		if (curves.back()->getEndPoint() == curves[0]->getStartPoint())
			curves.back()->nextEdge() = curves[0];

		// connect to verts
		for (AbstractGraphCurve* c : curves)
		{
			for (int i = 0; i < c->numKeyPoints(); i++)
				c->keyPoint(i)->edges().insert(c);
		}
	}

}