#include "GraphPoint.h"
#include "tinyxml\tinyxml.h"
#include "AbstractGraphCurve.h"
#include "GraphLoop.h"
namespace ldp
{
	GraphPoint g_graphPoint;

	GraphPoint::GraphPoint() : AbstractGraphObject()
	{
	
	}

	GraphPoint::GraphPoint(Float2 p) : AbstractGraphObject(), m_p(p)
	{

	}

	GraphPoint* GraphPoint::clone()const
	{
		GraphPoint* gp = (GraphPoint*)create(getType());
		gp->m_p = m_p;
		gp->m_edges = m_edges;
		return gp;
	}

	void GraphPoint::setPosition(Float2 p)
	{
		m_p = p;
		for (auto iter = edge_begin(); !iter.isEnd(); ++iter)
			iter->requireResample();
	}

	TiXmlElement* GraphPoint::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractGraphObject::toXML(parent);
		ele->SetDoubleAttribute("x", m_p[0]);
		ele->SetDoubleAttribute("y", m_p[1]);
		return ele;
	}

	void GraphPoint::fromXML(TiXmlElement* self)
	{
		AbstractGraphObject::fromXML(self);
		double x = 0, y = 0;
		if (!self->Attribute("x", &x))
			throw std::exception(("cannot find x for " + getTypeString()).c_str());
		if (!self->Attribute("y", &y))
			throw std::exception(("cannot find y for " + getTypeString()).c_str());
		m_p[0] = x;
		m_p[1] = y;
	}

}