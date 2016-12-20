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
		gp->m_edge = m_edge;
		return gp;
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

	/////////////////////////////////////////////////////////////////////////////
	AbstractGraphCurve* GraphPoint::nextEdge(AbstractGraphCurve* e)
	{
		if (e->getStartPoint() == this)
			return e->m_leftNextEdge;
		if (e->getEndPoint() == this)
			return e->m_rightNextEdge;
		return nullptr;
	}

	GraphLoop* GraphPoint::leftLoop(AbstractGraphCurve* e)
	{
		return e->m_leftLoop;
	}

	GraphLoop* GraphPoint::rightLoop(AbstractGraphCurve* e)
	{
		return e->m_rightLoop;
	}

	GraphPoint::EdgeIter::EdgeIter(GraphPoint* p, AbstractGraphCurve* s) : 
		m_point(p), m_curEdge(s), m_startEdge(s)
	{
	}

	GraphPoint::EdgeIter& GraphPoint::EdgeIter::operator++()
	{
		m_startEdgeVisited = true;
		m_curEdge = m_point->nextEdge(m_curEdge);
		return *this;
	}

	GraphPoint::LoopIter::LoopIter(GraphPoint* p, GraphLoop* s) : m_point(p), m_curLoop(s)
	{

	}

	GraphPoint::LoopIter& GraphPoint::LoopIter::operator++()
	{
		m_startEdgeVisited = true;
		return *this;
	}
}