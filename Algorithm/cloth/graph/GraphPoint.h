#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_vec.h"

namespace ldp
{
	class GraphPoint : public AbstractGraphObject
	{
		friend class Graph;
		friend class GraphLoop;
		friend class AbstractGraphCurve;
		friend class BGraph;
	public:
		class EdgeIter
		{
			GraphPoint* m_point = nullptr;
			std::hash_set<AbstractGraphCurve*>::iterator m_curIter;
		public:
			EdgeIter(GraphPoint* p) : m_point(p), m_curIter(p->m_edges.begin()) {}
			EdgeIter& operator++() { ++m_curIter; return *this; }
			EdgeIter operator++()const { return ++EdgeIter(*this); }
			bool isEnd()const { return m_curIter == m_point->m_edges.end(); }
			AbstractGraphCurve* operator ->() { return *m_curIter; }
			AbstractGraphCurve& operator*() { return **m_curIter; }
			operator AbstractGraphCurve* () { return *m_curIter; }
		};
	public:
		GraphPoint();
		GraphPoint(Float2 p);

		virtual GraphPoint* clone()const;
		virtual Type getType()const { return TypeGraphPoint; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		const Float2& getPosition()const { return m_p; }
		void setPosition(Float2 p);

		// iteration over all edges
		EdgeIter edge_begin() { return EdgeIter(this); }
		const EdgeIter edge_begin()const { return EdgeIter((GraphPoint*)this); }
	protected:
		// for winged edge
		AbstractGraphCurve* nextEdge(AbstractGraphCurve* e);
		GraphLoop* leftLoop(AbstractGraphCurve* e);
		GraphLoop* rightLoop(AbstractGraphCurve* e);
	private:
		Float2 m_p;
		std::hash_set<AbstractGraphCurve*> m_edges;
	};

	extern GraphPoint g_graphPoint;
	typedef std::shared_ptr<GraphPoint> GraphPointPtr;
}