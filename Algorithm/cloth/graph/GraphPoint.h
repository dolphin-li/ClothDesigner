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
	public:
		class EdgeIter
		{
			GraphPoint* m_point = nullptr;
			AbstractGraphCurve* m_curEdge = nullptr;
			bool m_startEdgeVisited = false;
		public:
			EdgeIter(GraphPoint* p, AbstractGraphCurve* s) : m_point(p), m_curEdge(s) {}
			EdgeIter& operator++();
			EdgeIter operator++()const { EdgeIter it(*this); return ++it; }
			bool isEnd()const { return m_curEdge == nullptr || isClosed(); }
			bool isClosed()const { return m_curEdge == m_point->m_edge && m_startEdgeVisited; }
			AbstractGraphCurve* operator ->() { return m_curEdge; }
			AbstractGraphCurve& operator*() { return *m_curEdge; }
			operator AbstractGraphCurve* () { return m_curEdge; }
		};

		class LoopIter
		{
			GraphPoint* m_point = nullptr;
			GraphLoop* m_curLoop = nullptr;
			GraphLoop* m_startLoop = nullptr;
			bool m_startEdgeVisited = false;
		public:
			LoopIter(GraphPoint* p, GraphLoop* s);
			LoopIter& operator++();
			LoopIter operator++()const { LoopIter it(*this); return ++it; }
			bool isEnd()const { return m_curLoop == nullptr || isClosed(); }
			bool isClosed()const { return m_curLoop == m_startLoop && m_startEdgeVisited; }
			GraphLoop* operator ->() { return m_curLoop; }
			GraphLoop& operator*() { return *m_curLoop; }
			operator GraphLoop*() { return m_curLoop; }
		};
	public:
		GraphPoint();
		GraphPoint(Float2 p);

		virtual GraphPoint* clone()const;
		virtual Type getType()const { return TypeGraphPoint; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		Float2& position() { return m_p; }
		const Float2& position()const { return m_p; }

		// iteration over all edges
		EdgeIter edge_begin() { return EdgeIter(this, m_edge); }
		const EdgeIter edge_begin()const { return EdgeIter((GraphPoint*)this, m_edge); }

		// iteration over all loops
		EdgeIter loop_begin() { return EdgeIter(this, m_edge); }
		const EdgeIter loop_begin()const { return EdgeIter((GraphPoint*)this, m_edge); }
	protected:
		// for winged edge
		AbstractGraphCurve* nextEdge(AbstractGraphCurve* e);
		GraphLoop* leftLoop(AbstractGraphCurve* e);
		GraphLoop* rightLoop(AbstractGraphCurve* e);
	private:
		Float2 m_p;
		AbstractGraphCurve* m_edge = nullptr;
	};

	extern GraphPoint g_graphPoint;
	typedef std::shared_ptr<GraphPoint> GraphPointPtr;
}