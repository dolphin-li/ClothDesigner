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
		struct EdgeIter
		{
			GraphPoint* point = nullptr;
			AbstractGraphCurve* curEdge = nullptr;
			bool startEdgeVisited = false;
			EdgeIter(GraphPoint* p, AbstractGraphCurve* s) : point(p), curEdge(s) {}
			EdgeIter& operator++();
			EdgeIter operator++()const;
			bool isEnd()const
			{
				return curEdge == nullptr || isClosed();
			}
			bool isClosed()const
			{
				return curEdge == point->m_edge && startEdgeVisited;
			}
			AbstractGraphCurve* operator ->()
			{
				return curEdge;
			}
			AbstractGraphCurve& operator*()
			{
				return *curEdge;
			}
		};
		struct LoopIter
		{
			LoopIter(GraphPoint* p, GraphLoop* s);
			GraphPoint* point = nullptr;
			GraphLoop* curLoop = nullptr;
			GraphLoop* startLoop = nullptr;
			bool startEdgeVisited = false;
			LoopIter& operator++();
			LoopIter operator++()const;
			bool isEnd()const
			{
				return curLoop == nullptr || isClosed();
			}
			bool isClosed()const
			{
				return curLoop == startLoop && startEdgeVisited;
			}
			GraphLoop* operator ->()
			{
				return curLoop;
			}
			GraphLoop& operator*()
			{
				return *curLoop;
			}
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