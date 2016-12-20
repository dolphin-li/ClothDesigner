#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_vec.h"
namespace ldp
{
	class GraphLoop : public AbstractGraphObject
	{
		friend class Graph;
		friend class AbstractGraphCurve;
		friend class GraphPoint;
	public:
		class EdgeIter
		{
			GraphLoop* m_loop = nullptr;
			AbstractGraphCurve* m_curEdge = nullptr;
			bool m_startEdgeVisited = false;
			bool m_shouldReverse = false;
		public:
			EdgeIter(GraphLoop* l, AbstractGraphCurve* s);
 			EdgeIter& operator++();
			EdgeIter operator++()const { EdgeIter iter(*this); return ++iter; }
			bool isEnd()const { return m_curEdge == nullptr || isClosed(); }
			bool isClosed()const { return m_curEdge == m_loop->m_startEdge && m_startEdgeVisited; }
			AbstractGraphCurve* operator ->() { return m_curEdge;}
			AbstractGraphCurve& operator*() { return *m_curEdge; }
			bool shouldReverse()const { return m_shouldReverse; }
		};
	public:
		GraphLoop();

		virtual GraphLoop* clone()const;
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);
		virtual Type getType()const { return TypeGraphLoop; }

		bool isClosed()const;
		bool isBoundingLoop()const { return m_isBoundingLoop; }
		void setBoundingLoop(bool b) { m_isBoundingLoop = b; }

		// sample points based on step and cos(angle)
		void samplePoints(std::vector<Float2>& pts, float step, float angleThreCos);

		// iteration over all edges
		EdgeIter edge_begin() { return EdgeIter(this, m_startEdge); }
		const EdgeIter edge_begin()const { return EdgeIter((GraphLoop*)this, m_startEdge); }
	protected:
		// for winged edge
		AbstractGraphCurve* getNextEdge(AbstractGraphCurve* curve);
	private:
		AbstractGraphCurve* m_startEdge = nullptr;
		bool m_isBoundingLoop = false;
		std::vector<AbstractGraphCurve*> m_loadedCurves; // buffer generated when loading, used in Graph to build topology
	};

	typedef std::shared_ptr<GraphLoop> GraphLoopPtr;
}