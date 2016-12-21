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
			EdgeIter() {}
			EdgeIter(GraphLoop* l, AbstractGraphCurve* s);
 			EdgeIter& operator++();
			EdgeIter operator++()const { EdgeIter iter(*this); return ++iter; }
			bool isEnd()const { return m_curEdge == nullptr || isClosed(); }
			bool isClosed()const { return m_curEdge == m_loop->m_startEdge && m_startEdgeVisited; }
			AbstractGraphCurve* operator ->() { return m_curEdge;}
			AbstractGraphCurve& operator*() { return *m_curEdge; }
			operator AbstractGraphCurve*() { return m_curEdge; }
			bool shouldReverse()const { return m_shouldReverse; }
		};
		
		class KeyPointIter
		{
			EdgeIter m_edgeIter;
			GraphPoint* m_curPoint = nullptr;
			int m_curPtPos = 0;
		public:			
			KeyPointIter(GraphLoop* l, AbstractGraphCurve* s);
			KeyPointIter& operator++();
			KeyPointIter operator++()const { KeyPointIter iter(*this); return ++iter; }
			bool isEnd()const { return m_edgeIter.isEnd(); }
			bool isClosed()const { return m_edgeIter.isClosed(); }
			GraphPoint* operator ->() { return m_curPoint; }
			GraphPoint& operator*() { return *m_curPoint; }
			operator GraphPoint*() { return m_curPoint; }
		};

		class SamplePointIter
		{
			EdgeIter m_edgeIter;
			const Float2* m_curSample = nullptr;
			int m_curPtPos = 0;
			float m_step = 0;
		public:
			SamplePointIter(GraphLoop* l, AbstractGraphCurve* s, float step);
			SamplePointIter& operator++();
			SamplePointIter operator++()const { SamplePointIter iter(*this); return ++iter; }
			bool isEnd()const { return m_edgeIter.isEnd(); }
			bool isClosed()const { return m_edgeIter.isClosed(); }
			const Float2* operator ->() { return m_curSample; }
			const Float2& operator*() { return *m_curSample; }
			operator const Float2* () { return m_curSample; }
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
		bool isSameCurves(const GraphLoop& rhs)const;
		bool contains(const GraphLoop& rhs)const;
		bool overlapped(const GraphLoop& rhs)const;
		bool contains(const std::vector<AbstractGraphCurve*>& rhs)const;
		bool overlapped(const std::vector<AbstractGraphCurve*>& rhs)const;

		// iteration over all edges
		EdgeIter edge_begin() { return EdgeIter(this, m_startEdge); }
		const EdgeIter edge_begin()const { return EdgeIter((GraphLoop*)this, m_startEdge); }

		// iteration over all key points, including end points and bezeir points
		KeyPointIter keyPoint_begin() { return KeyPointIter(this, m_startEdge); }
		const KeyPointIter keyPoint_begin()const { return KeyPointIter((GraphLoop*)this, m_startEdge); }

		// iteration over all sample points, including end points and bezeir points
		SamplePointIter samplePoint_begin(float step) { return SamplePointIter(this, m_startEdge, step); }
		const SamplePointIter samplePoint_begin(float step)const { return SamplePointIter((GraphLoop*)this, m_startEdge, step); }
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