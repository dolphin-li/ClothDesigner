#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_vec.h"
namespace ldp
{
	class AbstractGraphCurve;
	class GraphLoop : public AbstractGraphObject
	{
		friend class Graph;
		friend class AbstractGraphCurve;
	public:
		struct EdgeIter
		{
			GraphLoop* loop = nullptr;
			AbstractGraphCurve* curEdge = nullptr;
			bool startEdgeVisited = false;
			EdgeIter(GraphLoop* l, AbstractGraphCurve* s) : loop(l), curEdge(s) {}
			EdgeIter& operator++();
			EdgeIter operator++()const;
			bool isEnd()const
			{
				return curEdge == nullptr || isClosed();
			}
			bool isClosed()const
			{
				return curEdge == loop->m_startEdge && startEdgeVisited;
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
	public:
		GraphLoop();

		virtual GraphLoop* clone()const;
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);
		virtual Type getType()const { return TypeGraphLoop; }

		bool isClosed()const;
		bool isBoundingLoop()const { return m_isBoundingLoop; }
		void setBoundingLoop(bool b) { m_isBoundingLoop = b; }

		// iteration over all edges
		EdgeIter edge_begin() { return EdgeIter(this, m_startEdge); }
		const EdgeIter edge_begin()const { return EdgeIter((GraphLoop*)this, m_startEdge); }

		// sample points based on step and cos(angle)
		void samplePoints(std::vector<Float2>& pts, float step, float angleThreCos);
	protected:
		AbstractGraphCurve* getNextEdge(AbstractGraphCurve* curve);
	private:
		AbstractGraphCurve* m_startEdge = nullptr;
		bool m_isBoundingLoop = false;
		std::vector<AbstractGraphCurve*> m_loadedCurves; // buffer generated when loading, used in Graph to build topology
	};

	typedef std::shared_ptr<GraphLoop> GraphLoopPtr;
}