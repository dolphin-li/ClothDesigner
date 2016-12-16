#include "Graph.h"
#include "GraphPoint.h"
#include "AbstractGraphCurve.h"
#include "GraphLoop.h"
namespace ldp
{
	Graph::Graph() : AbstractGraphObject()
	{
	
	}

	Graph::Graph(size_t id) : AbstractGraphObject(id)
	{

	}

	void Graph::clear()
	{
		m_keyPoints.clear();
		m_curves.clear();
		m_loops.clear();
		m_ptrMapAfterClone.clear();
	}

	Graph* Graph::clone()const
	{
		m_ptrMapAfterClone.clear();

		Graph* graph = new Graph();

		for (auto iter : m_keyPoints)
		{
			std::shared_ptr<GraphPoint> kp(iter.second->clone());
			m_ptrMapAfterClone.insert(std::make_pair(iter.second.get(), kp.get()));
			graph->m_keyPoints.insert(std::make_pair(kp->getId(), kp));
		}
		for (auto iter : m_curves)
		{
			std::shared_ptr<AbstractGraphCurve> kp(iter.second->clone());
			m_ptrMapAfterClone.insert(std::make_pair(iter.second.get(), kp.get()));
			graph->m_curves.insert(std::make_pair(kp->getId(), kp));
		}
		for (auto iter : m_loops)
		{
			std::shared_ptr<GraphLoop> kp(iter.second->clone());
			m_ptrMapAfterClone.insert(std::make_pair(iter.second.get(), kp.get()));
			graph->m_loops.insert(std::make_pair(kp->getId(), kp));
		}

		for (auto iter : graph->m_keyPoints)
		{
			auto tmp = iter.second->edges();
			iter.second->edges().clear();
			for (auto e : tmp)
				iter.second->edges().insert((AbstractGraphCurve*)m_ptrMapAfterClone[e]);
		}
		for (auto iter : graph->m_curves)
		{
			iter.second->nextEdge() = (AbstractGraphCurve*)m_ptrMapAfterClone[iter.second->nextEdge()];
			for (int i = 0; i < iter.second->numKeyPoints(); i++)
				iter.second->keyPoint(i) = (GraphPoint*)m_ptrMapAfterClone[iter.second->keyPoint(i)];
		}
		for (auto iter : graph->m_loops)
		{
			iter.second->startEdge() = (AbstractGraphCurve*)m_ptrMapAfterClone[iter.second->startEdge()];
		}

		return graph;
	}

	TiXmlElement* Graph::toXML(TiXmlNode* parent)const
	{
		return nullptr; // not implemented
	}

	void Graph::fromXML(TiXmlElement* self)
	{

	}

	//////////////// topology operations: add units///////////////////////////////////////////
	GraphPoint* Graph::addKeyPoint(ldp::Float2 p)
	{
		GraphPointPtr kp(new GraphPoint(p));
		addKeyPoint(kp);
		return kp.get();
	}

	GraphPoint* Graph::addKeyPoint(const std::shared_ptr<GraphPoint>& kp)
	{
		auto iter = m_keyPoints.find(kp->getId());
		if (iter != m_keyPoints.end())
		{
			printf("warning: key point %d already existed!\n", kp->getId());
			return iter->second.get();
		}
		m_keyPoints.insert(std::make_pair(kp->getId(), kp));
		return kp.get();
	}

	AbstractGraphCurve* Graph::addCurve(const std::vector<GraphPoint*>& kpts)
	{
		AbstractGraphCurvePtr curve(AbstractGraphCurve::create(kpts));
		return addCurve(curve);
	}

	AbstractGraphCurve* Graph::addCurve(const std::shared_ptr<AbstractGraphCurve>& curve)
	{
		if (curve.get() == nullptr)
			return nullptr;
		auto iter = m_curves.find(curve->getId());
		if (iter != m_curves.end())
		{
			printf("warning: addCurve: curve %d already existed!\n", curve->getId());
			return curve.get();
		}

		for (int i = 0; i < curve->numKeyPoints(); i++)
		{
			auto kp = curve->keyPoint(i);
			if (m_keyPoints.find(kp->getId()) == m_keyPoints.end())
				throw std::exception("addCurve: keyPoints not exist!");
			kp->edges().insert(curve.get());
		} // end for i

		m_curves.insert(std::make_pair(curve->getId(), curve));
		return curve.get();
	}

	GraphLoop* Graph::addLoop(const std::vector<AbstractGraphCurve*>& curves)
	{
		if (curves.size() == 0)
			return nullptr;
		for (auto& c : curves)
		{
			if (m_curves.find(c->getId()) == m_curves.end())
				throw std::exception("addLoop: curve not exist!");
		}
		for (size_t i = 1; i < curves.size(); i++)
		{
			if (curves[i - 1]->getEndPoint() != curves[i]->getStartPoint())
				throw std::exception("addLoop: given curves not connected!");
			curves[i - 1]->nextEdge() = curves[i];
		}
		std::shared_ptr<GraphLoop> loop(new GraphLoop);
		loop->startEdge() = curves[0];
		return addLoop(loop);
	}

	GraphLoop* Graph::addLoop(const std::shared_ptr<GraphLoop>& loop)
	{
		auto iter = m_loops.find(loop->getId());
		if (iter != m_loops.end())
		{
			printf("warning: addCurve: curve %d already existed!\n", loop->getId());
			return loop.get();
		}
		m_loops.insert(std::make_pair(loop->getId(), loop));
		auto edge = loop->startEdge();
		do
		{
			edge->loop() = loop.get();
			edge = edge->nextEdge();
		} while (edge && edge != loop->startEdge());
		return loop.get();
	}

	///////////////////// topology operations: remove units, return false if not removed//////////////
	bool Graph::remove(size_t id)
	{
		auto obj = getObjByIdx(id);
		if (obj == nullptr)
			return false;
		if (obj->isCurve())
			return removeCurve((AbstractGraphCurve*)obj);
		if (obj->getType() == TypeGraphPoint)
			return removeKeyPoints((GraphPoint*)obj);
		if (obj->getType() == TypeGraphLoop)
			return removeLoop((GraphLoop*)obj);
		return false;
	}

	bool Graph::removeKeyPoints(const GraphPoint* kp)
	{
		if (kp == nullptr)
			return false;
		auto iter = m_keyPoints.find(kp->getId());
		if (iter == m_keyPoints.end())
			return false;

		// remove related curves
		for (auto & c : kp->edges())
			removeCurve(c);

		// remove self
		iter->second->edges().clear();
		m_keyPoints.erase(iter);
		return true;
	}

	bool Graph::removeCurve(const AbstractGraphCurve* curve)
	{
		if (curve == nullptr)
			return false;

		auto iter = m_curves.find(curve->getId());
		if (iter == m_curves.end())
			return false;

		// check the loop consistency
		assert(iter->second->loop());
		auto prevEdge = iter->second->loop()->startEdge();
		do
		{
			if (prevEdge->nextEdge() == iter->second.get())
				break;
			prevEdge = prevEdge->nextEdge();
		} while (prevEdge && prevEdge != iter->second->loop()->startEdge());
		auto nextEdge = iter->second->nextEdge();

		// only one edge, remove the empty loop
		if (prevEdge == nullptr && nextEdge == nullptr)
			m_loops.erase(m_loops.find(iter->second->loop()->getId()));
		// start edge removed
		else if (prevEdge == nullptr && nextEdge)
			iter->second->loop()->startEdge() = nextEdge;
		// end edge removed, no change
		else if (prevEdge && nextEdge == nullptr)
			prevEdge->nextEdge() = nullptr;
		// middle edge removed, split needed
		else
		{
			prevEdge->nextEdge() = nullptr;
			std::shared_ptr<GraphLoop> loop(new GraphLoop);
			loop->startEdge() = nextEdge;
			addLoop(loop);
		}

		// remove its point
		for (int i = 0; i < iter->second->numKeyPoints(); i++)
		{
			GraphPoint* p = iter->second->keyPoint(i);
			if (p->edges().size() == 1 && *p->edges().begin() == iter->second.get())
				m_keyPoints.erase(m_keyPoints.find(p->getId()));
		} // end for i

		// remove self
		m_curves.erase(iter);
		return true;
	}

	bool Graph::removeLoop(const GraphLoop* loop)
	{
		if (loop == nullptr)
			return false;
		auto iter = m_loops.find(loop->getId());
		if (iter == m_loops.end())
			return false;

		auto edge = iter->second->startEdge();
		std::hash_set<AbstractGraphCurve*> tmp;
		do
		{
			edge->loop() = nullptr;
			tmp.insert(edge);
			edge = edge->nextEdge();
		} while (edge && edge != iter->second->startEdge());

		// by removing all edges, the loop itself is removed
		for (auto e : tmp)
			removeCurve(e);

		return true;
	}


	/////////// split a curve into two, return the inserted point and the newCurve generated/////////////
	GraphPoint* Graph::splitEdgeMakePoint(AbstractGraphCurve* curveToSplit, 
		float splitPos, AbstractGraphCurve*& newCurve)
	{
		splitPos = std::min(1.f, std::max(0.f, splitPos));

		return nullptr; // not implemented
	}


	/////////// two curves can be merged into one iff they share a common point///////////////////
	bool Graph::mergeCurve(AbstractGraphCurve* curve1, AbstractGraphCurve* curve2)
	{
		return nullptr; // not implmented
	}

}