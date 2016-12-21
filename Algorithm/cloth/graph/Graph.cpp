#include "Graph.h"
#include "GraphPoint.h"
#include "AbstractGraphCurve.h"
#include "GraphLoop.h"
#include "GraphsSewing.h"
#include "tinyxml\tinyxml.h"
#include "cloth\definations.h"
#include "GraphsSewing.h"
namespace ldp
{
	Graph::Graph() : AbstractGraphObject()
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
		graph->setSelected(isSelected());
		graph->setHighlighted(false);
		// clone the objects
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

		// clone the graph relations
		for (auto iter : graph->m_keyPoints)
		{
			auto tmpEdges = iter.second->m_edges;
			iter.second->m_edges.clear();
			for (auto e : tmpEdges)
				iter.second->m_edges.insert((AbstractGraphCurve*)m_ptrMapAfterClone[e]);
			for (auto e : iter.second->m_edges)
			{
				if (e == nullptr)
					throw std::exception("point edge clone error");
			}
		}
		for (auto iter : graph->m_curves)
		{
			auto tmp = iter.second->m_graphLinks;
			iter.second->m_graphLinks.clear();
			for (auto lk : tmp)
			{
				lk.second.loop = (GraphLoop*)m_ptrMapAfterClone[lk.second.loop];
				lk.second.next = (AbstractGraphCurve*)m_ptrMapAfterClone[lk.second.next];
				lk.second.prev = (AbstractGraphCurve*)m_ptrMapAfterClone[lk.second.prev];
				iter.second->m_graphLinks.insert(std::make_pair(lk.second.loop, lk.second));
			}
			for (int i = 0; i < iter.second->numKeyPoints(); i++)
				iter.second->keyPoint(i) = (GraphPoint*)m_ptrMapAfterClone[iter.second->keyPoint(i)];
		}
		for (auto iter : graph->m_loops)
			iter.second->m_startEdge = (AbstractGraphCurve*)m_ptrMapAfterClone[iter.second->m_startEdge];

		graph->m_bbox[0] = m_bbox[0];
		graph->m_bbox[1] = m_bbox[1];
		return graph;
	}

	TiXmlElement* Graph::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractGraphObject::toXML(parent);

		TiXmlElement* ele_kpts = new TiXmlElement("key-points");
		ele->LinkEndChild(ele_kpts);
		for (auto& p : m_keyPoints)
			p.second->toXML(ele_kpts);
		
		TiXmlElement* ele_cvs = new TiXmlElement("curves");
		ele->LinkEndChild(ele_cvs);
		for (auto& p : m_curves)
			p.second->toXML(ele_cvs);

		TiXmlElement* ele_loops = new TiXmlElement("loops");
		ele->LinkEndChild(ele_loops);
		for (auto& p : m_loops)
			p.second->toXML(ele_loops);

		return ele;
	}

	void Graph::fromXML(TiXmlElement* self)
	{
		clear();
		AbstractGraphObject::fromXML(self);

		for (auto groups = self->FirstChildElement(); groups; groups = groups->NextSiblingElement())
		{
			if (groups->Value() == std::string("key-points"))
			{
				for (auto child = groups->FirstChildElement(); child; child = child->NextSiblingElement())
				{
					GraphPointPtr kpt((GraphPoint*)AbstractGraphObject::create(child->Value()));
					kpt->fromXML(child);
					s_idxObjMap_loading[kpt->m_loadedId] = addKeyPoint(kpt, false);
				}
			} // end if key-pts
			if (groups->Value() == std::string("curves"))
			{
				for (auto child = groups->FirstChildElement(); child; child = child->NextSiblingElement())
				{
					AbstractGraphCurvePtr kpt((AbstractGraphCurve*)AbstractGraphObject::create(child->Value()));
					kpt->fromXML(child);
					s_idxObjMap_loading[kpt->m_loadedId] = addCurve(kpt);
				}
			} // end if curves
			if (groups->Value() == std::string("loops"))
			{
				for (auto child = groups->FirstChildElement(); child; child = child->NextSiblingElement())
				{
					GraphLoopPtr kpt((GraphLoop*)AbstractGraphObject::create(child->Value()));
					kpt->fromXML(child);
					s_idxObjMap_loading[kpt->m_loadedId] = addLoop(kpt->m_loadedCurves, kpt->isBoundingLoop());
				}
			} // end if loops
		} // end for groups
		updateBound();

		// auto make bounding loop
		std::vector<GraphLoop*> closedLoops;
		for (auto iter : m_loops)
		if (iter.second->isClosed())
			closedLoops.push_back(iter.second.get());
		if (closedLoops.size() == 1)
			closedLoops[0]->setBoundingLoop(true);
	}

	bool Graph::select(int idx, SelectOp op)
	{
		if (op == SelectEnd)
			return false;
		std::set<int> idxSet;
		idxSet.insert(idx);
		return select(idxSet, op);
	}

	bool Graph::select(const std::set<int>& idxSet, SelectOp op)
	{
		if (op == SelectEnd)
			return false;
		static std::vector<AbstractGraphObject*> objs;
		objs.clear();
		for (auto& o : m_keyPoints)
			objs.push_back(o.second.get());
		for (auto& o : m_curves)
			objs.push_back(o.second.get());
		for (auto& o : m_loops)
			objs.push_back(o.second.get());
		objs.push_back(this);
		bool changed = false;
		for (auto& obj : objs)
		{
			bool oldSel = obj->isSelected();
			switch (op)
			{
			case ldp::AbstractGraphObject::SelectThis:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(true);
				else
					obj->setSelected(false);
				break;
			case ldp::AbstractGraphObject::SelectUnion:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(true);
				break;
			case ldp::AbstractGraphObject::SelectUnionInverse:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(!obj->isSelected());
				break;
			case ldp::AbstractGraphObject::SelectAll:
				obj->setSelected(true);
				break;
			case ldp::AbstractGraphObject::SelectNone:
				obj->setSelected(false);
				break;
			case ldp::AbstractGraphObject::SelectInverse:
				obj->setSelected(!obj->isSelected());
				break;
			default:
				break;
			}
			bool newSel = obj->isSelected();
			if (oldSel != newSel)
				changed = true;
		} // end for obj
		return changed;
	}

	void Graph::highLight(int idx, int lastIdx)
	{
		auto cur = getObjByIdx(idx);
		if (cur)
			cur->setHighlighted(true);
		if (idx != lastIdx)
		{
			auto pre = getObjByIdx(lastIdx);
			if (pre)
				pre->setHighlighted(false);
		}
	}

	void Graph::updateBound(Float2& bmin, Float2& bmax)
	{
		m_bbox[0] = FLT_MAX;
		m_bbox[1] = -FLT_MAX;

		for (const auto& o : m_curves)
			o.second->unionBound(m_bbox[0], m_bbox[1]);

		for (int k = 0; k < bmin.size(); k++)
		{
			bmin[k] = std::min(bmin[k], m_bbox[0][k]);
			bmax[k] = std::max(bmax[k], m_bbox[1][k]);
		}
	}

	bool Graph::makeGraphValid(std::vector<std::shared_ptr<GraphsSewing>>& graphSewings)
	{
		// 1. check the bounding loop

		return true;
	}
	//////////////// topology operations: add units///////////////////////////////////////////
	GraphPoint* Graph::addKeyPoint(ldp::Float2 p, bool isEndPoint)
	{
		GraphPointPtr kp(new GraphPoint(p));
		return addKeyPoint(kp, isEndPoint);
	}

	GraphPoint* Graph::addKeyPoint(const std::shared_ptr<GraphPoint>& kp, bool isEndPoint)
	{
		auto iter = m_keyPoints.find(kp->getId());
		if (iter != m_keyPoints.end())
		{
			printf("warning: key point %d already existed!\n", kp->getId());
			return iter->second.get();
		}

		// a point exist that is close enough with p
		if (isEndPoint)
		{
			for (auto iter : m_keyPoints)
			{
				if ((iter.second->getPosition() - kp->getPosition()).length() < g_designParam.pointMergeDistThre)
					return iter.second.get();
			} // end for iter
		}

		m_keyPoints.insert(std::make_pair(kp->getId(), kp));
		return kp.get();
	}

	AbstractGraphCurve* Graph::addCurve(const std::vector<std::shared_ptr<GraphPoint>>& kpts)
	{
		std::vector<GraphPoint*> ptr;
		for (const auto& kp : kpts)
			ptr.push_back(addKeyPoint(kp, kp == kpts.front() || kp == kpts.back()));
		return addCurve(ptr);
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

		// check curve id
		auto iter = m_curves.find(curve->getId());
		if (iter != m_curves.end())
		{
			printf("warning: addCurve: curve %d already existed!\n", curve->getId());
			return curve.get();
		}

		// check curve points, both the curve and its reverse version is viewed as the same curve
		for (auto iter : m_curves)
		{
			if (iter.second->isEndPointsSame(curve.get()) || iter.second->isEndPointsReversed(curve.get()))
				return iter.second.get();
		} // end for iter

		// add the curve
		for (int i = 0; i < curve->numKeyPoints(); i++)
		{
			auto kp = curve->keyPoint(i);
			if (m_keyPoints.find(kp->getId()) == m_keyPoints.end())
				throw std::exception("addCurve: keyPoints not exist!");
			kp->m_edges.insert(curve.get());
		} // end for i

		m_curves.insert(std::make_pair(curve->getId(), curve));
		return curve.get();
	}

	GraphLoop* Graph::addLoop(const std::vector<std::vector<GraphPointPtr>>& curves, bool isBoundingLoop)
	{
		std::vector<AbstractGraphCurve*> ptr;
		for (const auto& c : curves)
			ptr.push_back(addCurve(c));
		return addLoop(ptr, isBoundingLoop);
	}

	GraphLoop* Graph::addLoop(const std::vector<AbstractGraphCurvePtr>& curves, bool isBoundingLoop)
	{
		std::vector<AbstractGraphCurve*> ptr;
		for (const auto& c : curves)
			ptr.push_back(addCurve(c));
		return addLoop(ptr, isBoundingLoop);
	}

	void Graph::connectNextCurve(AbstractGraphCurve* curr, AbstractGraphCurve* next, GraphLoop* loop, bool reverse)
	{
		for (auto lk = curr->diskLink_begin(); !lk.isEnd(); ++lk)
		{
			if (lk.loop() == loop)
				lk.next() = next;
		}
	}

	void Graph::connectPrevCurve(AbstractGraphCurve* curr, AbstractGraphCurve* prev, GraphLoop* loop, bool reverse)
	{
		for (auto lk = curr->diskLink_begin(); !lk.isEnd(); ++lk)
		{
			if (lk.loop() == loop)
				lk.prev() = prev;
		}
	}

	GraphLoop* Graph::addLoop(std::vector<AbstractGraphCurve*>& curves, bool isBoundingLoop)
	{
		if (curves.size() == 0)
			return nullptr;
		for (auto& c : curves)
		{
			if (m_curves.find(c->getId()) == m_curves.end())
				throw std::exception("addLoop: curve not exist!");
		}

		// check the connectivity
		std::vector<int> shouldCurveReverse(curves.size(), 0);
		for (size_t i = 1; i < curves.size(); i++)
		{
			if (i == 1)
			{
				if (curves[0]->getEndPoint() != curves[1]->getStartPoint()
					&& curves[0]->getEndPoint() != curves[1]->getEndPoint())
					shouldCurveReverse[0] = 1;
			}
			auto lp = shouldCurveReverse[i - 1] ? curves[i - 1]->getStartPoint() : curves[i - 1]->getEndPoint();
			if (curves[i]->getStartPoint() == lp)
				shouldCurveReverse[i] = 0;
			else if (curves[i]->getEndPoint() == lp)
				shouldCurveReverse[i] = 1;
			else
				throw std::exception("addLoop: given curves not connected!");
		}

		// reverse and check if needed
		for (size_t i = 0; i < curves.size(); i++)
		{
			int& rev = shouldCurveReverse[i];
			if (curves[i]->m_graphLinks.empty() && rev)
			{
				curves[i]->reverse();
				rev = 0;
			} // end if not related to loops
		} // end for i

		// check contains
		for (auto iter : m_loops)
		{
			if (iter.second->contains(curves) || iter.second->containedBy(curves))
			{
				printf("warning: addLoop: loop %d and the given curves may contains each other!\n", iter.second->getId());
				return iter.second.get();
			}
		}

		// create the loop
		std::shared_ptr<GraphLoop> loop(new GraphLoop);
		loop->m_startEdge = curves[0];
		loop->setBoundingLoop(isBoundingLoop);

		// relate edge to left/right loop
		for (auto& c : curves)
		{
			GraphDiskLink link;
			link.loop = loop.get();
			c->m_graphLinks.insert(std::make_pair(link.loop, link));
		}

		// connect curves with each other
		for (size_t i = 1; i < curves.size(); i++)
		{
			connectNextCurve(curves[i - 1], curves[i], loop.get(), shouldCurveReverse[i - 1]);
			connectPrevCurve(curves[i], curves[i - 1], loop.get(), shouldCurveReverse[i]);
		}

		auto fs = shouldCurveReverse.front() ? curves.front()->getEndPoint() : curves.front()->getStartPoint();
		auto be = shouldCurveReverse.back() ? curves.back()->getStartPoint() : curves.back()->getEndPoint();
		if (fs == be && curves.back() != curves.front())
		{
			connectNextCurve(curves.back(), curves.front(), loop.get(), shouldCurveReverse.back());
			connectPrevCurve(curves.front(), curves.back(), loop.get(), shouldCurveReverse.front());
		}

		return addLoop(loop);
	}

	GraphLoop* Graph::addLoop(const std::shared_ptr<GraphLoop>& loop)
	{
		if (loop->isBoundingLoop() && !loop->isClosed())
			throw std::exception("error: addLoop: bounding loop must be closed!");

		// check id
		auto iter = m_loops.find(loop->getId());
		if (iter != m_loops.end())
		{
			printf("warning: addLoop: loop %d already existed!\n", loop->getId());
			return loop.get();
		}

		// check same loop
		for (auto iter : m_loops)
		{
			if (iter.second->isSameCurves(*loop))
				return iter.second.get();
		}

		// check that there is only one bounding loop
		int nBounds = loop->isBoundingLoop();
		for (auto l : m_loops)
			nBounds += l.second->isBoundingLoop();
		if (nBounds > 1)
			throw std::exception("addLoop error: there must be exactly 1 bounding loop!");

		// insert loop
		m_loops.insert(std::make_pair(loop->getId(), loop));
		return loop.get();
	}

	GraphLoop* Graph::getBoundingLoop()
	{
		if (m_loops.size() == 0)
			return nullptr;
		int n = 0;
		GraphLoop* lp = nullptr;
		for (auto& iter : m_loops)
		if (iter.second->isBoundingLoop())
		{
			lp = iter.second.get();
			n++;
		}
		if (n > 1)
			throw std::exception("getBoundingLoop: there must be exactly 1 bounding loop!\n");
		return lp;
	}

	const GraphLoop* Graph::getBoundingLoop()const
	{
		if (m_loops.size() == 0)
			return nullptr;
		int n = 0;
		GraphLoop* lp = nullptr;
		for (auto& iter : m_loops)
		if (iter.second->isBoundingLoop())
		{
			lp = iter.second.get();
			n++;
		}
		if (n > 1)
			throw std::exception("const getBoundingLoop: there must be exactly 1 bounding loop!\n");
		return lp;
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

		// check existence
		const int id = kp->getId();
		auto kp_iter = m_keyPoints.find(id);
		if (kp_iter == m_keyPoints.end())
			return false;

		// remove the associated cuves
		auto tmpEdges = kp_iter->second->m_edges;
		for (auto& edge : tmpEdges)
			removeCurve(edge);

		// remove self
		kp_iter = m_keyPoints.find(id);
		if (kp_iter != m_keyPoints.end())
			m_keyPoints.erase(kp_iter);
		return true;
	}

	bool Graph::removeCurve(const AbstractGraphCurve* curve)
	{
		if (curve == nullptr)
			return false;

		// check existence
		const int id = curve->getId();
		auto curve_iter = m_curves.find(id);
		if (curve_iter == m_curves.end())
			return false;

		// modify associated loops
		for (auto lk = curve_iter->second->diskLink_begin(); !lk.isEnd(); ++lk)
		{
			if (!lk.loop()->isClosed())
			{
				std::vector<AbstractGraphCurve*> newLoop;
				for (auto eiter = lk.loop()->edge_begin(); !eiter.isEnd(); ++eiter)
				{
					if (eiter == curve_iter->second.get())
						break;
					newLoop.push_back(eiter);
				} // end for eiter
				for (auto c : newLoop)
					c->m_graphLinks.erase((GraphLoop*)lk.loop());
				addLoop(newLoop, false);
			} // end if non-closed

			lk.loopStartEdge() = lk.next();
			if (lk.prev())
			{
				for (auto prev_lk = lk.prev()->diskLink_begin(); !prev_lk.isEnd(); ++prev_lk)
				if (prev_lk.loop() == lk.loop())
					prev_lk.next() = nullptr;
			}
			if (lk.next())
			{
				for (auto next_lk = lk.next()->diskLink_begin(); !next_lk.isEnd(); ++next_lk)
				if (next_lk.loop() == lk.loop())
					next_lk.prev() = nullptr;
			}
		} // end for lk

		// remove empty loops
		auto tmpLinks = curve_iter->second->m_graphLinks;
		for (auto& lk : tmpLinks)
		{
			if (lk.second.loop->m_startEdge == nullptr)
				removeLoop(lk.second.loop);
		}

		// remove isolated points
		auto tmpKps = curve_iter->second->m_keyPoints;
		for (auto& kp : tmpKps)
		{
			kp->m_edges.erase(curve_iter->second.get());
			if (kp->m_edges.empty())
				removeKeyPoints(kp);
		}

		// remove self
		curve_iter = m_curves.find(id);
		if (curve_iter != m_curves.end())
			m_curves.erase(curve_iter);
		return true;
	}

	bool Graph::removeLoop(const GraphLoop* loop)
	{
		if (loop == nullptr)
			return false;

		// check existence
		const int id = loop->getId();
		auto loop_iter = m_loops.find(id);
		if (loop_iter == m_loops.end())
			return false;

		// remove links related with it
		std::vector<AbstractGraphCurve*> curves;
		for (auto eiter = loop->edge_begin(); !eiter.isEnd(); ++eiter)
		{
			eiter->m_graphLinks.erase(loop_iter->second.get());
			curves.push_back(eiter);
		} // end for eiter

		//// remove isolated curves
		//for (auto c : curves)
		//{
		//	if (c->m_graphLinks.empty())
		//		removeCurve(c);
		//}

		// remove self
		loop_iter = m_loops.find(id);
		if (loop_iter != m_loops.end())
			m_loops.erase(loop_iter);
		return true;
	}

	/////////// split a curve into two, return the inserted point and the newCurve generated/////////////
	GraphPoint* Graph::splitEdgeMakePoint(AbstractGraphCurve* curveToSplit, 
		Float2 splitPos, AbstractGraphCurve*& newCurve)
	{
		newCurve = nullptr;
		const float step = g_designParam.curveSampleStep / curveToSplit->getLength();
		const auto& vec = curveToSplit->samplePointsOnShape(step);

		float minDist = FLT_MAX;
		float tSplit = 0;
		for (size_t i = 1; i < vec.size(); i++)
		{
			float dist = pointSegDistance(splitPos, vec[i - 1], vec[i]);
			if (dist < minDist)
			{
				minDist = dist;
				tSplit = (i - 1)*step + nearestPointOnSeg_getParam(splitPos, vec[i - 1], vec[i]) * step;
			}
		}

		// too close, do not split
		if (tSplit < 0.1 || tSplit > 0.9)
			return nullptr;

		return nullptr; // not implemented
	}

	/////////// two curves can be merged into one iff they share a common point///////////////////
	bool Graph::mergeCurve(AbstractGraphCurve* curve1, AbstractGraphCurve* curve2,
		AbstractGraphCurve*& mergedCurve)
	{
		mergedCurve = nullptr;
		// perform security checking
		if (curve1 == nullptr || curve2 == nullptr)
			return false;
		if (m_curves.find(curve1->getId()) == m_curves.end() 
			|| m_curves.find(curve2->getId()) == m_curves.end())
			throw std::exception("mergeCurve: given curve not in the graph!");
		if (curve1 == curve2)
			return false;
		if (curve1->isEndPointsSame(curve2))
			throw std::exception("mergeCurve: duplicated curves in the graph!");
		if (curve1->getEndPoint() != curve2->getStartPoint())
			std::swap(curve1, curve2);
		if (curve1->getEndPoint() != curve2->getStartPoint())
			return false; // not connected
		if (curve1->m_graphLinks.size() != curve2->m_graphLinks.size())
			return false; // not share same loops
		for (auto iter1 = curve1->m_graphLinks.begin(), iter2 = curve2->m_graphLinks.begin();
			iter1 != curve1->m_graphLinks.end() && iter2 != curve2->m_graphLinks.end(); ++iter1, ++iter2)
		{
			if (iter1->first != iter2->first)
				return false; // not share same loops
		}
		if (curve1->m_sewings.size() != curve2->m_sewings.size())
			return false; // not share same sewings
		for (auto iter1 = curve1->m_sewings.begin(), iter2 = curve2->m_sewings.begin();
			iter1 != curve1->m_sewings.end() && iter2 != curve2->m_sewings.end(); ++iter1, ++iter2)
		{
			if (*iter1 != *iter2)
				return false; // not share same sewings
		}
		
		// modify curve2 to fit curve1+curve2
		auto vec1 = curve1->samplePointsOnShape(0.1);
		auto vec2 = curve2->samplePointsOnShape(0.1);
		vec1.insert(vec1.end(), vec2.begin(), vec2.end());
		std::vector<GraphPointPtr> fittedCurvePts;
		AbstractGraphCurve::fittingOneCurve(fittedCurvePts, vec1, g_designParam.curveFittingThre);
		mergedCurve = addCurve(fittedCurvePts);
		if (mergedCurve == nullptr)
			return false;
		assert(mergedCurve->getStartPoint() == curve1->getStartPoint() 
			&& mergedCurve->getEndPoint() == curve2->getEndPoint());
		mergedCurve->m_sewings = curve1->m_sewings;
		mergedCurve->m_graphLinks = curve1->m_graphLinks;
		mergedCurve->setSelected(curve1->isSelected());

		// update the sewings
		for (auto& sew : mergedCurve->m_sewings)
		{
			sew->swapCurve(curve1, mergedCurve);
			sew->remove(curve2->getId());
		}

		// merge the links
		for (auto& lk1 : mergedCurve->m_graphLinks)
		{
			assert(lk1.second.next == curve2);
			lk1.second.next = curve2->m_graphLinks[lk1.first].next;
			if (lk1.second.loop->m_startEdge == curve1 || lk1.second.loop->m_startEdge == curve2)
				lk1.second.loop->m_startEdge = mergedCurve;
			if (lk1.second.prev)
				lk1.second.prev->m_graphLinks[lk1.first].next = mergedCurve;
			if (lk1.second.next)
				lk1.second.next->m_graphLinks[lk1.first].prev = mergedCurve;
		} // end for lk

		// remove old cuves
		curve1->m_graphLinks.clear();
		curve2->m_graphLinks.clear();
		removeCurve(curve1);
		removeCurve(curve2);
		
		return true;
	}

	// two points can be merged into one iff
	bool Graph::mergeKeyPoints(GraphPoint* p1, GraphPoint* p2)
	{
		if (p1 == p2 || p1 == nullptr || p2 == nullptr)
			return false;

		// only end point can be merged
		for (auto& e1 : p1->m_edges)
		if (e1->getStartPoint() != p1 && e1->getEndPoint() != p1)
			return false;
		for (auto& e2 : p2->m_edges)
		if (e2->getStartPoint() != p2 && e2->getEndPoint() != p2)
			return false;
		
		// if the two points share common edges, we cannot merge them
		for (auto& e1 : p1->m_edges)
		{
			if (p2->m_edges.find(e1) != p2->m_edges.end())
				return false;
			// if the two points share common loops, we cannot merge them
			for (auto& e2 : p2->m_edges)
			{
				for (auto lk : e1->m_graphLinks)
				if (e2->m_graphLinks.find(lk.first) != e2->m_graphLinks.end())
					return false;
			}
		}

		// merge p2 into p1 for p2_edges
		for (auto& e2 : p2->m_edges)
		{
			for (auto& e2_p : e2->m_keyPoints)
			if (e2_p == p2)
				e2_p = p1;
			e2->requireResample();
			p1->m_edges.insert(e2);
		}

		// remove p2
		p2->m_edges.clear();
		removeKeyPoints(p2);

		return true;
	}

	/////////////// ui operations///////////////////////////////////
	bool Graph::selectedCurvesToLoop(bool isBoundingLoop)
	{
		std::hash_map<AbstractGraphCurve*, GraphDiskLink> curves;
		for (auto iter = m_curves.begin(); iter != m_curves.end(); ++iter)
		if (iter->second->isSelected())
			curves.insert(std::make_pair(iter->second.get(), GraphDiskLink()));

		if (curves.size() < 1)
			return false;

		if (curves.size() == 1)
		{
			std::vector<AbstractGraphCurve*> tc;
			tc.push_back((AbstractGraphCurve*)(curves.begin()->first));
			return addLoop(tc, isBoundingLoop) != nullptr;
		}

		// find a start curve
		AbstractGraphCurve* startCurve = nullptr;
		for (auto& curve : curves)
		{
			for (auto& other : curves)
			{
				if (other.first == curve.first)
					continue;

				// find link
				if (curve.first->getEndPoint() == other.first->getStartPoint()
					|| curve.first->getEndPoint() == other.first->getEndPoint())
				{
					if (curve.second.next)
						return false; // we donot allow multi-connection
					curve.second.next = other.first;
				}
				if (curve.first->getStartPoint() == other.first->getStartPoint()
					|| curve.first->getStartPoint() == other.first->getEndPoint())
				{
					if (curve.second.prev)
						return false; // we donot allow multi-connection
					curve.second.prev = other.first;
				}
			} // end for other

			// we donot allow non-connected curves
			if (curve.second.prev == nullptr && curve.second.next == nullptr)
				return false;
		} // end for curve

		// set an end point as the start point
		for (auto& curve : curves)
		if (curve.second.prev == nullptr || curve.second.next == nullptr)
			startCurve = curve.first;

		// find start curve
		if (startCurve)
		{
			while (curves[startCurve].prev)
				startCurve = curves[startCurve].prev;
		}
		else
		{
			startCurve = curves.begin()->first;
		}
		

		// check connectivity and reorder
		std::hash_set<AbstractGraphCurve*> visited;
		std::vector<AbstractGraphCurve*> curvesOrdered;
		auto c = startCurve;
		do
		{
			curvesOrdered.push_back(c);
			visited.insert(c);
			if (visited.find(curves[c].next) == visited.end())
				c = curves[c].next;
			else if (visited.find(curves[c].prev) == visited.end())
				c = curves[c].prev;
			else
				break;
		} while (c && c != startCurve);
		
		if (addLoop(curvesOrdered, isBoundingLoop))
			return true;
		return false;
	}

	bool Graph::mergeSelectedCurves()
	{
		std::hash_map<AbstractGraphCurve*, GraphDiskLink> curves;
		for (auto iter = m_curves.begin(); iter != m_curves.end(); ++iter)
		if (iter->second->isSelected())
			curves.insert(std::make_pair(iter->second.get(), GraphDiskLink()));

		if (curves.size() < 2)
			return false;

		// find a start curve
		AbstractGraphCurve* startCurve = nullptr;
		for (auto& curve : curves)
		{
			for (auto& other : curves)
			{
				if (other.first == curve.first)
					continue;

				// all the curves must be in the same loops
				if (curve.first->m_graphLinks.size() != other.first->m_graphLinks.size())
					return false;
				for (auto iter1 = curve.first->m_graphLinks.begin(), iter2 = other.first->m_graphLinks.begin();
					iter1 != curve.first->m_graphLinks.end() && iter2 != other.first->m_graphLinks.end(); 
					++iter1, ++iter2)
				{
					if (iter1->first != iter2->first)
						return false;
				}

				// all the curves must be the same sewings
				if (curve.first->m_sewings.size() != other.first->m_sewings.size())
					return false;
				for (auto iter1 = curve.first->m_sewings.begin(), iter2 = other.first->m_sewings.begin();
					iter1 != curve.first->m_sewings.end() && iter2 != other.first->m_sewings.end();
					++iter1, ++iter2)
				{
					if (*iter1 != *iter2)
						return false;
				}

				// find link
				if (curve.first->getEndPoint() == other.first->getStartPoint()
					|| curve.first->getEndPoint() == other.first->getEndPoint())
				{
					if (curve.second.next)
						return false; // we donot allow multi-connection
					curve.second.next = other.first;
				}
				if (curve.first->getStartPoint() == other.first->getStartPoint()
					|| curve.first->getStartPoint() == other.first->getEndPoint())
				{
					if (curve.second.prev)
						return false; // we donot allow multi-connection
					curve.second.prev = other.first;
				}
			} // end for other

			// we donot allow non-connected curves
			if (curve.second.prev == nullptr && curve.second.next == nullptr)
				return false;

			// set an end point as the start point
			if (curve.second.prev == nullptr || curve.second.next == nullptr)
				startCurve = curve.first;
		} // end for curve

		// we do not allow closed polygon
		if (startCurve == nullptr)
			return false;
		while (curves[startCurve].prev)
			startCurve = curves[startCurve].prev;

		// check connectivity and reorder
		std::hash_set<AbstractGraphCurve*> visited;
		std::vector<AbstractGraphCurve*> curvesOrdered;
		auto c = startCurve;
		do
		{
			curvesOrdered.push_back(c);
			visited.insert(c);
			if (visited.find(curves[c].next) == visited.end())
				c = curves[c].next;
			else if (visited.find(curves[c].prev) == visited.end())
				c = curves[c].prev;
			else
				break;
		} while (c && c != startCurve);

		// perform merging
		AbstractGraphCurve* mergedCurve = curvesOrdered[0];
		for (size_t i = 1; i < curvesOrdered.size(); i++)
		if (!mergeCurve(mergedCurve, curvesOrdered[i], mergedCurve))
			return false;

		return true;
	}

	bool Graph::splitTheSelectedCurve(Float2 position)
	{
		std::hash_set<AbstractGraphCurve*> curves;
		for (auto iter = m_curves.begin(); iter != m_curves.end(); ++iter)
		if (iter->second->isSelected())
			curves.insert(iter->second.get());

		if (curves.size() != 1)
			return false;

		AbstractGraphCurve* newCurve = nullptr, *oldCurve = *curves.begin();
		auto kpt = splitEdgeMakePoint(oldCurve, position, newCurve);
		return kpt != nullptr;
	}

	bool Graph::mergeSelectedKeyPoints()
	{
		std::vector<GraphPoint*> points;
		for (auto iter = point_begin(); iter != point_end(); ++iter)
		if (iter->isSelected())
			points.push_back(iter);
		bool changed = false;
		for (size_t i = 1; i < points.size(); i++)
		{
			if (mergeKeyPoints(points[0], points[i]))
				changed = true;
			else
				return false;
		}
		return changed;
	}

}