#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_vec.h"
#include <set>
namespace ldp
{
	class GraphPoint;
	class AbstractGraphCurve;
	class GraphLoop;
	class PanelPolygon;
	class AbstractPanelObject;
	class GraphsSewing;
	class Graph : public AbstractGraphObject
	{
	public:
		typedef std::hash_map<size_t, std::shared_ptr<GraphPoint>> PointMap;
		typedef std::hash_map<size_t, std::shared_ptr<AbstractGraphCurve>> CurveMap;
		typedef std::hash_map<size_t, std::shared_ptr<GraphLoop>> LoopMap;
		typedef std::hash_map<AbstractGraphObject*, AbstractGraphObject*> PtrMap;
		class PointIter
		{
			PointMap::const_iterator m_iter;
		public:
			PointIter(PointMap::const_iterator iter) : m_iter(iter) {}
			PointIter& operator++() { ++m_iter; return *this; }
			PointIter operator++()const { PointIter it(*this); return ++it; }
			GraphPoint* operator->() { return m_iter->second.get(); }
			GraphPoint& operator*() { return *m_iter->second.get(); }
			bool operator == (const PointIter& r)const { return m_iter == r.m_iter; }
			bool operator != (const PointIter& r)const { return !((*this) == r); }
			operator GraphPoint*() { return m_iter->second.get(); }
		};
		class CurveIter
		{
			CurveMap::const_iterator m_iter;
		public:
			CurveIter(CurveMap::const_iterator iter) : m_iter(iter) {}
			CurveIter& operator++() { ++m_iter; return *this; }
			CurveIter operator++()const { CurveIter it(*this); return ++it; }
			AbstractGraphCurve* operator->() { return m_iter->second.get(); }
			AbstractGraphCurve& operator*() { return *m_iter->second.get(); }
			bool operator == (const CurveIter& r)const { return m_iter == r.m_iter; }
			bool operator != (const CurveIter& r)const { return !((*this) == r); }
			operator AbstractGraphCurve*() { return m_iter->second.get(); }
		};
		class LoopIter
		{
			LoopMap::const_iterator m_iter;
		public:
			LoopIter(LoopMap::const_iterator iter) : m_iter(iter) {}
			LoopIter& operator++() { ++m_iter; return *this; }
			LoopIter operator++()const { LoopIter it(*this); return ++it; }
			GraphLoop* operator->() { return m_iter->second.get(); }
			GraphLoop& operator*() { return *m_iter->second.get(); }
			bool operator == (const LoopIter& r)const { return m_iter == r.m_iter; }
			bool operator != (const LoopIter& r)const { return !((*this) == r); }
			operator GraphLoop*() { return m_iter->second.get(); }
		};
	public:
		Graph();

		virtual Type getType()const { return TypeGraph; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);
		virtual Graph* clone()const;
		const PtrMap& getPtrMapAfterClone() { return m_ptrMapAfterClone; } // for mapping update after clone

		void clear();

		// make the graph valid, that is:
		// 1. there must be exactly one bounding loop, all other loops must be inside the bounding loop
		// 2. if there is intersecting curves, split them and made the loops proper
		// 3. all inner loops must not be intersected
		// return false if the opeartion failed.
		bool makeGraphValid(std::vector<std::shared_ptr<GraphsSewing>>& graphSewings);

		// ui operations
		bool select(int idx, SelectOp op);
		bool select(const std::set<int>& indices, SelectOp op);
		void highLight(int idx, int lastIdx);
		void updateBound(Float2& bmin = Float2(), Float2& bmax = Float2());
		const Float2* bound()const { return m_bbox; }

		// getters
		int numKeyPoints()const { return (int)m_keyPoints.size(); }
		int numCurves()const { return (int)m_curves.size(); }
		int numLoops()const { return (int)m_loops.size(); }
		PointIter point_begin()const { return PointIter(m_keyPoints.begin()); }
		PointIter point_end()const { return PointIter(m_keyPoints.end()); }
		CurveIter curve_begin()const { return CurveIter(m_curves.begin()); }
		CurveIter curve_end()const { return CurveIter(m_curves.end()); }
		LoopIter loop_begin()const { return LoopIter(m_loops.begin()); }
		LoopIter loop_end()const { return LoopIter(m_loops.end()); }
		GraphPoint* getPointById(size_t id)
		{
			auto iter = m_keyPoints.find(id);
			if (iter == m_keyPoints.end())
				return nullptr;
			return iter->second.get();
		}
		const GraphPoint* getPointById(size_t id)const
		{
			auto iter = m_keyPoints.find(id);
			if (iter == m_keyPoints.end())
				return nullptr;
			return iter->second.get();
		}
		AbstractGraphCurve* getCurveById(size_t id)
		{
			auto iter = m_curves.find(id);
			if (iter == m_curves.end())
				return nullptr;
			return iter->second.get();
		}
		const AbstractGraphCurve* getCurveById(size_t id)const
		{
			auto iter = m_curves.find(id);
			if (iter == m_curves.end())
				return nullptr;
			return iter->second.get();
		}
		GraphLoop* getLoopById(size_t id)
		{
			auto iter = m_loops.find(id);
			if (iter == m_loops.end())
				return nullptr;
			return iter->second.get();
		}
		const GraphLoop* getLoopById(size_t id)const
		{
			auto iter = m_loops.find(id);
			if (iter == m_loops.end())
				return nullptr;
			return iter->second.get();
		}
		GraphLoop* getBoundingLoop();
		const GraphLoop* getBoundingLoop()const;

		// topology operations: add units
		GraphPoint* addKeyPoint(ldp::Float2 p, bool isEndPoint); // is end point means the front or back of the curve
		GraphPoint* addKeyPoint(const std::shared_ptr<GraphPoint>& kp, bool isEndPoint);
		AbstractGraphCurve* addCurve(const std::vector<std::shared_ptr<GraphPoint>>& kpts);
		AbstractGraphCurve* addCurve(const std::vector<GraphPoint*>& kpts);
		AbstractGraphCurve* addCurve(const std::shared_ptr<AbstractGraphCurve>& curve);
		// a graph has and must has one bounding loop, that is, all other loops must be inside it.
		GraphLoop* addLoop(const std::vector<std::vector<std::shared_ptr<GraphPoint>>>& curves, bool isBoundingLoop);
		GraphLoop* addLoop(const std::vector<std::shared_ptr<AbstractGraphCurve>>& curves, bool isBoundingLoop);
		GraphLoop* addLoop(std::vector<AbstractGraphCurve*>& curves, bool isBoundingLoop);
		GraphLoop* addLoop(const std::shared_ptr<GraphLoop>& loop);

		// topology operations: remove units, return false if not removed
		bool remove(size_t id);
		bool removeKeyPoints(const GraphPoint* kp);
		bool removeCurve(const AbstractGraphCurve* curve);
		bool removeLoop(const GraphLoop* loop, bool removeCurvesPoints = false);

		// split a curve into two, return the inserted point and the newCurve generated
		bool splitEdge(AbstractGraphCurve* curveToSplit, 
			Float2 splitPosition, AbstractGraphCurve* newCurves[2] = nullptr);

		// two curves can be merged into one iff they share a common point
		bool mergeCurve(AbstractGraphCurve* curve1, AbstractGraphCurve* curve2,
			AbstractGraphCurve*& mergedCurve);

		// two points can be merged into one iff
		bool mergeKeyPoints(GraphPoint* p1, GraphPoint* p2);

		// split the curve by the point and merge the point to it
		bool mergeCurvePoint(AbstractGraphCurve* curveToSplit, GraphPoint* p);

		// ui related
		bool selectedCurvesToLoop(bool isBoundingLoop=false);
		bool mergeSelectedCurves();
		bool splitTheSelectedCurve(Float2 position);
		bool mergeSelectedKeyPoints();
		bool mergeTheSelectedKeyPointToCurve();
	protected:
		void connectNextCurve(AbstractGraphCurve* curr, AbstractGraphCurve* next, GraphLoop* loop, bool reverse);
		void connectPrevCurve(AbstractGraphCurve* curr, AbstractGraphCurve* prev, GraphLoop* loop, bool reverse);
	private:
		ldp::Float2 m_bbox[2];
		PointMap m_keyPoints;
		CurveMap m_curves;
		LoopMap m_loops;
		mutable PtrMap m_ptrMapAfterClone;
	};
}