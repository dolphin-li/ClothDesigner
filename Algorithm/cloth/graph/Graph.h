#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_vec.h"
namespace ldp
{
	class GraphPoint;
	class AbstractGraphCurve;
	class GraphLoop;
	class PanelPolygon;
	class Graph : public AbstractGraphObject
	{
	public:
		typedef std::hash_map<size_t, std::shared_ptr<GraphPoint>> PointMap;
		typedef std::hash_map<size_t, std::shared_ptr<AbstractGraphCurve>> CurveMap;
		typedef std::hash_map<size_t, std::shared_ptr<GraphLoop>> LoopMap;
		typedef std::hash_map<AbstractGraphObject*, AbstractGraphObject*> PtrMap;
	public:
		Graph();

		virtual Type getType()const { return TypeGraph; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);
		virtual Graph* clone()const;
		const PtrMap& getPtrMapAfterClone() { return m_ptrMapAfterClone; } // for mapping update after clone

		void clear();

		void fromPanelPolygon(const PanelPolygon& panel);

		// getters
		int numKeyPoints()const { return (int)m_keyPoints.size(); }
		int numCurves()const { return (int)m_curves.size(); }
		int numLoops()const { return (int)m_loops.size(); }
		PointMap::const_iterator pointBegin()const { return m_keyPoints.begin(); }
		PointMap::const_iterator pointEnd()const { return m_keyPoints.end(); }
		CurveMap::const_iterator curveBegin()const { return m_curves.begin(); }
		CurveMap::const_iterator curveEnd()const { return m_curves.end(); }
		LoopMap::const_iterator loopBegin()const { return m_loops.begin(); }
		LoopMap::const_iterator loopEnd()const { return m_loops.end(); }
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

		// topology operations: add units
		GraphPoint* addKeyPoint(ldp::Float2 p);
		GraphPoint* addKeyPoint(const std::shared_ptr<GraphPoint>& kp);
		AbstractGraphCurve* addCurve(const std::vector<GraphPoint*>& kpts);
		AbstractGraphCurve* addCurve(const std::shared_ptr<AbstractGraphCurve>& curve);
		GraphLoop* addLoop(const std::vector<AbstractGraphCurve*>& curves);
		GraphLoop* addLoop(const std::shared_ptr<GraphLoop>& loop);

		// topology operations: remove units, return false if not removed
		bool remove(size_t id);
		bool removeKeyPoints(const GraphPoint* kp);
		bool removeCurve(const AbstractGraphCurve* curve);
		bool removeLoop(const GraphLoop* loop);

		// split a curve into two, return the inserted point and the newCurve generated
		GraphPoint* splitEdgeMakePoint(AbstractGraphCurve* curveToSplit, 
			float splitPosition, AbstractGraphCurve*& newCurve);

		// two curves can be merged into one iff they share a common point
		bool mergeCurve(AbstractGraphCurve* curve1, AbstractGraphCurve* curve2);
	private:
		PointMap m_keyPoints;
		CurveMap m_curves;
		LoopMap m_loops;
		mutable PtrMap m_ptrMapAfterClone;
	};
}