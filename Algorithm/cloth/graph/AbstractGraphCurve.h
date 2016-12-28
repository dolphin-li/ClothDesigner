#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_mat.h"
#include <set>
namespace ldp
{
	class GraphsSewing;
	class AbstractGraphCurve;
	class GraphLoop;
	struct GraphDiskLink
	{
		AbstractGraphCurve* prev = nullptr;
		AbstractGraphCurve* next = nullptr;
		GraphLoop* loop = nullptr;
	};
	class AbstractGraphCurve : public AbstractGraphObject
	{
		friend class Graph;
		friend class GraphLoop;
		friend class GraphPoint;
	public:
		class DiskLinkIter
		{
			AbstractGraphCurve* m_curve = nullptr;
			std::hash_map<GraphLoop*, GraphDiskLink>::iterator m_lkIter;
		public:
			DiskLinkIter(AbstractGraphCurve* s) :m_curve(s), m_lkIter(s->m_graphLinks.begin()) {}
			DiskLinkIter& operator++() { ++m_lkIter; return*this; }
			DiskLinkIter operator++()const { DiskLinkIter iter(*this); return ++iter; }
			bool isEnd()const { return m_curve->m_graphLinks.end() == m_lkIter; }
			const GraphLoop*const& loop()const { return m_lkIter->second.loop; }
			AbstractGraphCurve*& loopStartEdge();
			AbstractGraphCurve* curve() { return m_curve; }
			const AbstractGraphCurve* curve()const { return m_curve; }
			const AbstractGraphCurve*const& loopStartEdge()const;
			AbstractGraphCurve*& prev() { return m_lkIter->second.prev; }
			const AbstractGraphCurve*const& prev()const { return m_lkIter->second.prev; }
			AbstractGraphCurve*& next() { return m_lkIter->second.next; }
			const AbstractGraphCurve*const& next()const { return m_lkIter->second.next; }
		};
	public:
		AbstractGraphCurve();
		AbstractGraphCurve(const std::vector<GraphPoint*>& pts);
		virtual ~AbstractGraphCurve();

		virtual AbstractGraphCurve* clone()const;
		virtual Type getType()const = 0;
		virtual Float2 getPointByParam(float t)const = 0; // t \in [0, 1]
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);
		virtual const std::vector<Float2>& samplePointsOnShape(float step)const;

		// call this if you manually change the position of key points, without calling the interface functions
		void requireResample() { m_invalid = true; m_lengthInvalid = true; }

		static AbstractGraphCurve* create(const std::vector<GraphPoint*>& kpts);
		static void fittingCurves(std::vector<std::vector<std::shared_ptr<GraphPoint>>>& curves,
			const std::vector<Float2>& keyPoints, float fittingThre);
		static void fittingOneCurve(std::vector<std::shared_ptr<GraphPoint>>& curves,
			const std::vector<Float2>& keyPoints, float fittingThre);

		bool isEndPointsSame(const AbstractGraphCurve* r)const
		{
			return getStartPoint() == r->getStartPoint() && getEndPoint() == r->getEndPoint();
		}
		bool isEndPointsReversed(const AbstractGraphCurve* r)const
		{
			return getStartPoint() == r->getEndPoint() && getEndPoint() == r->getStartPoint();
		}
		const GraphPoint* getStartPoint()const
		{
			return m_keyPoints[0];
		}
		const GraphPoint* getEndPoint()const
		{
			return m_keyPoints.back();
		}
		int numKeyPoints()const
		{
			return (int)m_keyPoints.size();
		}
		const GraphPoint* const& keyPoint(int i)const
		{
			return m_keyPoints[i];
		}	
		GraphPoint*& keyPoint(int i)
		{
			m_invalid = true;
			return m_keyPoints[i];
		}
		void translateKeyPoint(int i, ldp::Float2 t);
		void translate(Float2 t);
		void rotate(const Mat2f& R);
		void rotateBy(const Mat2f& R, Float2 c);
		void scale(Float2 s);
		void scaleBy(Float2 s, Float2 c);
		void transform(const Mat3f& M);
		void unionBound(Float2& bmin, Float2& bmax);
		AbstractGraphCurve& reverse()
		{
			std::reverse(m_keyPoints.begin(), m_keyPoints.end());
			m_invalid = true;
			return *this;
		}
		float getLength()const
		{
			if (m_lengthInvalid)
				m_length = calcLength();
			m_lengthInvalid = false;
			return m_length;
		}

		// relate to sewings
		std::hash_set<GraphsSewing*>& graphSewings() { return m_sewings; }
		const std::hash_set<GraphsSewing*>& graphSewings()const { return m_sewings; }

		// winged edge related
		DiskLinkIter diskLink_begin() { return DiskLinkIter((AbstractGraphCurve*)this); }
		const DiskLinkIter diskLink_begin()const { return DiskLinkIter((AbstractGraphCurve*)this); }
	protected:
		virtual float calcLength()const;
	protected:
		std::vector<GraphPoint*> m_keyPoints;

		// for winged-edge data structure
		std::hash_map<GraphLoop*, GraphDiskLink> m_graphLinks;

		// relate to sewings
		std::hash_set<GraphsSewing*> m_sewings;
	private:
		mutable std::vector<Float2> m_samplePoints;
		mutable bool m_invalid = true;
		mutable bool m_lengthInvalid = true;
		mutable float m_lastSampleStep = 0;
		mutable float m_length = 0;
	};
	typedef std::shared_ptr<AbstractGraphCurve> AbstractGraphCurvePtr;
}