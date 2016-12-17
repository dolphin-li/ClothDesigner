#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_mat.h"
namespace ldp
{
	class GraphPoint;
	class GraphLoop;
	class AbstractGraphCurve : public AbstractGraphObject
	{
	public:
		AbstractGraphCurve();
		AbstractGraphCurve(const std::vector<GraphPoint*>& pts);

		virtual AbstractGraphCurve* clone()const;
		virtual Type getType()const = 0;
		virtual Float2 getPointByParam(float t)const = 0; // t \in [0, 1]
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);
		virtual const std::vector<Float2>& samplePointsOnShape(float step)const;

		static AbstractGraphCurve* create(const std::vector<GraphPoint*>& kpts);
		static void fittingCurves(std::vector<std::vector<std::shared_ptr<GraphPoint>>>& curves,
			const std::vector<Float2>& keyPoints, float fittingThre);
		static void fittingOneCurve(std::vector<std::shared_ptr<GraphPoint>>& curves,
			const std::vector<Float2>& keyPoints, float fittingThre);

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

		// for half-edge data structure
		AbstractGraphCurve*& nextEdge() { return m_next; }
		const AbstractGraphCurve* const& nextEdge()const { return m_next; }
		GraphLoop*& loop() { return m_loop; }
		const GraphLoop* const& loop()const { return m_loop; }
	protected:
		virtual float calcLength()const;
	protected:
		std::vector<GraphPoint*> m_keyPoints;

		// for half-edge data structure
		AbstractGraphCurve* m_next = nullptr;
		GraphLoop* m_loop = nullptr;
	private:
		mutable std::vector<Float2> m_samplePoints;
		mutable bool m_invalid = true;
		mutable bool m_lengthInvalid = true;
		mutable float m_lastSampleStep = 0;
		mutable float m_length = 0;
	};
	typedef std::shared_ptr<AbstractGraphCurve> AbstractGraphCurvePtr;
}