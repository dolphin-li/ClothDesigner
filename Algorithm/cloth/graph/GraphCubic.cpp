#include "GraphCubic.h"
#include "tinyxml\tinyxml.h"
#include "GraphPoint.h"
namespace ldp
{
	GraphCubic::GraphCubic() : AbstractGraphCurve()
	{
		m_keyPoints.resize(4, nullptr);
	}
	GraphCubic::GraphCubic(const std::vector<GraphPoint*>& pts) : AbstractGraphCurve(pts)
	{
		assert(pts.size() == 4);
	}

	Float2 GraphCubic::getPointByParam(float t)const
	{
		for (int k = 0; k < numKeyPoints(); k++)
		{
			assert(m_keyPoints[k]);
		}
		Float2 p1 = (1 - t) * ((1 - t)*m_keyPoints[0]->getPosition() + t*m_keyPoints[1]->getPosition())
			+ t * ((1 - t)*m_keyPoints[1]->getPosition() + t*m_keyPoints[2]->getPosition());
		Float2 p2 = (1 - t) * ((1 - t)*m_keyPoints[1]->getPosition() + t*m_keyPoints[2]->getPosition())
			+ t * ((1 - t)*m_keyPoints[2]->getPosition() + t*m_keyPoints[3]->getPosition());
		return (1 - t) * p1 + t * p2;
	}
}