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
		Float2 p1 = (1 - t) * ((1 - t)*m_keyPoints[0]->position() + t*m_keyPoints[1]->position())
			+ t * ((1 - t)*m_keyPoints[1]->position() + t*m_keyPoints[2]->position());
		Float2 p2 = (1 - t) * ((1 - t)*m_keyPoints[1]->position() + t*m_keyPoints[2]->position())
			+ t * ((1 - t)*m_keyPoints[2]->position() + t*m_keyPoints[3]->position());
		return (1 - t) * p1 + t * p2;
	}
}