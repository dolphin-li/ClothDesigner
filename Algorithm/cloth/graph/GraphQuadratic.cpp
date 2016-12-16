#include "GraphQuadratic.h"
#include "tinyxml\tinyxml.h"
#include "GraphPoint.h"
namespace ldp
{
	GraphQuadratic::GraphQuadratic() : AbstractGraphCurve()
	{
		m_keyPoints.resize(3, nullptr);
	}
	GraphQuadratic::GraphQuadratic(size_t id) : AbstractGraphCurve(id)
	{
		m_keyPoints.resize(3, nullptr);
	}
	GraphQuadratic::GraphQuadratic(const std::vector<GraphPoint*>& pts, size_t id) : AbstractGraphCurve(pts, id)
	{
		assert(pts.size() == 3);
	}

	Float2 GraphQuadratic::getPointByParam(float t)const
	{
		for (int k = 0; k < numKeyPoints(); k++)
		{
			assert(m_keyPoints[k]);
		}
		return (1 - t) * ((1 - t)*m_keyPoints[0]->position() + t*m_keyPoints[1]->position())
			+ t * ((1 - t)*m_keyPoints[1]->position() + t*m_keyPoints[2]->position());
	}
}