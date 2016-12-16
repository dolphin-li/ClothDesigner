#pragma once

#include "AbstractGraphCurve.h"
#include "ldpMat\ldp_basic_vec.h"

namespace ldp
{
	class GraphQuadratic : public AbstractGraphCurve
	{
	public:
		GraphQuadratic();
		GraphQuadratic(size_t id);
		GraphQuadratic(const std::vector<GraphPoint*>& pts, size_t id = 0);

		virtual Float2 getPointByParam(float t)const;
		virtual Type getType()const { return TypeGraphQuadratic; }
	private:
		
	};
}