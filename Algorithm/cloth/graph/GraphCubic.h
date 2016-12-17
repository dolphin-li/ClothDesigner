#pragma once

#include "AbstractGraphCurve.h"
#include "ldpMat\ldp_basic_vec.h"

namespace ldp
{
	class GraphCubic : public AbstractGraphCurve
	{
	public:
		GraphCubic();
		GraphCubic(const std::vector<GraphPoint*>& pts);

		virtual Float2 getPointByParam(float t)const;
		virtual Type getType()const { return TypeGraphCubic; }
	private:
		
	};
}