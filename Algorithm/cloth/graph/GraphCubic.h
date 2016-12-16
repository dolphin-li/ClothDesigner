#pragma once

#include "AbstractGraphCurve.h"
#include "ldpMat\ldp_basic_vec.h"

namespace ldp
{
	class GraphCubic : public AbstractGraphCurve
	{
	public:
		GraphCubic();
		GraphCubic(size_t id);
		GraphCubic(const std::vector<GraphPoint*>& pts, size_t id=0);

		virtual Float2 getPointByParam(float t)const;
		virtual Type getType()const { return TypeGraphCubic; }
	private:
		
	};
}