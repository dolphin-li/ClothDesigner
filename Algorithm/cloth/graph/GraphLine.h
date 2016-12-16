#pragma once

#include "AbstractGraphCurve.h"
#include "ldpMat\ldp_basic_vec.h"

namespace ldp
{
	class GraphLine : public AbstractGraphCurve
	{
	public:
		GraphLine();
		GraphLine(size_t id);
		GraphLine(const std::vector<GraphPoint*>& pts, size_t id = 0);

		virtual Float2 getPointByParam(float t)const;
		virtual Type getType()const { return TypeGraphLine; }
	protected:
		virtual float calcLength()const;
	private:
		
	};
}