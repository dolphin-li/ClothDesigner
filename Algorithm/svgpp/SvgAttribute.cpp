#include "SvgAttribute.h"

namespace svg
{
	SvgAttribute::SvgAttribute()
	{
		m_color = ldp::Float4(0, 0, 0, 1);
		m_transfrom.eye();
	}

	SvgAttribute::~SvgAttribute()
	{
	}
}