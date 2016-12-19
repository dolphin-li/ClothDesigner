#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_vec.h"

namespace ldp
{
	class AbstractGraphCurve;
	class GraphPoint : public AbstractGraphObject
	{
		friend class Graph;
		friend class GraphLoop;
		friend class AbstractGraphCurve;
	public:
		GraphPoint();
		GraphPoint(Float2 p);

		virtual GraphPoint* clone()const;
		virtual Type getType()const { return TypeGraphPoint; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		Float2& position() { return m_p; }
		const Float2& position()const { return m_p; }
	private:
		Float2 m_p;
		AbstractGraphCurve* m_edge = nullptr;
	};

	extern GraphPoint g_graphPoint;
	typedef std::shared_ptr<GraphPoint> GraphPointPtr;
}