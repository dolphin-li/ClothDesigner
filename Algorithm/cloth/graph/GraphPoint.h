#pragma once

#include "AbstractGraphObject.h"
#include "ldpMat\ldp_basic_vec.h"

namespace ldp
{
	class AbstractGraphCurve;
	class GraphPoint : public AbstractGraphObject
	{
	public:
		GraphPoint();
		GraphPoint(Float2 p);

		virtual GraphPoint* clone()const;
		virtual Type getType()const { return TypeGraphPoint; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		Float2& position() { return m_p; }
		const Float2& position()const { return m_p; }

		std::hash_set<AbstractGraphCurve*>& edges() { return m_edges; }
		const std::hash_set<AbstractGraphCurve*>& edges()const { return m_edges; }
	private:
		Float2 m_p;
		std::hash_set<AbstractGraphCurve*> m_edges;
	};

	extern GraphPoint g_graphPoint;
	typedef std::shared_ptr<GraphPoint> GraphPointPtr;
}