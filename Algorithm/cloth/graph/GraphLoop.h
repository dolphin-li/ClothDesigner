#pragma once

#include "AbstractGraphObject.h"

namespace ldp
{
	class AbstractGraphCurve;
	class GraphLoop : public AbstractGraphObject
	{
	public:
		GraphLoop();
		GraphLoop(size_t id);

		virtual GraphLoop* clone()const;
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);
		virtual Type getType()const { return TypeGraphLoop; }

		AbstractGraphCurve*& startEdge() { return m_startEdge; }
		const AbstractGraphCurve* const& startEdge()const { return m_startEdge; }
	private:
		AbstractGraphCurve* m_startEdge = nullptr;
	};

	typedef std::shared_ptr<GraphLoop> GraphLoopPtr;
}