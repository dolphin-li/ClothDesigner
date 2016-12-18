#pragma once

#include "AbstractGraphObject.h"

namespace ldp
{
	class AbstractGraphCurve;
	class GraphLoop : public AbstractGraphObject
	{
	public:
		GraphLoop();

		virtual GraphLoop* clone()const;
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);
		virtual Type getType()const { return TypeGraphLoop; }

		bool isClosed()const;
		bool isBoundingLoop()const { return m_isBoundingLoop; }
		void setBoundingLoop(bool b) { m_isBoundingLoop = b; }

		AbstractGraphCurve*& startEdge() { return m_startEdge; }
		const AbstractGraphCurve* const& startEdge()const { return m_startEdge; }
	private:
		AbstractGraphCurve* m_startEdge = nullptr;
		bool m_isBoundingLoop = false;
	};

	typedef std::shared_ptr<GraphLoop> GraphLoopPtr;
}