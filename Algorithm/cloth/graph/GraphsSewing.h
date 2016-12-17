#pragma once
#include "AbstractGraphObject.h"
#include <set>
class TiXmlElement;
class TiXmlNode;
namespace ldp
{
	class AbstractGraphCurve;
	class GraphsSewing : public AbstractGraphObject
	{
	public:
		struct Unit
		{
			AbstractGraphCurve* curve;
			bool reverse;
			Unit() :curve(0), reverse(false) {}
			Unit(AbstractGraphCurve* c, bool d) :curve(c), reverse(d) {}
		};
	public:
		GraphsSewing();

		virtual GraphsSewing* clone()const;
		virtual Type getType()const { return TypeGraphsSewing; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		void clear();
		bool empty()const { return m_firsts.size() == 0 || m_seconds.size() == 0; }
		const std::vector<Unit>& firsts()const { return m_firsts; }
		const std::vector<Unit>& seconds()const { return m_seconds; }
		std::vector<Unit>& firsts() { return m_firsts; }
		std::vector<Unit>& seconds() { return m_seconds; }
		void addFirst(Unit unit);
		void addSecond(Unit unit);
		void addFirsts(const std::vector<Unit>& unit);
		void addSeconds(const std::vector<Unit>& unit);
		void remove(size_t id);
		void remove(const std::set<size_t>& s);
		bool select(int idx, SelectOp op);
		bool select(const std::set<int>& indices, SelectOp op);
		void highLight(int idx, int lastIdx);
	protected:
		std::vector<Unit> m_firsts;
		std::vector<Unit> m_seconds;
	};
	typedef std::shared_ptr<GraphsSewing> GraphsSewingPtr;
}