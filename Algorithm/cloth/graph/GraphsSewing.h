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
		friend class HistoryStack;
		friend class ClothManager;
		struct Unit
		{
			AbstractGraphCurve* curve;
			bool reverse;
			Unit() :curve(0), reverse(false) {}
			Unit(AbstractGraphCurve* c, bool d) :curve(c), reverse(d) {}
		};
		enum SewingType
		{
			SewingTypeStitch = 0,	// two edges a, b are stitched means that they are topological one piece
			SewingTypePosition,		// the position of the two edges should be togither, but not topological 
			SewingTypeEnd			// end flag, not valid
		};
	public:
		GraphsSewing();
		virtual ~GraphsSewing();

		virtual Type getType()const { return TypeGraphsSewing; }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		void clear();
		bool empty()const { return m_firsts.size() == 0 || m_seconds.size() == 0; }
		SewingType getSewingType()const{ return m_sewingType; }
		const char* getSewingTypeStr()const; 
		void setSewingType(SewingType s){ m_sewingType = s; }
		const std::vector<Unit>& firsts()const { return m_firsts; }
		const std::vector<Unit>& seconds()const { return m_seconds; }
		bool firstsContains(AbstractGraphCurve* curve)const
		{
			for (auto &u : m_firsts)
			if (u.curve == curve)
				return true;
			return false;
		}
		bool secondsContains(AbstractGraphCurve* curve)const
		{
			for (auto &u : m_seconds)
			if (u.curve == curve)
				return true;
			return false;
		}
		bool contains(AbstractGraphCurve* curve)const
		{
			return firstsContains(curve) || secondsContains(curve);
		}
		void addFirst(Unit unit);
		void addSecond(Unit unit);
		void addFirsts(const std::vector<Unit>& unit);
		void addSeconds(const std::vector<Unit>& unit);
		void remove(size_t curveId);
		void remove(const std::set<size_t>& s);
		void reverse(size_t curveId);
		void reverseFirsts();
		void reverseSeconds();
		void swapCurve(AbstractGraphCurve* oldCurve, AbstractGraphCurve* newCurve);
		void swapCurve(AbstractGraphCurve* oldCurve, const std::vector<AbstractGraphCurve*>& newCurves);
		void swapUnit(Unit ou, Unit u);
		void swapUnit(Unit ou, const std::vector<Unit>& us);
		bool select(int idx, SelectOp op);
		bool select(const std::set<int>& indices, SelectOp op);
		void highLight(int idx, int lastIdx);
		void setAngleInDegree(float d){ m_angleInDegree = d; }
		float getAngleInDegree()const{ return m_angleInDegree; }
	protected:
		void add(std::vector<Unit>& units, Unit unit)const;
		void remove(std::vector<Unit>& units, size_t curveId)const;
		void swapUnit(std::vector<Unit>& units, Unit ou, Unit u);
		void swapUnit(std::vector<Unit>& units, Unit ou, const std::vector<Unit>& u);
		void swapCurve(std::vector<Unit>& units, AbstractGraphCurve* oldCurve, AbstractGraphCurve* newCurve);
		void swapCurve(std::vector<Unit>& units, AbstractGraphCurve* oldCurve, 
			const std::vector<AbstractGraphCurve*>& newCurve);
	private:
		// take care of this function: it is not self-completed,
		// it only clones the pointer, but not the object
		virtual GraphsSewing* clone()const;
	protected:
		std::vector<Unit> m_firsts;
		std::vector<Unit> m_seconds;
		float m_angleInDegree = 0.f;
		SewingType m_sewingType = SewingTypeStitch;
	};
	typedef std::shared_ptr<GraphsSewing> GraphsSewingPtr;
}