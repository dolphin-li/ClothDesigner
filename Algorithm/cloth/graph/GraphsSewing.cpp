#include "GraphsSewing.h"
#include "AbstractGraphCurve.h"
#include "tinyxml\tinyxml.h"
namespace ldp
{
	GraphsSewing::GraphsSewing() : AbstractGraphObject()
	{
	
	}

	GraphsSewing::~GraphsSewing()
	{
		clear();
	}

	void GraphsSewing::clear()
	{
		auto tmp = m_firsts;
		tmp.insert(tmp.end(), m_seconds.begin(), m_seconds.end());
		for (auto t : tmp)
			remove(t.curve->getId());
		m_firsts.clear();
		m_seconds.clear();
		m_sewingType = SewingTypeStitch;
	}

	void GraphsSewing::addFirst(Unit unit)
	{
		add(m_firsts, unit);
	}

	void GraphsSewing::addSecond(Unit unit)
	{
		add(m_seconds, unit);
	}

	void GraphsSewing::add(std::vector<Unit>& units, Unit unit)const
	{
		auto iter_same = units.end();
		for (auto iter = units.begin(); iter != units.end(); ++iter)
		{
			if (iter->curve == unit.curve)
			{
				iter_same = iter;
				break;
			}
		}
		if (iter_same == units.end())
		{
			units.push_back(unit);
			unit.curve->graphSewings().insert((GraphsSewing*)this);
		}
		else
			iter_same->reverse = unit.reverse;
	}

	void GraphsSewing::addFirsts(const std::vector<Unit>& unit)
	{
		for (const auto& u : unit)
			addFirst(u);
	}

	void GraphsSewing::addSeconds(const std::vector<Unit>& unit)
	{
		for (const auto& u : unit)
			addSecond(u);
	}

	void GraphsSewing::remove(size_t id)
	{
		remove(m_firsts, id);
		remove(m_seconds, id);
	}

	void GraphsSewing::remove(const std::set<size_t>& s)
	{
		for (auto id : s)
			remove(id);
	}

	void GraphsSewing::remove(std::vector<Unit>& units, size_t curveId)const
	{
		for (auto iter = units.begin(); iter != units.end(); ++iter)
		{
			if (iter->curve->getId() != curveId)
				continue;
			auto siter = iter->curve->graphSewings().find((GraphsSewing*)this);
			if (siter == iter->curve->graphSewings().end())
			{
				printf("GraphsSewing remove warning: curve %d not relate to sew %d",
					iter->curve->getId(), getId());
			}
			else
				iter->curve->graphSewings().erase(siter);
			units.erase(iter);
			break;
		}
	}

	void GraphsSewing::swapCurve(AbstractGraphCurve* oldCurve, AbstractGraphCurve* newCurve)
	{
		swapCurve(m_firsts, oldCurve, newCurve);
		swapCurve(m_seconds, oldCurve, newCurve);
	}

	void GraphsSewing::swapCurve(AbstractGraphCurve* oldCurve, const std::vector<AbstractGraphCurve*>& newCurves)
	{
		swapCurve(m_firsts, oldCurve, newCurves);
		swapCurve(m_seconds, oldCurve, newCurves);
	}

	void GraphsSewing::swapUnit(Unit ou, Unit nu)
	{
		swapUnit(m_firsts, ou, nu);
		swapUnit(m_seconds, ou, nu);
	}

	void GraphsSewing::swapUnit(Unit ou, const std::vector<Unit>& nus)
	{
		swapUnit(m_firsts, ou, nus);
		swapUnit(m_seconds, ou, nus);
	}

	void GraphsSewing::swapUnit(std::vector<Unit>& units, Unit ou, Unit nu)
	{
		for (auto& u : units)
		{
			if (u.curve == ou.curve)
			{
				if (u.curve->graphSewings().find(this) == u.curve->graphSewings().end())
					printf("GraphsSewing::swapUnit warning: curve %d does not relate to sew %d\n",
					u.curve->getId(), getId());
				else
					u.curve->graphSewings().erase(this);
				u = nu;
				u.curve->graphSewings().insert(this);
			}
		} // end for u
	}

	void GraphsSewing::swapUnit(std::vector<Unit>& units, Unit ou, const std::vector<Unit>& nus)
	{
		std::vector<Unit> tmpUnits;
		for (const auto& u : units)
		{
			if (u.curve == ou.curve)
			{
				if (u.curve->graphSewings().find(this) == u.curve->graphSewings().end())
					printf("GraphsSewing::swapUnit warning: curve %d does not relate to sew %d\n",
					u.curve->getId(), getId());
				else
					u.curve->graphSewings().erase(this);
				for (auto nu : nus)
				{
					nu.curve->graphSewings().insert(this);
					tmpUnits.push_back(nu);
				} // end for nu
			}
			else
				tmpUnits.push_back(u);
		} // end for u
		units = tmpUnits;
	}

	void GraphsSewing::swapCurve(std::vector<Unit>& units, AbstractGraphCurve* oldCurve, AbstractGraphCurve* newCurve)
	{
		Unit ou, nu;
		for (auto& u : units)
		{
			if (u.curve == oldCurve)
			{
				ou = u;
				break;
			}
		}

		nu.curve = newCurve;
		nu.reverse = ou.reverse;

		if (ou.curve)
			swapUnit(units, ou, nu);
	}

	void GraphsSewing::swapCurve(std::vector<Unit>& units, AbstractGraphCurve* oldCurve, 
		const std::vector<AbstractGraphCurve*>& newCurves)
	{
		Unit ou;
		for (auto& u : units)
		{
			if (u.curve == oldCurve)
			{
				ou = u;
				break;
			}
		}

		std::vector<Unit> nus;
		for (auto c : newCurves)
			nus.push_back(Unit(c, ou.reverse));

		if (ou.curve)
		{
			if (ou.reverse)
				std::reverse(nus.begin(), nus.end());
			swapUnit(units, ou, nus);
		}
	}


	void GraphsSewing::reverse(size_t curveId)
	{
		for (auto& u : m_firsts)
		if (u.curve->getId() == curveId)
		{
			u.reverse = !u.reverse;
			return;
		}
		for (auto& u : m_seconds)
		if (u.curve->getId() == curveId)
		{
			u.reverse = !u.reverse;
			return;
		}
	}

	void GraphsSewing::reverseFirsts()
	{
		for (auto& u : m_firsts)
			u.reverse = !u.reverse;
		std::reverse(m_firsts.begin(), m_firsts.end());
	}

	void GraphsSewing::reverseSeconds()
	{
		for (auto& u : m_seconds)
			u.reverse = !u.reverse;
		std::reverse(m_seconds.begin(), m_seconds.end());
	}

	bool GraphsSewing::select(int idx, SelectOp op)
	{
		if (op == SelectEnd)
			return false;
		std::set<int> idxSet;
		idxSet.insert(idx);
		return select(idxSet, op);
	}

	bool GraphsSewing::select(const std::set<int>& idxSet, SelectOp op)
	{
		if (op == SelectEnd)
			return false;
		static std::vector<AbstractGraphObject*> objs;
		objs.clear();
		objs.push_back(this);
		bool changed = false;
		for (auto& obj : objs)
		{
			bool oldSel = obj->isSelected();
			switch (op)
			{
			case AbstractGraphObject::SelectThis:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(true);
				else
					obj->setSelected(false);
				break;
			case AbstractGraphObject::SelectUnion:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(true);
				break;
			case AbstractGraphObject::SelectUnionInverse:
				if (idxSet.find(obj->getId()) != idxSet.end())
					obj->setSelected(!obj->isSelected());
				break;
			case AbstractGraphObject::SelectAll:
				obj->setSelected(true);
				break;
			case AbstractGraphObject::SelectNone:
				obj->setSelected(false);
				break;
			case AbstractGraphObject::SelectInverse:
				obj->setSelected(!obj->isSelected());
				break;
			default:
				break;
			}
			bool newSel = obj->isSelected();
			if (oldSel != newSel)
				changed = true;
		} // end for obj
		return changed;
	}

	void GraphsSewing::highLight(int idx, int lastIdx)
	{
		auto cur = getObjByIdx(idx);
		if (cur)
			cur->setHighlighted(true);
		if (idx != lastIdx)
		{
			auto pre = getObjByIdx(lastIdx);
			if (pre)
				pre->setHighlighted(false);
		}
	}

	GraphsSewing* GraphsSewing::clone() const
	{
		GraphsSewing* r = (GraphsSewing*)AbstractGraphObject::create(getType());
		r->m_firsts = m_firsts;
		r->m_seconds = m_seconds;
		r->m_sewingType = m_sewingType;
		r->setSelected(isSelected());
		return r;
	}

	static const char* sewingTypeToStr(GraphsSewing::SewingType s)
	{
		switch (s)
		{
		case ldp::GraphsSewing::SewingTypeStitch:
			return "stitch";
		case ldp::GraphsSewing::SewingTypePosition:
			return "position";
		case ldp::GraphsSewing::SewingTypeEnd:
		default:
			throw std::exception("sewingTypeToStr(): unknown type!");
		}
	}

	const char* GraphsSewing::getSewingTypeStr()const
	{
		return sewingTypeToStr(m_sewingType);
	}

	TiXmlElement* GraphsSewing::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractGraphObject::toXML(parent);
		ele->SetAttribute("type", getSewingTypeStr());
		TiXmlElement* fele = new TiXmlElement("Firsts");
		ele->LinkEndChild(fele);
		for (const auto& f : m_firsts)
		{
			TiXmlElement* uele = new TiXmlElement("unit");
			fele->LinkEndChild(uele);
			uele->SetAttribute("shape_id", f.curve->getId());
			uele->SetAttribute("reverse", f.reverse);
		}
		TiXmlElement* sele = new TiXmlElement("Seconds");
		ele->LinkEndChild(sele);
		for (const auto& s : m_seconds)
		{
			TiXmlElement* uele = new TiXmlElement("unit");
			sele->LinkEndChild(uele);
			uele->SetAttribute("shape_id", s.curve->getId());
			uele->SetAttribute("reverse", s.reverse);
		}
		return ele;
	}

	void GraphsSewing::fromXML(TiXmlElement* self)
	{
		AbstractGraphObject::fromXML(self);
		clear();
		const char* sewTypeStr = self->Attribute("type");
		if (sewTypeStr)
		{
			for (size_t i = 0; i < (size_t)SewingTypeEnd; i++)
			{
				if (sewingTypeToStr((SewingType)i) == std::string(sewTypeStr))
				{
					m_sewingType = (SewingType)i;
					break;
				}
			}
		} // end if sewTypeStr

		for (auto child = self->FirstChildElement(); child; child = child->NextSiblingElement())
		{
			if (child->Value() == std::string("Firsts"))
			{
				for (auto child1 = child->FirstChildElement(); child1; child1 = child1->NextSiblingElement())
				{
					if (child1->Value() == std::string("unit"))
					{
						Unit u;
						int tmp = 0;
						if (!child1->Attribute("shape_id", &tmp))
							throw std::exception("unit id lost");
						auto obj = getObjByIdx_loading(tmp);
						if (!obj->isCurve())
							throw std::exception("sewing loading error: units type not matched!\n");
						u.curve = (AbstractGraphCurve*)obj;
						if (!child1->Attribute("reverse", &tmp))
							throw std::exception("unit reverse lost");
						u.reverse = !!tmp;
						addFirst(u);
					}
				} // end for child1
			} // end if firsts
			if (child->Value() == std::string("Seconds"))
			{
				for (auto child1 = child->FirstChildElement(); child1; child1 = child1->NextSiblingElement())
				{
					if (child1->Value() == std::string("unit"))
					{
						Unit u;
						int tmp = 0;
						if (!child1->Attribute("shape_id", &tmp))
							throw std::exception("unit id lost");
						auto obj = getObjByIdx_loading(tmp);
						if (!obj->isCurve())
							throw std::exception("sewing loading error: units type not matched!\n");
						u.curve = (AbstractGraphCurve*)obj;
						if (!child1->Attribute("reverse", &tmp))
							throw std::exception("unit reverse lost");
						u.reverse = !!tmp;
						addSecond(u);
					}
				} // end for child1
			} // end if seconds
		} // end for child
	}
}