#include "GraphsSewing.h"
#include "AbstractGraphCurve.h"
#include "tinyxml\tinyxml.h"
namespace ldp
{
	GraphsSewing::GraphsSewing() : AbstractGraphObject()
	{
	
	}

	void GraphsSewing::clear()
	{
		m_firsts.clear();
		m_seconds.clear();
	}

	void GraphsSewing::addFirst(Unit unit)
	{
		auto iter_same = m_firsts.end();
		for (auto iter = m_firsts.begin(); iter != m_firsts.end(); ++iter)
		{
			if (iter->curve == unit.curve)
			{
				iter_same = iter;
				break;
			}
		}
		if (iter_same == m_firsts.end())
			m_firsts.push_back(unit);
		else
			iter_same->reverse = unit.reverse;
	}

	void GraphsSewing::addSecond(Unit unit)
	{
		auto iter_same = m_seconds.end();
		for (auto iter = m_seconds.begin(); iter != m_seconds.end(); ++iter)
		{
			if (iter->curve == unit.curve)
			{
				iter_same = iter;
				break;
			}
		}
		if (iter_same == m_seconds.end())
			m_seconds.push_back(unit);
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
		std::set<size_t> shapes;
		shapes.insert(id);
		remove(shapes);
	}

	void GraphsSewing::remove(const std::set<size_t>& s)
	{
		auto tmp = m_firsts;
		m_firsts.clear();
		for (auto& f : tmp)
		{
			if (s.find(f.curve->getId()) == s.end())
				m_firsts.push_back(f);
		}
		tmp = m_seconds;
		m_seconds.clear();
		for (auto& f : tmp)
		{
			if (s.find(f.curve->getId()) == s.end())
				m_seconds.push_back(f);
		}
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
		r->setSelected(isSelected());
		return r;
	}

	TiXmlElement* GraphsSewing::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = AbstractGraphObject::toXML(parent);
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
						m_firsts.push_back(u);
					}
				}
			}
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
						m_seconds.push_back(u);
					}
				}
			}
		}
	}
}