#include "GL\glew.h"
#include "SvgGroup.h"
namespace svg
{
#undef min
#undef max
	SvgGroup::SvgGroup()
	{
		m_boxColor = ldp::Float3(0.8, 0.6, 0.4);
		m_boxStrokeWidth = 3;
	}

	SvgGroup::~SvgGroup()
	{
	}

	void SvgGroup::render()
	{
		if (ancestorAfterRoot() == this && isSelected())
			renderBounds(false);

		for (size_t i = 0; i < m_children.size(); i++)
			m_children[i]->render();
	}

	void SvgGroup::renderId()
	{
		for (size_t i = 0; i < m_children.size(); i++)
			m_children[i]->renderId();
		if (ancestorAfterRoot() == this && isSelected())
			renderBounds(true);
	}

	void SvgGroup::updateBoundFromGeometry()
	{
		ldp::Float4 box(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX);

		for (auto child : m_children)
		{
			child->updateBoundFromGeometry();
			box = child->unionBound(box);
		}

		setBound(box);
	}

	TiXmlElement* SvgGroup::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = new TiXmlElement("g");
		parent->LinkEndChild(ele);
		for (auto child : m_children)
			child->toXML(ele);
		return ele;
	}

	void SvgGroup::copyTo(SvgAbstractObject* obj)const
	{
		SvgAbstractObject::copyTo(obj);
		if (obj->objectType() == SvgAbstractObject::Group)
		{
			auto g = (SvgGroup*)obj;
			g->m_children = m_children;
		}
	}

	bool SvgGroup::hasSelectedChildren()const
	{
		for (auto c : m_children)
		if (c->hasSelectedChildren() || c->isSelected())
			return true;
		return false;
	}

	std::shared_ptr<SvgAbstractObject> SvgGroup::clone(bool selectedOnly)const
	{
		std::shared_ptr<SvgAbstractObject> g(new SvgGroup());
		auto gptr = (SvgGroup*)g.get();
		if (selectedOnly)
		{
			if (!(hasSelectedChildren() || isSelected()))
				throw std::exception("ERROR: SvgGroup::clone(), mis-called");
		}

		copyTo(gptr);
		gptr->m_children.clear();
		for (auto c : m_children)
		{
			if (selectedOnly)
			{
				if (!(c->hasSelectedChildren() || c->isSelected()))
					continue;
			}
			auto newC = c->clone(selectedOnly);
			newC->setParent(gptr);
			gptr->m_children.push_back(newC);
		}

		return g;
	}

	std::shared_ptr<SvgAbstractObject> SvgGroup::deepclone(bool selectedOnly)const
	{
		std::shared_ptr<SvgAbstractObject> g(new SvgGroup());
		auto gptr = (SvgGroup*)g.get();
		if (selectedOnly)
		{
			if (!(hasSelectedChildren() || isSelected()))
				throw std::exception("ERROR: SvgGroup::clone(), mis-called");
		}

		copyTo(gptr);
		gptr->m_children.clear();
		for (auto c : m_children)
		{
			if (selectedOnly)
			{
				if (!(c->hasSelectedChildren() || c->isSelected()))
					continue;
			}
			auto newC = c->deepclone(selectedOnly);
			newC->setParent(gptr);
			gptr->m_children.push_back(newC);
		}

		return g;
	}

	void SvgGroup::collectObjects(ObjectType type, 
		std::vector<std::shared_ptr<SvgAbstractObject>>& objects,
		bool selectedOnly)const
	{
		for (auto c : m_children)
		{
			if (c->objectType() == Group)
				((SvgGroup*)c.get())->collectObjects(type, objects, selectedOnly);
			else
			{
				if (c->objectType() == type)
				if (!selectedOnly || c->isSelected())
					objects.push_back(c);
			}
		}
	}
}