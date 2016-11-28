#pragma once

#include "SvgAbstractObject.h"
namespace svg
{
	class SvgGroup : public SvgAbstractObject
	{
	public:
		SvgGroup();
		virtual ~SvgGroup();
		ObjectType objectType()const { return ObjectType::Group; }

		virtual void render();
		virtual void renderId();
		virtual std::shared_ptr<SvgAbstractObject> clone(bool selectedOnly = false)const;
		virtual std::shared_ptr<SvgAbstractObject> deepclone(bool selectedOnly = false)const;
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void copyTo(SvgAbstractObject* obj)const;
		virtual bool hasSelectedChildren()const;

		virtual void updateBoundFromGeometry();

		void collectObjects(ObjectType type, std::vector<std::shared_ptr<SvgAbstractObject>>& objects, bool selectedOnly)const;
	public:
		std::vector<std::shared_ptr<SvgAbstractObject>> m_children;
	};
}