#include "AbstractGraphObject.h"
#include "tinyxml\tinyxml.h"

#include "GraphPoint.h"
#include "GraphLine.h"
#include "GraphQuadratic.h"
#include "GraphCubic.h"
#include "GraphLoop.h"
#include "Graph.h"
#include "GraphsSewing.h"
namespace ldp
{
	AbstractGraphObject::IdxObjMap AbstractGraphObject::s_idxObjMap;
	AbstractGraphObject::IdxSet AbstractGraphObject::s_freeIdx;
	size_t AbstractGraphObject::s_nextIdx = 1;
	AbstractGraphObject::IdxObjMap AbstractGraphObject::s_idxObjMap_loading;

	AbstractGraphObject::TypeStringMap AbstractGraphObject::generateTypeStringMap()
	{
		TypeStringMap map;
		map.clear();
		map[TypeGraphPoint] = "point";
		map[TypeGraphLine] = "line";
		map[TypeGraphQuadratic] = "quadratic";
		map[TypeGraphCubic] = "cubic";
		map[TypeGraphLoop] = "loop";
		map[TypeGraph] = "graph";
		map[TypeGraphsSewing] = "sewing";
		return map;
	}
	AbstractGraphObject::TypeStringMap AbstractGraphObject::s_typeStringMap = AbstractGraphObject::generateTypeStringMap();

	AbstractGraphObject::AbstractGraphObject()
	{
		requireIdx();
	}

	AbstractGraphObject::~AbstractGraphObject()
	{
		freeIdx();
	}

	void AbstractGraphObject::requireIdx()
	{
		if (!s_freeIdx.empty())
		{
			m_id = *s_freeIdx.begin();
			s_freeIdx.erase(s_freeIdx.begin());
		}
		else
			m_id = s_nextIdx++;

		s_nextIdx = std::max(s_nextIdx, m_id + 1);
		s_idxObjMap.insert(std::make_pair(m_id, this));
	}

	void AbstractGraphObject::freeIdx()
	{
		auto iter = s_idxObjMap.find(m_id);
		if (iter == s_idxObjMap.end())
		{
			printf("error: freeIdx not existed %d\n", m_id);
			return;
		}
		s_idxObjMap.erase(iter);
		s_freeIdx.insert(m_id);
		m_id = 0;
	}

	std::string AbstractGraphObject::getTypeString()const
	{
		return s_typeStringMap[getType()];
	}

	TiXmlElement* AbstractGraphObject::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = new TiXmlElement(getTypeString().c_str());
		parent->LinkEndChild(ele);
		ele->SetAttribute("id", m_id);
		return ele;
	}

	void AbstractGraphObject::fromXML(TiXmlElement* self)
	{
		int id = 0;
		if (!self->Attribute("id", &id))
			throw std::exception(("cannot find id for " + getTypeString()).c_str());
		m_loadedId = id;
		s_idxObjMap_loading[id] = this;
	}

	AbstractGraphObject* AbstractGraphObject::clone()const
	{
		auto obj = create(getType());
		obj->setSelected(isSelected());
		return obj;
	}

	/////////////////////////////////////////////////////////////////////////////////
	AbstractGraphObject* AbstractGraphObject::create(Type type)
	{
		switch (type)
		{
		case ldp::AbstractGraphObject::TypeGraphPoint:
			return new GraphPoint();
		case ldp::AbstractGraphObject::TypeGraphLine:
			return new GraphLine();
		case ldp::AbstractGraphObject::TypeGraphQuadratic:
			return new GraphQuadratic();
		case ldp::AbstractGraphObject::TypeGraphCubic:
			return new GraphCubic();
		case ldp::AbstractGraphObject::TypeGraphLoop:
			return new GraphLoop();
		case ldp::AbstractGraphObject::TypeGraph:
			return new Graph();
		case ldp::AbstractGraphObject::TypeGraphsSewing:
			return new GraphsSewing();
		case ldp::AbstractGraphObject::Type_End:
		default:
			return nullptr;
		}
		return nullptr;
	}

	AbstractGraphObject* AbstractGraphObject::create(std::string typeString)
	{
		for (auto& iter : s_typeStringMap)
		{
			if (iter.second == typeString)
				return create(iter.first);
		}
		return nullptr;
	}
}