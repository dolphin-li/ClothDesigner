#include "AbstractGraphObject.h"
#include "tinyxml\tinyxml.h"

#include "GraphPoint.h"
#include "GraphLine.h"
#include "GraphQuadratic.h"
#include "GraphCubic.h"
namespace ldp
{
	AbstractGraphObject::IdxObjMap AbstractGraphObject::s_idxObjMap;
	AbstractGraphObject::IdxSet AbstractGraphObject::s_freeIdx;
	size_t AbstractGraphObject::s_nextIdx = 1;

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
		map[TypeSewing] = "sewing";
		return map;
	}
	AbstractGraphObject::TypeStringMap AbstractGraphObject::s_typeStringMap = AbstractGraphObject::generateTypeStringMap();

	AbstractGraphObject::AbstractGraphObject()
	{
		size_t id = 0;
		if (!s_freeIdx.empty())
		{
			id = *s_freeIdx.begin();
			s_freeIdx.erase(s_freeIdx.begin());
		}
		else
		{
			id = s_nextIdx++;
		}
		s_idxObjMap.insert(std::make_pair(id, this));
	}

	AbstractGraphObject::AbstractGraphObject(size_t id)
	{
		if (id > 0)
		{
			m_id = id;
			auto iter = s_idxObjMap.find(m_id);
			if (iter != s_idxObjMap.end())
				throw std::exception(std::string("IdxPool, duplicate index required: "
				+ std::to_string(m_id)).c_str());
		}
		else
		{
			size_t id = 0;
			if (!s_freeIdx.empty())
			{
				id = *s_freeIdx.begin();
				s_freeIdx.erase(s_freeIdx.begin());
			}
			else
			{
				id = s_nextIdx++;
			}
			s_idxObjMap.insert(std::make_pair(id, this));
		}
	}

	AbstractGraphObject::~AbstractGraphObject()
	{
		auto iter = s_idxObjMap.find(m_id);
		if (iter == s_idxObjMap.end())
			throw std::exception(std::string("IdxPool, freeIdx not existed: "
			+ std::to_string(m_id)).c_str());
		s_idxObjMap.erase(iter);
		s_freeIdx.insert(m_id);
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
		m_id = id;
		auto iter = s_idxObjMap.find(m_id);
		if (iter != s_idxObjMap.end())
			throw std::exception(std::string("IdxPool, duplicate index required: "
			+ std::to_string(m_id)).c_str());
	}

	AbstractGraphObject* AbstractGraphObject::clone()const
	{
		return create(getType(), getId());
	}

	/////////////////////////////////////////////////////////////////////////////////
	AbstractGraphObject* AbstractGraphObject::create(Type type, size_t id)
	{
		switch (type)
		{
		case ldp::AbstractGraphObject::TypeGraphPoint:
			return new GraphPoint(id);
		case ldp::AbstractGraphObject::TypeGraphLine:
			return new GraphLine(id);
		case ldp::AbstractGraphObject::TypeGraphQuadratic:
			return new GraphQuadratic(id);
		case ldp::AbstractGraphObject::TypeGraphCubic:
			return new GraphCubic(id);
		case ldp::AbstractGraphObject::TypeGraphLoop:
			break;
		case ldp::AbstractGraphObject::TypeGraph:
			break;
		case ldp::AbstractGraphObject::TypeSewing:
			break;
		case ldp::AbstractGraphObject::Type_End:
		default:
			return nullptr;
		}
		return nullptr;
	}

	AbstractGraphObject* AbstractGraphObject::create(std::string typeString, size_t id)
	{
		for (auto& iter : s_typeStringMap)
		{
			if (iter.second == typeString)
				return create(iter.first, id);
		}
		return nullptr;
	}
}