#pragma once

#include <string>
#include <hash_map>
#include <hash_set>
#include <memory>
class TiXmlElement;
class TiXmlNode;
namespace ldp
{
	// NOT THREAD SAFE
	class AbstractGraphObject
	{
	public:
		enum Type
		{
			TypeGraphPoint,
			TypeGraphLine,
			TypeGraphQuadratic,
			TypeGraphCubic,
			TypeGraphLoop,
			TypeGraph,
			TypeGraphsSewing,
			Type_End
		};
		enum SelectOp
		{
			SelectThis,
			SelectUnion,
			SelectUnionInverse,
			SelectAll,
			SelectNone,
			SelectInverse,
			SelectEnd
		};
		typedef std::hash_map<size_t, AbstractGraphObject*> IdxObjMap;
		typedef std::hash_map<Type, std::string> TypeStringMap;
		typedef std::hash_set<size_t> IdxSet;
	public:
		AbstractGraphObject();
		virtual ~AbstractGraphObject();

		virtual AbstractGraphObject* clone()const;
		virtual Type getType()const = 0;
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		static AbstractGraphObject* create(Type type);
		static AbstractGraphObject* create(std::string typeString);

		std::string getTypeString()const;
		size_t getId()const { return m_id; }
		bool isSelected()const { return m_selected; }
		void setSelected(bool s) { m_selected = s; }
		bool isHighlighted()const { return m_highlighted; }
		void setHighlighted(bool h) { m_highlighted = h; }
		static AbstractGraphObject* getObjByIdx(size_t id)
		{
			auto iter = s_idxObjMap.find(id);
			if (iter == s_idxObjMap.end())
				return nullptr;
			return iter->second;
		}

		bool isCurve()const { return getType() == TypeGraphCubic 
			|| getType() == TypeGraphLine || getType() == TypeGraphQuadratic; }
		bool operator <(const AbstractGraphObject& rhs)const { return m_id < rhs.m_id; }
	protected:
		virtual AbstractGraphObject& operator = (const AbstractGraphObject& rhs) 
		{ 
			throw std::exception("AbstractGraphObject: assign not allowed!"); 
		}
		AbstractGraphObject(const AbstractGraphObject& rhs) 
		{
			throw std::exception("AbstractGraphObject: assign not allowed!");
		}
		void requireIdx();
		void freeIdx();
	private:
		size_t m_id = 0;
		bool m_selected = false;
		bool m_highlighted = false;
	private:
		static IdxObjMap s_idxObjMap;
		static IdxSet s_freeIdx;
		static size_t s_nextIdx;
		static TypeStringMap s_typeStringMap;
		static TypeStringMap generateTypeStringMap();

		// for tmp idx maping when loading from files.
		static IdxObjMap s_idxObjMap_loading;
	protected:
		static AbstractGraphObject* getObjByIdx_loading(size_t id)
		{
			auto iter = s_idxObjMap_loading.find(id);
			if (iter == s_idxObjMap_loading.end())
				return nullptr;
			return iter->second;
		}
	};
	typedef std::shared_ptr<AbstractGraphObject> AbstractGraphObjectPtr;
}