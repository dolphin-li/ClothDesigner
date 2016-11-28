#pragma once

#include <string>
#include <vector>
#include <memory>
#include "ldpMat\ldp_basic_mat.h"
#include "tinyxml\tinyxml.h"
namespace svg
{
	class SvgAttribute;
	class SvgGroup;
	class SvgAbstractObject
	{
	public:
		enum ObjectType
		{
			Path,
			Text,
			Group,
			PolyPath
		};
		enum
		{
			INDEX_BEGIN = 1
		};
	public:
		SvgAbstractObject();
		virtual ~SvgAbstractObject();

		virtual ObjectType objectType()const = 0;
		virtual void render() = 0;
		virtual void renderId() = 0;
		virtual std::shared_ptr<SvgAbstractObject> clone(bool selectedOnly = false)const = 0;
		virtual std::shared_ptr<SvgAbstractObject> deepclone(bool selectedOnly = false)const { return clone(); }
		virtual TiXmlElement* toXML(TiXmlNode* parent)const = 0;
		void invalid(){ m_invalid = true; } // call me when render context updated.
		virtual void copyTo(SvgAbstractObject* obj)const;
		virtual bool hasSelectedChildren()const{ return false; }

		void setId(int id) { m_id = id; }
		int getId()const { return m_id; }
		// number of indices this object needed
		// e.g., a quad may need 4 indices for each of its edge
		virtual int numId()const{ return 1; }
		virtual bool isMyValidIdRange(int id)const{ return id - m_id < numId() && id - m_id >= 0; }

		virtual void setSelected(bool s, int idx = -1){ m_selected = s; }
		bool isSelected()const{ return m_selected; }
		virtual void setHighlighted(bool s, int idx = -1){ m_highlighted = s; }
		bool isHighlighted()const{ return m_highlighted; }

		SvgGroup* parent();
		const SvgGroup* parent()const;
		void setParent(SvgGroup* p);
		SvgGroup* root();
		const SvgGroup* root()const;
		// find the ancestor that is the child of root
		const SvgAbstractObject* ancestorAfterRoot()const;
		SvgAbstractObject* ancestorAfterRoot();

		SvgAttribute* attribute(){ return m_attribute.get(); }
		const SvgAttribute* attribute()const{ return m_attribute.get(); }

		ldp::Float4 getBound()const{ return m_bbox; }
		ldp::Float2 getCenter()const{ return ldp::Float2(m_bbox[0]+m_bbox[1], m_bbox[2]+m_bbox[3]) * 0.5f; }
		void setBound(ldp::Float4 b){ m_bbox = b; }
		virtual void updateBoundFromGeometry(){  }
		void resetBound(){ m_bbox = ldp::Float4(FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX); }
		ldp::Float4 unionBound(ldp::Float4 b)const;
		ldp::Float4 unionBound(ldp::Float2 point)const;
		ldp::Float4 intersectBound(ldp::Float4 b)const;
		ldp::Float2 getOrigion()const{ return ldp::Float2(m_bbox[0], m_bbox[2]); }
		ldp::Float2 getSize()const { return ldp::Float2(width(), height()); }
		virtual ldp::Float2 getStartPoint()const{ return std::numeric_limits<float>::quiet_NaN(); }
		virtual ldp::Float2 getEndPoint()const{ return std::numeric_limits<float>::quiet_NaN(); }
		float width()const{ return m_bbox[1] - m_bbox[0]; }
		float height()const{ return m_bbox[3] - m_bbox[2]; }
	protected:
		virtual void renderBounds(bool index_mode);
	public:
		static SvgAbstractObject* create(ObjectType type);
		static void printGLError(const char* label = "");
		static ldp::Float4 color_from_index(int index)
		{
			int r = index & 0xff;
			int g = (index >> 8) & 0xff;
			int b = (index >> 16) & 0xff;
			int a = (index >> 24) & 0xff;
			return ldp::Float4(r, g, b, a) / 255.f;
		}
		static int index_from_color(ldp::Float4 c)
		{
			int r = c[0] * 255.f;
			int g = c[1] * 255.f;
			int b = c[2] * 255.f;
			int a = c[3] * 255.f;
			return r + (g << 8) + (b << 16) + (a << 24);
		}
	protected:
		int m_id;
		SvgGroup* m_parent;
		std::shared_ptr<SvgAttribute> m_attribute;
		ldp::Float4 m_bbox;
		bool m_selected;
		bool m_highlighted;
		bool m_invalid;
		ldp::Float3 m_boxColor;
		float m_boxStrokeWidth;
	};
}