#include "SvgAbstractObject.h"
#include "SvgAttribute.h"
#include "SvgPath.h"
#include "SvgText.h"
#include "SvgGroup.h"
#include "SvgPolyPath.h"

#undef min
#undef max

namespace svg
{
	SvgAbstractObject::SvgAbstractObject()
	{
		m_id = -1;
		m_parent = nullptr;
		m_selected = false;
		m_highlighted = false;
		m_invalid = true;
		resetBound();
		m_attribute = std::shared_ptr<SvgAttribute>(new SvgAttribute());
		m_boxColor = 0.f;
		m_boxStrokeWidth = 1;
	}

	SvgAbstractObject::~SvgAbstractObject()
	{
	}

	void SvgAbstractObject::copyTo(SvgAbstractObject* obj)const
	{
		obj->m_id = m_id;
		obj->m_parent = m_parent;
		obj->m_selected = m_selected;
		obj->m_highlighted = m_highlighted;
		obj->m_invalid = true; // always require valid
		obj->m_bbox = m_bbox;
		obj->m_attribute = m_attribute;
		obj->m_boxColor = m_boxColor;
		obj->m_boxStrokeWidth = m_boxStrokeWidth;
	}

	SvgAbstractObject* SvgAbstractObject::create(ObjectType type)
	{
		switch (type)
		{
		case svg::SvgAbstractObject::Path:
			return (SvgAbstractObject*)new SvgPath();
		case svg::SvgAbstractObject::Text:
			return (SvgAbstractObject*)new SvgText();
		case svg::SvgAbstractObject::Group:
			return (SvgAbstractObject*)new SvgGroup();
		case svg::SvgAbstractObject::PolyPath:
			return (SvgAbstractObject*)new SvgPolyPath();
		default:
			return nullptr;
			break;
		}
	}

	const SvgGroup* SvgAbstractObject::parent()const
	{
		return m_parent;
	}
	SvgGroup* SvgAbstractObject::parent()
	{
		return m_parent;
	}
	void SvgAbstractObject::setParent(SvgGroup* p)
	{
		m_parent = p;
	}
	const SvgGroup* SvgAbstractObject::root()const
	{
		const SvgGroup* p = parent();
		const SvgGroup* p1 = p;
		while (p != nullptr)
		{
			p1 = p;
			p = p->parent();
		}
		if (p1 == nullptr)
		{
			assert(this->objectType() == ObjectType::Group);
			p1 = (const SvgGroup*)this;
		}
		return p1;
	}
	SvgGroup* SvgAbstractObject::root()
	{
		SvgGroup* p = parent();
		SvgGroup* p1 = p;
		while (p != nullptr)
		{
			p1 = p;
			p = p->parent();
		}
		if (p1 == nullptr)
		{
			assert(this->objectType() == ObjectType::Group);
			p1 = (SvgGroup*)this;
		}
		return p1;
	}
	const SvgAbstractObject* SvgAbstractObject::ancestorAfterRoot()const
	{
		const SvgAbstractObject* p = this;

		// this is root itself, thus it cannot find ancestorAfterRoot
		if (p->parent() == nullptr) return nullptr;

		while (p->parent()->parent()) p = p->parent();
		return p;
	}
	SvgAbstractObject* SvgAbstractObject::ancestorAfterRoot()
	{
		SvgAbstractObject* p = this;

		// this is root itself, thus it cannot find ancestorAfterRoot
		if (p->parent() == nullptr) return nullptr;

		while (p->parent()->parent()) p = p->parent();
		return p;
	}

	ldp::Float4 SvgAbstractObject::unionBound(ldp::Float4 b)const
	{
		ldp::Float4 out;
		out[0] = std::min(m_bbox[0], b[0]);
		out[1] = std::max(m_bbox[1], b[1]);
		out[2] = std::min(m_bbox[2], b[2]);
		out[3] = std::max(m_bbox[3], b[3]);
		return out;
	}

	ldp::Float4 SvgAbstractObject::unionBound(ldp::Float2 point)const
	{
		ldp::Float4 out;
		out[0] = std::min(m_bbox[0], point[0]);
		out[1] = std::max(m_bbox[1], point[0]);
		out[2] = std::min(m_bbox[2], point[1]);
		out[3] = std::max(m_bbox[3], point[1]);
		return out;
	}

	ldp::Float4 SvgAbstractObject::intersectBound(ldp::Float4 b)const
	{
		ldp::Float4 out;
		out[0] = std::max(m_bbox[0], b[0]);
		out[1] = std::min(m_bbox[1], b[1]);
		out[2] = std::max(m_bbox[2], b[2]);
		out[3] = std::min(m_bbox[3], b[3]);
		return out;
	}

	void SvgAbstractObject::printGLError(const char* label)
	{
		if (label == nullptr)
			label = "";
		GLenum er = glGetError();
		if (er != GL_NO_ERROR)
			printf("[GLError][%s]: %s\n", label, glewGetErrorString(er));
	}

	void SvgAbstractObject::renderBounds(bool index_mode)
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glDisable(GL_STENCIL_TEST);

		if (index_mode)
		{
			ldp::Float4 c = color_from_index(m_id);
			glVertex4fv(c.ptr());
		}

		float sz = 1;
		if (isHighlighted())
			sz = 2;

		ldp::Float2 p[4] = {
			ldp::Float2(m_bbox[0], m_bbox[2]),
			ldp::Float2(m_bbox[1], m_bbox[2]),
			ldp::Float2(m_bbox[1], m_bbox[3]),
			ldp::Float2(m_bbox[0], m_bbox[3])
		};
		const float r = sz * std::min(m_bbox[1] - m_bbox[0], m_bbox[3] - m_bbox[2]) * 0.001f;
		glLineWidth(m_boxStrokeWidth * sz);

		// main box
		if (!index_mode)
			glColor4f(m_boxColor[0], m_boxColor[1], m_boxColor[2], 1);
		glBegin(GL_LINE_LOOP);
		for (int k = 0; k < 4; k++)
			glVertex2fv(p[k].ptr());
		glEnd();

		glPopAttrib();
	}
}