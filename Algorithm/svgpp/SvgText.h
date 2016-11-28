#pragma once

#include "SvgAbstractObject.h"
#include <map>
#include <string>
namespace svg
{
	class SvgText : public SvgAbstractObject
	{
	protected:
		struct FontFace{
			static const int em_scale = 1;
			GLenum target;
			int num_chars;
			std::string font_name;
			GLuint glyph_base;
			std::vector<GLfloat> horizontal_advance;
			GLfloat y_min, y_max;
			GLfloat underline_position, underline_thickness;
			FontFace(const char *name);
			static int gl_path_id;
		};
		typedef std::shared_ptr<FontFace> FontFacePtr;
	public:
		SvgText();
		SvgText(bool generate_resource);
		virtual ~SvgText();
		ObjectType objectType()const { return ObjectType::Text; }

		virtual void render();
		virtual void renderId();
		virtual std::shared_ptr<SvgAbstractObject> clone(bool selectedOnly = false)const;
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void copyTo(SvgAbstractObject* obj)const;

		virtual void updateBoundFromGeometry();
		void updateText();
	public:
		std::string m_font;
		std::string m_text;
		float m_font_size;
	private:
		ldp::Float4 m_bbox_before_transform;
		std::vector<float> m_hori_shifts;
		float m_total_advance;
		FontFacePtr m_font_face;
		static std::map<std::string, FontFacePtr> m_font_face_map;
		static FontFacePtr requireFontFace(std::string name);
	};
}