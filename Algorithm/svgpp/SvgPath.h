#pragma once
#include "GL\glew.h"
#include "SvgAbstractObject.h"
#include "PathStyle.hpp"
namespace svg
{
	class SvgPath : public SvgAbstractObject
	{
	public:
		SvgPath();
		virtual ~SvgPath();
		ObjectType objectType()const { return ObjectType::Path; }

		virtual void render();
		virtual void renderId();
		virtual std::shared_ptr<SvgAbstractObject> clone(bool selectedOnly = false)const;
		virtual std::shared_ptr<SvgAbstractObject> deepclone(bool selectedOnly = false)const;
		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void copyTo(SvgAbstractObject* obj)const;
		virtual ldp::Float2 getStartPoint()const;
		virtual ldp::Float2 getEndPoint()const;
		virtual int numId()const{ return 1; }

		// the path is closed or open
		bool isClosed()const;

		virtual void updateBoundFromGeometry();
		void setPathWidth(float w);
		float getPathWidth()const { return m_pathStyle.stroke_width; }

		// split all segments to individual paths
		// if there is only one segment, we do not split and return an nullptr
		// else we return grouped splittings
		std::shared_ptr<SvgAbstractObject> splitToSegments(bool to_single_segs = true)const;

		// extract path from cmdsBegin(include) to cmdsEnd(exclude)
		// to_single_segment will convert 'M-L-L-L' to 'M-L M-L M-L'
		std::shared_ptr<SvgAbstractObject> subPath(int cmdsBegin, int cmdsEnd,
			bool to_single_segment = false)const;
		std::shared_ptr<SvgAbstractObject> subPath(const std::vector<int>& cmdsBegins,
			const std::vector<int>& cmdsEnds, bool to_single_segment = false)const;

		// check the intersection with other and insert a point if intersected
		bool insertPointByIntersection(const SvgPath* other, float thre);
	protected:
		void cacheNvPaths();
		void renderSelection();
		void configNvParams();
	public:
		std::vector<GLubyte> m_cmds;
		std::vector<GLfloat> m_coords;
		GLenum m_gl_fill_rull;
		PathStyle m_pathStyle;

		// return number of coordinates associated with a cmd
		static int numCoords(GLubyte cmd);
		static char svgCmd(GLubyte cmd);
		static const char* strokeFillMap(int fill);
		static const char* strokeLineCapMap(int cap);
		static const char* strokeLineJoinMap(int join);
		static GLenum lineJoinConverter(const SvgPath *path);
		static GLenum lineCapConverter(const SvgPath *path);
	public:
		struct GLPathResource{
			GLuint id;
			GLPathResource();
			~GLPathResource();
		};
		std::shared_ptr<GLPathResource> m_gl_path_res;
	};
}