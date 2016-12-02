#include "TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "ldpMat\Quaternion.h"
#include <sstream>
namespace ldp
{
	TransformInfo::TransformInfo()
	{
		m_T.zeros();

		// by default, the conversion is from (x,y,0) to (x,0,y)
		m_T.setRotationPart(ldp::QuaternionF().fromAngles(ldp::Float3(ldp::PI_S/2, 0, 0)).toRotationMatrix3());

		// then move foward
		m_T(1, 3) = -0.3; // in meters

		// to handel CCW/CW issues of [triangle]
		m_flipNormal = true;
	}

	TransformInfo::~TransformInfo()
	{
	
	}

	void TransformInfo::flipNormal()
	{
		m_flipNormal = !m_flipNormal;
	}

	void TransformInfo::apply(ObjMesh& mesh)
	{
		if (m_flipNormal)
		{
			for (auto& f : mesh.face_list)
				std::reverse(f.vertex_index, f.vertex_index + f.vertex_count);
		}
		mesh.transform(m_T);
	}


	TiXmlElement* TransformInfo::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = new TiXmlElement("TransformInfo");
		parent->LinkEndChild(ele);
		ele->SetAttribute("FlipNormal", m_flipNormal);
		std::stringstream ts;
		ts << m_T;
		std::string s;
		ts >> s;
		ele->SetAttribute("T", s.c_str());
		return ele;
	}

	void TransformInfo::fromXML(TiXmlElement* self)
	{
		int tmp = 0;
		if (!self->Attribute("FlipNormal", &tmp))
			printf("warning: transformInfo.flipNormal lost");
		m_flipNormal = !!tmp;
		std::string s = self->Attribute("T");
		if (s.empty())
			printf("warning: transformInfo.T lost");
		std::stringstream ts(s);
		m_T.eye();
		for (int i = 0; i < m_T.nRow(); i++)
		for (int j = 0; j < m_T.nCol(); j++)
			ts >> m_T(i, j);
	}
}