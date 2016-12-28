#include "TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "ldpMat\Quaternion.h"
#include <sstream>
namespace ldp
{
	TransformInfo::TransformInfo()
	{
		setIdentity();
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
		// 1. flip normal if needed
		if (m_flipNormal)
		{
			for (auto& f : mesh.face_list)
				std::reverse(f.vertex_index, f.vertex_index + f.vertex_count);
		}

		// 2. apply cylinder transform
		if (hasCylinderTransform())
		{
			const Float3 center = mesh.getCenter();
			const float radius = m_cylinderTrans.radius;
			const Float3 y_dir(m_cylinderTrans.axis.normalize());
			Float3 x_dir(mesh.getBoundingBox(1) - center);
			const Float3 z_dir = x_dir.cross(y_dir).normalize();
			x_dir = y_dir.cross(z_dir).normalize();
			for (auto& v : mesh.vertex_list)
			{
				float y = (v - center).dot(y_dir);
				float x = (v - center).dot(x_dir);
				float x1 = radius * sin(x / radius);
				float z1 = radius * (1-cos(x / radius));
				v = y * y_dir + x1 * x_dir + z1 * z_dir + center;
			} // end for v
		} // end if hasCylinderTransform

		// 3. apply linear transform
		mesh.transform(m_T);
	}

	void TransformInfo::setIdentity()
	{
		m_T.eye();
		m_flipNormal = false;
	}

	void TransformInfo::translate(Float3 t)
	{
		m_T.setTranslationPart(m_T.getTranslationPart() + t);
	}

	void TransformInfo::rotate(Mat3f R, Float3 center)
	{
		m_T.setRotationPart(R*m_T.getRotationPart());
		m_T.setTranslationPart(R * (m_T.getTranslationPart()-center) + center);
	}

	void TransformInfo::scale(Float3 S, Float3 center)
	{
		ldp::Mat3f SM = ldp::Mat3f().eye();
		for (int k = 0; k < 3; k++)
			SM(k, k) = S[k];
		m_T.setRotationPart(SM*m_T.getRotationPart());
		m_T.setTranslationPart(SM * (m_T.getTranslationPart() - center) + center);
	}

	TiXmlElement* TransformInfo::toXML(TiXmlNode* parent)const
	{
		TiXmlElement* ele = new TiXmlElement(getTypeString().c_str());
		parent->LinkEndChild(ele);

		// flip normal
		ele->SetAttribute("FlipNormal", m_flipNormal);

		// transform
		std::string s;
		for (int i = 0; i < m_T.nRow(); i++)
		for (int j = 0; j < m_T.nCol(); j++)
			s += std::to_string(m_T(i, j)) + " ";
		ele->SetAttribute("T", s.c_str());

		// cylinder
		if (hasCylinderTransform())
		{
			std::string s = std::to_string(m_cylinderTrans.axis[0]) 
				+ " " + std::to_string(m_cylinderTrans.axis[1])
				+ " " + std::to_string(m_cylinderTrans.axis[2]);
			ele->SetAttribute("cylinder_axis", s.c_str());
			ele->SetAttribute("cylinder_radius", std::to_string(m_cylinderTrans.radius).c_str());
		}
		return ele;
	}

	void TransformInfo::fromXML(TiXmlElement* self)
	{
		// flip normal
		int tmp = 0;
		if (!self->Attribute("FlipNormal", &tmp))
			printf("warning: transformInfo.flipNormal lost");
		m_flipNormal = !!tmp;

		// transform
		if (self->Attribute("T"))
		{
			std::stringstream ts(self->Attribute("T"));
			m_T.eye();
			for (int i = 0; i < m_T.nRow(); i++)
			for (int j = 0; j < m_T.nCol(); j++)
				ts >> m_T(i, j);
		}

		// cylinder
		if (self->Attribute("cylinder_axis"))
		{
			std::stringstream ts(self->Attribute("cylinder_axis"));
			ts >> m_cylinderTrans.axis[0] >> m_cylinderTrans.axis[1] >> m_cylinderTrans.axis[2];
		}
		if (self->Attribute("cylinder_radius"))
		{
			std::stringstream ts(self->Attribute("cylinder_radius"));
			ts >> m_cylinderTrans.radius;
		}
	}

	float TransformInfo::cylinderCalcAngleFromRadius(const ObjMesh& mesh, float radius)
	{
		if (std::isnan(radius) || std::isinf(radius))
			return 0;

		if (radius == 0)
			return std::numeric_limits<float>::infinity();

		const Float3 center = mesh.getCenter();
		const Float3 y_dir(m_cylinderTrans.axis.normalize());
		Float3 x_dir(mesh.getBoundingBox(1) - center);
		const Float3 z_dir = x_dir.cross(y_dir).normalize();
		x_dir = y_dir.cross(z_dir).normalize();

		float maxX = -FLT_MAX;
		for (auto& v : mesh.vertex_list)
			maxX = std::max((v - center).dot(x_dir), maxX);

		return maxX / radius * ldp::PI_S / 2;
	}

	float TransformInfo::cylinderCalcRadiusFromAngle(const ObjMesh& mesh, float angle)
	{
		if (angle == 0)
			return std::numeric_limits<float>::quiet_NaN();
		if (std::isinf(angle))
			return 0;

		const Float3 center = mesh.getCenter();
		const Float3 y_dir(m_cylinderTrans.axis.normalize());
		Float3 x_dir(mesh.getBoundingBox(1) - center);
		const Float3 z_dir = x_dir.cross(y_dir).normalize();
		x_dir = y_dir.cross(z_dir).normalize();

		float maxX = -FLT_MAX;
		for (auto& v : mesh.vertex_list)
			maxX = std::max((v - center).dot(x_dir), maxX);

		return maxX / angle * ldp::PI_S / 2;
	}
}