#pragma once

#include "ldpMat\ldp_basic_mat.h"
#include "tinyxml\tinyxml.h"
class ObjMesh;
namespace ldp
{
	class TransformInfo
	{
	public:
		struct CylinderTransform
		{
			Float3 axis = Float3(0, 1, 0);
			float radius = std::numeric_limits<float>::quiet_NaN();
			CylinderTransform() {}
			CylinderTransform(Float3 a, float r) : axis(a), radius(r) {}
		};
	public:
		TransformInfo();
		~TransformInfo();

		void apply(ObjMesh& mesh);

		void setIdentity();
		void translate(Float3 t);
		void rotate(Mat3f R, Float3 center);
		void scale(Float3 S, Float3 center);

		ldp::Mat4f& transform() { return m_T; }
		const ldp::Mat4f& transform()const { return m_T; }

		bool hasCylinderTransform()const { return !std::isnan(m_cylinderTrans.radius); }
		const CylinderTransform& cylinderTransform()const { return m_cylinderTrans; }
		CylinderTransform& cylinderTransform() { return m_cylinderTrans; }
		void disableCylinderTransform() { m_cylinderTrans.radius = std::numeric_limits<float>::quiet_NaN(); }
		float cylinderCalcAngleFromRadius(const ObjMesh& mesh, float radius);
		float cylinderCalcRadiusFromAngle(const ObjMesh& mesh, float angle);

		void flipNormal();
		bool isFlipNormal() { return m_flipNormal; }

		virtual TiXmlElement* toXML(TiXmlNode* parent)const;
		virtual void fromXML(TiXmlElement* self);

		std::string getTypeString()const { return "TransformInfo"; }
	private:
		ldp::Mat4f m_T;
		bool m_flipNormal;

		CylinderTransform m_cylinderTrans;
	};
}