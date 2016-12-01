#include "TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "ldpMat\Quaternion.h"
namespace ldp
{
	TransformInfo::TransformInfo()
	{
		m_T.zeros();

		// by default, the conversion is from (x,y,0) to (x,0,y)
		m_T.setRotationPart(ldp::QuaternionF().fromAngles(ldp::Float3(ldp::PI_S/2, 0, 0)).toRotationMatrix3());

		// then move foward
		m_T(1, 3) = -0.3; // in meters
	}

	TransformInfo::~TransformInfo()
	{
	
	}

	void TransformInfo::apply(ObjMesh& mesh)
	{
		mesh.transform(m_T);
	}
}