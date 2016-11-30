#include "TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "ldpMat\Quaternion.h"
namespace ldp
{
	TransformInfo::TransformInfo()
	{
		m_T.eye();
	}

	TransformInfo::~TransformInfo()
	{
	
	}

	void TransformInfo::apply(ObjMesh& mesh)
	{
		mesh.transform(m_T);
	}
}