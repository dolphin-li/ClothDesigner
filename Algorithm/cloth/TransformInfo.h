#pragma once

#include "ldpMat\ldp_basic_mat.h"
class ObjMesh;
namespace ldp
{
	class TransformInfo
	{
	public:
		TransformInfo();
		~TransformInfo();

		void apply(ObjMesh& mesh);

		ldp::Mat4f& transform() { return m_T; }
		const ldp::Mat4f& transform()const { return m_T; }
	private:
		ldp::Mat4f m_T;
	};
}