#pragma once

#include "GLHelper.h"
#include "ldpMat\ldp_basic_mat.h"
class ObjMesh;

struct MeshModel
{
	cml::GLVertexVBO	m_VertexVBO;
	cml::GLVertexVBO	m_NormalVBO;
	std::vector<ldp::Int3> m_faces;

	MeshModel(){ createVBO(); }
	~MeshModel(){ deleteVBO(); }

	void createVBO();
	void deleteVBO();
	void fromObjMesh(const ObjMesh* mesh);

	void drawPlane()const;
	void drawShade()const;
};

class MeshRender
{
public:

	bool	Initialize();
	bool	Cleanup();

	void updateGeometry(ObjMesh** meshModelsA, int meshNumA,
		ObjMesh** meshModelsB, int meshNumB);

	void	PrepareLightMatrix(
		ldp::Float3 sceneRangeMin, ldp::Float3 sceneRangeMax,
		const ldp::Float3 * lightDirections,
		ldp::Mat4f* modelViewMatrix, ldp::Mat4f* projectionMatrix) const;

	void	RenderShadowMap(
		ldp::Mat4f* modelViewMatrix, ldp::Mat4f* projectionMatrix,
		cml::GLFBO& shadowMapFBO) const;

	void	RenderMesh(
			ldp::Float3 camPos,
			const ldp::Float3* lightColors, const ldp::Float3* lightDirections,
			const ldp::Mat4f* modelViewMatrix, const ldp::Mat4f* modelViewProjectionMatrix,
			float ambient, float diffuse, float specular, float shadow,
			ldp::Float3 colorA, ldp::Float3 colorB,
			cml::GLMSFBO& pFramebufferFBO,
			cml::GLTexture& pDOMDepthMap,
			int showType);

	void	RenderComposite(cml::GLMSTexture& pColorTexture);

private:

	bool	LoadShaders();
	void	ReleaseShaders();

	cml::GLProgram m_DepthMapProgram;
	cml::GLProgram m_ShadeMeshProgram;
	cml::GLProgram m_CompositeProgram;

	std::vector<MeshModel> m_meshesA;
	std::vector<MeshModel> m_meshesB;
};
