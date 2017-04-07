#include "MeshRender.h"

#include "Renderable\ObjMesh.h"

using namespace cml;

bool MeshRender::Initialize()
{
	if (!m_DepthMapProgram.CreateProgramFromFiles("shaders/ShadowMap.vert", "shaders/ShadowMap.frag")){
		printf("Failed to load DepthMap shader.");
		return false;
	}
	if (!m_ShadeMeshProgram.CreateProgramFromFiles("shaders/ShadeHead.vert", "shaders/ShadeHead.frag")){
		printf("Failed to load ShadeHead shader.\n");
		return false;
	}
	if (!m_CompositeProgram.CreateProgramFromFiles("shaders/Composite.vert", "shaders/Composite.frag")){
		printf("Failed to load Composite shader.\n");
		return false;
	}
	return true;
}

bool MeshRender::Cleanup()
{
	m_DepthMapProgram.DeleteProgram();
	m_ShadeMeshProgram.DeleteProgram();
	m_CompositeProgram.DeleteProgram();
	m_meshesA.clear();
	m_meshesB.clear();
	return true;
}

void MeshRender::updateGeometry(ObjMesh** meshModelsA, int meshNumA,
	ObjMesh** meshModelsB, int meshNumB)
{
	m_meshesA.resize(meshNumA);
	m_meshesB.resize(meshNumB);

	for (size_t i = 0; i < m_meshesA.size(); i++)
		m_meshesA[i].fromObjMesh(meshModelsA[i]);
	for (size_t i = 0; i < m_meshesB.size(); i++)
		m_meshesB[i].fromObjMesh(meshModelsB[i]);
}

void MeshRender::PrepareLightMatrix(
	ldp::Float3 sceneRangeMin, ldp::Float3 sceneRangeMax,
	const ldp::Float3 * lightDirections,
	ldp::Mat4f* modelViewMatrix, ldp::Mat4f* projectionMatrix) const
{
	for (int lightI = 0; lightI < 4; lightI++)
	{
		const ldp::Float3& curLightDirection = lightDirections[lightI];
		ldp::Mat4f& curModelView = modelViewMatrix[lightI];
		ldp::Mat4f& curProjection = projectionMatrix[lightI];

		ldp::Float3 target = (sceneRangeMin + sceneRangeMax) * 0.5f;
		float dist = (sceneRangeMax - sceneRangeMin).length();
		ldp::Float3 center = target + curLightDirection * dist;
		ldp::Float3 right = 0.f;

		int right_axis = (fabs(curLightDirection[1]) < fabs(curLightDirection[0])) ? 1 : 0;
		right_axis = (fabs(curLightDirection[2]) < fabs(curLightDirection[right_axis])) ? 2 : right_axis;
		right[right_axis] = 1.f;

		ldp::Float3 up = right.cross(curLightDirection).normalize();

		// Set the world matrix
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		gluLookAt(
			center[0], center[1], center[2],
			target[0], target[1], target[2],
			up[0], up[1], up[2]);
		glGetFloatv(GL_MODELVIEW_MATRIX, curModelView.ptr());
		glPopMatrix();

		// Transform the bbox into the light source's view space
		// xydim is the maximal distance of the bbox's vertex to the center
		// it's the size of the texture
		// zmax is the maximal depth
		float zmin, zmax;
		float xydim = 0.f;
		ldp::Float3 bboxvertex[8] = 
		{
			ldp::Float3(sceneRangeMin[0], sceneRangeMin[1], sceneRangeMin[2]),
			ldp::Float3(sceneRangeMin[0], sceneRangeMin[1], sceneRangeMax[2]),
			ldp::Float3(sceneRangeMin[0], sceneRangeMax[1], sceneRangeMin[2]),
			ldp::Float3(sceneRangeMin[0], sceneRangeMax[1], sceneRangeMax[2]),
			ldp::Float3(sceneRangeMax[0], sceneRangeMin[1], sceneRangeMin[2]),
			ldp::Float3(sceneRangeMax[0], sceneRangeMin[1], sceneRangeMax[2]),
			ldp::Float3(sceneRangeMax[0], sceneRangeMax[1], sceneRangeMin[2]),
			ldp::Float3(sceneRangeMax[0], sceneRangeMax[1], sceneRangeMax[2])
		};

		for (int vid = 0; vid < 8; ++vid)
		{
			ldp::Float3 vview = curModelView.getRotationPart() * bboxvertex[vid] 
				+ curModelView.getTranslationPart();
			vview[2] = -vview[2];
			xydim = std::max(xydim, fabs(vview[0]));
			xydim = std::max(xydim, fabs(vview[1]));
			zmin = (vid == 0) ? vview[2] : std::min(vview[2], zmin);
			zmax = (vid == 0) ? vview[2] : std::max(vview[2], zmax);
		}

		// Set the projection matrix
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-xydim, xydim, -xydim, xydim, zmin - 100.f, zmax + 100.f);
		// Save the light view projection matrix
		glGetFloatv(GL_PROJECTION_MATRIX, curProjection.ptr());
		glPopMatrix();

		glMatrixMode(GL_MODELVIEW);
	}
}

void MeshRender::RenderShadowMap(
	ldp::Mat4f* modelViewMatrix,
	ldp::Mat4f* projectionMatrix,
	GLFBO& depthMapFBO) const
{
	glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_VIEWPORT_BIT);

	int depthMapWidth, depthMapHeight;
	depthMapFBO.GetTextureSize(depthMapWidth, depthMapHeight);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	depthMapFBO.BindFBO();
	m_DepthMapProgram.UseProgram();
	glDepthRange(0.f, 1.f);
	glClearColor(1e30f, 1e30f, 1e30f, 1.f);
	glClearDepth(1.f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_TRUE);
	glDisable(GL_MULTISAMPLE);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (int lightI = 0; lightI < 4; lightI++){
		glViewport((lightI % 2) * depthMapWidth / 2, (lightI / 2) * depthMapHeight / 2, depthMapWidth / 2, depthMapHeight / 2);

		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(projectionMatrix[lightI].ptr());

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(modelViewMatrix[lightI].ptr());

		for (size_t meshI = 0; meshI < m_meshesA.size(); meshI++)
			m_meshesA[meshI].drawShade();

		for (size_t meshI = 0; meshI < m_meshesB.size(); meshI++)
			m_meshesB[meshI].drawShade();
	}

	glEnable(GL_MULTISAMPLE);
	m_DepthMapProgram.DisuseProgram();
	depthMapFBO.UnbindFBO();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glPopAttrib();
}

void MeshRender::RenderMesh(
	ldp::Float3 camPos,
	const ldp::Float3* lightColors, const ldp::Float3* lightDirections,
	const ldp::Mat4f* modelViewMatrix, const ldp::Mat4f* modelViewProjectionMatrix,
	float ambient, float diffuse, float specular, float shadow,
	ldp::Float3 colorA, ldp::Float3 colorB,
	cml::GLMSFBO& pFramebufferFBO,
	cml::GLTexture& pDOMDepthMap,
	int showType)
{
	glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	int width, height;
	pFramebufferFBO.GetTextureSize(width, height);

	glViewport(0, 0, width, height);

	pFramebufferFBO.BindFBO();

	m_ShadeMeshProgram.UseProgram();

	int texColor = 0;
	int texDepth = 1;

	///pColorTexture.BindTextureUnit(texColor);
	pDOMDepthMap.BindTextureUnit(texDepth);

	m_ShadeMeshProgram.SetUniform3f("ViewPoint", camPos[0], camPos[1], camPos[2]);

	m_ShadeMeshProgram.SetUniform3f("LightDir1", lightDirections[0][0], lightDirections[0][1], lightDirections[0][2]);
	m_ShadeMeshProgram.SetUniform3f("LightDir2", lightDirections[1][0], lightDirections[1][1], lightDirections[1][2]);
	m_ShadeMeshProgram.SetUniform3f("LightDir3", lightDirections[2][0], lightDirections[2][1], lightDirections[2][2]);
	m_ShadeMeshProgram.SetUniform3f("LightDir4", lightDirections[3][0], lightDirections[3][1], lightDirections[3][2]);

	m_ShadeMeshProgram.SetUniform3f("LightColor1", lightColors[0][0], lightColors[0][1], lightColors[0][2]);
	m_ShadeMeshProgram.SetUniform3f("LightColor2", lightColors[1][0], lightColors[1][1], lightColors[1][2]);
	m_ShadeMeshProgram.SetUniform3f("LightColor3", lightColors[2][0], lightColors[2][1], lightColors[2][2]);
	m_ShadeMeshProgram.SetUniform3f("LightColor4", lightColors[3][0], lightColors[3][1], lightColors[3][2]);

	m_ShadeMeshProgram.SetUniformMatrix4fv("LightModelViewMatrix1", modelViewMatrix[0].ptr());
	m_ShadeMeshProgram.SetUniformMatrix4fv("LightModelViewMatrix2", modelViewMatrix[1].ptr());
	m_ShadeMeshProgram.SetUniformMatrix4fv("LightModelViewMatrix3", modelViewMatrix[2].ptr());
	m_ShadeMeshProgram.SetUniformMatrix4fv("LightModelViewMatrix4", modelViewMatrix[3].ptr());

	m_ShadeMeshProgram.SetUniformMatrix4fv("LightModelViewProjMatrix1", modelViewProjectionMatrix[0].ptr());
	m_ShadeMeshProgram.SetUniformMatrix4fv("LightModelViewProjMatrix2", modelViewProjectionMatrix[1].ptr());
	m_ShadeMeshProgram.SetUniformMatrix4fv("LightModelViewProjMatrix3", modelViewProjectionMatrix[2].ptr());
	m_ShadeMeshProgram.SetUniformMatrix4fv("LightModelViewProjMatrix4", modelViewProjectionMatrix[3].ptr());

	m_ShadeMeshProgram.SetUniform1f("Ambient", ambient);
	m_ShadeMeshProgram.SetUniform1f("Diffuse", diffuse);
	m_ShadeMeshProgram.SetUniform1f("Specular", specular);
	m_ShadeMeshProgram.SetUniform1f("Shadow", shadow);

	m_ShadeMeshProgram.SetUniform1i("ScalpColor", texColor);
	m_ShadeMeshProgram.SetUniform1i("DepthMap", texDepth);

	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClearDepth(1.f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	m_ShadeMeshProgram.SetUniform3f("Color", colorA[0], colorA[1], colorA[2]);

	for (size_t meshI = 0; meshI < m_meshesA.size(); meshI++)
		m_meshesA[meshI].drawShade();

	m_ShadeMeshProgram.SetUniform3f("Color", colorB[0], colorB[1], colorB[2]);

	for (size_t meshI = 0; meshI < m_meshesB.size(); meshI++)
		m_meshesB[meshI].drawShade();

	///pColorTexture.UnbindTextureUnit(texColor);
	pDOMDepthMap.UnbindTextureUnit(texDepth);

	pDOMDepthMap.ActiveTexture(0);

	m_ShadeMeshProgram.DisuseProgram();

	pFramebufferFBO.UnbindFBO();

	glPopAttrib();
}

void MeshRender::RenderComposite(GLMSTexture& pColorTexture)
{
	glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	m_CompositeProgram.UseProgram();

	int texHeadMap = 0;

	pColorTexture.BindTextureUnit(texHeadMap);

	m_CompositeProgram.SetUniform1i("HeadMap", texHeadMap);

	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	glBegin(GL_QUADS);
	glTexCoord2f(0.f, 0.f);
	glVertex2f(-1.f, -1.f);
	glTexCoord2f(1.f, 0.f);
	glVertex2f(1.f, -1.f);
	glTexCoord2f(1.f, 1.f);
	glVertex2f(1.f, 1.f);
	glTexCoord2f(0.f, 1.f);
	glVertex2f(-1.f, 1.f);
	glEnd();

	pColorTexture.UnbindTextureUnit(texHeadMap);

	pColorTexture.ActiveTexture(0);

	m_CompositeProgram.DisuseProgram();

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);

	glPopAttrib();
}

void MeshModel::createVBO()
{
	if (m_VertexVBO.GetVBO() == 0)
		m_VertexVBO.CreateVBO();
	if (m_NormalVBO.GetVBO() == 0)
		m_NormalVBO.CreateVBO();
}

void MeshModel::deleteVBO()
{
	if (m_VertexVBO.GetVBO())
		m_VertexVBO.DeleteVBO();
	if (m_NormalVBO.GetVBO())
		m_NormalVBO.DeleteVBO();
}

void MeshModel::fromObjMesh(const ObjMesh* mesh)
{
	m_VertexVBO.FillVBO(mesh->vertex_list.data(), mesh->vertex_list.size() * 3 * sizeof(float), true);
	m_NormalVBO.FillVBO(mesh->vertex_normal_list.data(), mesh->vertex_normal_list.size() * 3 * sizeof(float), true);
	m_faces.clear();
	for (const auto& f : mesh->face_list)
	{
		for (int k = 0; k < f.vertex_count - 2; k++)
			m_faces.push_back(ldp::Int3(f.vertex_index[0], f.vertex_index[k+1], f.vertex_index[k+2]));
	}
}

void MeshModel::drawPlane()const
{
	m_VertexVBO.BindVBO();
	glVertexPointer(3, GL_FLOAT, sizeof(float)* 3, 0);
	m_VertexVBO.UnbindVBO();

	glEnableClientState(GL_VERTEX_ARRAY);

	glEnable(GL_TEXTURE_2D);
	glDrawElements(GL_TRIANGLES, m_faces.size() * 3, GL_UNSIGNED_INT, m_faces.data());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void MeshModel::drawShade()const
{
	m_VertexVBO.BindVBO();
	glVertexPointer(3, GL_FLOAT, sizeof(float)* 3, 0);
	m_VertexVBO.UnbindVBO();

	m_NormalVBO.BindVBO();
	glNormalPointer(GL_FLOAT, sizeof(float)* 3, 0);
	m_NormalVBO.UnbindVBO();

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	glEnable(GL_TEXTURE_2D);
	glDrawElements(GL_TRIANGLES, m_faces.size() * 3, GL_UNSIGNED_INT, m_faces.data());

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
}