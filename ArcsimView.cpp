#include <GL\glew.h>
#include <GL\glut.h>
#include "ArcsimView.h"
#include "ldpMat\Quaternion.h"
#include "cloth\SmplManager.h"
#include "cloth\TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "cloth\clothManager.h"
#include "arcsim\ArcSimManager.h"
#include <eigen\Dense>
#include "global_data_holder.h"
#include "Shader\ShaderManager.h"
#include "CmlShadowMap\MeshRender.h"
#include "CmlShadowMap\GPUBuffers.h"

#pragma region --mat_utils

inline Eigen::Matrix3d convert(ldp::Mat3d A)
{
	Eigen::Matrix3d B;
	for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		B(i, j) = A(i, j);
	return B;
}

inline ldp::Mat3d convert(Eigen::Matrix3d A)
{
	ldp::Mat3d B;
	for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		B(i, j) = A(i, j);
	return B;
}

inline Eigen::Vector3d convert(ldp::Double3 v)
{
	return Eigen::Vector3d(v[0], v[1], v[2]);
}

inline ldp::Double3 convert(Eigen::Vector3d v)
{
	return ldp::Double3(v[0], v[1], v[2]);
}

inline ldp::Mat3f angles2rot(ldp::Float3 v)
{
	float theta = v.length();
	if (theta == 0)
		return ldp::Mat3f().eye();
	v /= theta;
	return ldp::QuaternionF().fromAngleAxis(theta, v).toRotationMatrix3();
}

inline ldp::Float3 rot2angles(ldp::Mat3f R)
{
	ldp::QuaternionF q;
	q.fromRotationMatrix(R);
	ldp::Float3 v;
	float ag;
	q.toAngleAxis(v, ag);
	v *= ag;
	return v;
}

static GLUquadric* get_quadric()
{
	static GLUquadric* q = gluNewQuadric();
	return q;
}

static ldp::Mat4f get_z2x_rot()
{
	static ldp::Mat4f R = ldp::QuaternionF().fromRotationVecs(ldp::Float3(0, 0, 1),
		ldp::Float3(1, 0, 0)).toRotationMatrix();
	return R;
}

static ldp::Mat4f get_z2y_rot()
{
	static ldp::Mat4f R = ldp::QuaternionF().fromRotationVecs(ldp::Float3(0, 0, 1),
		ldp::Float3(0, 1, 0)).toRotationMatrix();
	return R;
}

static void solid_axis(float base, float length)
{
	GLUquadric* q = get_quadric();
	gluCylinder(q, base, base, length, 32, 32);
	glTranslatef(0, 0, length);
	gluCylinder(q, base*2.5f, 0.f, length* 0.2f, 32, 32);
	glTranslatef(0, 0, -length);
}

inline int colorToSelectId(ldp::Float4 c)
{
	ldp::UInt4 cl = c*255.f;
	return (cl[0] << 24) + (cl[1] << 16) + (cl[2] << 8) + cl[3];
}

inline ldp::Float4 selectIdToColor(unsigned int id)
{
	int r = (id >> 24) & 0xff;
	int g = (id >> 16) & 0xff;
	int b = (id >> 8) & 0xff;
	int a = id & 0xff;
	return ldp::Float4(r, g, b, a) / 255.f;
}

static int CheckGLError(const std::string& file, int line)
{
	GLenum glErr;
	int    retCode = 0;

	glErr = glGetError();
	while (glErr != GL_NO_ERROR)
	{
		const GLubyte* sError = glewGetErrorString(glErr);

		if (sError)
			std::cout << "GL Error #" << glErr << "(" << gluErrorString(glErr) << ") " 
			<< " in File " << file.c_str() << " at line: " << line << std::endl;
		else
			std::cout << "GL Error #" << glErr << " (no message available)" << " in File " 
			<< file.c_str() << " at line: " << line << std::endl;

		retCode = 1;
		glErr = glGetError();
	}
	return retCode;
}

#define CHECK_GL_ERROR() CheckGLError(__FILE__, __LINE__)

#pragma endregion

ArcsimView::ArcsimView(QWidget *parent)
: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	setMouseTracking(true);
	m_shaderManager.reset(new CShaderManager());
}

ArcsimView::~ArcsimView()
{

}

void ArcsimView::init(arcsim::ArcSimManager* manager)
{
	m_arcsimManager = manager;
}

void ArcsimView::resetCamera()
{
	m_camera.setViewPort(0, width(), 0, height());
	m_camera.setModelViewMatrix(ldp::Mat4f().eye());
	m_camera.setPerspective(60, float(width()) / float(height()), 0.1, 10000);
	ldp::Float3 c = 0.f;
	float l = 1.f;
	if (m_arcsimManager)
	{
		ldp::Float3 bmin, bmax;
		m_arcsimManager->calcBoundingBox(bmin.ptr(), bmax.ptr());
		c = (bmax + bmin) / 2.f;
		l = (bmax - bmin).length();
	}
	m_camera.lookAt(ldp::Float3(0, l, 0)*1 + c, c, ldp::Float3(0, 0, 1));
	m_camera.arcballSetCenter(c);
}

void ArcsimView::initializeGL()
{
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_FRONT_AND_BACK);
	//glEnable(GL_COLOR_MATERIAL);
	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

	m_showType = Renderable::SW_F | Renderable::SW_SMOOTH | Renderable::SW_TEXTURE
		| Renderable::SW_LIGHTING | Renderable::SW_SKELETON;

	resetCamera();

	// shaders
	//m_shaderManager->create("shaders");

	// shadow map
	initShadowMap();

	CHECK_GL_ERROR();
}

void ArcsimView::resizeGL(int w, int h)
{
	m_camera.setViewPort(0, w, 0, h);
	m_camera.setPerspective(m_camera.getFov(), float(w) / float(h), 
		m_camera.getFrustumNear(), m_camera.getFrustumFar());

	// shadow map
	m_GPUBuffers->ResetFrameSize(w, h);
}

void ArcsimView::paintGL()
{
	QGLFunctions func(QGLContext::currentContext());

	// then we do formal rendering=========================
	glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m_camera.apply();

	if (m_arcsimManager == nullptr)
		return;
	arcsim::Simulation* sim = m_arcsimManager->getSimulator();
	if (sim == nullptr)
		return;

	if (m_showShadow)
	{
		renderWithShadowMap();
	}
	else
	{
		//m_shaderManager->bind(CShaderManager::phong);
		if (m_showBody)
			m_arcsimManager->getBodyMesh()->render(m_showType);
		m_arcsimManager->getClothMesh()->render(m_showType);
		//m_shaderManager->unbind();
	}

	// debug ldp: render texmap...
	if (g_dataholder.m_arcsim_show_texcoord)
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glViewport(0, 0, width() / 4, height() / 4);
		auto cloth = m_arcsimManager->getClothMesh();
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glBegin(GL_TRIANGLES);
		for (const auto& f : cloth->face_list)
		{
			glVertex2fv(cloth->vertex_texture_list[f.texture_index[0]].ptr());
			glVertex2fv(cloth->vertex_texture_list[f.texture_index[1]].ptr());
			glVertex2fv(cloth->vertex_texture_list[f.texture_index[2]].ptr());
		}
		glEnd();
		glPopAttrib();
	}
}

void ArcsimView::mousePressEvent(QMouseEvent *ev)
{
	setFocus();
	m_lastPos = ev->pos();

	if (ev->buttons() == Qt::LeftButton)
		camera().arcballClick(ldp::Float2(ev->x(), ev->y()));

	updateGL();
}

void ArcsimView::keyPressEvent(QKeyEvent*ev)
{
	switch (ev->key())
	{
	default:
		break;
	case Qt::Key_E:
		m_showType ^= Renderable::SW_E;
		break;
	case Qt::Key_F:
		m_showType ^= Renderable::SW_F;
		break;
	case Qt::Key_T:
		m_showType ^= Renderable::SW_TEXTURE;
		break;
	case Qt::Key_V:
		m_showType ^= Renderable::SW_V;
		break;
	case Qt::Key_L:
		if (m_shadowMapInitialized)
			m_showShadow = !m_showShadow;
		break;
	case Qt::Key_S:
		m_showType ^= Renderable::SW_SMOOTH;
		m_showType ^= Renderable::SW_FLAT;
		break;
	}
	updateGL();
}

void ArcsimView::keyReleaseEvent(QKeyEvent*ev)
{

	//
	updateGL();
}

void ArcsimView::mouseReleaseEvent(QMouseEvent *ev)
{

}

void ArcsimView::mouseMoveEvent(QMouseEvent*ev)
{
	if (ev->buttons() == Qt::LeftButton)
		camera().arcballDrag(ldp::Float2(ev->x(), ev->y()));
	if (ev->buttons() == Qt::MidButton && m_arcsimManager)
	{
		QPoint dif = ev->pos() - m_lastPos;
		ldp::Float3 bmin, bmax;
		m_arcsimManager->calcBoundingBox(bmin.ptr(), bmax.ptr());
		float len = (bmax - bmin).length() / sqrt(3.f);
		ldp::Float3 t(-(float)dif.x() / width(), (float)dif.y() / height(), 0);
		camera().translate(t * len);
		camera().arcballSetCenter((bmin + bmax) / 2.f + t * len);
	}

	// backup last position
	m_lastPos = ev->pos();
	updateGL();
}

void ArcsimView::mouseDoubleClickEvent(QMouseEvent *ev)
{
	if (ev->button() == Qt::MouseButton::MiddleButton)
		resetCamera();

	//
	updateGL();
}

void ArcsimView::wheelEvent(QWheelEvent*ev)
{
	float s = 1.2;
	if (ev->delta() < 0)
		s = 1.f / s;

	ldp::Camera& cam = camera();
	ldp::Float3 c = cam.getLocation();
	ldp::Float3 c0 = cam.arcballGetCenter();
	cam.setLocation((c - c0)*s + c0);
	//
	updateGL();
}

////////////////////////////////////////////////////////////////////////
inline int lightDivUp(int a, int b)
{
	return (a + b - 1) / b;
}

void ArcsimView::initShadowMap()
{
	QString lightFile = "Env/ironman_op25.dir";
	if (!loadLight(lightFile))
	{
		printf("lighting file not found: %s\n", lightFile.toStdString().c_str());
		return;
	}
	for (auto& ld : m_lightDirections)
	{
		ldp::Mat3f R = ldp::QuaternionF().fromAngleAxis(
			-ldp::PI_S / 2.f,
			ldp::Float3(1, 0, 0)).toRotationMatrix3();
		ld = R * ld;
	}


	m_MeshRender.reset(new MeshRender());
	if (!m_MeshRender->Initialize())
		return;
	m_GPUBuffers.reset(new GPUBuffers());
	m_GPUBuffers->SetShadowSize(1024 * 2, 1024 * 2);
	m_GPUBuffers->SetFrameSize(width(), height());
	m_GPUBuffers->CreateTextures();
	m_GPUBuffers->CreateFBOs();
	m_GPUBuffers->AttachFBOTextures();
	m_shadowMapInitialized = true;
}

void ArcsimView::renderWithShadowMap()
{
	if (!m_shadowMapInitialized)
		return;

	glClearColor(0., 0., 0., 0.);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_MULTISAMPLE);

	m_camera.apply();

	const float shadeLight = 0.2;
	const float shadeShadow = 1;
	const float shadeAmbient = 0.1 / m_lightNum;
	const float shadeDiffuse = 1;
	const float shadeSpecular = 0.5;
	const ldp::Float3 clothColor(0.7, 0.9, 1);
	const ldp::Float3 bodyColor(1, 1, 1);

	if (m_arcsimManager)
	{
		// collect meshes
		std::vector<ObjMesh*> meshes;
		meshes.push_back(m_arcsimManager->getBodyMesh());
		meshes.push_back(m_arcsimManager->getClothMesh());
		m_MeshRender->updateGeometry(meshes.data() + 1, meshes.size() - 1, meshes.data(), 1);

		for (int lightI = 0; lightI < m_lightNum; ++lightI)
			m_lightShadeColors[lightI] = m_lightOriginalColors[lightI] * shadeLight;

		std::vector<ldp::Float3> lightDirsRot = m_lightDirections;
		for (auto& ld : lightDirsRot)
			ld = m_camera.getModelViewMatrix().getRotationPart().inv() * ld;

		// render each passes
		for (int passI = 0; passI < lightDivUp(m_lightNum, MAX_LIGHTS_PASS); passI++){
			m_MeshRender->PrepareLightMatrix(
				meshes[0]->boundingBox[0], meshes[0]->boundingBox[1],
				lightDirsRot.data() + passI * MAX_LIGHTS_PASS,
				m_LightModelViewMatrix, m_LightProjectionMatrix);

			m_MeshRender->RenderShadowMap(
				m_LightModelViewMatrix,
				m_LightProjectionMatrix,
				m_GPUBuffers->GetShadowFBO());

			for (int lightI = 0; lightI < MAX_LIGHTS_PASS; lightI++)
				m_LightModelViewProjectionMatrix[lightI] =
				m_LightProjectionMatrix[lightI] * m_LightModelViewMatrix[lightI];

			m_MeshRender->RenderMesh(
				m_camera.getLocation(),
				m_lightShadeColors.data() + passI * MAX_LIGHTS_PASS,
				lightDirsRot.data() + passI * MAX_LIGHTS_PASS,
				m_LightModelViewMatrix,
				m_LightModelViewProjectionMatrix,
				shadeAmbient, shadeDiffuse, shadeSpecular, shadeShadow,
				clothColor, bodyColor,
				m_GPUBuffers->GetFrameFBO(),
				m_GPUBuffers->GetShadowFBO().GetColorTexture(),
				m_showType);

			m_MeshRender->RenderComposite(m_GPUBuffers->GetFrameFBO().GetColorTexture());
		}
	} // end if clothManager
}

bool ArcsimView::loadLight(QString fileName)
{
	QFile lightFile(fileName);
	if (!lightFile.open(QIODevice::ReadOnly | QIODevice::Text))
		return false;
	QTextStream lightStream(&lightFile);
	lightStream >> m_lightNum;
	m_lightOriginalColors.clear();
	m_lightShadeColors.clear();
	m_lightDirections.clear();
	m_lightOriginalColors.resize(lightDivUp(m_lightNum, MAX_LIGHTS_PASS)*MAX_LIGHTS_PASS);
	m_lightShadeColors.resize(lightDivUp(m_lightNum, MAX_LIGHTS_PASS)*MAX_LIGHTS_PASS);
	m_lightDirections.resize(lightDivUp(m_lightNum, MAX_LIGHTS_PASS)*MAX_LIGHTS_PASS, ldp::Float3(0, 0, -1));
	float lightIntensity = 0.f;
	for (int lightI = 0; lightI < m_lightNum; ++lightI){
		lightStream >> m_lightOriginalColors[lightI][0] >> m_lightOriginalColors[lightI][1] >> m_lightOriginalColors[lightI][2];
		lightStream >> m_lightDirections[lightI][0] >> m_lightDirections[lightI][1] >> m_lightDirections[lightI][2];
		lightStream >> lightIntensity;
		m_lightOriginalColors[lightI] *= lightIntensity;
		m_lightShadeColors[lightI] = m_lightOriginalColors[lightI];
		m_lightDirections[lightI].normalize();
		m_lightDirections[lightI] *= -1.f;
	}
	lightFile.close();
	return true;
}

