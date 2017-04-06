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

	CHECK_GL_ERROR();
}

void ArcsimView::resizeGL(int w, int h)
{
	m_camera.setViewPort(0, w, 0, h);
	m_camera.setPerspective(m_camera.getFov(), float(w) / float(h), 
		m_camera.getFrustumNear(), m_camera.getFrustumFar());
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

	if (m_showBody)
		m_arcsimManager->getBodyMesh()->render(m_showType);
	m_arcsimManager->getClothMesh()->render(m_showType);

	// debug ldp: render texmap...
	if (g_dataholder.m_arcsim_show_texcoord)
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glViewport(0, 0, width() / 2, height() / 2);
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

