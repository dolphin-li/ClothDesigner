#include <GL\glew.h>
#include <GL\glut.h>
#include "TrainingImageRenderView.h"
#include "ldpMat\Quaternion.h"
#include "cloth\SmplManager.h"
#include "cloth\TransformInfo.h"
#include "Renderable\ObjMesh.h"
#include "cloth\clothManager.h"
#include <eigen\Dense>
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

TrainingImageRenderView::TrainingImageRenderView(QWidget *parent)
: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	setMouseTracking(true);
	m_buttons = Qt::MouseButton::NoButton;
	m_fbo = nullptr;
}

TrainingImageRenderView::~TrainingImageRenderView()
{

}

void TrainingImageRenderView::init(ldp::ClothManager* clothManager, ObjMesh* clothMeshLoaded)
{
	m_clothManager = clothManager;
	m_clothMeshLoaded = clothMeshLoaded;
}

void TrainingImageRenderView::resetCamera()
{
	m_camera.setViewPort(0, width(), 0, height());
	m_camera.setModelViewMatrix(ldp::Mat4f().eye());
	m_camera.setPerspective(60, float(width()) / float(height()), 0.1, 10000);
	ldp::Float3 c = 0.f;
	float l = 1.f;
	if (m_clothManager)
	{
		ldp::Float3 bmin, bmax;
		bmin = m_clothManager->bodyMesh()->getBoundingBox(0);
		bmax = m_clothManager->bodyMesh()->getBoundingBox(1);
		c = (bmax + bmin) / 2.f;
		l = (bmax - bmin).length();
	}
	m_camera.lookAt(ldp::Float3(0, l, 0)*1 + c, c, ldp::Float3(0, 0, 1));
	m_camera.arcballSetCenter(c);
}

void TrainingImageRenderView::initializeGL()
{
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_FRONT_AND_BACK);
	glEnable(GL_COLOR_MATERIAL);
	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

	m_showType = Renderable::SW_F | Renderable::SW_SMOOTH | Renderable::SW_TEXTURE
		| Renderable::SW_LIGHTING | Renderable::SW_SKELETON;

	resetCamera();

	// fbo
	QGLFramebufferObjectFormat fmt;
	fmt.setAttachment(QGLFramebufferObject::CombinedDepthStencil);
	m_fbo = new QGLFramebufferObject(width(), height(), fmt);
	if (!m_fbo->isValid())
		printf("error: invalid depth fbo!\n");

	CHECK_GL_ERROR();
}

void TrainingImageRenderView::resizeGL(int w, int h)
{
	m_camera.setViewPort(0, w, 0, h);
	m_camera.setPerspective(m_camera.getFov(), float(w) / float(h), 
		m_camera.getFrustumNear(), m_camera.getFrustumFar());

	// re-init fbo
	if (m_fbo)
		delete m_fbo;
	QGLFramebufferObjectFormat fmt;
	fmt.setAttachment(QGLFramebufferObject::CombinedDepthStencil);
	fmt.setMipmap(true);
	m_fbo = new QGLFramebufferObject(width(), height(), fmt);
}

void TrainingImageRenderView::paintGL()
{
	QGLFunctions func(QGLContext::currentContext());

	// we first render for selection
	renderSelectionOnFbo();

	// then we do formal rendering=========================
	glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m_camera.apply();

	// show cloth simulation=============================
	if (m_clothManager)
	{
		glColor3f(0.6, 0.8, 1.0);
		m_clothManager->bodyMesh()->render(m_showType);
		glColor3f(0.8, 0.8, 0.8);
		auto smpl = m_clothManager->bodySmplManager();
		if (smpl)
		{
			auto T = m_clothManager->getBodyMeshTransform().transform();
			glPushMatrix();
			glMultMatrixf(T.ptr());
			int stype = m_showType & Renderable::SW_SKELETON;
			m_clothManager->bodySmplManager()->render(stype);
			glPopMatrix();
		}
	}
	if (m_clothMeshLoaded)
	{
		glColor3f(0.8, 0.8, 0.8);
		m_clothMeshLoaded->render(m_showType);
	}
}

void TrainingImageRenderView::renderSelectionOnFbo()
{
	m_fbo->bind();
	glClearColor(0.f, 0.f, 0.f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	m_camera.apply();

	if (m_clothManager)
	{
		glColor4f(1, 0, 0, 1);
		m_clothManager->bodyMesh()->render(Renderable::SW_F | Renderable::SW_FLAT);
	}
	if (m_clothMeshLoaded)
	{
		glColor4f(0, 1, 0, 1);
		m_clothMeshLoaded->render(Renderable::SW_F | Renderable::SW_FLAT);
	}

	m_fboImage = m_fbo->toImage();

	glPopAttrib();
	m_fbo->release();
}

void TrainingImageRenderView::mousePressEvent(QMouseEvent *ev)
{
	setFocus();
	m_lastPos = ev->pos();
	m_buttons = ev->buttons();

	if (ev->buttons() == Qt::LeftButton)
		camera().arcballClick(ldp::Float2(ev->x(), ev->y()));

	updateGL();
}

void TrainingImageRenderView::keyPressEvent(QKeyEvent*ev)
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

void TrainingImageRenderView::keyReleaseEvent(QKeyEvent*ev)
{

	//
	updateGL();
}

void TrainingImageRenderView::mouseReleaseEvent(QMouseEvent *ev)
{

	// clear buttons
	m_buttons = Qt::NoButton;
	updateGL();
}

void TrainingImageRenderView::mouseMoveEvent(QMouseEvent*ev)
{
	if (ev->buttons() == Qt::LeftButton)
		camera().arcballDrag(ldp::Float2(ev->x(), ev->y()));
	if (ev->buttons() == Qt::MidButton && m_clothManager)
	{
		QPoint dif = ev->pos() - lastMousePos();
		ldp::Float3 bmin, bmax;
		bmin = m_clothManager->bodyMesh()->getBoundingBox(0);
		bmax = m_clothManager->bodyMesh()->getBoundingBox(1);
		float len = (bmax - bmin).length() / sqrt(3.f);
		ldp::Float3 t(-(float)dif.x() / width(), (float)dif.y() / height(), 0);
		camera().translate(t * len);
		camera().arcballSetCenter((bmin + bmax) / 2.f + t * len);
	}

	// backup last position
	m_lastPos = ev->pos();
	updateGL();
}

void TrainingImageRenderView::mouseDoubleClickEvent(QMouseEvent *ev)
{
	if (ev->button() == Qt::MouseButton::MiddleButton)
		resetCamera();

	//
	updateGL();
}

void TrainingImageRenderView::wheelEvent(QWheelEvent*ev)
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

void TrainingImageRenderView::generateDistMap_x9(std::vector<QImage>& distMaps)
{
	distMaps.clear();
	if (m_clothManager == nullptr)
		throw std::exception("distMap: not initialized!");
	auto smpl = m_clothManager->bodySmplManager();
	if (smpl == nullptr)
		throw std::exception("distMap: no smpl model!");
	auto T = m_clothManager->getBodyMeshTransform().transform();

	// 1. project 3d joint nodes into image space
	std::vector<ldp::Float2> nodes2d;
	for (int iNode = 0; iNode < smpl->numPoses(); iNode++)
	{
		auto p = T.getRotationPart() * convert(smpl->getCurNodeCenter(iNode)) + T.getTranslationPart();
		p = m_camera.getScreenCoords(p);
		nodes2d.push_back(ldp::Float2(p[0], height()-1-p[1]));
	} // end for iNode

	// 2. compute 9 bone segments
	std::vector<ldp::Float2> bones;
	// bone 0
	bones.push_back(nodes2d[17]);
	bones.push_back(nodes2d[19]);
	// bone 1
	bones.push_back(nodes2d[19]);
	bones.push_back(nodes2d[21]);
	// bone 2
	bones.push_back(nodes2d[16]);
	bones.push_back(nodes2d[18]);
	// bone 3
	bones.push_back(nodes2d[18]);
	bones.push_back(nodes2d[20]);
	// bone 4
	bones.push_back(nodes2d[2]);
	bones.push_back(nodes2d[5]);
	// bone 5
	bones.push_back(nodes2d[5]);
	bones.push_back(nodes2d[8]);
	// bone 6
	bones.push_back(nodes2d[1]);
	bones.push_back(nodes2d[4]);
	// bone 7
	bones.push_back(nodes2d[4]);
	bones.push_back(nodes2d[7]);
	// bone 8
	bones.push_back((nodes2d[12] + nodes2d[9])*0.5f);
	bones.push_back(nodes2d[0]);

	// distance used to normalize the computed distance map
	const float dist_max = (bones[8 * 2] - bones[8 * 2 + 1]).length();

	// 3. compute 9 dist maps + 1 visualization map
	distMaps.resize(1 + 9);
	for (size_t iMap = 0; iMap < distMaps.size(); iMap++)
	{
		// 3.1 compute 1 visualization map
		auto& D = distMaps[iMap];
		D = QImage(width(), height(), QImage::Format_ARGB32);

		// initialize map, filling those masked.
		for (int y = 0; y < height(); y++)
		{
			const QRgb* f_scan = (const QRgb*)m_fboImage.scanLine(y);
			QRgb* D_scan = (QRgb*)D.scanLine(y);
			for (int x = 0; x < width(); x++)
			{
				if (qGreen(f_scan[x]) == 255)
					D_scan[x] = qRgb(128, 128, 128);
				else
					D_scan[x] = qRgb(0, 0, 0);
			} // end for x
		} // end for y

		if (iMap == 0)
		{
			QPainter painter(&D);
			QPen pen(QColor(255, 255, 255));
			pen.setWidth(5);
			for (size_t iBone = 0; iBone < bones.size(); iBone += 2)
			{
				auto p1 = bones[iBone];
				auto p2 = bones[iBone + 1];
				int idx = (iBone + 2) * 255 / bones.size();
				pen.setColor(QColor(idx, idx, idx));
				painter.setPen(pen);
				painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]));
			} // end for iBone
		} // end if iMap == 0
		else
		{
			const int iBone = iMap - 1;
			ldp::Float2 bone_s = bones[iBone * 2];
			ldp::Float2 bone_e = bones[iBone * 2 + 1];
			for (int y = 0; y < height(); y++)
			{
				QRgb* D_scan = (QRgb*)D.scanLine(y);
				for (int x = 0; x < width(); x++)
				{
					if (qGreen(D_scan[x]) == 0)
						continue;
					float dist = ldp::pointSegDistance(ldp::Float2(x, y), bone_s, bone_e);
					dist /= dist_max;
					int idx = std::min(255, int(dist*255.f));
					D_scan[x] = qRgb(idx, idx, idx);
				} // end for x
			} // end for y
		} // else if iMap != 0
	} // end for iMap
}

