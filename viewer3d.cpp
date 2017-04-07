#include <GL\glew.h>
#include <GL\glut.h>
#include "Viewer3d.h"
#include "ldpMat\Quaternion.h"
#include "cloth\clothPiece.h"
#include "cloth\graph\Graph.h"
#include "cloth\SmplManager.h"
#include "cloth\TransformInfo.h"
#include "cloth\graph\GraphsSewing.h"
#include "Renderable\ObjMesh.h"
#include "CmlShadowMap\MeshRender.h"
#include "CmlShadowMap\GPUBuffers.h"
#pragma region --mat_utils

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

static int CheckGLError(const string& file, int line)
{
	GLenum glErr;
	int    retCode = 0;

	glErr = glGetError();
	while (glErr != GL_NO_ERROR)
	{
		const GLubyte* sError = glewGetErrorString(glErr);

		if (sError)
			cout << "GL Error #" << glErr << "(" << gluErrorString(glErr) << ") " << " in File " << file.c_str() << " at line: " << line << endl;
		else
			cout << "GL Error #" << glErr << " (no message available)" << " in File " << file.c_str() << " at line: " << line << endl;

		retCode = 1;
		glErr = glGetError();
	}
	return retCode;
}

#define CHECK_GL_ERROR() CheckGLError(__FILE__, __LINE__)

#pragma endregion

Viewer3d::Viewer3d(QWidget *parent)
: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	setMouseTracking(true);
	m_eventHandles.resize((size_t)Abstract3dEventHandle::ProcessorTypeEnd, nullptr);
	for (size_t i = (size_t)Abstract3dEventHandle::ProcessorTypeGeneral;
		i < (size_t)Abstract3dEventHandle::ProcessorTypeEnd; i++)
	{
		m_eventHandles[i] = std::shared_ptr<Abstract3dEventHandle>(
			Abstract3dEventHandle::create(Abstract3dEventHandle::ProcessorType(i), this));
	}
	setEventHandleType(Abstract3dEventHandle::ProcessorTypeGeneral);
}

Viewer3d::~Viewer3d()
{

}

void Viewer3d::init(ldp::ClothManager* clothManager, ClothDesigner* ui)
{
	m_clothManager = clothManager;
	m_mainUI = ui;
	getEventHandle(getEventHandleType())->resetSelection();
	resetCamera();
}

void Viewer3d::resetCamera()
{
	m_camera.setModelViewMatrix(ldp::Mat4f().eye());
	m_camera.setPerspective(60, float(width()) / float(height()), 0.1, 10000);
	ldp::Float3 c = 0.f;
	float l = 1.f;
	if (m_clothManager)
	{
		ldp::Float3 bmin, bmax;
		getModelBound(bmin, bmax);
		c = (bmax + bmin) / 2.f;
		l = (bmax - bmin).length();
	}
	m_camera.lookAt(ldp::Float3(0, l, 0)*2 + c, c, ldp::Float3(0, 0, 1));
	m_camera.arcballSetCenter(c);
}

void Viewer3d::initializeGL()
{
	glewInit();

	m_showType = Renderable::SW_F | Renderable::SW_SMOOTH | Renderable::SW_TEXTURE
		| Renderable::SW_LIGHTING | Renderable::SW_SKELETON;

	resetCamera();

	// fbo
	QGLFramebufferObjectFormat fmt;
	fmt.setAttachment(QGLFramebufferObject::CombinedDepthStencil);
	m_fbo = new QGLFramebufferObject(width(), height(), fmt);
	if (!m_fbo->isValid())
		printf("error: invalid depth fbo!\n");

	// shadow map
	initShadowMap();

	CHECK_GL_ERROR();
}

void Viewer3d::resizeGL(int w, int h)
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

	m_GPUBuffers->ResetFrameSize(w, h);
}

void Viewer3d::timerEvent(QTimerEvent* ev)
{
	//updateGL();
}

void Viewer3d::paintGL()
{
	QGLFunctions func(QGLContext::currentContext());

	// we first render for selection
	renderSelectionOnFbo();

	// then we do formal rendering=========================
	glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m_camera.apply();

	int showType = m_showType;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	// show cloth simulation=============================
	if (m_showShadow)
	{
		renderWithShadowMap();
	} // end if show shadow
	else
	{
		glEnable(GL_TEXTURE_2D);
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_FRONT_AND_BACK);
		//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
		glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1, 1);
		if (m_clothManager)
		{
			m_clothManager->bodyMesh()->render(showType);
			for (int i = 0; i < m_clothManager->numClothPieces(); i++)
			{
				const auto& piece = m_clothManager->clothPiece(i);
				if (piece->mesh3d().material_list.size())
				{
					if (piece->graphPanel().isHighlighted())
						piece->mesh3d().material_list[0].diff = ldp::Float3(0.0, 0.6, 0.8);
					else if (piece->graphPanel().isSelected())
						piece->mesh3d().material_list[0].diff = ldp::Float3(0.8, 0.6, 0);
					else
						piece->mesh3d().material_list[0].diff = ldp::Float3(1, 1, 1);
				}
				piece->mesh3d().render(showType);
			}
			if (isSmplMode() && m_clothManager->bodySmplManager())
			{
				auto T = m_clothManager->getBodyMeshTransform().transform();
				glPushMatrix();
				glMultMatrixf(T.ptr());
				int stype = showType & Renderable::SW_SKELETON;
				m_clothManager->bodySmplManager()->render(stype);
				glPopMatrix();
			} // end if smpl mode
		} // end if m_clothManager
		renderStitches();
		renderTrackBall(false);
		renderDragBox();
		renderGroupPlane();
	} // end else not show shadow

	glPopAttrib();
}

void Viewer3d::renderGroupPlane()
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glDisable(GL_LIGHTING);
	const float bg = -3, ed = 3, grid=0.1;
	glColor3f(0.4, 0.4, 0.4);
	glLineWidth(1);
	glBegin(GL_LINES);
	for (float x = bg; x <= ed; x += grid)
	{
		glVertex2f(x, bg);
		glVertex2f(x, ed);
	}
	for (float y = bg; y < ed; y += grid)
	{
		glVertex2f(bg, y);
		glVertex2f(ed, y);
	}
	glEnd();
	glLineWidth(2);
	glColor3f(0.5, 0.5, 0.5);
	glBegin(GL_LINES);
	glVertex2f(0, bg);
	glVertex2f(0, ed);
	glVertex2f(bg, 0);
	glVertex2f(ed, 0);
	glEnd();

	glPopAttrib();
}

void Viewer3d::renderStitches()
{
	if (!m_clothManager)
		return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	for (int is = 0; is < m_clothManager->numStitches(); is++)
	{
		auto vp = m_clothManager->getStitchPos(is);
		ldp::GraphsSewing::SewingType type = (ldp::GraphsSewing::SewingType)m_clothManager->getStitchType(is);
		if (type == ldp::GraphsSewing::SewingTypeStitch)
			glColor3f(0, 1, 0);
		else if (type == ldp::GraphsSewing::SewingTypePosition)
			glColor3f(1, 0, 0);
		else
			glColor3f(1, 1, 1);
		glVertex3fv(vp.first.ptr());
		glVertex3fv(vp.second.ptr());
	} // end for is
	glEnd();

	//// ldp debug
	//if (m_clothManager->numClothPieces())
	//{
	//	glPointSize(5);
	//	glBegin(GL_POINTS);
	//	glColor3f(1, 0, 0);
	//	glVertex3fv(m_clothManager->getVertexByGlobalId(1136).ptr());
	//	glColor3f(1, 1, 0);
	//	glVertex3fv(m_clothManager->getVertexByGlobalId(1137).ptr());
	//	glColor3f(1, 0, 1);
	//	glVertex3fv(m_clothManager->getVertexByGlobalId(1138).ptr());
	//	glEnd();
	//}

	glPopAttrib();
}

void Viewer3d::renderSelectionOnFbo()
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

	if (isSmplMode() && m_clothManager->bodySmplManager())
	{
		auto T = m_clothManager->getBodyMeshTransform().transform();
		glPushMatrix();
		glMultMatrixf(T.ptr());
		int stype = m_showType & Renderable::SW_SKELETON;
		m_clothManager->bodySmplManager()->renderForSelection(stype, SmplJointIndex);
		glPopMatrix();
	} // end if smpl mode
	else
		renderMeshForSelection();
	renderTrackBall(true);

	m_fboImage = m_fbo->toImage();

	glPopAttrib();
	m_fbo->release();
}

void Viewer3d::renderMeshForSelection()
{
	if (m_clothManager == nullptr)
		return;
	int curIdx = FaceIndex;
	auto mesh = m_clothManager->bodyMesh();
	glBegin(GL_TRIANGLES); 
	for (const auto& f : mesh->face_list)
	{
		for (int k = 0; k < 3; k++)
		{
			glColor4fv(selectIdToColor(curIdx).ptr());
			glVertex3fv(mesh->vertex_list[f.vertex_index[k]].ptr());
		}
		curIdx++;
	}
	for (int iMesh = 0; iMesh < m_clothManager->numClothPieces(); iMesh++)
	{
		auto mesh = &m_clothManager->clothPiece(iMesh)->mesh3d();
		for (const auto& f : mesh->face_list)
		{
			for (int k = 0; k < 3; k++)
			{
				glColor4fv(selectIdToColor(curIdx).ptr());
				glVertex3fv(mesh->vertex_list[f.vertex_index[k]].ptr());
			}
			curIdx++;
		}
	}
	glEnd();
}

void Viewer3d::mousePressEvent(QMouseEvent *ev)
{
	setFocus();
	m_lastPos = ev->pos();
	m_buttons = ev->buttons();

	m_currentEventHandle->mousePressEvent(ev);

	updateGL();
}

void Viewer3d::keyPressEvent(QKeyEvent*ev)
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
	case Qt::Key_L:
		if (m_shadowMapInitialized)
			m_showShadow = !m_showShadow;
		break;
	}
	m_currentEventHandle->keyPressEvent(ev);
	updateGL();
}

void Viewer3d::keyReleaseEvent(QKeyEvent*ev)
{
	m_currentEventHandle->keyReleaseEvent(ev);
	updateGL();
}

void Viewer3d::mouseReleaseEvent(QMouseEvent *ev)
{
	m_currentEventHandle->mouseReleaseEvent(ev);

	// clear buttons
	m_buttons = Qt::NoButton;
	updateGL();
}

void Viewer3d::mouseMoveEvent(QMouseEvent*ev)
{
	m_currentEventHandle->mouseMoveEvent(ev);

	// backup last position
	m_lastPos = ev->pos();
	updateGL();
}

void Viewer3d::mouseDoubleClickEvent(QMouseEvent *ev)
{
	m_currentEventHandle->mouseDoubleClickEvent(ev);

	updateGL();
}

void Viewer3d::simulateWheeling(int delta)
{	
	float s = 1.2;
	if (delta < 0)
		s = 1.f / s;
	ldp::Float3 c = m_camera.getLocation();
	ldp::Float3 c0 = m_camera.arcballGetCenter();
	m_camera.setLocation((c - c0)*s + c0);
}
void Viewer3d::wheelEvent(QWheelEvent*ev)
{
	m_currentEventHandle->wheelEvent(ev);

	updateGL();
}

Abstract3dEventHandle::ProcessorType Viewer3d::getEventHandleType()const
{
	return m_currentEventHandle->type();
}

void Viewer3d::setEventHandleType(Abstract3dEventHandle::ProcessorType type)
{
	if (m_currentEventHandle)
		m_currentEventHandle->handleLeave();
	m_currentEventHandle = m_eventHandles[size_t(type)].get();
	m_currentEventHandle->handleEnter();
	setCursor(m_currentEventHandle->cursor());
}

const Abstract3dEventHandle* Viewer3d::getEventHandle(Abstract3dEventHandle::ProcessorType type)const
{
	return m_eventHandles[size_t(type)].get();
}

Abstract3dEventHandle* Viewer3d::getEventHandle(Abstract3dEventHandle::ProcessorType type)
{
	return m_eventHandles[size_t(type)].get();
}

void Viewer3d::beginDragBox(QPoint p)
{
	m_dragBoxBegin = p;
	m_isDragBox = true;
}

void Viewer3d::endDragBox()
{
	m_isDragBox = false;
}

void Viewer3d::renderDragBox()
{
	if (!m_isDragBox)
		return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	float l = camera().getFrustumLeft();
	float r = camera().getFrustumRight();
	float t = camera().getFrustumTop();
	float b = camera().getFrustumBottom();
	float x0 = std::min(m_dragBoxBegin.x(), m_lastPos.x()) / float(width()) * (r - l) + l;
	float x1 = std::max(m_dragBoxBegin.x(), m_lastPos.x()) / float(width()) * (r - l) + l;
	float y0 = std::min(m_dragBoxBegin.y(), m_lastPos.y()) / float(height()) * (b - t) + t;
	float y1 = std::max(m_dragBoxBegin.y(), m_lastPos.y()) / float(height()) * (b - t) + t;

	glDisable(GL_STENCIL_TEST);
	glColor3f(0, 1, 0);
	glLineWidth(2);
	//glEnable(GL_LINE_STIPPLE);
	glLineStipple(0xAAAA, 1);
	glBegin(GL_LINE_LOOP);
	glVertex2f(x0, y0);
	glVertex2f(x0, y1);
	glVertex2f(x1, y1);
	glVertex2f(x1, y0);
	glEnd();

	glPopAttrib();
}

void Viewer3d::getModelBound(ldp::Float3& bmin, ldp::Float3& bmax)const
{
	bmin = FLT_MAX;
	bmax = -FLT_MAX;
	if (m_clothManager)
	{
		bmin = m_clothManager->bodyMesh()->boundingBox[0];
		bmax = m_clothManager->bodyMesh()->boundingBox[1];
	}
}

void Viewer3d::beginTrackBall(TrackBallMode mode, ldp::Float3 p, ldp::Mat3f R, float scale)
{
	m_trackBallPos = p;
	m_trackBallR = R;
	m_trackBallScale = scale;
	m_trackBallMode = mode;
	m_activeTrackBallAxis = -1;
	m_hoverTrackBallAxis = -1;
}

void Viewer3d::rotateTrackBall(ldp::Mat3d R)
{
	m_trackBallR = R * m_trackBallR;
}

void Viewer3d::translateTrackBall(ldp::Double3 t)
{
	m_trackBallPos += t;
}

void Viewer3d::endTrackBall()
{
	m_trackBallMode = TrackBall_None;
	m_activeTrackBallAxis = -1;
	m_hoverTrackBallAxis = -1;
}

int Viewer3d::fboRenderedIndex(QPoint p)const
{
	if (m_fboImage.rect().contains(p))
	{
		QRgb c = m_fboImage.pixel(p);
		return colorToSelectId(ldp::Float4(qRed(c), qGreen(c), qBlue(c), qAlpha(c))/255.f);
	}
	return 0;
}

void Viewer3d::renderTrackBall(bool indexMode)
{
	if (m_trackBallMode == TrackBall_None)
		return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);
	if (!indexMode)
	{
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	glPushMatrix();
	glTranslatef(m_trackBallPos[0], m_trackBallPos[1], m_trackBallPos[2]);
	ldp::Mat4f M = ldp::Mat4f().eye();
	M.setRotationPart(m_trackBallR/powf(m_trackBallR.det(), 1.f/3.f));
	glMultMatrixf(M.ptr());

	// x axis
	glColor3f(1, 0, 0);
	if (m_activeTrackBallAxis == TrackBallIndex_X || (m_hoverTrackBallAxis == 
		TrackBallIndex_X && m_activeTrackBallAxis < 0))
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(TrackBallIndex_X).ptr());
	glMultMatrixf(get_z2x_rot().ptr());
	if (m_trackBallMode == TrackBall_Rot)
		glutSolidTorus(m_trackBallScale * 0.03, m_trackBallScale, 16, 128);
	else if (m_trackBallMode == TrackBall_Trans)
		solid_axis(m_trackBallScale * 0.03, m_trackBallScale);
	else if (m_trackBallMode == TrackBall_Cylinder)
	{
		solid_axis(m_trackBallScale * 0.03, m_trackBallScale);
	}
	glMultMatrixf(get_z2x_rot().trans().ptr());

	// y axis
	glColor3f(0, 1, 0);
	if (m_activeTrackBallAxis == TrackBallIndex_Y || (m_hoverTrackBallAxis == 
		TrackBallIndex_Y && m_activeTrackBallAxis < 0))
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(TrackBallIndex_Y).ptr());
	glMultMatrixf(get_z2y_rot().ptr());
	if (m_trackBallMode == TrackBall_Rot)
		glutSolidTorus(m_trackBallScale * 0.03, m_trackBallScale, 16, 128);
	else if (m_trackBallMode == TrackBall_Trans)
		solid_axis(m_trackBallScale * 0.03, m_trackBallScale);
	else if (m_trackBallMode == TrackBall_Cylinder)
	{
		glutSolidTorus(m_trackBallScale * 0.03, m_trackBallScale, 16, 128);
	}
	glMultMatrixf(get_z2y_rot().trans().ptr());

	// z axis
	glColor3f(0, 0, 1);
	if (m_activeTrackBallAxis == TrackBallIndex_Z || (m_hoverTrackBallAxis == 
		TrackBallIndex_Z && m_activeTrackBallAxis < 0))
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(TrackBallIndex_Z).ptr());
	if (m_trackBallMode == TrackBall_Rot)
		glutSolidTorus(m_trackBallScale * 0.03, m_trackBallScale, 16, 128);
	else if (m_trackBallMode == TrackBall_Trans)
		solid_axis(m_trackBallScale * 0.03, m_trackBallScale);
	else if (m_trackBallMode == TrackBall_Cylinder)
	{
		glutSolidTorus(m_trackBallScale * 0.03, m_trackBallScale, 16, 128);
	}

	// sphere
	if (!indexMode && m_trackBallMode == TrackBall_Rot)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glColor4f(0.6, 0.6, 0.6, 0.5);
		glutSolidSphere(m_trackBallScale, 32, 32);
		glDisable(GL_BLEND);
	}

	glPopMatrix();
	glPopAttrib();
}

////////////////////////////////////////////////////////////////////////
inline int lightDivUp(int a, int b)
{
	return (a + b - 1) / b;
}

void Viewer3d::initShadowMap()
{
	QString lightFile = "Env/ironman_op50.dir";
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

void Viewer3d::renderWithShadowMap()
{
	if (!m_shadowMapInitialized)
		return;

	glClearColor(0., 0., 0., 0.);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_MULTISAMPLE);

	m_camera.apply();

	const float shadeLight = 0.4;
	const float shadeShadow = 1;
	const float shadeAmbient = 0.1 / m_lightNum;
	const float shadeDiffuse = 1.;
	const float shadeSpecular = 0.5;
	const ldp::Float3 clothColor(0.7, 0.9, 1);
	const ldp::Float3 bodyColor(1, 1, 1);

	if (m_clothManager)
	{
		// collect meshes
		std::vector<ObjMesh*> meshes;
		meshes.push_back(m_clothManager->bodyMesh());
		for (int i = 0; i < m_clothManager->numClothPieces(); i++)
		{
			const auto& piece = m_clothManager->clothPiece(i);
			if (piece->mesh3d().material_list.size())
			{
				if (piece->graphPanel().isHighlighted())
					piece->mesh3d().material_list[0].diff = ldp::Float3(0.0, 0.6, 0.8);
				else if (piece->graphPanel().isSelected())
					piece->mesh3d().material_list[0].diff = ldp::Float3(0.8, 0.6, 0);
				else
					piece->mesh3d().material_list[0].diff = ldp::Float3(1, 1, 1);
			}
			meshes.push_back(&piece->mesh3d());
		}

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

bool Viewer3d::loadLight(QString fileName)
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