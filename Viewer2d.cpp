#include <GL\glew.h>
#include <GL\glut.h>
#include "Viewer2d.h"
#include "ldpMat\Quaternion.h"
#include "cloth\clothPiece.h"
#include "cloth\graph\Graph.h"
#include "cloth\graph\AbstractGraphCurve.h"
#include "cloth\graph\GraphPoint.h"
#include "cloth\graph\GraphLoop.h"
#include "cloth\graph\GraphsSewing.h"
#include "Renderable\ObjMesh.h"

#pragma region --mat_utils

const static float BODY_Z = -50;
const static float BACK_Z = -10;
const static float MESH_Z = -1;
const static float EDGE_Z = 0;
const static float SEW_EDGE_Z = 1;
const static float SEW_CONTACT_Z = 1.1;
const static float KEYPT_Z = 2;
const static float DRAGBOX_Z = 50;
const static float VERT_SEW_BAR_LEN = 0.01;//1cm

const static float EDGE_RENDER_WIDTH = 2;
const static float EDGE_SELECT_WIDTH = 6;
const static float KEYPT_RENDER_WIDTH = 5;
const static float KEYPT_SELECT_WIDTH = 6;
const static float SEW_RENDER_WIDTH = 4;
const static float SEW_SELECT_WIDTH = 6;

const static float SELECT_COLOR[4] = { 1, 1, 0, 1 };
const static float HIGHLIGHT_COLOR[4] = { 0, 1, 1, 1 };
const static float DEFAULT_COLOR[4] = { 0, 0, 0, 1 };

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

static std::vector<ldp::Float3> create_color_table()
{
	std::vector<ldp::Float3> table;
	table.push_back(ldp::Float3(0.8, 0.6, 0.4));
	table.push_back(ldp::Float3(0.8, 0.4, 0.6));
	table.push_back(ldp::Float3(0.6, 0.8, 0.4));
	table.push_back(ldp::Float3(0.6, 0.4, 0.8));
	table.push_back(ldp::Float3(0.4, 0.6, 0.8));
	table.push_back(ldp::Float3(0.4, 0.8, 0.6));

	table.push_back(ldp::Float3(0.2, 0.4, 0.6));
	table.push_back(ldp::Float3(0.2, 0.6, 0.4));
	table.push_back(ldp::Float3(0.4, 0.2, 0.6));
	table.push_back(ldp::Float3(0.4, 0.6, 0.2));
	table.push_back(ldp::Float3(0.6, 0.2, 0.4));
	table.push_back(ldp::Float3(0.6, 0.4, 0.2));

	table.push_back(ldp::Float3(0.7, 0.4, 0.1));
	table.push_back(ldp::Float3(0.7, 0.1, 0.4));
	table.push_back(ldp::Float3(0.4, 0.1, 0.7));
	table.push_back(ldp::Float3(0.4, 0.7, 0.1));
	table.push_back(ldp::Float3(0.1, 0.7, 0.4));
	table.push_back(ldp::Float3(0.1, 0.4, 0.7));

	table.push_back(ldp::Float3(0.7, 0.8, 0.9));
	table.push_back(ldp::Float3(0.7, 0.9, 0.8));
	table.push_back(ldp::Float3(0.9, 0.8, 0.7));
	table.push_back(ldp::Float3(0.9, 0.7, 0.8));
	table.push_back(ldp::Float3(0.8, 0.7, 0.9));
	table.push_back(ldp::Float3(0.8, 0.9, 0.7));

	return table;
}
static ldp::Float3 color_table(int i)
{
	static std::vector<ldp::Float3> table = create_color_table();
	return table.at(i%table.size());
}

static ldp::Float3 color_table()
{
	static int a = 0;
	return color_table(a++);
}

#pragma endregion

Viewer2d::Viewer2d(QWidget *parent)
: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	setMouseTracking(true);
	m_buttons = Qt::MouseButton::NoButton;
	m_isDragBox = false;
	m_currentEventHandle = nullptr;
	m_fbo = nullptr;
	m_clothManager = nullptr;
	m_mainUI = nullptr;
	m_isSewingMode = false;

	m_eventHandles.resize((size_t)Abstract2dEventHandle::ProcessorTypeEnd, nullptr);
	for (size_t i = (size_t)Abstract2dEventHandle::ProcessorTypeGeneral;
		i < (size_t)Abstract2dEventHandle::ProcessorTypeEnd; i++)
	{
		m_eventHandles[i] = std::shared_ptr<Abstract2dEventHandle>(
			Abstract2dEventHandle::create(Abstract2dEventHandle::ProcessorType(i), this));
	}
	setEventHandleType(Abstract2dEventHandle::ProcessorTypeGeneral);
}

Viewer2d::~Viewer2d()
{

}

void Viewer2d::init(ldp::ClothManager* clothManager, ClothDesigner* ui)
{
	m_clothManager = clothManager;
	m_mainUI = ui;
	resetCamera();
}

void Viewer2d::resetCamera()
{
	m_camera.setViewPort(0, width(), 0, height());
	m_camera.enableOrtho(true);
	const float as = float(width()) / float(height());
	m_camera.setFrustum(-as, as, -1, 1, -100, 100);
	m_camera.lookAt(ldp::Float3(0, 0, 0), ldp::Float3(0, 0, -1), ldp::Float3(0, 1, 0));
	//if (m_clothManager)
	//{
	//	ldp::Float2 bmin, bmax;
	//	m_clothManager->get2dBound(bmin, bmax);
	//	if (bmin != bmax)
	//	{
	//		float x0 = bmin[0], x1 = bmax[0], y0 = bmin[1], y1 = bmax[1];
	//		float bw = (x1 - x0) / 2, bh = (y1 - y0) / 2, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
	//		if (bw / bh < as)
	//		{
	//			x0 = mx - bh * as;
	//			x1 = mx + bh * as;
	//		}
	//		else
	//		{
	//			y0 = my - bw / as;
	//			y1 = my + bw / as;
	//		}
	//		m_camera.setFrustum(x0, x1, y0, y1, -1, 1);
	//	}
	//}
}

void Viewer2d::initializeGL()
{
	glewInit();
	glDisable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glEnable(GL_FRONT_AND_BACK);
	m_showType = Renderable::SW_V | Renderable::SW_F | Renderable::SW_E | Renderable::SW_TEXTURE;
	resetCamera();
}

void Viewer2d::resizeGL(int w, int h)
{
	const float x0 = m_camera.getFrustumLeft();
	const float x1 = m_camera.getFrustumRight();
	const float y0 = m_camera.getFrustumTop();
	const float y1 = m_camera.getFrustumBottom();
	const float xc = (x0 + x1) / 2;
	const float yc = (y0 + y1) / 2;
	resetCamera();
	const float nx0 = m_camera.getFrustumLeft();
	const float nx1 = m_camera.getFrustumRight();
	const float ny0 = m_camera.getFrustumTop();
	const float ny1 = m_camera.getFrustumBottom();
	const float nxc = -(nx0 + nx1) / 2 + xc;
	const float nyc = -(ny0 + ny1) / 2 + yc;
	m_camera.setFrustum(nx0 + nxc, nx1 + nxc, ny0 + nyc, ny1 + nyc, 
		m_camera.getFrustumNear(), m_camera.getFrustumFar());

	if (m_fbo)
		delete m_fbo;
	QGLFramebufferObjectFormat fmt;
	fmt.setAttachment(QGLFramebufferObject::CombinedDepthStencil);
	fmt.setMipmap(true);
	m_fbo = new QGLFramebufferObject(width(), height(), fmt);
	updateGL();
}

void Viewer2d::paintGL()
{
	// we first render for selection
	renderSelectionOnFbo();

	// then we do formal rendering=========================
	glClearColor(0.8f, 0.8f, 0.8f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// show cloth simulation=============================
	m_camera.apply();

	renderBackground();

	m_camera.apply();
	renderMeshes(false);
	m_camera.apply();
	renderClothsPanels(false);
	renderDragBox();
}

void Viewer2d::renderSelectionOnFbo()
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

	renderMeshes(true);
	renderClothsPanels(true);

	m_fboImage = m_fbo->toImage();
	m_fbo->release();

	glPopAttrib();
}

void Viewer2d::renderBackground()
{
	ldp::Float3 lt = m_camera.getWorldCoords(ldp::Float3(0, height(), m_camera.getFrustumNear()));
	ldp::Float3 rb = m_camera.getWorldCoords(ldp::Float3(width(), 0, m_camera.getFrustumNear()));
	float gridSz = 0.1;
	if ((rb-lt).length() / gridSz < 10)
		gridSz /= 10;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glColor4f(0.7, 0.7, 0.7, 1);
	glLineWidth(1);
	glBegin(GL_LINES);
	for (float x = std::floor(lt[0]/gridSz)*gridSz; x < std::ceil(rb[0]/gridSz)*gridSz; x += gridSz)
	{
		if (fabs(x) < gridSz*0.01)
			continue;
		glVertex3f(x, lt[1], BACK_Z);
		glVertex3f(x, rb[1], BACK_Z);
	}
	for (float y = std::floor(lt[1] / gridSz)*gridSz; y < std::ceil(rb[1] / gridSz)*gridSz;y += gridSz)
	{
		if (fabs(y) < gridSz*0.01)
			continue;
		glVertex3f(lt[0], y, BACK_Z);
		glVertex3f(rb[0], y, BACK_Z);
	}
	glEnd();
	glLineWidth(3);
	glColor4f(0.5, 0.5, 0.5, 1);
	glBegin(GL_LINES);
	glVertex3f(0, lt[1], BACK_Z);
	glVertex3f(0, rb[1], BACK_Z);
	glVertex3f(lt[0], 0, BACK_Z);
	glVertex3f(rb[0], 0, BACK_Z);
	glEnd();

	// render the body as two d
	glEnable(GL_DEPTH_TEST);
	glPushMatrix();
	m_camera.apply();
	glRotatef(90, 1, 0, 0);
	glTranslatef(0, BODY_Z, 0);
	glColor3f(0.6, 0.6, 0.6);
	if (m_clothManager)
	{
		m_clothManager->bodyMesh()->render(Renderable::SW_F | Renderable::SW_SMOOTH);
	}
	glPopMatrix();

	glPopAttrib();
}

void Viewer2d::mousePressEvent(QMouseEvent *ev)
{
	setFocus();
	m_lastPos = ev->pos();
	m_buttons = ev->buttons();

	m_currentEventHandle->mousePressEvent(ev);

	updateGL();
}

void Viewer2d::keyPressEvent(QKeyEvent*ev)
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
	}
	m_currentEventHandle->keyPressEvent(ev);
	updateGL();
}

void Viewer2d::keyReleaseEvent(QKeyEvent*ev)
{
	m_currentEventHandle->keyReleaseEvent(ev);
	updateGL();
}

void Viewer2d::mouseReleaseEvent(QMouseEvent *ev)
{
	m_currentEventHandle->mouseReleaseEvent(ev);

	// clear buttons
	m_buttons = Qt::NoButton;
	updateGL();
}

void Viewer2d::mouseMoveEvent(QMouseEvent*ev)
{
	m_currentEventHandle->mouseMoveEvent(ev);

	// backup last position
	m_lastPos = ev->pos();
	updateGL();
}

void Viewer2d::mouseDoubleClickEvent(QMouseEvent *ev)
{
	m_currentEventHandle->mouseDoubleClickEvent(ev);

	updateGL();
}

void Viewer2d::wheelEvent(QWheelEvent*ev)
{
	m_currentEventHandle->wheelEvent(ev);

	updateGL();
}

Abstract2dEventHandle::ProcessorType Viewer2d::getEventHandleType()const
{
	return m_currentEventHandle->type();
}

void Viewer2d::setEventHandleType(Abstract2dEventHandle::ProcessorType type)
{
	if (m_currentEventHandle)
		m_currentEventHandle->handleLeave();
	m_currentEventHandle = m_eventHandles[size_t(type)].get();
	m_currentEventHandle->handleEnter();
	setCursor(m_currentEventHandle->cursor());
}

const Abstract2dEventHandle* Viewer2d::getEventHandle(Abstract2dEventHandle::ProcessorType type)const
{
	return m_eventHandles[size_t(type)].get();
}

Abstract2dEventHandle* Viewer2d::getEventHandle(Abstract2dEventHandle::ProcessorType type)
{
	return m_eventHandles[size_t(type)].get();
}

void Viewer2d::beginDragBox(QPoint p)
{
	m_dragBoxBegin = p;
	m_isDragBox = true;
}

void Viewer2d::endDragBox()
{
	m_isDragBox = false;
}

void Viewer2d::renderDragBox()
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

	glColor3f(0, 1, 0);
	glLineWidth(2);
	glBegin(GL_LINE_LOOP);
	glVertex3f(x0, y0, DRAGBOX_Z);
	glVertex3f(x0, y1, DRAGBOX_Z);
	glVertex3f(x1, y1, DRAGBOX_Z);
	glVertex3f(x1, y0, DRAGBOX_Z);
	glEnd();

	glPopAttrib();
}

void Viewer2d::getModelBound(ldp::Float3& bmin, ldp::Float3& bmax)const
{
	bmin = FLT_MAX;
	bmax = -FLT_MAX;
	if (m_clothManager)
	{
		bmin = m_clothManager->bodyMesh()->boundingBox[0];
		bmax = m_clothManager->bodyMesh()->boundingBox[1];
	}
}

int Viewer2d::fboRenderedIndex(QPoint p)const
{
	if (m_fboImage.rect().contains(p))
	{
		QRgb c = m_fboImage.pixel(p);
		return colorToSelectId(ldp::Float4(qRed(c), qGreen(c), qBlue(c), qAlpha(c))/255.f);
	}
	return 0;
}

void Viewer2d::renderClothsPanels(bool idxMode)
{
	if (!m_clothManager)
		return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	if (!idxMode)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		glEnable(GL_MULTISAMPLE);
	}
	else
	{
		glDisable(GL_BLEND);
		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_MULTISAMPLE);
		glHint(GL_LINE_SMOOTH_HINT, GL_NEAREST);
	}

	for (int iPiece = 0; iPiece < m_clothManager->numClothPieces(); iPiece++)
	{
		const auto piece = m_clothManager->clothPiece(iPiece);
		renderClothsPanels_Edge(piece, idxMode);
		renderClothsPanels_KeyPoint(piece, idxMode);
	} // end for iPiece

	renderClothsSewing(idxMode);

	glPopAttrib();
}

void Viewer2d::renderClothsPanels_Edge(const ldp::ClothPiece* piece, bool idxMode)
{
	const float step = m_clothManager->getClothDesignParam().curveSampleStep;
	const auto& panel = piece->graphPanel();

	if (idxMode)
		glLineWidth(EDGE_SELECT_WIDTH);
	else
		glLineWidth(EDGE_RENDER_WIDTH);
	glBegin(GL_LINES);
	for (auto iter = panel.loopBegin(); iter != panel.loopEnd(); ++iter)
	{
		auto loop = iter->second.get();
		auto shape = loop->startEdge();
		do
		{
			if (!idxMode)
			{
				if (shape->isHighlighted() || panel.isHighlighted())
					glColor4fv(HIGHLIGHT_COLOR);
				else if (shape->isSelected() || panel.isSelected())
					glColor4fv(SELECT_COLOR);
				else
					glColor4fv(DEFAULT_COLOR);
			}
			else
				glColor4fv(selectIdToColor(shape->getId()).ptr());
			const auto& pts = shape->samplePointsOnShape(step / shape->getLength());
			for (size_t i = 1; i < pts.size(); i++)
			{
				glVertex3f(pts[i - 1][0], pts[i - 1][1], EDGE_Z);
				glVertex3f(pts[i][0], pts[i][1], EDGE_Z);
			}
			shape = shape->nextEdge();
		} while (shape && shape != loop->startEdge());
	}
	glEnd();
}

void Viewer2d::renderClothsPanels_KeyPoint(const ldp::ClothPiece* piece, bool idxMode)
{
	if (!(m_showType & Renderable::SW_V))
		return;
	const float step = m_clothManager->getClothDesignParam().curveSampleStep;
	const auto& panel = piece->graphPanel();

	if (idxMode)
		glPointSize(KEYPT_SELECT_WIDTH);
	else
		glPointSize(KEYPT_RENDER_WIDTH);
	glBegin(GL_POINTS);
	for (auto iter = panel.curveBegin(); iter != panel.curveEnd(); ++iter)
	{
		auto shape = iter->second;
		for (int i = 0; i < shape->numKeyPoints(); i++)
		{
			const auto& p = *shape->keyPoint(i);

			if (!idxMode)
			{
				if (p.isHighlighted() || shape->isHighlighted() || panel.isHighlighted())
					glColor4fv(HIGHLIGHT_COLOR);
				else if (p.isSelected() || shape->isSelected() || panel.isSelected())
					glColor4fv(SELECT_COLOR);
				else
					glColor4fv(DEFAULT_COLOR);
			}
			else
				glColor4fv(selectIdToColor(p.getId()).ptr());
			const auto& x = p.position();
			glVertex3f(x[0], x[1], KEYPT_Z);
		}
	}
	glEnd();
}

void Viewer2d::beginSewingMode()
{
	m_isSewingMode = true;
}

void Viewer2d::endSewingMode()
{
	m_isSewingMode = false;
}

void Viewer2d::renderClothsSewing(bool idxMode)
{
	if (!m_isSewingMode)
		return;
	if (!m_clothManager)
		return;

	const float step = m_clothManager->getClothDesignParam().curveSampleStep;
	std::vector<float> lLens, rLens;
	if (idxMode)
		glLineWidth(SEW_SELECT_WIDTH);
	else
		glLineWidth(SEW_RENDER_WIDTH);
	glBegin(GL_LINES);
	for (int iSewing = 0; iSewing < m_clothManager->numGraphSewings(); iSewing++)
	{
		const auto sew = m_clothManager->graphSewing(iSewing);
		if (sew->empty())
			continue;

		const auto& firsts = sew->firsts();
		const auto& seconds = sew->seconds();

		if (!idxMode)
		{
			if (sew->isSelected())
				glColor4fv(SELECT_COLOR);
			else if (sew->isHighlighted())
				glColor4fv(HIGHLIGHT_COLOR);
			else
			{
				auto c = color_table(sew->getId());
				glColor4f(c[0], c[1], c[2], 1);
			}
		}
		else
			glColor4fv(selectIdToColor(sew->getId()).ptr());

		ldp::Float2 fb, fe, sb, se;

		// draw the first sew
		for (size_t iShape = 0; iShape < firsts.size(); iShape++)
		{
			const auto& f = firsts[iShape];
			const auto& shape = f.curve;
			assert(shape);

			// extract b/e
			if (iShape == 0)
				fb = f.reverse ? shape->getEndPoint()->position() : shape->getStartPoint()->position();
			if (iShape + 1 == firsts.size())
				fe = f.reverse ? shape->getStartPoint()->position() : shape->getEndPoint()->position();

			// draw self
			const auto& pts = shape->samplePointsOnShape(step / shape->getLength());
			for (size_t i = 1; i < pts.size(); i++)
			{
				glVertex3f(pts[i - 1][0], pts[i - 1][1], SEW_EDGE_Z);
				glVertex3f(pts[i][0], pts[i][1], SEW_EDGE_Z);
			}

			// draw vertical bar
			const float t = f.reverse ? 0.8 : 0.2;
			ldp::Float2 ct = shape->getPointByParam(t);
			ldp::Float2 pre = shape->getPointByParam(t - 0.1);
			ldp::Float2 nxt = shape->getPointByParam(t + 0.1);
			ldp::Float2 dir = (pre - nxt).normalize();
			dir = ldp::Float2(-dir[1], dir[0]);
			glVertex3f(ct[0] - dir[0] * VERT_SEW_BAR_LEN, ct[1] - dir[1] * VERT_SEW_BAR_LEN, SEW_EDGE_Z);
			glVertex3f(ct[0] + dir[0] * VERT_SEW_BAR_LEN, ct[1] + dir[1] * VERT_SEW_BAR_LEN, SEW_EDGE_Z);
		}

		// draw the second sew
		for (size_t iShape = 0; iShape < seconds.size(); iShape++)
		{
			const auto& s = seconds[iShape];
			const auto& shape = s.curve;
			assert(shape);
			// extract b/e
			if (iShape == 0)
				sb = s.reverse ? shape->getEndPoint()->position() : shape->getStartPoint()->position();
			if (iShape + 1 == seconds.size())
				se = s.reverse ? shape->getStartPoint()->position() : shape->getEndPoint()->position();

			// draw self
			const auto& pts = shape->samplePointsOnShape(step / shape->getLength());
			for (size_t i = 1; i < pts.size(); i++)
			{
				glVertex3f(pts[i - 1][0], pts[i - 1][1], SEW_EDGE_Z);
				glVertex3f(pts[i][0], pts[i][1], SEW_EDGE_Z);
			}

			// draw vertical bar
			const float t = s.reverse ? 0.8 : 0.2;
			ldp::Float2 ct = shape->getPointByParam(t);
			ldp::Float2 pre = shape->getPointByParam(t - 0.1);
			ldp::Float2 nxt = shape->getPointByParam(t + 0.1);
			ldp::Float2 dir = (pre - nxt).normalize();
			dir = ldp::Float2(-dir[1], dir[0]);
			glVertex3f(ct[0] - dir[0] * VERT_SEW_BAR_LEN, ct[1] - dir[1] * VERT_SEW_BAR_LEN, SEW_EDGE_Z);
			glVertex3f(ct[0] + dir[0] * VERT_SEW_BAR_LEN, ct[1] + dir[1] * VERT_SEW_BAR_LEN, SEW_EDGE_Z);
		}

		// draw contacts
		if ((sew->isSelected() || sew->isHighlighted()) && !idxMode)
		{
			if (sew->isSelected())
				glColor4f(0, 1, 0, 1);
			else
				glColor4f(0, 1, 0, 0.5);
			glVertex3f(fb[0], fb[1], SEW_CONTACT_Z);
			glVertex3f(sb[0], sb[1], SEW_CONTACT_Z);
			glVertex3f(fe[0], fe[1], SEW_CONTACT_Z);
			glVertex3f(se[0], se[1], SEW_CONTACT_Z);
		}
	}// end for iSewing
	glEnd();

}

void Viewer2d::renderMeshes(bool idxMode)
{
	if (!m_clothManager)
		return;
	if (!(m_showType & Renderable::SW_F))
		return;

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	if (!idxMode)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
		glEnable(GL_MULTISAMPLE);
		glDisable(GL_LIGHTING);
		glColor4f(0.8, 0.8, 0.8, 0.8);
	}
	else
	{
		glDisable(GL_BLEND);
		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_MULTISAMPLE);
		glHint(GL_LINE_SMOOTH_HINT, GL_NEAREST);
		glDisable(GL_LIGHTING);
		glColor4f(0, 0, 0, 1);
	}

	glTranslatef(0, 0, MESH_Z);

	if (!idxMode)
	{
		for (int iMesh = 0; iMesh < m_clothManager->numClothPieces(); iMesh++)
		{
			auto piece = m_clothManager->clothPiece(iMesh);
			auto& mesh = piece->mesh2d();
			if (piece->graphPanel().isHighlighted())
				glColor4f(HIGHLIGHT_COLOR[0], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[2], 0.3);
			else if (piece->graphPanel().isSelected())
				glColor4f(SELECT_COLOR[0], SELECT_COLOR[1], SELECT_COLOR[2], 0.3);
			else
				glColor4f(DEFAULT_COLOR[0], DEFAULT_COLOR[1], DEFAULT_COLOR[2], 0.3);
			int showType = ((m_showType|Renderable::SW_V)^Renderable::SW_V);
			mesh.render(showType);
		}
	} // end if not idxMode
	else
	{
		glBegin(GL_TRIANGLES);
		for (int iMesh = 0; iMesh < m_clothManager->numClothPieces(); iMesh++)
		{
			auto piece = m_clothManager->clothPiece(iMesh);
			auto& mesh = piece->mesh2d();
			const auto& v = mesh.vertex_list;
			auto id = piece->graphPanel().getId();
			glColor4fv(selectIdToColor(id).ptr());
			for (const auto& f : mesh.face_list)
			{
				glVertex3fv(v[f.vertex_index[0]].ptr());
				glVertex3fv(v[f.vertex_index[1]].ptr());
				glVertex3fv(v[f.vertex_index[2]].ptr());
			} // end for f
		} // end for iMesh
		glEnd();
	} // end else idxMode

	glPopAttrib();
}