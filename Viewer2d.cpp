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
#include "../clothdesigner.h"

#pragma region --mat_utils

const static float BODY_Z = -50;
const static float BACK_Z = -10;
const static float MESH_Z = 0;
const static float EDGE_Z = 0.1;
const static float EDGE_Z_ADDURVE = 0.11;
const static float SEW_EDGE_Z = 1;
const static float SEW_CONTACT_Z = 1.1;
const static float KEYPT_Z = 2;
const static float KEYPT_Z_ADDCURVE = 2.01;
const static float DRAGBOX_Z = 50;
const static float VERT_SEW_BAR_LEN = 0.01;//1cm

const static float EDGE_RENDER_WIDTH = 2;
const static float EDGE_SELECT_WIDTH = 6;
const static float KEYPT_RENDER_WIDTH = 5;
const static float KEYPT_SELECT_WIDTH = 6;
const static float SEW_RENDER_WIDTH = 4;
const static float SEW_SELECT_WIDTH = 6;

const static float SELECT_COLOR[4] = { 1, 1, 0, 0.8 };
const static float HIGHLIGHT_COLOR[4] = { 0, 1, 1, 0.8 };
const static float DEFAULT_COLOR[4] = { 0, 0, 0, 1 };
const static float ADDCURVE_COLOR[4] = { 0, 1, 0, 1 };

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

static std::vector<ldp::Float4> create_color_table()
{
	std::vector<ldp::Float4> table;
	table.push_back(ldp::Float4(0.8, 0.6, 0.4, 0.8));
	table.push_back(ldp::Float4(0.8, 0.4, 0.6, 0.8));
	table.push_back(ldp::Float4(0.6, 0.8, 0.4, 0.8));
	table.push_back(ldp::Float4(0.6, 0.4, 0.8, 0.8));
	table.push_back(ldp::Float4(0.4, 0.6, 0.8, 0.8));
	table.push_back(ldp::Float4(0.4, 0.8, 0.6, 0.8));
	table.push_back(ldp::Float4(0.2, 0.4, 0.6, 0.8));
	table.push_back(ldp::Float4(0.2, 0.6, 0.4, 0.8));
	table.push_back(ldp::Float4(0.4, 0.2, 0.6, 0.8));
	table.push_back(ldp::Float4(0.4, 0.6, 0.2, 0.8));
	table.push_back(ldp::Float4(0.6, 0.2, 0.4, 0.8));
	table.push_back(ldp::Float4(0.6, 0.4, 0.2, 0.8));
	table.push_back(ldp::Float4(0.7, 0.4, 0.1, 0.8));
	table.push_back(ldp::Float4(0.7, 0.1, 0.4, 0.8));
	table.push_back(ldp::Float4(0.4, 0.1, 0.7, 0.8));
	table.push_back(ldp::Float4(0.4, 0.7, 0.1, 0.8));
	table.push_back(ldp::Float4(0.1, 0.7, 0.4, 0.8));
	table.push_back(ldp::Float4(0.1, 0.4, 0.7, 0.8));
	table.push_back(ldp::Float4(0.7, 0.8, 0.9, 0.8));
	table.push_back(ldp::Float4(0.7, 0.9, 0.8, 0.8));
	table.push_back(ldp::Float4(0.9, 0.8, 0.7, 0.8));
	table.push_back(ldp::Float4(0.9, 0.7, 0.8, 0.8));
	table.push_back(ldp::Float4(0.8, 0.7, 0.9, 0.8));
	table.push_back(ldp::Float4(0.8, 0.9, 0.7, 0.8));

	return table;
}
static ldp::Float4 color_table(int i)
{
	static std::vector<ldp::Float4> table = create_color_table();
	return table.at(i%table.size());
}

static ldp::Float4 color_table()
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
	m_isAddCurveMode = false;

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
	getEventHandle(getEventHandleType())->resetSelection();
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
	glEnable(GL_LIGHT0);
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
	try
	{
		m_camera.apply();
		renderBackground();
		m_camera.apply();
		renderMeshes(false);
		m_camera.apply();
		renderClothsPanels(false);
		m_camera.apply();
		renderUiCurves();
		m_camera.apply();
		renderDragBox();
	} catch (std::exception e)
	{
		std::cout << "paintGL: " << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "paintGL: unknown error" << std::endl;
	}
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
	glRotatef(180, 0, 0, 1);
	glTranslatef(0, -BODY_Z, 0);
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
	try
	{
		m_currentEventHandle->mousePressEvent(ev);
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error" << std::endl;
	}
	updateGL();
}

void Viewer2d::keyPressEvent(QKeyEvent*ev)
{
	try
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
			m_showType ^= Renderable::SW_LIGHTING;
			break;
		case Qt::Key_V:
			m_showType ^= Renderable::SW_V;
			break;
		}
		m_currentEventHandle->keyPressEvent(ev);
		updateGL();
	}catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error" << std::endl;
	}
}

void Viewer2d::keyReleaseEvent(QKeyEvent*ev)
{
	try
	{
		m_currentEventHandle->keyReleaseEvent(ev);
		updateGL();
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error" << std::endl;
	}
}

void Viewer2d::mouseReleaseEvent(QMouseEvent *ev)
{
	try
	{
		m_currentEventHandle->mouseReleaseEvent(ev);
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error" << std::endl;
	}

	// clear buttons
	m_buttons = Qt::NoButton;
	updateGL();
}

void Viewer2d::mouseMoveEvent(QMouseEvent*ev)
{
	try
	{
		m_currentEventHandle->mouseMoveEvent(ev);
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error" << std::endl;
	}
	// backup last position
	m_lastPos = ev->pos();
	updateGL();
}

void Viewer2d::mouseDoubleClickEvent(QMouseEvent *ev)
{
	try
	{
		m_currentEventHandle->mouseDoubleClickEvent(ev);
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error" << std::endl;
	}
	updateGL();
}

void Viewer2d::wheelEvent(QWheelEvent*ev)
{
	try 
	{
		m_currentEventHandle->wheelEvent(ev);
	} catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	} catch (...)
	{
		std::cout << "unknown error" << std::endl;
	}

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
		glLineWidth(EDGE_SELECT_WIDTH + isSewingMode() * SEW_SELECT_WIDTH);
	else
		glLineWidth(EDGE_RENDER_WIDTH);
	glBegin(GL_LINES);
	for (auto edge_iter = panel.curve_begin(); edge_iter != panel.curve_end(); ++edge_iter)
	{
		if (!idxMode)
		{
			if (edge_iter->isHighlighted() || panel.isHighlighted())
				glColor4fv(HIGHLIGHT_COLOR);
			else if (edge_iter->isSelected() || panel.isSelected())
				glColor4fv(SELECT_COLOR);
			else
				glColor4fv(DEFAULT_COLOR);
		}
		else
			glColor4fv(selectIdToColor(edge_iter->getId()).ptr());
		const auto& pts = edge_iter->samplePointsOnShape(step / edge_iter->getLength());
		for (size_t i = 1; i < pts.size(); i++)
		{
			glVertex3f(pts[i - 1][0], pts[i - 1][1], EDGE_Z);
			glVertex3f(pts[i][0], pts[i][1], EDGE_Z);
		}
	} // end for edge_iter
	glEnd();
}

void Viewer2d::renderClothsPanels_KeyPoint(const ldp::ClothPiece* piece, bool idxMode)
{
	if (!(m_showType & Renderable::SW_V))
		return;
	const float step = m_clothManager->getClothDesignParam().curveSampleStep;
	const auto& panel = piece->graphPanel();

	if (idxMode)
		glPointSize(KEYPT_SELECT_WIDTH + m_isAddCurveMode * 3);
	else
		glPointSize(KEYPT_RENDER_WIDTH + m_isAddCurveMode * 3);
	glBegin(GL_POINTS);
	for (auto point_iter = panel.point_begin(); point_iter != panel.point_end(); ++point_iter)
	{
		const ldp::GraphPoint& p = *point_iter;
		bool selected = p.isSelected() || panel.isSelected();
		bool highlighted = p.isHighlighted() || panel.isHighlighted();
		for (auto e_iter = point_iter->edge_begin(); !e_iter.isEnd(); ++e_iter)
		{
			selected |= e_iter->isSelected();
			highlighted |= e_iter->isHighlighted();
		}
		if (!idxMode)
		{
			if (highlighted)
				glColor4fv(HIGHLIGHT_COLOR);
			else if (selected)
				glColor4fv(SELECT_COLOR);
			else
				glColor4fv(DEFAULT_COLOR);
		}
		else
			glColor4fv(selectIdToColor(p.getId()).ptr());
		const auto& x = p.getPosition();
		glVertex3f(x[0], x[1], KEYPT_Z);
	}
	glEnd();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
void Viewer2d::beginSewingMode()
{
	m_isSewingMode = true;
}

void Viewer2d::endSewingMode()
{
	m_isSewingMode = false;
}

UiSewChanges Viewer2d::addCurrentUISew()
{
	if (m_uiSews.firsts.empty() || m_uiSews.seconds.empty())
		return UiSewNoChange;

	bool added = false;

	ldp::GraphsSewingPtr sew(new ldp::GraphsSewing);
	sew->addFirsts(m_uiSews.firsts);
	sew->addSeconds(m_uiSews.seconds);
	sew->setSelected(true);
	if (m_clothManager->addGraphSewing(sew))
		added = true;
	deleteCurrentUISew();
	if (added)
		return UiSewAddedToPanel;
	return UiSewUiTmpChanged;
}

UiSewChanges Viewer2d::deleteCurrentUISew()
{
	m_uiSews.firsts.clear();
	m_uiSews.seconds.clear();
	m_uiSews.f.curve = nullptr;
	m_uiSews.s.curve = nullptr;
	m_uiSews.state = UiSewAddingEnd;
	if (getMainUI())
		getMainUI()->pushHistory("delete curent ui sew", ldp::HistoryStack::TypeUiSewChanged);
	return UiSewUiTmpChanged;
}

UiSewChanges Viewer2d::makeSewUnit(ldp::AbstractGraphCurve* curve, QPoint pos, bool tmp)
{
	if (m_uiSews.state == UiSewAddingEnd)
		return UiSewNoChange;

	// check existence
	for (auto& f : m_uiSews.firsts)
	if (f.curve == curve)
		curve = nullptr;
	for (auto& f : m_uiSews.seconds)
	if (f.curve == curve)
		curve = nullptr;

	if (curve == nullptr)
	{
		if (tmp)
		{
			m_uiSews.f.curve = nullptr;
			m_uiSews.s.curve = nullptr;
		}
		return UiSewNoChange;
	}

	ldp::Float3 lp3(lastMousePos().x(), height() - 1 - lastMousePos().y(), 1);
	lp3 = camera().getWorldCoords(lp3);
	ldp::Float2 splitPos(lp3[0], lp3[1]);

	const float step = ldp::g_designParam.curveSampleStep / curve->getLength();
	const auto& vec = curve->samplePointsOnShape(step);

	float minDist = FLT_MAX;
	float tSplit = 0;
	for (size_t i = 1; i < vec.size(); i++)
	{
		float dist = pointSegDistance(splitPos, vec[i - 1], vec[i]);
		if (dist < minDist)
		{
			minDist = dist;
			tSplit = (i - 1)*step + nearestPointOnSeg_getParam(splitPos, vec[i - 1], vec[i]) * step;
		}
	} // end for i in vec

	// too close, do not split
	bool reverse = tSplit < 0.5f;
	ldp::GraphsSewing::Unit unit(curve, reverse);

	if (tmp)
	{
		m_uiSews.f.curve = nullptr;
		m_uiSews.s.curve = nullptr;
		if (m_uiSews.state == UiSewAddingFirst)
			m_uiSews.f = unit;
		else if (m_uiSews.state == UiSewAddingSecond)
			m_uiSews.s = unit;
	}
	else
	{
		if (m_uiSews.state == UiSewAddingFirst)
			m_uiSews.firsts.push_back(unit);
		else if (m_uiSews.state == UiSewAddingSecond)
			m_uiSews.seconds.push_back(unit);
		if (getMainUI())
			getMainUI()->pushHistory("add one ui sew unit", ldp::HistoryStack::TypeUiSewChanged);
	}
	return UiSewUiTmpChanged;
}

UiSewChanges Viewer2d::setSewAddingState(UiSewAddingState s)
{
	m_uiSews.state = s;
	switch (m_uiSews.state)
	{
	case UiSewAddingFirst:
		m_uiSews.f.curve = nullptr;
		m_uiSews.s.curve = nullptr;
		return UiSewNoChange;
	case UiSewAddingSecond:
		m_uiSews.f.curve = nullptr;
		m_uiSews.s.curve = nullptr;
		return UiSewNoChange;
	case UiSewAddingEnd:
		return addCurrentUISew();
	default:
		break;
	}
	return UiSewNoChange;
}

UiSewChanges Viewer2d::setNextSewAddingState()
{
	if (m_uiSews.state == UiSewAddingFirst)
		m_uiSews.state = UiSewAddingSecond;
	else if (m_uiSews.state == UiSewAddingSecond)
		m_uiSews.state = UiSewAddingEnd;
	else if (m_uiSews.state == UiSewAddingEnd)
		m_uiSews.state = UiSewAddingFirst;

	return setSewAddingState(m_uiSews.state);
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

	// render the existed sewings
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
				glColor4fv(color_table(sew->getId()).ptr());
		}
		else
			glColor4fv(selectIdToColor(sew->getId()).ptr());

		renderOneSew(sew, idxMode);
	}// end for iSewing

	// render the ui sewing
	ldp::GraphsSewingPtr sew(new ldp::GraphsSewing);
	for (auto& f : m_uiSews.firsts)
	if (f.curve)
		sew->addFirst(f);
	if (m_uiSews.f.curve)
		sew->addFirst(m_uiSews.f);
	for (auto& s : m_uiSews.seconds)
	if (s.curve)
		sew->addSecond(s);
	if (m_uiSews.s.curve)
		sew->addSecond(m_uiSews.s);
	sew->setHighlighted(true);
	if (!idxMode)
	{
		if (sew->isSelected())
			glColor4fv(SELECT_COLOR);
		else if (sew->isHighlighted())
			glColor4fv(HIGHLIGHT_COLOR);
		else
			glColor4fv(color_table(sew->getId()).ptr());
		renderOneSew(sew.get(), idxMode);
	}

	glEnd();

}

void Viewer2d::renderOneSew(const ldp::GraphsSewing* sew, bool idxMode)
{
	if (sew == nullptr)
		return;

	const float step = m_clothManager->getClothDesignParam().curveSampleStep;
	ldp::Float2 fb, fe, sb, se;

	// draw the first sew
	for (size_t iShape = 0; iShape < sew->firsts().size(); iShape++)
	{
		const auto& f = sew->firsts()[iShape];
		const auto& shape = f.curve;
		assert(shape);

		// extract b/e
		if (iShape == 0)
			fb = f.reverse ? shape->getEndPoint()->getPosition() : shape->getStartPoint()->getPosition();
		if (iShape + 1 == sew->firsts().size())
			fe = f.reverse ? shape->getStartPoint()->getPosition() : shape->getEndPoint()->getPosition();

		// draw self
		renderOneSewUnit(f, idxMode);
	}

	// draw the second sew
	for (size_t iShape = 0; iShape < sew->seconds().size(); iShape++)
	{
		const auto& s = sew->seconds()[iShape];
		const auto& shape = s.curve;
		assert(shape);
		// extract b/e
		if (iShape == 0)
			sb = s.reverse ? shape->getEndPoint()->getPosition() : shape->getStartPoint()->getPosition();
		if (iShape + 1 == sew->seconds().size())
			se = s.reverse ? shape->getStartPoint()->getPosition() : shape->getEndPoint()->getPosition();

		// draw self
		renderOneSewUnit(s, idxMode);
	}

	// draw contacts
	if ((sew->isSelected() || sew->isHighlighted()) && !idxMode && !sew->empty())
	{
		if (sew->isSelected())
			glColor4f(0, 1, 0, 0.8);
		else
			glColor4f(0, 1, 0, 0.5);
		glVertex3f(fb[0], fb[1], SEW_CONTACT_Z);
		glVertex3f(sb[0], sb[1], SEW_CONTACT_Z);
		glVertex3f(fe[0], fe[1], SEW_CONTACT_Z);
		glVertex3f(se[0], se[1], SEW_CONTACT_Z);
	}
}

void Viewer2d::renderOneSewUnit(const ldp::GraphsSewing::Unit& unit, bool idxMode)
{
	if (unit.curve == nullptr)
		return;

	const auto& f = unit;
	const auto& shape = f.curve;
	assert(shape);

	const float step = m_clothManager->getClothDesignParam().curveSampleStep;

	// draw self
	const auto& pts = shape->samplePointsOnShape(step / shape->getLength());
	for (size_t i = 1; i < pts.size(); i++)
	{
		glVertex3f(pts[i - 1][0], pts[i - 1][1], SEW_EDGE_Z);
		glVertex3f(pts[i][0], pts[i][1], SEW_EDGE_Z);
	}

	// draw vertical bar
	const float t = f.reverse ? 0.2 : 0.8;
	ldp::Float2 ct = shape->getPointByParam(t);
	ldp::Float2 pre = shape->getPointByParam(t - 0.1);
	ldp::Float2 nxt = shape->getPointByParam(t + 0.1);
	ldp::Float2 dir = (pre - nxt).normalize();
	dir = ldp::Float2(-dir[1], dir[0]);
	glVertex3f(ct[0] - dir[0] * VERT_SEW_BAR_LEN, ct[1] - dir[1] * VERT_SEW_BAR_LEN, SEW_EDGE_Z);
	glVertex3f(ct[0] + dir[0] * VERT_SEW_BAR_LEN, ct[1] + dir[1] * VERT_SEW_BAR_LEN, SEW_EDGE_Z);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
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
		glColor4f(0.8, 0.8, 0.8, 0.8);
	}
	else
	{
		glDisable(GL_BLEND);
		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_MULTISAMPLE);
		glHint(GL_LINE_SMOOTH_HINT, GL_NEAREST);
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
			int showType = ((m_showType | Renderable::SW_V) ^ Renderable::SW_V);
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

////////////////////////////////////////////////////////////////////////////////////////////////////////
void Viewer2d::beginAddCurveMode()
{
	m_uiCurves.points.clear();
	m_uiCurves.renderedIds.clear();
	m_uiCurves.tmpPoint.reset((ldp::GraphPoint*)nullptr);
	m_isAddCurveMode = true;
}

void Viewer2d::endAddCurveMode()
{
	m_isAddCurveMode = false;
	m_uiCurves.points.clear();
	m_uiCurves.renderedIds.clear();
	m_uiCurves.tmpPoint.reset((ldp::GraphPoint*)nullptr);
}

bool Viewer2d::beginNewCurve(QPoint pos)
{
	m_uiCurves.points.clear();
	m_uiCurves.renderedIds.clear();
	m_uiCurves.tmpPoint.reset((ldp::GraphPoint*)nullptr);
	return addCurvePoint(pos, false);
}

bool Viewer2d::addCurvePoint(QPoint pos, bool tmp)
{
	ldp::Float3 p3(pos.x(), height() - 1 - pos.y(), 1);
	p3 = camera().getWorldCoords(p3);
	ldp::Float2 p(p3[0], p3[1]);
	int renderId = fboRenderedIndex(pos);
	if (renderId <= 0)
		return false; // cannot add curve without on an existed panel

	// check if the given position is on a given object
	auto obj = ldp::AbstractGraphObject::getObjByIdx(renderId);
	if (obj)
	{
		if (obj->getType() == ldp::AbstractGraphObject::TypeGraphPoint)
			p = ((ldp::GraphPoint*)obj)->getPosition();
		else if (obj->isCurve())
			p = ((ldp::AbstractGraphCurve*)obj)->getNearestPoint(p);
	} // end if obj
	
	auto kpt = ldp::GraphPointPtr(new ldp::GraphPoint(p));

	if (tmp)
	{
		m_uiCurves.tmpPoint = kpt;
	}
	else
	{
		m_uiCurves.points.push_back(kpt);
		m_uiCurves.renderedIds.push_back(renderId);
	}
	return true;
}

bool Viewer2d::endCurve()
{
	// add to the cloth
	if (!m_clothManager)
		return false;
	if (!m_clothManager->addCurveOnAPanel(m_uiCurves.points, m_uiCurves.renderedIds))
		return false;

	// clear
	m_uiCurves.points.clear();
	m_uiCurves.renderedIds.clear();
	m_uiCurves.tmpPoint.reset((ldp::GraphPoint*)nullptr);
	return true;
}

bool Viewer2d::giveupCurve()
{
	m_uiCurves.points.clear();
	m_uiCurves.renderedIds.clear();
	m_uiCurves.tmpPoint.reset((ldp::GraphPoint*)nullptr);
	return true;
}

void Viewer2d::renderUiCurves()
{
	if (!m_isAddCurveMode)
		return;
	if (m_uiCurves.points.empty())
		return;
	const float step = m_clothManager->getClothDesignParam().curveSampleStep;

	std::vector<ldp::GraphPoint*> keyPts;
	for (auto& p : m_uiCurves.points)
		keyPts.push_back(p.get());
	if (m_uiCurves.tmpPoint && keyPts.size() < ldp::AbstractGraphCurve::maxKeyPointsNum())
		keyPts.push_back(m_uiCurves.tmpPoint.get());

	// render points
	glPointSize(KEYPT_RENDER_WIDTH);
	glBegin(GL_POINTS);
	for (auto& p : keyPts)
	{
		glColor4fv(DEFAULT_COLOR);
		const auto& x = p->getPosition();
		glVertex3f(x[0], x[1], KEYPT_Z_ADDCURVE);
	}
	glEnd();

	std::shared_ptr<ldp::AbstractGraphCurve> curve(ldp::AbstractGraphCurve::create(keyPts));
	if (curve == nullptr)
		return;

	// render curve
	glLineWidth(EDGE_RENDER_WIDTH);
	glBegin(GL_LINES);
	glColor4fv(ADDCURVE_COLOR);
	const auto& pts = curve->samplePointsOnShape(step / curve->getLength());
	for (size_t i = 1; i < pts.size(); i++)
	{
		glVertex3f(pts[i - 1][0], pts[i - 1][1], EDGE_Z);
		glVertex3f(pts[i][0], pts[i][1], EDGE_Z);
	}
	glEnd();
}
