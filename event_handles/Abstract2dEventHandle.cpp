#include <QEvent>
#include <GL\glew.h>
#include "Viewer2d.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
#include "Renderable\ObjMesh.h"
#include "cloth\graph\Graph.h"

#include "Abstract2dEventHandle.h"
#include "Edit2dPatternEventHandle.h"
#include "Transform2dPatternEventHandle.h"
#include "Sewing2dPatternEventHandle.h"

Abstract2dEventHandle::Abstract2dEventHandle(Viewer2d* v)
{
	m_viewer = v;
	m_cursor = QCursor(Qt::CursorShape::ArrowCursor);
	m_iconFile = "";
	m_inactiveIconFile = "";
	m_toolTips = "general handle";
}

Abstract2dEventHandle::~Abstract2dEventHandle()
{

}

QString Abstract2dEventHandle::iconFile()const
{
	return m_iconFile;
}

QString Abstract2dEventHandle::inactiveIconFile()const
{
	return m_inactiveIconFile;
}

void Abstract2dEventHandle::handleEnter()
{
	m_viewer->setFocus();
}
void Abstract2dEventHandle::handleLeave()
{
	m_highLightInfo.clear();
	m_pickInfo.clear();
	m_viewer->clearFocus();

	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;
	for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
	{
		auto piece = manager->clothPiece(iPiece);
		auto& panel = piece->graphPanel();
		panel.highLight(0, m_highLightInfo.lastId);
	} // end for iPiece
}

QString Abstract2dEventHandle::toolTips()const
{
	return m_toolTips;
}

void Abstract2dEventHandle::pick(QPoint pos)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	m_pickInfo.renderId = m_viewer->fboRenderedIndex(pos);
}

void Abstract2dEventHandle::highLight(QPoint pos)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	m_highLightInfo.lastId = m_highLightInfo.renderId;
	m_highLightInfo.renderId = m_viewer->fboRenderedIndex(pos);
	for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
	{
		auto piece = manager->clothPiece(iPiece);
		auto& panel = piece->graphPanel();
		panel.highLight(m_highLightInfo.renderId, m_highLightInfo.lastId);
	} // end for iPiece
}

Abstract2dEventHandle* Abstract2dEventHandle::create(ProcessorType type, Viewer2d* v)
{
	switch (type)
	{
	case Abstract2dEventHandle::ProcessorTypeGeneral:
		return new Abstract2dEventHandle(v);
	case Abstract2dEventHandle::ProcessorTypeEditPattern:
		return new Edit2dPatternEventHandle(v);
	case Abstract2dEventHandle::ProcessorTypeTransformPattern:
		return new Transform2dPatternEventHandle(v);
	case Abstract2dEventHandle::ProcessorTypeSewingPattern:
		return new Sewing2dPatternEventHandle(v);
	case Abstract2dEventHandle::ProcessorTypeEnd:
	default:
		return nullptr;
	}
}

void Abstract2dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	m_mouse_press_pt = ev->pos();
}

void Abstract2dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{

}

void Abstract2dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	if (ev->button() == Qt::MouseButton::MiddleButton)
		m_viewer->resetCamera();
}

void Abstract2dEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	if (ev->buttons() == Qt::MidButton)
	{
		float l = m_viewer->camera().getFrustumLeft();
		float r = m_viewer->camera().getFrustumRight();
		float t = m_viewer->camera().getFrustumTop();
		float b = m_viewer->camera().getFrustumBottom();
		float dx = (r - l) / float(m_viewer->width()) * (ev->pos().x() - m_viewer->lastMousePos().x());
		float dy = (b - t) / float(m_viewer->height()) * (ev->pos().y() - m_viewer->lastMousePos().y());
		l -= dx;
		r -= dx;
		t -= dy;
		b -= dy;
		m_viewer->camera().setFrustum(l, r, t, b,
			m_viewer->camera().getFrustumNear(),
			m_viewer->camera().getFrustumFar());
	}

	if (ev->buttons() == Qt::NoButton)
		highLight(ev->pos());
}

void Abstract2dEventHandle::wheelEvent(QWheelEvent *ev)
{
	float s = 1.2f;
	if (ev->delta() < 0)
		s = 1.f / s;

	float l = m_viewer->camera().getFrustumLeft();
	float r = m_viewer->camera().getFrustumRight();
	float t = m_viewer->camera().getFrustumTop();
	float b = m_viewer->camera().getFrustumBottom();
	float dx = float(ev->pos().x()) / float(m_viewer->width()) * (r - l);
	float dy = float(ev->pos().y()) / float(m_viewer->height()) * (b - t);

	r = dx + l + (r - l - dx) * s;
	l = dx + l - dx * s;
	b = dy + t + (b - t - dy) * s;
	t = dy + t - dy * s;
	m_viewer->camera().setFrustum(l, r, t, b,
		m_viewer->camera().getFrustumNear(), m_viewer->camera().getFrustumFar());
}

void Abstract2dEventHandle::keyPressEvent(QKeyEvent *ev)
{

}

void Abstract2dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{

}
