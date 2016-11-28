#include <QEvent>
#include <GL\glew.h>
#include "Viewer2d.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"

#include "Abstract2dEventHandle.h"
Abstract2dEventHandle::Abstract2dEventHandle(Viewer2d* v)
{
	m_viewer = v;
	m_lastHighlightShapeId = -1;
	m_currentSelectedId = -1;
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
	m_viewer->clearFocus();
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

	int renderedId = m_viewer->fboRenderedIndex(pos);
}

Abstract2dEventHandle* Abstract2dEventHandle::create(ProcessorType type, Viewer2d* v)
{
	switch (type)
	{
	case Abstract2dEventHandle::ProcessorTypeGeneral:
		return new Abstract2dEventHandle(v);
	case Abstract2dEventHandle::ProcessorTypeEnd:
	default:
		return nullptr;
	}
}

void Abstract2dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	m_mouse_press_pt = ev->pos();

	// arcball drag
	if (ev->buttons() == Qt::LeftButton)
		m_viewer->camera().arcballClick(ldp::Float2(ev->x(), ev->y()));
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
	if (ev->buttons() == Qt::LeftButton)
		m_viewer->camera().arcballDrag(ldp::Float2(ev->x(), ev->y()));
	if (ev->buttons() == Qt::MidButton)
	{
		QPoint dif = ev->pos() - m_viewer->lastMousePos();
		ldp::Float3 bmin, bmax;
		m_viewer->getModelBound(bmin, bmax);
		float len = (bmax - bmin).length() / sqrt(3.f);
		ldp::Float3 t(-(float)dif.x() / m_viewer->width(), (float)dif.y() / m_viewer->height(), 0);
		m_viewer->camera().translate(t * len);
		m_viewer->camera().arcballSetCenter((bmin + bmax) / 2.f + t * len);
	}
}

void Abstract2dEventHandle::wheelEvent(QWheelEvent *ev)
{
	float s = 1.1;
	if (ev->delta() < 0)
		s = 1.f / s;

	float fov = std::max(1e-3f, std::min(160.f, m_viewer->camera().getFov()*s));
	m_viewer->camera().setPerspective(fov, m_viewer->camera().getAspect(),
		m_viewer->camera().getFrustumNear(), m_viewer->camera().getFrustumFar());
}

void Abstract2dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	switch (ev->key())
	{
	default:
		break;
	}
}

void Abstract2dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{

}
