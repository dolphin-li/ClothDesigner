#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"

#include "Select3dEventHandle.h"


Select3dEventHandle::Select3dEventHandle(Viewer3d* v) : Abstract3dEventHandle(v)
{
	m_cursor = QCursor(Qt::CursorShape::ArrowCursor);
	m_iconFile = "icons/selection.png";
	m_inactiveIconFile = "icons/selection_inactive.png";
	m_toolTips = "select handle";
}

Select3dEventHandle::~Select3dEventHandle()
{

}

void Select3dEventHandle::handleEnter()
{
	Abstract3dEventHandle::handleEnter();
}

void Select3dEventHandle::handleLeave()
{
	Abstract3dEventHandle::handleLeave();
}

void Select3dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mousePressEvent(ev);
	if (m_viewer->buttons() == Qt::LeftButton)
	{
		pick(ev->pos());
	} // end if initial_location and left button
}

void Select3dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseReleaseEvent(ev);
}

void Select3dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseDoubleClickEvent(ev);
}

void Select3dEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseMoveEvent(ev);
}

void Select3dEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract3dEventHandle::wheelEvent(ev);
}

void Select3dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyPressEvent(ev);
}

void Select3dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyReleaseEvent(ev);
}
