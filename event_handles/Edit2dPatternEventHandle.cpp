#include <QEvent>
#include <GL\glew.h>
#include "Viewer2d.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"

#include "Edit2dPatternEventHandle.h"
Edit2dPatternEventHandle::Edit2dPatternEventHandle(Viewer2d* v)
: Abstract2dEventHandle(v)
{
	m_cursor = QCursor(Qt::CursorShape::CrossCursor);
	m_iconFile = "icons/groupArrow.png";
	m_inactiveIconFile = "icons/groupArrow.png";
	m_toolTips = "edit pattern";
}

Edit2dPatternEventHandle::~Edit2dPatternEventHandle()
{

}

void Edit2dPatternEventHandle::handleEnter()
{
	Abstract2dEventHandle::handleEnter();
	m_viewer->setFocus();
}
void Edit2dPatternEventHandle::handleLeave()
{
	m_viewer->clearFocus();
	Abstract2dEventHandle::handleLeave();
}

void Edit2dPatternEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mousePressEvent(ev);

	if (ev->buttons() == Qt::LeftButton)
	{

	} // end if left button
}

void Edit2dPatternEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseReleaseEvent(ev);
}

void Edit2dPatternEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseDoubleClickEvent(ev);
}

void Edit2dPatternEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseMoveEvent(ev);
}

void Edit2dPatternEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract2dEventHandle::wheelEvent(ev);
}

void Edit2dPatternEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyPressEvent(ev);
}

void Edit2dPatternEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyReleaseEvent(ev);
}
