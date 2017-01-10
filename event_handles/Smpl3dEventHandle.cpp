#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"

#include "Smpl3dEventHandle.h"
#include "cloth\clothManager.h"
#include "cloth\SmplManager.h"
#include "Renderable\ObjMesh.h"

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

Smpl3dEventHandle::Smpl3dEventHandle(Viewer3d* v) : Abstract3dEventHandle(v)
{
	m_cursor = QCursor(Qt::CursorShape::ArrowCursor);
	m_iconFile = "icons/smpl.png";
	m_inactiveIconFile = "icons/smpl.png";
	m_toolTips = "smpl handle";
}

Smpl3dEventHandle::~Smpl3dEventHandle()
{

}

void Smpl3dEventHandle::handleEnter()
{
	Abstract3dEventHandle::handleEnter();
	m_pickInfo.mesh = nullptr;
	m_viewer->setSmplMode(true);
}

void Smpl3dEventHandle::handleLeave()
{
	Abstract3dEventHandle::handleLeave();
	m_pickInfo.mesh = nullptr;
	m_viewer->setSmplMode(false);
}

void Smpl3dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mousePressEvent(ev);
	m_pickInfo.mesh = nullptr;
	if (m_viewer->buttons() == Qt::LeftButton)
	{
		pick(ev->pos());

		auto manager = m_viewer->getManager();
		if (manager && manager->bodySmplManager())
		{
			manager->bodySmplManager()->selectAction(selectIdToColor(pickInfo().renderId), Renderable::MOUSE_L_PRESS, 0);
		} // end if manager
	} // end if initial_location and left button
}

void Smpl3dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseReleaseEvent(ev);

	auto manager = m_viewer->getManager();
	if (manager && manager->bodySmplManager())
	{
		manager->bodySmplManager()->selectAction(selectIdToColor(pickInfo().renderId), Renderable::MOUSE_L_RELEASE, 0);
	} // end if manager

	m_pickInfo.mesh = nullptr;
	m_pickInfo.piece = nullptr;
}

void Smpl3dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseDoubleClickEvent(ev);
}

void Smpl3dEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	bool valid_op = false;
	if (m_viewer->buttons() == Qt::NoButton)
	{
		int idx = m_viewer->fboRenderedIndex(ev->pos());
		auto manager = m_viewer->getManager();
		if (manager && manager->bodySmplManager() && idx > 0)
		{
			manager->bodySmplManager()->selectAction(selectIdToColor(idx), Renderable::MOUSE_MOVE, 0);
		} // end if manager
	} // end if initial_location and left button

	if (!valid_op)
		Abstract3dEventHandle::mouseMoveEvent(ev);
}

void Smpl3dEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract3dEventHandle::wheelEvent(ev);
}

void Smpl3dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyPressEvent(ev);
}

void Smpl3dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyReleaseEvent(ev);
}
