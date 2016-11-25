#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"

#include "Rotate3dEventHandle.h"


Rotate3dEventHandle::Rotate3dEventHandle(Viewer3d* v) : Abstract3dEventHandle(v)
{
	m_cursor = QCursor(Qt::CursorShape::CrossCursor);
	m_iconFile = "icons/rotation.png";
	m_inactiveIconFile = "icons/rotation_inactive.png";
	m_toolTips = "rotate handle";
	m_axisScale = 0.3;
}

Rotate3dEventHandle::~Rotate3dEventHandle()
{

}

void Rotate3dEventHandle::handleEnter()
{
	Abstract3dEventHandle::handleEnter();
	if (m_pickInfo.mesh){
		auto box = m_pickInfo.mesh->boundingBox;
		m_viewer->beginTrackBall(Viewer3d::TrackBall_Rot, (box[0]+box[1])*0.5f, 
			ldp::Mat3d().eye(), (box[1] - box[0]).length()* m_axisScale);
	}
}

void Rotate3dEventHandle::handleLeave()
{
	m_viewer->endTrackBall();
	Abstract3dEventHandle::handleLeave();
}

void Rotate3dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mousePressEvent(ev);
	if (m_viewer->buttons() == Qt::LeftButton)
	{
		int sid = m_viewer->fboRenderedIndex(ev->pos());
		m_viewer->setHoverTrackBallAxis(sid);
		if (sid >= Viewer3d::TrackBallIndex_X && sid <= Viewer3d::TrackBallIndex_Z)
			m_viewer->setActiveTrackBallAxis(sid);
		else
			pick(ev->pos());
	} // end if initial_location and left button
}

void Rotate3dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseReleaseEvent(ev);
	if (m_viewer->buttons() == Qt::LeftButton)
	{
		if (m_viewer->getActiveTrackBallAxis() > 0)
		{
			m_viewer->setActiveTrackBallAxis(-1);
		} // end if track ball axis active
		else
		{
			if (m_pickInfo.mesh)
			{
				auto box = m_pickInfo.mesh->boundingBox;
				m_viewer->beginTrackBall(Viewer3d::TrackBall_Trans, (box[0] + box[1])*0.5f,
					ldp::Mat3d().eye(), (box[1] - box[0]).length()* m_axisScale);
			}
		} // end else
	} // end if left button and initial_cloth
}

void Rotate3dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseDoubleClickEvent(ev);
}

void Rotate3dEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	bool valid_op = false;

	int sid = m_viewer->getActiveTrackBallAxis();
	if (sid < Viewer3d::TrackBallIndex_X || sid > Viewer3d::TrackBallIndex_Z)
	{
		int hid = m_viewer->fboRenderedIndex(ev->pos());
		if (hid >= Viewer3d::TrackBallIndex_X && hid <= Viewer3d::TrackBallIndex_Z)
			m_viewer->setHoverTrackBallAxis(hid);
		else
			m_viewer->setHoverTrackBallAxis(0);
	}
	else if (m_viewer->buttons() == Qt::LeftButton)
	{
		const ldp::Camera& cam = m_viewer->camera();
		if (m_pickInfo.mesh)
		{
			QPoint lp = m_viewer->lastMousePos();
			ldp::Float3 axis = 0;
			switch (sid)
			{
			default:
				break;
			case (int)Viewer3d::TrackBallIndex_X:
				axis[0] = 1;
				break;
			case (int)Viewer3d::TrackBallIndex_Y:
				axis[1] = 1;
				break;
			case (int)Viewer3d::TrackBallIndex_Z:
				axis[2] = 1;
				break;
			}
			ldp::Double3 wp = cam.getWorldCoords(ldp::Float3(ev->x(), m_viewer->height() - 1 - ev->y(), m_pickInfo.screenPos[2]));
			ldp::Double3 wlp = cam.getWorldCoords(ldp::Float3(lp.x(), m_viewer->height() - 1 - lp.y(), m_pickInfo.screenPos[2]));
			ldp::Double3 dir = (wp - wlp)*axis;
			m_pickInfo.mesh->translate(dir);
			m_viewer->translateTrackBall(dir);
			valid_op = true;
		} // end if getPickedMeshFrameInfo
	} // end if initial_cloth and left button
	if (!valid_op)
		Abstract3dEventHandle::mouseMoveEvent(ev);
}

void Rotate3dEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract3dEventHandle::wheelEvent(ev);
}

void Rotate3dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyPressEvent(ev);
}

void Rotate3dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyReleaseEvent(ev);
}
