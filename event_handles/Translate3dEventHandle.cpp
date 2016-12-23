#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"
#include "Renderable\ObjMesh.h"
#include "Translate3dEventHandle.h"

#include "../clothdesigner.h"
#include "../Viewer2d.h"

#include "cloth\clothPiece.h"
#include "cloth\TransformInfo.h"

Translate3dEventHandle::Translate3dEventHandle(Viewer3d* v) : Abstract3dEventHandle(v)
{
	m_cursor = QCursor(Qt::CursorShape::SizeAllCursor);
	m_iconFile = "icons/translation.png";
	m_inactiveIconFile = "icons/translation_inactive.png";
	m_toolTips = "translate handle";
	m_axisScale = 0.3;
}

Translate3dEventHandle::~Translate3dEventHandle()
{

}

void Translate3dEventHandle::handleEnter()
{
	Abstract3dEventHandle::handleEnter();
	if (m_viewer->getManager())
		m_viewer->getManager()->simulationDestroy();
	if (m_pickInfo.mesh)
	{
		auto box = m_pickInfo.mesh->boundingBox;
		m_viewer->beginTrackBall(Viewer3d::TrackBall_Trans, m_pickInfo.pickPos, 
			ldp::Mat3d().eye(), (box[1] - box[0]).length()* m_axisScale);
	}
	m_transformed = false;
}

void Translate3dEventHandle::handleLeave()
{
	m_viewer->endTrackBall();
	Abstract3dEventHandle::handleLeave();
}

void Translate3dEventHandle::mousePressEvent(QMouseEvent *ev)
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

void Translate3dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseReleaseEvent(ev);

	if (m_transformed)
	{
		if (m_viewer->getMainUI())
			m_viewer->getMainUI()->pushHistory("translate piece", ldp::HistoryStack::Type3dTransform);
		m_transformed = false;
	}

	if (m_viewer->buttons() == Qt::LeftButton)
	{
		if (m_viewer->getActiveTrackBallAxis() > 0)
		{
			m_viewer->setActiveTrackBallAxis(-1);
		} // end if track ball axis active
		else
		{
			if (m_pickInfo.mesh && ev->pos() == m_mouse_press_pt)
			{
				auto box = m_pickInfo.mesh->boundingBox;
				m_viewer->beginTrackBall(Viewer3d::TrackBall_Trans, m_pickInfo.pickPos,
					ldp::Mat3d().eye(), (box[1] - box[0]).length()* m_axisScale);
			}
		} // end else
	} // end if left button and initial_cloth
}

void Translate3dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseDoubleClickEvent(ev);
}

void Translate3dEventHandle::mouseMoveEvent(QMouseEvent *ev)
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
			if (m_pickInfo.piece) // cloth mesh
			{
				m_pickInfo.piece->transformInfo().translate(dir);
				m_viewer->getManager()->updateCloths3dMeshBy2d();
			}
			else // body mesh
			{
				auto tr = m_viewer->getManager()->getBodyMeshTransform();
				tr.translate(dir);
				m_viewer->getManager()->setBodyMeshTransform(tr);
			}
			m_transformed = true;
			m_viewer->translateTrackBall(dir);
			if (m_viewer->getMainUI())
				m_viewer->getMainUI()->viewer2d()->updateGL();
			valid_op = true;
		} // end if getPickedMeshFrameInfo
	} // end if initial_cloth and left button
	if (!valid_op)
		Abstract3dEventHandle::mouseMoveEvent(ev);
}

void Translate3dEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract3dEventHandle::wheelEvent(ev);
}

void Translate3dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyPressEvent(ev);
}

void Translate3dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyReleaseEvent(ev);
}
