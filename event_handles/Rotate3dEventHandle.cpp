#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"
#include "ldpMat\Quaternion.h"
#include "Rotate3dEventHandle.h"
#include "Renderable\ObjMesh.h"
#include "cloth\clothManager.h"

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
	if (m_viewer->getManager())
		m_viewer->getManager()->simulationDestroy();
	if (m_pickInfo.mesh)
	{
		auto box = m_pickInfo.mesh->boundingBox;
		m_trackBallMouseClickR.eye();
		if (m_accumulatedRots.find(m_pickInfo.mesh) == m_accumulatedRots.end())
			m_accumulatedRots[m_pickInfo.mesh] = m_trackBallMouseClickR;
		else
			m_trackBallMouseClickR = m_accumulatedRots[m_pickInfo.mesh];
		m_viewer->beginTrackBall(Viewer3d::TrackBall_Rot, m_pickInfo.meshCenter,
			m_trackBallMouseClickR, (box[1] - box[0]).length()* m_axisScale);
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
		{
			m_viewer->setActiveTrackBallAxis(sid);

			m_trackBallMouseClickR.eye();
			if (m_accumulatedRots.find(m_pickInfo.mesh) == m_accumulatedRots.end())
				m_accumulatedRots[m_pickInfo.mesh] = m_trackBallMouseClickR;
			else
				m_trackBallMouseClickR = m_accumulatedRots[m_pickInfo.mesh];
		}
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
			if (m_pickInfo.mesh && ev->pos() == m_mouse_press_pt)
			{
				auto box = m_pickInfo.mesh->boundingBox;
				m_trackBallMouseClickR.eye();
				if (m_accumulatedRots.find(m_pickInfo.mesh) == m_accumulatedRots.end())
					m_accumulatedRots[m_pickInfo.mesh] = m_trackBallMouseClickR;
				else
					m_trackBallMouseClickR = m_accumulatedRots[m_pickInfo.mesh];
				m_viewer->beginTrackBall(Viewer3d::TrackBall_Rot, m_pickInfo.meshCenter,
					m_trackBallMouseClickR, (box[1] - box[0]).length()* m_axisScale);
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
			QPoint lp = m_mouse_press_pt;
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
			axis = m_trackBallMouseClickR * axis;
			const ldp::Float3 c = m_pickInfo.meshCenter;
			ldp::Float3 c1 = c + axis;
			ldp::Float3 c_uvd = m_viewer->camera().getScreenCoords(c);
			ldp::Float3 c1_uvd = m_viewer->camera().getScreenCoords(c1);
			ldp::Float2 c_uv(c_uvd[0] / c_uvd[2], c_uvd[1] / c_uvd[2]);
			c_uv[1] = m_viewer->camera().getViewPortBottom() - c_uv[1];
			ldp::Float2 d1 = (ldp::Float2(ev->x(), ev->y()) - c_uv).normalize();
			ldp::Float2 d2 = (ldp::Float2(lp.x(), lp.y()) - c_uv).normalize();
			float ag = atan2(d1.cross(d2), d1.dot(d2));
			if (c_uvd[2] < c1_uvd[2]) ag = -ag;
			auto R = ldp::QuaternionF().fromAngleAxis(ag, axis).toRotationMatrix3() * m_trackBallMouseClickR;

			auto lastR = m_accumulatedRots[m_pickInfo.mesh];
			m_pickInfo.mesh->rotateBy(R*lastR.trans(), m_pickInfo.meshCenter);
			m_viewer->rotateTrackBall(R*lastR.trans());
			m_viewer->getManager()->updateCurrentClothsToInitial();
			m_accumulatedRots[m_pickInfo.mesh] = R;
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
