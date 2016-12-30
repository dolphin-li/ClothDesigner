#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"
#include "ldpMat\Quaternion.h"
#include "Rotate3dEventHandle.h"
#include "Renderable\ObjMesh.h"
#include "cloth\clothManager.h"
#include "../clothdesigner.h"
#include "../Viewer2d.h"

#include "cloth\clothPiece.h"
#include "cloth\TransformInfo.h"

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
		if (m_pickInfo.piece) // cloth mesh
			m_trackBallMouseClickR = m_pickInfo.piece->transformInfo().transform().getRotationPart();
		else
			m_trackBallMouseClickR = m_viewer->getManager()->getBodyMeshTransform().transform().getRotationPart();
		m_viewer->beginTrackBall(Viewer3d::TrackBall_Rot, m_pickInfo.meshCenter,
			m_trackBallMouseClickR, (box[1] - box[0]).length()* m_axisScale);
	}
	m_transformed = false;
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
			if (m_pickInfo.piece) // cloth mesh
				m_trackBallMouseClickR = m_pickInfo.piece->transformInfo().transform().getRotationPart();
			else
				m_trackBallMouseClickR = m_viewer->getManager()->getBodyMeshTransform().transform().getRotationPart();
		}
	} // end if initial_location and left button
}

void Rotate3dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseReleaseEvent(ev);

	if (m_transformed)
	{
		if (m_viewer->getMainUI())
			m_viewer->getMainUI()->pushHistory("rotate piece", ldp::HistoryStack::Type3dTransform);
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
			if (ev->pos() == m_mouse_press_pt)
			{
				pick(ev->pos());
				if (m_pickInfo.mesh)
				{
					auto box = m_pickInfo.mesh->boundingBox;
					m_trackBallMouseClickR.eye();

					if (m_pickInfo.piece) // cloth mesh
						m_trackBallMouseClickR = m_pickInfo.piece->transformInfo().transform().getRotationPart();
					else
						m_trackBallMouseClickR = m_viewer->getManager()->getBodyMeshTransform().transform().getRotationPart();
					m_viewer->beginTrackBall(Viewer3d::TrackBall_Rot, m_pickInfo.meshCenter,
						m_trackBallMouseClickR, (box[1] - box[0]).length()* m_axisScale);
				}
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
			ldp::Float3 c_uvd = m_viewer->camera().getScreenCoords(c);
			ldp::Float3 lp3(lp.x(), m_viewer->height() - 1 - lp.y(), c_uvd[2]);
			ldp::Float3 p3(ev->x(), m_viewer->height() - 1 - ev->y(), c_uvd[2]);
			lp3 = m_viewer->camera().getWorldCoords(lp3);
			p3 = m_viewer->camera().getWorldCoords(p3);
			ldp::Float3 dl = lp3 - c, d = p3 - c;
			dl = (dl - dl.dot(axis)*axis).normalize();
			d = (d - d.dot(axis)*axis).normalize();
			auto R = ldp::QuaternionF().fromRotationVecs(dl, d).toRotationMatrix3() * m_trackBallMouseClickR;

			if (m_pickInfo.piece) // cloth mesh
			{
				auto lastR = m_pickInfo.piece->transformInfo().transform().getRotationPart();
				m_pickInfo.piece->transformInfo().rotate(R*lastR.inv(), m_pickInfo.meshCenter);
				m_viewer->getManager()->updateCloths3dMeshBy2d();
				m_viewer->rotateTrackBall(R*lastR.inv());
			}
			else // body mesh
			{
				auto tr = m_viewer->getManager()->getBodyMeshTransform();
				auto lastR = tr.transform().getRotationPart();
				tr.rotate(R*lastR.inv(), m_pickInfo.meshCenter);
				m_viewer->getManager()->setBodyMeshTransform(tr);
				m_viewer->rotateTrackBall(R*lastR.inv());
			}
			m_transformed = true;

			if (m_viewer->getMainUI())
				m_viewer->getMainUI()->viewer2d()->updateGL();
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
