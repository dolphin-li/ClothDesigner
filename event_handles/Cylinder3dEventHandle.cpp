#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"
#include "ldpMat\Quaternion.h"
#include "Cylinder3dEventHandle.h"
#include "Renderable\ObjMesh.h"
#include "cloth\clothManager.h"
#include "../clothdesigner.h"
#include "../Viewer2d.h"

#include "cloth\clothPiece.h"
#include "cloth\TransformInfo.h"

Cylinder3dEventHandle::Cylinder3dEventHandle(Viewer3d* v) : Abstract3dEventHandle(v)
{
	m_cursor = QCursor(Qt::CursorShape::CrossCursor);
	m_iconFile = "icons/cylinder.png";
	m_inactiveIconFile = "icons/cylinder.png";
	m_toolTips = "make cylinder handle";
	m_axisScale = 0.3;
}

Cylinder3dEventHandle::~Cylinder3dEventHandle()
{

}


inline ldp::Mat3f rotationFromCylinderAxis(const ldp::TransformInfo& info)
{
	auto axis = info.cylinderTransform().axis;
	auto saxis = ldp::Float3(0, 1, 0);
	return info.transform().getRotationPart() * ldp::QuaternionF().fromRotationVecs(axis, saxis).toRotationMatrix3();
}

void Cylinder3dEventHandle::handleEnter()
{
	Abstract3dEventHandle::handleEnter();
	if (m_viewer->getManager())
		m_viewer->getManager()->simulationDestroy();
	if (m_pickInfo.piece)
	{
		auto box = m_pickInfo.piece->mesh3d().boundingBox;
		m_cylinderAxisMouseClick = m_pickInfo.piece->transformInfo().cylinderTransform().axis;
		m_cylinderRadiusMouseClick = m_pickInfo.piece->transformInfo().cylinderTransform().radius;
		m_viewer->beginTrackBall(Viewer3d::TrackBall_Cylinder, m_pickInfo.meshCenter,
			rotationFromCylinderAxis(m_pickInfo.piece->transformInfo()), (box[1] - box[0]).length()* m_axisScale);
	}
	m_transformed = false;
}

void Cylinder3dEventHandle::handleLeave()
{
	m_viewer->endTrackBall();
	Abstract3dEventHandle::handleLeave();
}

void Cylinder3dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mousePressEvent(ev);
	if (m_viewer->buttons() == Qt::LeftButton)
	{
		int sid = m_viewer->fboRenderedIndex(ev->pos());
		m_viewer->setHoverTrackBallAxis(sid);
		if (sid >= Viewer3d::TrackBallIndex_X && sid <= Viewer3d::TrackBallIndex_Z && m_pickInfo.piece)
		{
			m_viewer->setActiveTrackBallAxis(sid);
			m_cylinderAxisMouseClick = m_pickInfo.piece->transformInfo().cylinderTransform().axis;
			m_cylinderRadiusMouseClick = m_pickInfo.piece->transformInfo().cylinderTransform().radius;
		}
		else
			pick(ev->pos());
	} // end if initial_location and left button
}

void Cylinder3dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseReleaseEvent(ev);

	if (m_transformed)
	{
		if (m_viewer->getMainUI())
			m_viewer->getMainUI()->pushHistory("change piece cylinder", ldp::HistoryStack::Type3dTransform);
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
			if (m_pickInfo.piece && ev->pos() == m_mouse_press_pt)
			{
				auto box = m_pickInfo.piece->mesh3d().boundingBox;
				m_cylinderAxisMouseClick = m_pickInfo.piece->transformInfo().cylinderTransform().axis;
				m_cylinderRadiusMouseClick = m_pickInfo.piece->transformInfo().cylinderTransform().radius;
				m_viewer->beginTrackBall(Viewer3d::TrackBall_Cylinder, m_pickInfo.meshCenter,
					rotationFromCylinderAxis(m_pickInfo.piece->transformInfo()), (box[1] - box[0]).length()* m_axisScale);
			}
		} // end else
	} // end if left button and initial_cloth
}

void Cylinder3dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseDoubleClickEvent(ev);
}

void Cylinder3dEventHandle::mouseMoveEvent(QMouseEvent *ev)
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
		if (m_pickInfo.piece)
		{
			QPoint lp = m_mouse_press_pt;

			auto& info = pickInfo().piece->transformInfo();
			auto& cTrans = info.cylinderTransform();
			const ldp::Float3 c = m_pickInfo.meshCenter;
			auto lastR = rotationFromCylinderAxis(info);

			//// compute the angle changed
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
			auto partR = ldp::Mat3f().eye();
			if (m_pickInfo.piece) // cloth mesh
				partR = m_pickInfo.piece->transformInfo().transform().getRotationPart();
			else
				partR = m_viewer->getManager()->getBodyMeshTransform().transform().getRotationPart();
			axis = partR * axis;
			ldp::Float3 c_uvd = m_viewer->camera().getScreenCoords(c);
			ldp::Float3 lp3(lp.x(), m_viewer->height() - 1 - lp.y(), c_uvd[2]);
			ldp::Float3 p3(ev->x(), m_viewer->height() - 1 - ev->y(), c_uvd[2]);
			lp3 = m_viewer->camera().getWorldCoords(lp3);
			p3 = m_viewer->camera().getWorldCoords(p3);
			ldp::Float3 dl = lp3 - c, d = p3 - c;
			dl = (dl - dl.dot(axis)*axis).normalize();
			d = (d - d.dot(axis)*axis).normalize();
			auto Q = ldp::QuaternionF().fromRotationVecs(dl, d);
			float ag = 0;
			ldp::Float3 axis1;
			Q.toAngleAxis(axis1, ag);
			if ((axis1 - axis).length() < 1e-2)
				ag = -ag;

			// if z selected, we rotate the cylinder axis
			if (sid == (int)Viewer3d::TrackBallIndex_Z)
			{
				cTrans.axis = ldp::QuaternionF().fromAngleAxis(ag, ldp::Float3(0, 0, 1)).
					toRotationMatrix3() * m_cylinderAxisMouseClick;
			} // end if z selected
			// if y selected, we change the cylinder radius
			else if (sid == (int)Viewer3d::TrackBallIndex_Y)
			{
				cTrans.radius = info.cylinderCalcRadiusFromAngle(m_pickInfo.piece->mesh2d(),
					info.cylinderCalcAngleFromRadius(m_pickInfo.piece->mesh2d(), 
					m_cylinderRadiusMouseClick) + ag);
			} // end if y selected

			// update the piece
			m_viewer->getManager()->updateCloths3dMeshBy2d();
			m_viewer->rotateTrackBall(rotationFromCylinderAxis(info) * lastR.inv());
			m_transformed = true;

			if (m_viewer->getMainUI())
				m_viewer->getMainUI()->viewer2d()->updateGL();
			valid_op = true;
		} // end if getPickedMeshFrameInfo
	} // end if initial_cloth and left button
	if (!valid_op)
		Abstract3dEventHandle::mouseMoveEvent(ev);
}

void Cylinder3dEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract3dEventHandle::wheelEvent(ev);
}

void Cylinder3dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyPressEvent(ev);
}

void Cylinder3dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyReleaseEvent(ev);
}
