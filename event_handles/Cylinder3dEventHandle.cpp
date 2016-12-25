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

			// compute the angle changed
			ldp::Float3 c1 = c + m_cylinderAxisMouseClick;
			ldp::Float3 c_uvd = m_viewer->camera().getScreenCoords(c);
			ldp::Float3 c1_uvd = m_viewer->camera().getScreenCoords(c1);
			ldp::Float2 c_uv(c_uvd[0] / c_uvd[2], c_uvd[1] / c_uvd[2]);
			c_uv[1] = m_viewer->camera().getViewPortBottom() - c_uv[1];
			ldp::Float2 d1 = (ldp::Float2(ev->x(), ev->y()) - c_uv).normalize();
			ldp::Float2 d2 = (ldp::Float2(lp.x(), lp.y()) - c_uv).normalize();
			float ag = atan2(d1.cross(d2), d1.dot(d2));
			if (c_uvd[2] < c1_uvd[2]) ag = -ag;

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
