#include <QEvent>
#include <GL\glew.h>
#include "Viewer2d.h"
#include "viewer3d.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
#include "cloth\graph\AbstractGraphCurve.h"
#include "cloth\graph\Graph.h"
#include "cloth\HistoryStack.h"
#include "Renderable\ObjMesh.h"
#include "..\clothdesigner.h"
#include "AddCurve2dEventHandle.h"
AddCurve2dEventHandle::AddCurve2dEventHandle(Viewer2d* v)
: Abstract2dEventHandle(v)
{
	QString name = "icons/curve.png";
	m_cursor = Qt::CursorShape::CrossCursor;
	m_iconFile = name;
	m_inactiveIconFile = name;
	m_toolTips = "add curve";
}

AddCurve2dEventHandle::~AddCurve2dEventHandle()
{

}

void AddCurve2dEventHandle::handleEnter()
{
	Abstract2dEventHandle::handleEnter();
	m_viewer->setFocus();
	m_viewer->beginAddCurveMode();
}
void AddCurve2dEventHandle::handleLeave()
{
	m_viewer->endAddCurveMode();
	m_viewer->clearFocus();
	Abstract2dEventHandle::handleLeave();
}

void AddCurve2dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mousePressEvent(ev);

	if (ev->buttons() == Qt::LeftButton)
	{
		pick(ev->pos());
	} // end if left button
}

void AddCurve2dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	if (m_viewer->buttons() & Qt::LeftButton)
	{
		if (ev->pos() == m_mouse_press_pt)
		{
			if (m_viewer->getUiCurveData().points.size() == 0)
				m_viewer->beginNewCurve(ev->pos());
			else if (m_viewer->getUiCurveData().points.size() < ldp::AbstractGraphCurve::maxKeyPointsNum())
				m_viewer->addCurvePoint(ev->pos(), false);
			if (m_viewer->getUiCurveData().points.size() == ldp::AbstractGraphCurve::maxKeyPointsNum())
			{
				if (m_viewer->endCurve())
				{
					if (m_viewer->getMainUI())
					{
						manager->triangulate();
						m_viewer->getMainUI()->viewer3d()->updateGL();
						m_viewer->getMainUI()->pushHistory("add a curve", ldp::HistoryStack::TypeGeneral);
					}
				}
			}
		} // end if single selection
	}
	m_viewer->endDragBox();
	Abstract2dEventHandle::mouseReleaseEvent(ev);
}

void AddCurve2dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseDoubleClickEvent(ev);
}

void AddCurve2dEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseMoveEvent(ev);
	m_viewer->addCurvePoint(ev->pos(), true);
}

void AddCurve2dEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract2dEventHandle::wheelEvent(ev);
}

void AddCurve2dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyPressEvent(ev);
	auto manager = m_viewer->getManager();
	if (!manager)
		return;
	ldp::AbstractGraphObject::SelectOp op = ldp::AbstractGraphObject::SelectEnd;
	bool changed = false;
	bool shouldTriangulate = false;
	QString str;
	switch (ev->key())
	{
	default:
		break;
	case Qt::Key_Return:
		changed = m_viewer->endCurve();
		shouldTriangulate = changed;
		str = "add a curve";
		break;
	case Qt::Key_Escape:
		m_viewer->giveupCurve();
		break;
	}

	if (m_viewer->getMainUI() && changed)
	{
		if (shouldTriangulate)
			manager->triangulate();
		m_viewer->getMainUI()->viewer3d()->updateGL();
		m_viewer->getMainUI()->pushHistory(str, ldp::HistoryStack::TypeGeneral);
	}
}

void AddCurve2dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyReleaseEvent(ev);
}
