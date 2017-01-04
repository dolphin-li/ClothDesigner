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
#include "Loop2dEventHandle.h"
Loop2dEventHandle::Loop2dEventHandle(Viewer2d* v)
: Abstract2dEventHandle(v)
{
	QString name = "icons/loop.png";
	m_cursor = Qt::CursorShape::ArrowCursor;
	m_iconFile = name;
	m_inactiveIconFile = name;
	m_toolTips = "edit loop";
}

Loop2dEventHandle::~Loop2dEventHandle()
{

}

void Loop2dEventHandle::handleEnter()
{
	Abstract2dEventHandle::handleEnter();
	m_viewer->setFocus();
	m_viewer->beginEditLoopMode();
}
void Loop2dEventHandle::handleLeave()
{
	m_viewer->endEditLoopMode();
	m_viewer->clearFocus();
	Abstract2dEventHandle::handleLeave();
}

void Loop2dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mousePressEvent(ev);

	if (ev->buttons() == Qt::LeftButton)
	{
		pick(ev->pos());
		if (pickInfo().renderId == 0)
			m_viewer->beginDragBox(ev->pos());
	} // end if left button
}

void Loop2dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	if (m_viewer->buttons() & Qt::LeftButton)
	{

		auto op = ldp::AbstractGraphObject::SelectThis;
		if (ev->modifiers() & Qt::SHIFT)
			op = ldp::AbstractGraphObject::SelectUnion;
		if (ev->modifiers() & Qt::CTRL)
			op = ldp::AbstractGraphObject::SelectUnionInverse;
		if (ev->pos() == m_mouse_press_pt)
		{
			bool changed = false;
			for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
			{
				auto piece = manager->clothPiece(iPiece);
				auto& panel = piece->graphPanel();
				if (panel.select(pickInfo().renderId, op))
					changed = true;
			} // end for iPiece
			if (m_viewer->getMainUI() && changed)
			{
				m_viewer->getMainUI()->viewer3d()->updateGL();
				m_viewer->getMainUI()->updateUiByParam();
				m_viewer->getMainUI()->pushHistory(QString().sprintf("pattern select: %d",
					pickInfo().renderId), ldp::HistoryStack::TypePatternSelect);
			}
		} // end if single selection
		else
		{
			const QImage& I = m_viewer->fboImage();
			std::set<int> ids;
			float x0 = std::max(0, std::min(m_mouse_press_pt.x(), ev->pos().x()));
			float x1 = std::min(I.width() - 1, std::max(m_mouse_press_pt.x(), ev->pos().x()));
			float y0 = std::max(0, std::min(m_mouse_press_pt.y(), ev->pos().y()));
			float y1 = std::min(I.height() - 1, std::max(m_mouse_press_pt.y(), ev->pos().y()));
			for (int y = y0; y <= y1; y++)
			for (int x = x0; x <= x1; x++)
				ids.insert(m_viewer->fboRenderedIndex(QPoint(x, y)));
			bool changed = false;
			for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
			{
				auto piece = manager->clothPiece(iPiece);
				auto& panel = piece->graphPanel();
				if (panel.select(ids, op))
					changed = true;
			} // end for iPiece
			if (m_viewer->getMainUI() && changed)
			{
				m_viewer->getMainUI()->viewer3d()->updateGL();
				m_viewer->getMainUI()->updateUiByParam();
				m_viewer->getMainUI()->pushHistory(QString().sprintf("pattern select: %d...",
					*ids.begin()), ldp::HistoryStack::TypePatternSelect);
			}
		} // end else group selection
	}
	m_viewer->endDragBox();
	Abstract2dEventHandle::mouseReleaseEvent(ev);
}

void Loop2dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseDoubleClickEvent(ev);
}

void Loop2dEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseMoveEvent(ev);
	m_viewer->addCurvePoint(ev->pos(), true);
}

void Loop2dEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract2dEventHandle::wheelEvent(ev);
}

void Loop2dEventHandle::keyPressEvent(QKeyEvent *ev)
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
	case Qt::Key_Delete:
		if (ev->modifiers() == Qt::NoModifier)
		{
			if (manager->removeSelectedLoops())
			{
				changed = true;
				shouldTriangulate = true;
				str = "loop removed";
			}
		}
		break;
	case Qt::Key_L:
		if (ev->modifiers() == Qt::NoModifier)
		{
			if (manager->makeSelectedCurvesToLoop())
			{
				changed = true;
				shouldTriangulate = true;
				str = "curves to loop";
			}
		}
		if (ev->modifiers() == Qt::SHIFT)
		{
			if (manager->removeLoopsOfSelectedCurves())
			{
				changed = true;
				shouldTriangulate = true;
				str = "remove loop";
			}
		}
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

void Loop2dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyReleaseEvent(ev);
}
