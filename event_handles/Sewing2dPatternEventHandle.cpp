#include <QEvent>
#include <GL\glew.h>
#include "Viewer2d.h"
#include "Viewer3d.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
#include "Renderable\ObjMesh.h"
#include "cloth\graph\GraphsSewing.h"
#include "../clothdesigner.h"
#include "Sewing2dPatternEventHandle.h"
Sewing2dPatternEventHandle::Sewing2dPatternEventHandle(Viewer2d* v)
: Abstract2dEventHandle(v)
{
	QString name = "icons/pattern_sewing.png";
	QPixmap img(name);
	img = img.scaledToWidth(32, Qt::TransformationMode::SmoothTransformation);
	m_cursor = QCursor(img, 1, 1);
	m_iconFile = name;
	m_inactiveIconFile = name;
	m_toolTips = "edit sewing";
}

Sewing2dPatternEventHandle::~Sewing2dPatternEventHandle()
{

}

void Sewing2dPatternEventHandle::handleEnter()
{
	Abstract2dEventHandle::handleEnter();
	m_viewer->setFocus();
	m_viewer->beginSewingMode();
	m_viewer->deleteCurrentUISew();
}

void Sewing2dPatternEventHandle::handleLeave()
{
	m_viewer->deleteCurrentUISew();
	m_viewer->endSewingMode();
	m_viewer->clearFocus();
	m_viewer->endDragBox();
	Abstract2dEventHandle::handleLeave();
}

void Sewing2dPatternEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mousePressEvent(ev);

	if (ev->buttons() == Qt::LeftButton)
	{
		pick(ev->pos());
		if (pickInfo().renderId == 0)
			m_viewer->beginDragBox(ev->pos());
	} // end if left button
}

void Sewing2dPatternEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	if (m_viewer->buttons() == Qt::LeftButton)
	{
		auto op = ldp::AbstractGraphObject::SelectThis;
		if (ev->modifiers() & Qt::SHIFT)
			op = ldp::AbstractGraphObject::SelectUnion;
		if (ev->modifiers() & Qt::CTRL)
			op = ldp::AbstractGraphObject::SelectUnionInverse;
		if (ev->pos() == m_mouse_press_pt)
		{
			bool changed = false;
			for (size_t iSewing = 0; iSewing < manager->numGraphSewings(); iSewing++)
			if (manager->graphSewing(iSewing)->select(pickInfo().renderId, op))
				changed = true;
			if (m_viewer->getMainUI() && changed)
			{
				m_viewer->getMainUI()->pushHistory(QString().sprintf("sew select: %d",
					pickInfo().renderId), ldp::HistoryStack::TypePatternSelect);
				m_viewer->getMainUI()->updateUiByParam();
			}

			auto obj = ldp::GraphsSewing::getObjByIdx(highLightInfo().renderId);
			if (obj)
			{
				if (obj->isCurve())
					m_viewer->makeSewUnit((ldp::AbstractGraphCurve*)obj, ev->pos());
			}
		}
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
			for (size_t iSewing = 0; iSewing < manager->numGraphSewings(); iSewing++)
			if (manager->graphSewing(iSewing)->select(ids, op))
				changed = true;
			if (m_viewer->getMainUI() && changed)
			{
				m_viewer->getMainUI()->pushHistory(QString().sprintf("sew select: %d",
				pickInfo().renderId), ldp::HistoryStack::TypePatternSelect);
				m_viewer->getMainUI()->updateUiByParam();
			}
		}
	}
	m_viewer->endDragBox();
	Abstract2dEventHandle::mouseReleaseEvent(ev);
}

void Sewing2dPatternEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseDoubleClickEvent(ev);
}

void Sewing2dPatternEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseMoveEvent(ev);

	bool valid = false;
	auto obj = ldp::GraphsSewing::getObjByIdx(highLightInfo().renderId);
	if (obj)
	{
		if (obj->isCurve())
		{
			m_viewer->makeSewUnit((ldp::AbstractGraphCurve*)obj, ev->pos(), true);
			valid = true;
		}
	}
	if (!valid)
		m_viewer->makeSewUnit(nullptr, ev->pos(), true);
}

void Sewing2dPatternEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract2dEventHandle::wheelEvent(ev);
}

void Sewing2dPatternEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyPressEvent(ev);
	auto manager = m_viewer->getManager();
	if (!manager)
		return;
	ldp::AbstractGraphObject::SelectOp op = ldp::AbstractGraphObject::SelectEnd;
	switch (ev->key())
	{
	default:
		break;
	case Qt::Key_Escape:
		if (ev->modifiers() == Qt::NoModifier)
		{
			m_viewer->deleteCurrentUISew();
		}
		break;
	case Qt::Key_A:
		if (ev->modifiers() == Qt::CTRL)
			op = ldp::AbstractGraphObject::SelectAll;
		if (ev->modifiers() == Qt::NoModifier)
		{
			auto r = m_viewer->setNextSewAddingState();
			if (m_viewer->getMainUI())
			{
				if (r == UiSewAddedToPanel)
				{
					m_viewer->getMainUI()->viewer3d()->updateGL();
					m_viewer->getMainUI()->pushHistory("add a sew", ldp::HistoryStack::TypeGeneral);
				}
				if (r == UiSewUiTmpChanged)
				{
					m_viewer->getMainUI()->viewer3d()->updateGL();
					m_viewer->getMainUI()->pushHistory("add a sew on ui", ldp::HistoryStack::TypeUiSewChanged);
				}
			}
		}
		break;
	case Qt::Key_D:
		if (ev->modifiers() == Qt::CTRL)
			op = ldp::AbstractGraphObject::SelectNone;
		break;
	case Qt::Key_I:
		if (ev->modifiers() == (Qt::CTRL | Qt::SHIFT))
			op = ldp::AbstractGraphObject::SelectInverse;
		break;
	case Qt::Key_Delete:
		if (ev->modifiers() == Qt::NoModifier)
		{
			bool change = manager->removeSelectedSewings();
			if (m_viewer->getMainUI() && change)
			{
				m_viewer->getMainUI()->viewer3d()->updateGL();
				m_viewer->getMainUI()->pushHistory(QString().sprintf("sewing removed",
				op), ldp::HistoryStack::TypeGeneral);
			}
		}
		break;
	case Qt::Key_R:
		if (ev->modifiers() == Qt::NoModifier)
		{
			bool change = manager->reverseSelectedSewings();
			if (m_viewer->getMainUI() && change)
			{
				m_viewer->getMainUI()->viewer3d()->updateGL();
				m_viewer->getMainUI()->pushHistory(QString().sprintf("sewing reversed",
					op), ldp::HistoryStack::TypeGeneral);
			}
		}
		break;
	case Qt::Key_P:
		if (ev->modifiers() == Qt::NoModifier)
		{
			bool change = manager->toggleSelectedSewingsType();
			if (m_viewer->getMainUI() && change)
			{
				m_viewer->getMainUI()->viewer3d()->updateGL();
				m_viewer->getMainUI()->pushHistory(QString().sprintf("sewing type changed",
					op), ldp::HistoryStack::TypeGeneral);
			}
		}
		break;
	}

	bool changed = false;
	for (size_t iSewing = 0; iSewing < manager->numGraphSewings(); iSewing++)
	if (manager->graphSewing(iSewing)->select(0, op))
		changed = true;

	if (m_viewer->getMainUI() && changed)
		m_viewer->getMainUI()->pushHistory(QString().sprintf("sew select: all(%d)",
		op), ldp::HistoryStack::TypePatternSelect);
}

void Sewing2dPatternEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyReleaseEvent(ev);
}
