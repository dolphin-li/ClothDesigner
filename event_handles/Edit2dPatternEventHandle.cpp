#include <QEvent>
#include <GL\glew.h>
#include "Viewer2d.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"

#include "Edit2dPatternEventHandle.h"
Edit2dPatternEventHandle::Edit2dPatternEventHandle(Viewer2d* v)
: Abstract2dEventHandle(v)
{
	m_cursor = QCursor(Qt::CursorShape::CrossCursor);
	m_iconFile = "icons/groupArrow.png";
	m_inactiveIconFile = "icons/groupArrow.png";
	m_toolTips = "edit pattern";
}

Edit2dPatternEventHandle::~Edit2dPatternEventHandle()
{

}

void Edit2dPatternEventHandle::handleEnter()
{
	Abstract2dEventHandle::handleEnter();
	m_viewer->setFocus();
}
void Edit2dPatternEventHandle::handleLeave()
{
	m_viewer->clearFocus();
	m_viewer->endDragBox();
	Abstract2dEventHandle::handleLeave();
}

void Edit2dPatternEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mousePressEvent(ev);

	if (ev->buttons() == Qt::LeftButton)
	{
		pick(ev->pos());
		if (pickInfo().renderId == 0)
			m_viewer->beginDragBox(ev->pos());
	} // end if left button
}

void Edit2dPatternEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	if (m_viewer->buttons() & Qt::LeftButton)
	{
		auto op = ldp::AbstractPanelObject::SelectThis;
		if (ev->modifiers() & Qt::SHIFT)
			op = ldp::AbstractPanelObject::SelectUnion;
		if (ev->modifiers() & Qt::CTRL)
			op = ldp::AbstractPanelObject::SelectUnionInverse;
		if (ev->pos() == m_mouse_press_pt)
		{
			for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
			{
				auto piece = manager->clothPiece(iPiece);
				auto& panel = piece->panel();
				panel.select(pickInfo().renderId, op);
			} // end for iPiece
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
			{
				ids.insert(m_viewer->fboRenderedIndex(QPoint(x,y)));
			}
			for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
			{
				auto piece = manager->clothPiece(iPiece);
				auto& panel = piece->panel();
				panel.select(ids, op);
			} // end for iPiece
		}
	}
	m_viewer->endDragBox();
	Abstract2dEventHandle::mouseReleaseEvent(ev);
}

void Edit2dPatternEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseDoubleClickEvent(ev);
}

void Edit2dPatternEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseMoveEvent(ev);
}

void Edit2dPatternEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract2dEventHandle::wheelEvent(ev);
}

void Edit2dPatternEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyPressEvent(ev);
	auto manager = m_viewer->getManager();
	if (!manager)
		return;
	ldp::AbstractPanelObject::SelectOp op = ldp::AbstractPanelObject::SelectEnd;
	switch (ev->key())
	{
	default:
		break;
	case Qt::Key_A:
		if (ev->modifiers() == Qt::CTRL)
			op = ldp::AbstractPanelObject::SelectAll;
		break;
	case Qt::Key_D:
		if (ev->modifiers() == Qt::CTRL)
			op = ldp::AbstractPanelObject::SelectNone;
		break;
	case Qt::Key_I:
		if (ev->modifiers() == (Qt::CTRL | Qt::SHIFT))
			op = ldp::AbstractPanelObject::SelectInverse;
		break;
	}

	for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
	{
		auto piece = manager->clothPiece(iPiece);
		auto& panel = piece->panel();
		panel.select(0, op);
	} // end for iPiece
}

void Edit2dPatternEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyReleaseEvent(ev);
}
