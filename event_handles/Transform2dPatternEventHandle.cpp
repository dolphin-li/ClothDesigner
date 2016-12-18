#include <QEvent>
#include <GL\glew.h>
#include "Viewer2d.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
#include "cloth\HistoryStack.h"
#include "cloth\graph\Graph.h"
#include "cloth\graph\GraphPoint.h"
#include "cloth\graph\AbstractGraphCurve.h"
#include "cloth\TransformInfo.h"
#include "..\clothdesigner.h"

#include "Transform2dPatternEventHandle.h"

Transform2dPatternEventHandle::Transform2dPatternEventHandle(Viewer2d* v)
: Abstract2dEventHandle(v)
{
	QString name = "icons/pattern_transform.png";
	QPixmap img(name);
	img = img.scaledToWidth(32, Qt::TransformationMode::SmoothTransformation);
	m_cursor = QCursor(img, 1, 1);
	m_iconFile = name;
	m_inactiveIconFile = name;
	m_toolTips = "transform pattern";
}

Transform2dPatternEventHandle::~Transform2dPatternEventHandle()
{

}

void Transform2dPatternEventHandle::handleEnter()
{
	Abstract2dEventHandle::handleEnter();
	m_viewer->setFocus();
	m_transformed = false;
}

void Transform2dPatternEventHandle::handleLeave()
{
	m_viewer->clearFocus();
	m_viewer->endDragBox();
	Abstract2dEventHandle::handleLeave();
	m_transformed = false;
}

void Transform2dPatternEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mousePressEvent(ev);

	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	// handle selection for all buttons
	if (ev->buttons() != Qt::MidButton)
	{
		pick(ev->pos());
		if (pickInfo().renderId == 0)
			m_viewer->beginDragBox(ev->pos());
	}

	// translation start positon
	ldp::Float3 p3(ev->x(), m_viewer->height() - 1 - ev->y(), 1);
	p3 = m_viewer->camera().getWorldCoords(p3);
	m_translateStart = ldp::Float2(p3[0], p3[1]);

	// rotation center postion
	ldp::Float2 bMin = FLT_MAX, bMax = -FLT_MAX;
	for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
	{
		auto piece = manager->clothPiece(iPiece);
		auto& panel = piece->graphPanel();
		if (pickInfo().renderId == panel.getId() || panel.isSelected())
		{
			for (int k = 0; k < 3; k++)
			{
				bMin[k] = std::min(bMin[k], panel.bound()[0][k]);
				bMax[k] = std::max(bMax[k], panel.bound()[1][k]);
			}
		}
	} // end for iPiece
	m_rotateCenter = (bMin + bMax) / 2.f;
}

void Transform2dPatternEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	// handle transfrom triangulation update
	if (m_viewer->getMainUI() && m_transformed)
	{
		m_viewer->getManager()->triangulate();
		m_viewer->getMainUI()->pushHistory(QString().sprintf("pattern transform: %d",
			pickInfo().renderId), ldp::HistoryStack::TypeGeneral);
		m_transformed = false;
	}

	// handle selection ------------------------------------------------------------------
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
			m_viewer->getMainUI()->pushHistory(QString().sprintf("pattern select: %d",
			pickInfo().renderId), ldp::HistoryStack::TypePatternSelect);
	} // end if single selection
	else if (m_viewer->isDragBoxMode())
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
			m_viewer->getMainUI()->pushHistory(QString().sprintf("pattern select: %d...",
			*ids.begin()), ldp::HistoryStack::TypePatternSelect);
	} // end else group selection

	m_viewer->endDragBox();
	Abstract2dEventHandle::mouseReleaseEvent(ev);
}

void Transform2dPatternEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseDoubleClickEvent(ev);
}

void Transform2dPatternEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	Abstract2dEventHandle::mouseMoveEvent(ev);

	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	if (!m_viewer->isDragBoxMode())
	{
		if (panelLevelTransform_MouseMove(ev))
			return;
		if (curveLevelTransform_MouseMove(ev))
			return;
		if (pointLevelTransform_MouseMove(ev))
			return;
	}
}

void Transform2dPatternEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract2dEventHandle::wheelEvent(ev);
}

void Transform2dPatternEventHandle::keyPressEvent(QKeyEvent *ev)
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
	case Qt::Key_A:
		if (ev->modifiers() == Qt::CTRL)
			op = ldp::AbstractGraphObject::SelectAll;
		break;
	case Qt::Key_D:
		if (ev->modifiers() == Qt::CTRL)
			op = ldp::AbstractGraphObject::SelectNone;
		break;
	case Qt::Key_I:
		if (ev->modifiers() == (Qt::CTRL | Qt::SHIFT))
			op = ldp::AbstractGraphObject::SelectInverse;
		break;
	}

	for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
	{
		auto piece = manager->clothPiece(iPiece);
		auto& panel = piece->graphPanel();
		panel.select(0, op);
	} // end for iPiece
}

void Transform2dPatternEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract2dEventHandle::keyReleaseEvent(ev);
}

bool Transform2dPatternEventHandle::panelLevelTransform_MouseMove(QMouseEvent* ev)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return false;
	ldp::Float3 lp3(m_viewer->lastMousePos().x(), m_viewer->height() - 1 - m_viewer->lastMousePos().y(), 1);
	ldp::Float3 p3(ev->x(), m_viewer->height() - 1 - ev->y(), 1);
	lp3 = m_viewer->camera().getWorldCoords(lp3);
	p3 = m_viewer->camera().getWorldCoords(p3);
	ldp::Float2 lp(lp3[0], lp3[1]);
	ldp::Float2 p(p3[0], p3[1]);

	bool changed = false;
	// left button, translate ------------------------------------------------------
	if (ev->buttons() == Qt::LeftButton && !(ev->modifiers() & Qt::ALT))
	{
		for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
		{
			auto piece = manager->clothPiece(iPiece);
			auto& panel = piece->graphPanel();
			if (panel.isHighlighted() || panel.isSelected())
			{
				for (auto iter = panel.pointBegin(); iter != panel.pointEnd(); ++iter)
				{
					auto kp = iter->second.get();
					kp->position() += p - lp;
				}
				panel.updateBound();
				m_transformed = true;
				changed = true;
				// reverse move 3D piece if wanted
				if (ev->modifiers() & Qt::SHIFT)
				{
					auto R3 = piece->transformInfo().transform().getRotationPart();
					auto dif = p - lp;
					piece->transformInfo().translate(R3 * ldp::Float3(-dif[0], -dif[1], 0));
				}
			}
		} // end for iPiece
	} // end if left button

	// right button, rotate -------------------------------------------------
	if (ev->buttons() == Qt::RightButton && !(ev->modifiers() & Qt::ALT))
	{
		ldp::Float2 ld = (lp - m_rotateCenter).normalize();
		ldp::Float2 d = (p - m_rotateCenter).normalize();
		float ltheta = atan2(ld[1], ld[0]);
		float theta = atan2(d[1], d[0]);
		float dr = theta - ltheta;
		ldp::Mat2f R;
		R(0, 0) = cos(dr);		R(0, 1) = -sin(dr);
		R(1, 0) = sin(dr);		R(1, 1) = cos(dr);
		for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
		{
			auto piece = manager->clothPiece(iPiece);
			auto& panel = piece->graphPanel();
			if (panel.isHighlighted() || panel.isSelected())
			{
				for (auto iter = panel.pointBegin(); iter != panel.pointEnd(); ++iter)
				{
					auto kp = iter->second.get();
					kp->position() = R*(kp->position()-m_rotateCenter) + m_rotateCenter;
				}
				panel.updateBound();
				m_transformed = true;
				changed = true;
				// reverse move 3D piece if wanted
				if (ev->modifiers() & Qt::SHIFT)
				{
					auto R3 = piece->transformInfo().transform().getRotationPart();
					auto t3 = piece->transformInfo().transform().getTranslationPart();
					ldp::Float3 c(m_rotateCenter[0], m_rotateCenter[1], 0);
					ldp::Mat3f R2 = ldp::Mat3f().eye();
					R2(0, 0) = R(0, 0);		R2(0, 1) = R(0, 1);
					R2(1, 0) = R(1, 0);		R2(1, 1) = R(1, 1);
					piece->transformInfo().rotate(R3*R2.inv()*R3.inv(), t3 + R3*c - R3*R2*c);
					piece->transformInfo().translate(R3*R2*c - R3*c);
				}
			}
		} // end for iPiece
	} // end if right button

	// right button + ALT, scale -------------------------------------------------
	if (ev->buttons() == Qt::RightButton && (ev->modifiers() & Qt::ALT))
	{
		float ld = (lp - m_rotateCenter).length();
		float d = (p - m_rotateCenter).length();
		float s = d / ld;
		for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
		{
			auto piece = manager->clothPiece(iPiece);
			auto& panel = piece->graphPanel();
			if (pickInfo().renderId == panel.getId() || panel.isSelected())
			{
				for (auto iter = panel.pointBegin(); iter != panel.pointEnd(); ++iter)
				{
					auto kp = iter->second.get();
					kp->position() = s*(kp->position() - m_rotateCenter) + m_rotateCenter;
				}
				panel.updateBound();
				m_transformed = true;
				changed = true;
				// reverse move 3D piece if wanted
				if (ev->modifiers() & Qt::SHIFT)
				{
					auto R3 = piece->transformInfo().transform().getRotationPart();
					auto t3 = piece->transformInfo().transform().getTranslationPart();
					ldp::Float3 c(m_rotateCenter[0], m_rotateCenter[1], 0);
					piece->transformInfo().scale(1.f / s, t3 + R3*c - R3*s*c);
					piece->transformInfo().translate(R3*s*c - R3*c);
				}
			}
		} // end for iPiece
	} // end if right button + ALT
	return changed;
} 

bool Transform2dPatternEventHandle::curveLevelTransform_MouseMove(QMouseEvent* ev)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return false;
	ldp::Float3 lp3(m_viewer->lastMousePos().x(), m_viewer->height() - 1 - m_viewer->lastMousePos().y(), 1);
	ldp::Float3 p3(ev->x(), m_viewer->height() - 1 - ev->y(), 1);
	lp3 = m_viewer->camera().getWorldCoords(lp3);
	p3 = m_viewer->camera().getWorldCoords(p3);
	ldp::Float2 lp(lp3[0], lp3[1]);
	ldp::Float2 p(p3[0], p3[1]);

	bool changed = false;
	// left button, translate ------------------------------------------------------
	if (ev->buttons() == Qt::LeftButton && !(ev->modifiers() & Qt::ALT))
	{
		std::hash_set<ldp::GraphPoint*> validKpts;
		for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
		{
			ldp::ClothPiece* piece = manager->clothPiece(iPiece);
			ldp::Graph& panel = piece->graphPanel();
			for (auto iter = panel.curveBegin(); iter != panel.curveEnd(); ++iter)
			{
				auto cv = iter->second.get();
				if (cv->isSelected() || cv->isHighlighted())
				{
					for (int i = 0; i < cv->numKeyPoints(); i++)
						validKpts.insert(cv->keyPoint(i));
					m_transformed = true;
					changed = true;
				}
			}
		} // end for iPiece
		for (auto iter : validKpts)
			iter->position() += p - lp;
	} // end if left button
	return changed;
}

bool Transform2dPatternEventHandle::pointLevelTransform_MouseMove(QMouseEvent* ev)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return false;
	ldp::Float3 lp3(m_viewer->lastMousePos().x(), m_viewer->height() - 1 - m_viewer->lastMousePos().y(), 1);
	ldp::Float3 p3(ev->x(), m_viewer->height() - 1 - ev->y(), 1);
	lp3 = m_viewer->camera().getWorldCoords(lp3);
	p3 = m_viewer->camera().getWorldCoords(p3);
	ldp::Float2 lp(lp3[0], lp3[1]);
	ldp::Float2 p(p3[0], p3[1]);

	bool changed = false;
	// left button, translate ------------------------------------------------------
	if (ev->buttons() == Qt::LeftButton && !(ev->modifiers() & Qt::ALT))
	{
		for (size_t iPiece = 0; iPiece < manager->numClothPieces(); iPiece++)
		{
			ldp::ClothPiece* piece = manager->clothPiece(iPiece);
			ldp::Graph& panel = piece->graphPanel();
			for (auto iter = panel.pointBegin(); iter != panel.pointEnd(); ++iter)
			{
				auto kp = iter->second.get();
				if (kp->isSelected() || kp->isHighlighted())
				{
					kp->position() += p - lp;
					m_transformed = true;
					changed = true;
				}
			}
		} // end for iPiece
	} // end if left button
	return changed;
}
