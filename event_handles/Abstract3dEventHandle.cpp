#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"
#include "cloth\clothManager.h"
#include "cloth\clothPiece.h"
#include "Renderable\ObjMesh.h"

#include "Abstract3dEventHandle.h"
#include "Translate3dEventHandle.h"
#include "Select3dEventHandle.h"
#include "Rotate3dEventHandle.h"
Abstract3dEventHandle::Abstract3dEventHandle(Viewer3d* v)
{
	m_viewer = v;
	m_lastHighlightShapeId = -1;
	m_currentSelectedId = -1;
	m_cursor = QCursor(Qt::CursorShape::ArrowCursor);
	m_iconFile = "";
	m_inactiveIconFile = "";
	m_toolTips = "general handle";
}

Abstract3dEventHandle::~Abstract3dEventHandle()
{

}

QString Abstract3dEventHandle::iconFile()const
{
	return m_iconFile;
}

QString Abstract3dEventHandle::inactiveIconFile()const
{
	return m_inactiveIconFile;
}

void Abstract3dEventHandle::handleEnter()
{
	m_viewer->setFocus();
	m_pickInfo.mesh = nullptr;
}
void Abstract3dEventHandle::handleLeave()
{
	m_viewer->clearFocus();
	m_pickInfo.mesh = nullptr;
}

QString Abstract3dEventHandle::toolTips()const
{
	return m_toolTips;
}

void Abstract3dEventHandle::getSelectionRay(QPoint mousePos, ldp::Float3& p, ldp::Float3& q)const
{
	p = m_viewer->camera().getWorldCoords(ldp::Float3(mousePos.x(), m_viewer->height() - 1 - mousePos.y(), -1));
	q = m_viewer->camera().getWorldCoords(ldp::Float3(mousePos.x(), m_viewer->height() - 1 - mousePos.y(), 1));
}

void Abstract3dEventHandle::pick(QPoint pos)
{
	auto manager = m_viewer->getManager();
	if (manager == nullptr)
		return;

	int renderedId = m_viewer->fboRenderedIndex(pos);
	if (renderedId >= m_viewer->FaceIndex && renderedId < m_viewer->TrackBallIndex_X)
	{
		// 1. pick the body mesh
		bool picked = false;
		int curIdx = m_viewer->FaceIndex;
		auto mesh0 = manager->bodyMesh();
		if (renderedId >= curIdx && renderedId < curIdx + mesh0->face_list.size())
		{
			m_pickInfo.mesh = mesh0;
			m_pickInfo.faceId = renderedId - curIdx;
			picked = true;
		}
		curIdx += mesh0->face_list.size();

		// 2. pick the cloth meshes
		if (!picked)
		{
			for (int iMesh = 0; iMesh < manager->numClothPieces(); iMesh++)
			{
				auto mesh = &manager->clothPiece(iMesh)->mesh3d();
				if (renderedId >= curIdx && renderedId < curIdx + mesh->face_list.size())
				{
					m_pickInfo.mesh = mesh;
					m_pickInfo.faceId = renderedId - curIdx;
					picked = true;
					break;
				}
				curIdx += mesh->face_list.size();
			}
		}

		// 3. if picked, we compute the detailed info
		if (picked)
		{
			ldp::Float3 v[3]; // world pos
			for (int k = 0; k < 3; k++)
				v[k] = m_pickInfo.mesh->vertex_list[m_pickInfo.mesh->face_list[m_pickInfo.faceId].vertex_index[k]];
			ldp::Float3 vs[3]; // screen pos
			for (int k = 0; k < 3; k++)
				vs[k] = m_viewer->camera().getScreenCoords(v[k]);
			ldp::Float3 area;
			float totalArea = 0;
			ldp::Float2 s(pos.x(), m_viewer->height() - 1 - pos.y());
			for (int k = 0; k < 3; k++)
			{
				ldp::Float2 e1(vs[k][0] - s[0], vs[k][1] - s[1]);
				int k1 = (k + 1) % 3;
				ldp::Float2 e2(vs[k1][0] - s[0], vs[k1][1] - s[1]);
				area[k] = e1.cross(e2);
				totalArea += area[k];
			}
			if (fabs(totalArea) < std::numeric_limits<float>::epsilon())
			{
				m_pickInfo.mesh = nullptr;
				return;
			}
			area /= totalArea;
			m_pickInfo.pickInnerCoords = area;
			m_pickInfo.screenPos = vs[0] * area[0] + vs[1] * area[1] + vs[2] * area[2];
			m_pickInfo.pickPos = v[0] * area[0] + v[1] * area[1] + v[2] * area[2];

			auto box = m_pickInfo.mesh->boundingBox;
			m_pickInfo.meshCenter = (box[0] + box[1]) * 0.5f;
		}
	} // end for id
}

Abstract3dEventHandle* Abstract3dEventHandle::create(ProcessorType type, Viewer3d* v)
{
	switch (type)
	{
	case Abstract3dEventHandle::ProcessorTypeGeneral:
		return new Abstract3dEventHandle(v);
	case Abstract3dEventHandle::ProcessorTypeSelect:
		return new Select3dEventHandle(v);
	case Abstract3dEventHandle::ProcessorTypeTranslate:
		return new Translate3dEventHandle(v);
	case Abstract3dEventHandle::ProcessorTypeRotate:
		return new Rotate3dEventHandle(v);
	case Abstract3dEventHandle::ProcessorTypeEnd:
	default:
		return nullptr;
	}
}

void Abstract3dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	m_mouse_press_pt = ev->pos();

	// arcball drag
	if (ev->buttons() == Qt::LeftButton)
		m_viewer->camera().arcballClick(ldp::Float2(ev->x(), ev->y()));
}

void Abstract3dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{

}

void Abstract3dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	if (ev->button() == Qt::MouseButton::MiddleButton)
		m_viewer->resetCamera();
}

void Abstract3dEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	if (ev->buttons() == Qt::LeftButton)
		m_viewer->camera().arcballDrag(ldp::Float2(ev->x(), ev->y()));
	if (ev->buttons() == Qt::MidButton)
	{
		QPoint dif = ev->pos() - m_viewer->lastMousePos();
		ldp::Float3 bmin, bmax;
		m_viewer->getModelBound(bmin, bmax);
		float len = (bmax - bmin).length() / sqrt(3.f);
		ldp::Float3 t(-(float)dif.x() / m_viewer->width(), (float)dif.y() / m_viewer->height(), 0);
		m_viewer->camera().translate(t * len);
		m_viewer->camera().arcballSetCenter((bmin + bmax) / 2.f + t * len);
	}
}

void Abstract3dEventHandle::wheelEvent(QWheelEvent *ev)
{
	float s = 1.2;
	if (ev->delta() < 0)
		s = 1.f / s;

	ldp::Camera& cam = m_viewer->camera();
	ldp::Float3 c = cam.getLocation();
	ldp::Float3 c0 = cam.arcballGetCenter();
	cam.setLocation((c-c0)*s + c0);
	//float fov = std::max(1e-3f, std::min(160.f, m_viewer->camera().getFov()*s));
	//m_viewer->camera().setPerspective(fov, m_viewer->camera().getAspect(),
	//	m_viewer->camera().getFrustumNear(), m_viewer->camera().getFrustumFar());
}

void Abstract3dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	switch (ev->key())
	{
	default:
		break;
	}
}

void Abstract3dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{

}
