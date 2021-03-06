#include <QEvent>
#include <GL\glew.h>
#include "Viewer3d.h"

#include "Select3dEventHandle.h"
#include "cloth\clothManager.h"
#include "BatchSimulateManager.h"
#include "Renderable\ObjMesh.h"

Select3dEventHandle::Select3dEventHandle(Viewer3d* v) : Abstract3dEventHandle(v)
{
	m_cursor = QCursor(Qt::CursorShape::ArrowCursor);
	m_iconFile = "icons/selection.png";
	m_inactiveIconFile = "icons/selection_inactive.png";
	m_toolTips = "select handle";
}

Select3dEventHandle::~Select3dEventHandle()
{

}

void Select3dEventHandle::handleEnter()
{
	Abstract3dEventHandle::handleEnter();
	m_pickInfo.mesh = nullptr;
	auto manager = m_viewer->getManager();
	if (manager)
	{
		manager->dragEnd();
	}
}

void Select3dEventHandle::handleLeave()
{
	Abstract3dEventHandle::handleLeave();
	m_pickInfo.mesh = nullptr;
	auto manager = m_viewer->getManager();
	if (manager)
	{
		manager->dragEnd();
	}
}

void Select3dEventHandle::mousePressEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mousePressEvent(ev);
	m_pickInfo.mesh = nullptr;
	if (m_viewer->buttons() == Qt::LeftButton)
	{
		pick(ev->pos());

		auto manager = m_viewer->getManager();
		if (manager && m_pickInfo.mesh && ev->modifiers() == Qt::CTRL)
		{
			ldp::DragInfo info;
			info.selected_cloth = m_pickInfo.mesh;
			int vpos = 0;
			for (int i = 0; i < 3; i++)
			if (m_pickInfo.pickInnerCoords[i] > m_pickInfo.pickInnerCoords[vpos])
				vpos = i;
			info.selected_vert_id = m_pickInfo.mesh->face_list[m_pickInfo.faceId].vertex_index[vpos];
			printf("drag vert: local=%d, global=%d\n", info.selected_vert_id,
				manager->pieceVertId2GlobalVertId(m_pickInfo.mesh, info.selected_vert_id));
			auto v = info.selected_cloth->vertex_list[info.selected_vert_id];
			ldp::Float3	p, q;
			getSelectionRay(ev->pos(), p, q);
			auto dir = (q - p).normalize();
			auto diff = v - p;
			auto dist = diff.dot(dir);
			info.target = p + dist*dir;
			manager->dragBegin(info);
		} // end if manager
	} // end if initial_location and left button
}

void Select3dEventHandle::mouseReleaseEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseReleaseEvent(ev);

	auto manager = m_viewer->getManager();
	if (manager)
	{
		manager->dragEnd();
	}

	m_pickInfo.mesh = nullptr;
	m_pickInfo.piece = nullptr;
}

void Select3dEventHandle::mouseDoubleClickEvent(QMouseEvent *ev)
{
	Abstract3dEventHandle::mouseDoubleClickEvent(ev);
}

void Select3dEventHandle::mouseMoveEvent(QMouseEvent *ev)
{
	bool valid_op = false;
	if (m_viewer->buttons() == Qt::LeftButton)
	{
		auto manager = m_viewer->getManager();
		if (manager && m_pickInfo.mesh && ev->modifiers() == Qt::CTRL)
		{
			ldp::DragInfo info;
			info.selected_cloth = m_pickInfo.mesh;
			info.selected_vert_id = m_pickInfo.mesh->face_list[m_pickInfo.faceId].vertex_index[0];
			auto v = info.selected_cloth->vertex_list[info.selected_vert_id];
			ldp::Float3	p, q;
			getSelectionRay(ev->pos(), p, q);
			auto dir = (q - p).normalize();
			auto diff = v - p;
			auto dist = diff.dot(dir);
			info.target = p + dist*dir;
			manager->dragMove(info.target);
			valid_op = true;
		} // end if manager
	} // end if initial_location and left button

	if (!valid_op)
		Abstract3dEventHandle::mouseMoveEvent(ev);
}

void Select3dEventHandle::wheelEvent(QWheelEvent *ev)
{
	Abstract3dEventHandle::wheelEvent(ev);
}

void Select3dEventHandle::keyPressEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyPressEvent(ev);

	switch (ev->key())
	{
	case Qt::Key_Space:
	{
						  if (m_viewer->getBatchSimManager())
						  {
							  BatchSimulateManager* batchSimManager = m_viewer->getBatchSimManager();
							  if (batchSimManager->m_batchSimMode == ldp::BatchSimNotInit)
							  {
								  if (m_viewer->getManager())
								  {
									  if (m_viewer->getManager()->getSimulationMode() == ldp::SimulationNotInit)
										  m_viewer->getManager()->simulationInit();
									  if (m_viewer->getManager()->getSimulationMode() == ldp::SimulationOn)
										  m_viewer->getManager()->setSimulationMode(ldp::SimulationPause);
									  else if (m_viewer->getManager()->getSimulationMode() == ldp::SimulationPause)
										  m_viewer->getManager()->setSimulationMode(ldp::SimulationOn);
								  }
							  }
							  else if (batchSimManager->m_batchSimMode == ldp::BatchSimOn)
							  {
								  batchSimManager->m_batchSimMode = ldp::BatchSimPause;
								  std::cout << "batch sim pause" << std::endl;
							  }
							  else if (batchSimManager->m_batchSimMode == ldp::BatchSimPause)
							  {
								  batchSimManager->m_batchSimMode = ldp::BatchSimOn;
								  std::cout << "batch sim on" << std::endl;
							  }
						  }
						  break;
	}
	case Qt::Key_Escape:
	{
						   BatchSimulateManager* batchSimManager = m_viewer->getBatchSimManager();
						   if (batchSimManager)
						   {
							   if (batchSimManager->m_batchSimMode == ldp::BatchSimPause || batchSimManager->m_batchSimMode == ldp::BatchSimOn)
								   batchSimManager->m_batchSimMode = ldp::BatchSimFinished;
						   }
						   break;
	}
	case Qt::Key_B:
	{
					  if (m_viewer->getManager())
						  m_viewer->getManager()->simulationInit();
					  break;
	}
	case Qt::Key_D:
	{
					  if (m_viewer->getManager())
						  m_viewer->getManager()->clearBindClothesToSmplJoints();
					  break;
	}
	default:
		break;
	}
}

void Select3dEventHandle::keyReleaseEvent(QKeyEvent *ev)
{
	Abstract3dEventHandle::keyReleaseEvent(ev);
}
