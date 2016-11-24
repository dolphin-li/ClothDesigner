#pragma once

#include "Renderable\ObjMesh.h"
#include "panelPolygon.h"
namespace ldp
{
	class ClothPiece
	{
	public:
		ClothPiece();
		~ClothPiece();

		const ObjMesh& mesh3d()const { return m_mesh3d; }
		ObjMesh& mesh3d() { return m_mesh3d; }
		const ObjMesh& mesh2d()const { return m_mesh2d; }
		ObjMesh& mesh2d() { return m_mesh2d; }
		const PanelPolygon& panel()const { return m_panel; }
		PanelPolygon& panel() { return m_panel; }
	private:
		ObjMesh m_mesh3d;
		ObjMesh m_mesh2d;
		PanelPolygon m_panel;
	};
}