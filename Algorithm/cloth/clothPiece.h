#pragma once

#include "Renderable\ObjMesh.h"
#include "panelPolygon.h"
#include <set>
namespace ldp
{
	class ClothPiece
	{
	public:
		ClothPiece();
		~ClothPiece();

		const ObjMesh& mesh3d()const { return m_mesh3d; }
		ObjMesh& mesh3d() { return m_mesh3d; }
		const ObjMesh& mesh3dInit()const { return m_mesh3dInit; }
		ObjMesh& mesh3dInit() { return m_mesh3dInit; }
		const ObjMesh& mesh2d()const { return m_mesh2d; }
		ObjMesh& mesh2d() { return m_mesh2d; }
		const PanelPolygon& panel()const { return m_panel; }
		PanelPolygon& panel() { return m_panel; }

		std::string getName()const { return m_name; }
		std::string setName(std::string name); // return a unique name
	public:
		static std::string generateUniqueName(std::string nameHints);
	private:
		ObjMesh m_mesh3d;
		ObjMesh m_mesh3dInit;
		ObjMesh m_mesh2d;
		PanelPolygon m_panel;
		std::string m_name;
		static std::set<std::string> s_nameSet;
	};
}