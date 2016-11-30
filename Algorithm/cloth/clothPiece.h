#pragma once

#include <set>
#include <string>
#include "ldpMat\ldp_basic_mat.h"
class ObjMesh;
namespace ldp
{
	class PanelPolygon;
	class ClothPiece
	{
	public:
		ClothPiece();
		~ClothPiece();

		const ObjMesh& mesh3d()const { return *m_mesh3d; }
		ObjMesh& mesh3d() { return *m_mesh3d; }
		const ObjMesh& mesh3dInit()const { return *m_mesh3dInit; }
		ObjMesh& mesh3dInit() { return *m_mesh3dInit; }
		const ObjMesh& mesh2d()const { return *m_mesh2d; }
		ObjMesh& mesh2d() { return *m_mesh2d; }
		const PanelPolygon& panel()const { return *m_panel; }
		PanelPolygon& panel() { return *m_panel; }

		std::string getName()const { return m_name; }
		std::string setName(std::string name); // return a unique name

		ClothPiece* clone()const;
		ClothPiece* lightClone()const;
	public:
		static std::string generateUniqueName(std::string nameHints);
	private:
		std::shared_ptr<ObjMesh> m_mesh3d;
		std::shared_ptr<ObjMesh> m_mesh3dInit;
		std::shared_ptr<ObjMesh> m_mesh2d;
		std::shared_ptr<PanelPolygon> m_panel;
		ldp::Mat4f m_transfrom;
		std::string m_name;
		static std::set<std::string> s_nameSet;
	};
}