#include "clothPiece.h"

#include "Renderable\ObjMesh.h"
#include "PanelObject\panelPolygon.h"
#include "TransformInfo.h"
#include "graph\Graph.h"
namespace ldp
{
	std::set<std::string> ClothPiece::s_nameSet;

	ClothPiece::ClothPiece()
	{
		m_name = generateUniqueName("default");
		m_mesh2d.reset(new ObjMesh);
		m_mesh3d.reset(new ObjMesh);
		m_mesh3dInit.reset(new ObjMesh);
		m_panel.reset(new PanelPolygon);
		m_transfromInfo.reset(new TransformInfo);
		m_graphPanel.reset(new Graph);
	}

	ClothPiece::~ClothPiece()
	{
	
	}

	std::string ClothPiece::setName(std::string name)
	{
		m_name = generateUniqueName(name);
		return m_name;
	}

	std::string ClothPiece::generateUniqueName(std::string nameHints)
	{
		auto iter = s_nameSet.find(nameHints);
		if (iter == s_nameSet.end())
		{
			s_nameSet.insert(nameHints);
			return nameHints;
		}

		for (int k = 0; k < 999999; k++)
		{
			std::string nm = nameHints + "_" + std::to_string(k);
			auto iter = s_nameSet.find(nm);
			if (iter == s_nameSet.end())
			{
				s_nameSet.insert(nm);
				return nm;
			}
		} // end for k
		throw std::exception("ClothPiece : cannot generate a unique name!");
	}

	ClothPiece* ClothPiece::clone()const
	{
		ClothPiece* piece = new ClothPiece(*this);
		piece->m_mesh2d->cloneFrom(m_mesh2d.get());
		piece->m_mesh3d->cloneFrom(m_mesh3d.get());
		piece->m_mesh3dInit->cloneFrom(m_mesh3dInit.get());
		piece->m_panel.reset((PanelPolygon*)m_panel->clone());
		piece->m_transfromInfo.reset(new TransformInfo(*m_transfromInfo));
		piece->m_graphPanel.reset((Graph*)m_graphPanel->clone());
		return piece;
	}

	ClothPiece* ClothPiece::lightClone()const
	{
		ClothPiece* piece = new ClothPiece(*this);
		piece->m_panel.reset((PanelPolygon*)m_panel->clone());
		piece->m_transfromInfo.reset(new TransformInfo(*m_transfromInfo.get()));
		piece->m_graphPanel.reset((Graph*)m_graphPanel->clone());
		return piece;
	}
}