#pragma once

#include <vector>
#include <memory>
#include "Renderable\ObjMesh.h"
namespace ldp
{
	class ClothPiece;
	class PanelPolygon;
	class ClothManager
	{
	public:
		enum SimulationMode
		{
			SimulationInit,
			SimulationOn,
			SimulationPause,
		};
	public:
		ClothManager();
		~ClothManager();

		SimulationMode getSimulationMode()const { return m_simulationMode; }
		void setSimulationMode(SimulationMode mode);

		void simulationUpdate();

		const ObjMesh* bodyMesh()const { return m_bodyMesh.get(); }
		ObjMesh* bodyMesh() { return m_bodyMesh.get(); }
		int numClothPieces()const { return (int)m_clothPieces.size(); }
		const ClothPiece* clothPiece(int i)const { return m_clothPieces.at(i).get(); }
		ClothPiece* clothPiece(int i) { return m_clothPieces.at(i).get(); }
		void clearClothPieces() { m_clothPieces.clear(); }
		void addClothPiece(std::shared_ptr<ClothPiece> piece) { m_clothPieces.push_back(piece); }
		void removeClothPiece(int i) { m_clothPieces.erase(m_clothPieces.begin() + i); }
	private:
		std::vector<std::shared_ptr<ClothPiece>> m_clothPieces;
		std::shared_ptr<ObjMesh> m_bodyMesh;
		SimulationMode m_simulationMode;
	};
}
