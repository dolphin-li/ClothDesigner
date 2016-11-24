#include "clothManager.h"

namespace ldp
{
	ClothManager::ClothManager()
	{
		m_bodyMesh.reset(new ObjMesh);
	}

	ClothManager::~ClothManager()
	{

	}

	void ClothManager::simulationUpdate()
	{
		if (m_simulationMode != SimulationOn)
			return;

	}

	void ClothManager::setSimulationMode(SimulationMode mode)
	{
		m_simulationMode = mode;
	}
}