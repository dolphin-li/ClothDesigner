#pragma once

#include "adaptiveCloth\simulation.hpp"
#include "ldpmat/ldp_basic_mat.h"
namespace arcsim
{
	class ArcSimManager
	{
	public:
		ArcSimManager();
		void loadFromJsonConfig(std::string configFileName);
		void clear();
		Simulation* getSimulator() { return m_sim.get(); }
		const Simulation* getSimulator()const { return m_sim.get(); }
		void simulate(int nsteps);
		void saveCurrentCloth(std::string name);
	private:
		std::shared_ptr<Simulation> m_sim;
	};

}