#pragma once

#include "adaptiveCloth\simulation.hpp"
#include "ldp_basic_mat.h"
namespace svg
{
	class SvgManager;
}
namespace arcsim
{
	class SimulationManager
	{
	public:
		SimulationManager();
		void init(const svg::SvgManager& svgManager, std::string configFileName, int triangleNumWanted);
		void clear();
		Simulation* getSimulator() { return m_sim.get(); }
		const Simulation* getSimulator()const { return m_sim.get(); }
		void simulate(int nsteps);
		void saveCurrentCloth(std::string name);
	protected:
		void extractFromSvg();
		void triangulate(const std::vector<std::vector<ldp::Float2>>& polyVerts, 
			std::vector<ldp::Float2>& triVerts, std::vector<ldp::Int3>& tris, double triAreaWanted)const;
		void addToMesh(const std::vector<ldp::Float3>& triVerts, const std::vector<ldp::Int3>& tris,
			arcsim::Mesh& mesh)const;
	private:
		std::shared_ptr<Simulation> m_sim;
		const svg::SvgManager* m_svgManager;
		int m_triangleNumWanted;
	};

}