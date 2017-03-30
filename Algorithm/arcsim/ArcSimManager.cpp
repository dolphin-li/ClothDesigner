#include "ArcSimManager.h"
#include "adaptiveCloth\conf.hpp"
#include "adaptiveCloth\separateobs.hpp"
#include "adaptiveCloth\io.hpp"
#include "ldputil.h"

namespace arcsim
{
	const static double g_num_edge_sample_angle_thre = 15 * ldp::PI_D / 180.;
	const static double g_num_edge_sample_dist_thre = 0.01;

	ArcSimManager::ArcSimManager()
	{

	}

	void ArcSimManager::clear()
	{
		if (m_sim.get())
		{
			for (auto& h : m_sim->handles)
			{
				delete h;
			}
			for (auto& c : m_sim->cloths)
			{
				delete_mesh(c.mesh);
				for (auto m : c.materials)
					delete m;
			}
			for (auto& m : m_sim->morphs)
			{
				for (auto& s : m.targets)
					delete_mesh(s);
			}
			for (auto& o : m_sim->obstacles)
			{
				delete_mesh(o.base_mesh);
				delete_mesh(o.curr_state_mesh);
			}
			m_sim.reset((Simulation*)nullptr);
		}
	}

	void ArcSimManager::loadFromJsonConfig(std::string bodyMeshFileName)
	{
		clear();
		m_sim.reset(new Simulation);
		load_json(bodyMeshFileName, *m_sim.get());

		// prepare for simulation
		printf("arssim: preparing...\n");
		prepare(*m_sim);
		printf("arssim: separating obstacles...\n");
		separate_obstacles(m_sim->obstacle_meshes, m_sim->cloth_meshes);
		printf("arssim: relaxing initial state...\n");
		relax_initial_state(*m_sim);
	}

	void ArcSimManager::saveCurrentCloth(std::string fullname)
	{
		if (m_sim.get() == nullptr)
			return;
		std::string path, name, ext;
		ldp::fileparts(fullname, path, name, ext);
		for (size_t i = 0; i < m_sim->cloths.size(); i++)
		{
			const auto& cloth = m_sim->cloths[i];
			std::string nm = fullname;
			if (m_sim->cloths.size() > 1)
				nm = ldp::fullfile(path, name + "_" + std::to_string(i) + ext);
			arcsim::save_obj(cloth.mesh, nm);
		} // end for cloth
	}

	void ArcSimManager::simulate(int nsteps)
	{
		if (m_sim.get() == nullptr)
			return;
		Timer fps;
		fps.tick();
		for (int k = 0; k < nsteps; k++)
			advance_step(*m_sim);
		fps.tock();
		printf("step, time=%f/%f, tic=%f, physics=%f, collision=%f\n", 
			m_sim->time, m_sim->end_time, fps.last,
			m_sim->timers[Simulation::Physics].last,
			m_sim->timers[Simulation::Collision].last);
	}
}