#include "ArcSimManager.h"
#include "adaptiveCloth\conf.hpp"
#include "adaptiveCloth\separateobs.hpp"
#include "adaptiveCloth\io.hpp"
#include "ldputil.h"
#include "Renderable\ObjMesh.h"
namespace arcsim
{
	const static double g_num_edge_sample_angle_thre = 15 * ldp::PI_D / 180.;
	const static double g_num_edge_sample_dist_thre = 0.01;

	ArcSimManager::ArcSimManager()
	{
		m_bodyMesh.reset(new ObjMesh());
		m_clothMesh.reset(new ObjMesh());
		m_timeStamp.reset(new ldp::TimeStamp());
		m_timeStamp->Prefix("arcsim");
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
		m_timeStamp->Reset();

		clear();
		m_sim.reset(new Simulation);
		load_json(bodyMeshFileName, *m_sim.get());
		m_timeStamp->Stamp("json loaded");

		// prepare for simulation
		prepare(*m_sim);
		m_timeStamp->Stamp("prepared");
		separate_obstacles(m_sim->obstacle_meshes, m_sim->cloth_meshes);
		m_timeStamp->Stamp("obstacles seperated");
		///relax_initial_state(*m_sim);
		///m_timeStamp->Stamp("initial state relaxed\n");

		updateBodyMesh();
		updateClothMesh();
		m_timeStamp->Stamp("obj mesh updated");
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
		for (int k = 0; k < nsteps; k++)
			advance_step(*m_sim);
		updateBodyMesh();
		updateClothMesh();
		m_timeStamp->Stamp("step, time=%f/%f, phys=%f, collision=%f",
			m_sim->time, m_sim->end_time,
			m_sim->timers[Simulation::Physics].last,
			m_sim->timers[Simulation::Collision].last);
	}

	void ArcSimManager::convertToObj(const std::vector<arcsim::Mesh*>& meshes, ObjMesh& omesh)
	{
		omesh.clear();
		int t_cnt = 0, v_cnt = 0;
		for (const auto& mesh : meshes)
		{
			for (const auto& tri : mesh->faces)
			{
				ObjMesh::obj_face f;
				f.material_index = -1;
				f.vertex_count = 3;
				for (int k = 0; k < 3; k++)
				{
					f.vertex_index[k] = tri->v[k]->node->index + v_cnt;
					f.texture_index[k] = tri->v[k]->index + t_cnt;
				}
				omesh.face_list.push_back(f);
			}
			for (const auto& tex : mesh->verts)
				omesh.vertex_texture_list.push_back(ldp::Float2(tex->u[0], tex->u[1]));
			for (const auto& node : mesh->nodes)
				omesh.vertex_list.push_back(ldp::Float3(node->x[0], node->x[1], node->x[2]));
			t_cnt += (int)mesh->verts.size();
			v_cnt += (int)mesh->nodes.size();
		}
		omesh.updateNormals();
		omesh.updateBoundingBox();
	}

	void ArcSimManager::calcBoundingBox(float bmin[3], float bmax[3])
	{
		for (int k = 0; k < 3; k++)
		{
			bmin[k] = FLT_MAX;
			bmax[k] = -FLT_MAX;
		}
		if (m_sim == nullptr)
			return;
		
		for (int k = 0; k < 3; k++)
		{
			bmin[k] = m_bodyMesh->boundingBox[0][k];
			bmax[k] = m_bodyMesh->boundingBox[1][k];
		}
	}

	void ArcSimManager::updateBodyMesh()
	{
		m_bodyMesh->clear();
		if (m_sim == nullptr)
			return;
		convertToObj(m_sim->obstacle_meshes, *m_bodyMesh);
	}

	void ArcSimManager::updateClothMesh()
	{
		m_clothMesh->clear();
		if (m_sim == nullptr)
			return;
		convertToObj(m_sim->cloth_meshes, *m_clothMesh);
	}
}