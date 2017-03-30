#pragma once

#include "adaptiveCloth\simulation.hpp"
#include "ldpmat/ldp_basic_mat.h"
class ObjMesh;
namespace ldp
{
	class TimeStamp;
}
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

		void calcBoundingBox(float bmin[3], float bmax[3]);
		ObjMesh* getBodyMesh(){ return m_bodyMesh.get(); }
		const ObjMesh* getBodyMesh()const{ return m_bodyMesh.get(); }
		ObjMesh* getClothMesh(){ return m_clothMesh.get(); }
		const ObjMesh* getClothMesh()const{ return m_clothMesh.get(); }
	protected:
		void updateBodyMesh();
		void updateClothMesh();
		void convertToObj(const std::vector<arcsim::Mesh*>& meshes, ObjMesh& omesh);
	private:
		std::shared_ptr<Simulation> m_sim;
		std::shared_ptr<ObjMesh> m_bodyMesh, m_clothMesh;
		std::shared_ptr<ldp::TimeStamp> m_timeStamp;
	};

}