#pragma once

#include "adaptiveCloth\simulation.hpp"
#include "ldpmat/ldp_basic_mat.h"
#include <thread>
#include <mutex>
class ObjMesh;
namespace ldp
{
	class TimeStamp;
	class ClothManager;
	class GpuSim;
}
namespace arcsim
{
	class ArcSimManager;
	void simulate_thread_loop(ArcSimManager* threadData);
	class ArcSimManager
	{
	public:
		ArcSimManager();
		~ArcSimManager();
		void loadFromJsonConfig(std::string configFileName);
		void loadFromClothManager(ldp::ClothManager* clothManager);
		void clear();
		void reset();	// reset to initial state
		Simulation* getSimulator() { return m_sim.get(); }
		const Simulation* getSimulator()const { return m_sim.get(); }
		void simulate(int nsteps);
		void saveCurrentCloth(std::string name);

		void calcBoundingBox(float bmin[3], float bmax[3]);
		ObjMesh* getBodyMesh(){ return m_bodyMesh.get(); }
		const ObjMesh* getBodyMesh()const{ return m_bodyMesh.get(); }
		ObjMesh* getClothMesh(){ return m_clothMesh.get(); }
		const ObjMesh* getClothMesh()const{ return m_clothMesh.get(); }
		bool updateMesh();
	public:
		////////////////////////////////// multi-thread handling
		friend void arcsim::simulate_thread_loop(ArcSimManager* threadData);
		void start_simulate_loop_otherthread();
		void stop_simulate_loop_otherthread();
	protected:
		void convertToObj(const std::vector<arcsim::Mesh*>& meshes, ObjMesh& omesh);
	private:
		std::shared_ptr<Simulation> m_sim;
		std::shared_ptr<ObjMesh> m_bodyMesh, m_clothMesh;
		std::shared_ptr<ldp::TimeStamp> m_timeStamp;
		bool m_needUpdateMesh = false;
		std::shared_ptr<ldp::GpuSim> m_gpuSim;
	private:
		std::shared_ptr<std::thread> m_threadLoop;
		std::shared_ptr<std::mutex> m_threadMutex;
		bool m_thread_running = true;
	};

}