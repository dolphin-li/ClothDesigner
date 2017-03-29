#include "SimulationManager.h"
#include "adaptiveCloth\conf.hpp"
#include "adaptiveCloth\separateobs.hpp"
#include "adaptiveCloth\io.hpp"
#include "SvgManager.h"
#include "SvgPolyPath.h"
#include "ldputil.h"
extern "C"{
#include "triangle.h"
};

namespace arcsim
{
	const static double g_num_edge_sample_angle_thre = 15 * ldp::PI_D / 180.;
	const static double g_num_edge_sample_dist_thre = 0.01;

	SimulationManager::SimulationManager()
	{
		m_svgManager = nullptr;
	}

	void SimulationManager::clear()
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

		m_svgManager = nullptr;
	}

	void SimulationManager::init(const svg::SvgManager& svgManager, std::string bodyMeshFileName, int triangleNumWanted)
	{
		clear();
		m_svgManager = &svgManager;
		m_triangleNumWanted = triangleNumWanted;
		m_sim.reset(new Simulation);
		load_json(bodyMeshFileName, *m_sim.get());
		//extractFromSvg();
		//load_json("conf/dress-blue.json", *m_sim.get());

		// prepare for simulation
		printf("arssim: preparing...\n");
		prepare(*m_sim);
		printf("arssim: separating obstacles...\n");
		separate_obstacles(m_sim->obstacle_meshes, m_sim->cloth_meshes);
		printf("arssim: relaxing initial state...\n");
		//relax_initial_state(*m_sim);
	}

	void SimulationManager::saveCurrentCloth(std::string fullname)
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

	void SimulationManager::extractFromSvg()
	{
		auto polyPaths = m_svgManager->collectPolyPaths(true);
		auto edgeGroups = m_svgManager->collectEdgeGroups(true);
		const float pixel2meter = m_svgManager->getPixelToMeters();
		if (polyPaths.size() == 0)
			throw std::exception("no selected polypaths given!");
		if (pixel2meter == 1.f)
			printf("warning: are you sure 1 pixel = 1 meter? \npossibly you should select \
				   			   			   the standard-4cm rectangle and click the \"pixel to meter\" button\n");

		// 1.0 compute poly area -----------------------------------------------------------------
		double totalPolyArea = 0;
		for (size_t i_polyPath = 0; i_polyPath < polyPaths.size(); i_polyPath++)
		{
			const auto & polyPath = polyPaths[i_polyPath];
			// if non-closed, we ignore it this time
			if (!polyPath->isClosed())
				continue;
			totalPolyArea += fabs(polyPath->calcAreaCornerWise()) * pixel2meter * pixel2meter;
		} // end for polyPath
		const double avgTriangleArea = totalPolyArea / m_triangleNumWanted;

		// 1.1 add closed polygons as loops ------------------------------------------------------
		for (size_t i_polyPath = 0; i_polyPath < polyPaths.size(); i_polyPath++)
		{
			const auto & polyPath = polyPaths[i_polyPath];
			// if non-closed, we ignore it this time
			if (!polyPath->isClosed())
				continue;

			const auto& R = polyPath->get3dRot().toRotationMatrix3();
			const auto& C = polyPath->get3dCenter() * pixel2meter;
			const auto& P = polyPath->getCenter() * pixel2meter;
			
			// colect all poly edges, including curve edges
			std::vector<std::vector<ldp::Float2>> polyVerts(polyPath->numCornerEdges());
			for (size_t iEdge = 0; iEdge < polyPath->numCornerEdges(); iEdge++)
			{
				const auto& coords = polyPath->getEdgeCoords(iEdge);
				for (size_t i = 0; i < coords.size(); i += 2)
					polyVerts[iEdge].push_back(ldp::Float2(coords[i], coords[i+1]) * pixel2meter);
			} // end for iEdge

			// triangulate the poly
			std::vector<ldp::Float2> triVerts;
			std::vector<ldp::Int3> tris;
			triangulate(polyVerts, triVerts, tris, avgTriangleArea);

			// convert to 3D
			std::vector<ldp::Float3> triVerts3d(triVerts.size());
			for (size_t i = 0; i < triVerts.size(); i++)
			{
				// ldp commit: convert from ZXY to XYZ, to adapt the data of LiBaoJian
				ldp::Float3 p(0, triVerts[i][0] - P[0], triVerts[i][1] - P[1]);
				triVerts3d[i] = R * p + C;
			}

			// push to cloth
			auto& cloth = m_sim->cloths.at(0);
			addToMesh(triVerts3d, tris, cloth.mesh);
			//loopId2svgIdMap.insert(std::make_pair(polyLoops.back().id_l_add, polyPath->getId()));
			//svg2loopMap.insert(std::make_pair(polyPath->getId(), &polyLoops.back()));
		} // end for polyPath

		for (auto& cloth : m_sim->cloths)
		{
			mark_nodes_to_preserve(cloth.mesh);
			compute_ms_data(cloth.mesh);
		}
	}

	void SimulationManager::simulate(int nsteps)
	{
		if (m_sim.get() == nullptr)
			return;
		Timer fps;
		fps.tick();
		for (int k = 0; k < nsteps; k++)
			advance_step(*m_sim);
		fps.tock();
		printf("step, time=%f/%f, tic=%f\n", m_sim->time, m_sim->end_time, fps.last);
	}

#pragma region -- triangle
	static void destroyMem_trianglulateio(struct triangulateio *io)
	{
		if (io->pointlist)  free(io->pointlist);                                               /* In / out */
		if (io->pointattributelist) free(io->pointattributelist);                                      /* In / out */
		if (io->pointmarkerlist) free(io->pointmarkerlist);                                          /* In / out */

		if (io->trianglelist) free(io->trianglelist);                                             /* In / out */
		if (io->triangleattributelist) free(io->triangleattributelist);                                   /* In / out */
		if (io->trianglearealist) free(io->trianglearealist);                                    /* In only */
		if (io->neighborlist) free(io->neighborlist);                                           /* Out only */

		if (io->segmentlist) free(io->segmentlist);                                              /* In / out */
		if (io->segmentmarkerlist) free(io->segmentmarkerlist);                             /* In / out */

		if (io->holelist) free(io->holelist);                        /* In / pointer to array copied out */

		if (io->regionlist) free(io->regionlist);                      /* In / pointer to array copied out */

		if (io->edgelist) free(io->edgelist);                                                 /* Out only */
		if (io->edgemarkerlist) free(io->edgemarkerlist);           /* Not used with Voronoi diagram; out only */
		if (io->normlist) free(io->normlist);              /* Used only with Voronoi diagram; out only */

		//all set to 0
		init_trianglulateio(io);
	}

	static void init_triangulateIO_from_poly(triangulateio *pIO, const std::vector<ldp::Double2> &poly, int close_flag)
	{
		init_trianglulateio(pIO);
		pIO->numberofpoints = (int)poly.size();
		pIO->pointlist = (REAL *)malloc(pIO->numberofpoints * 2 * sizeof(REAL));
		for (int i = 0; i<pIO->numberofpoints; i++)
		{
			pIO->pointlist[i * 2] = poly[i][0];
			pIO->pointlist[i * 2 + 1] = poly[i][1];
		}
		/*********input segments************/
		pIO->numberofsegments = (int)poly.size() - 1;
		if (close_flag) pIO->numberofsegments++;
		pIO->segmentlist = (int *)malloc(pIO->numberofsegments * 2 * sizeof(int));
		for (int i = 0; i<(int)poly.size() - 1; i++)
		{
			pIO->segmentlist[i * 2] = i;
			pIO->segmentlist[i * 2 + 1] = i + 1;
		}
		if (close_flag)
		{
			pIO->segmentlist[pIO->numberofsegments * 2 - 2] = (int)poly.size() - 1;
			pIO->segmentlist[pIO->numberofsegments * 2 - 1] = 0;
		}
	}

	static int tess_poly_2D(const std::vector<ldp::Double2> &poly, 
		std::vector<ldp::Float2>& triVerts, std::vector<ldp::Int3> &vTriangle, 
		int close_flag, double triAreaWanted)
	{
		vTriangle.resize(0);

		triangulateio input, triout, vorout;

		// Define input points
		arcsim::init_triangulateIO_from_poly(&input, poly, close_flag);

		// Make necessary initializations so that Triangle can return a 
		// triangulation in `mid' and a voronoi diagram in `vorout'.  
		init_trianglulateio(&triout);
		init_trianglulateio(&vorout);

		// triangulate
		char cmd[1024];
		// p: polygon mode
		// z: zero based indexing
		// q%d: minimum angle %d
		// D: delauney
		// a%f: maximum triangle area %f
		sprintf_s(cmd, "pzq%dDa%f", 30, triAreaWanted);
		triangulate(cmd, &input, &triout, &vorout);
		vTriangle.resize(triout.numberoftriangles);
		memcpy(vTriangle.data(), triout.trianglelist, sizeof(int)*triout.numberoftriangles * 3);
		const ldp::Double2* vptr = (const ldp::Double2*)triout.pointlist;
		triVerts.resize(triout.numberofpoints);
		for (int i = 0; i < triout.numberofpoints; i++)
			triVerts[i] = vptr[i];
		arcsim::destroyMem_trianglulateio(&input);
		arcsim::destroyMem_trianglulateio(&triout);
		arcsim::destroyMem_trianglulateio(&vorout);
		return vTriangle.size();
	}
#pragma endregion

	void SimulationManager::triangulate(const std::vector<std::vector<ldp::Float2>>& polyVerts,
		std::vector<ldp::Float2>& triVerts, std::vector<ldp::Int3>& tris, double triAreaWanted)const
	{
		double totalLen = 0;
		for (const auto& verts : polyVerts)
		{
			for (size_t i = 1; i < verts.size(); i++)
				totalLen += (verts[i] - verts[i - 1]).length();
		} // verts
		const double dist_thre = g_num_edge_sample_dist_thre * totalLen;
		const double cos_angle_thre = cos(g_num_edge_sample_angle_thre);

		std::vector<ldp::Double2> polyVertsD;
		for (const auto& verts : polyVerts)
		{
			if (verts.size() < 2)
				continue;
			for (size_t i = 0; i < verts.size() - 1; i++)
			{
				if (polyVertsD.size())
				{
					const auto& d1 = verts[i + 1] - verts[i];
					const auto& d0 = verts[i] - polyVertsD.back();
					double d0_len = d0.length();
					double d1_len = d1.length();
					if (d0_len < std::numeric_limits<float>::epsilon())
						continue;
					double cos_angle = d1.dot(d0) / (d0_len * d1_len);
					if (d0_len < dist_thre && cos_angle > cos_angle_thre)
						continue;
				}
				polyVertsD.push_back(verts[i]);
			} // end for i
		} // verts

		tess_poly_2D(polyVertsD, triVerts, tris, true, triAreaWanted);
	}

	void SimulationManager::addToMesh(const std::vector<ldp::Float3>& triVerts, 
		const std::vector<ldp::Int3>& tris, arcsim::Mesh& mesh)const
	{
		int nOldVerts = mesh.nodes.size();
		for (const auto& v : triVerts)
		{
			mesh.add(new Node(arcsim::Vec3(v[0], v[1], v[2]), Vec3(0)));
			mesh.add(new Vert(project<2>(mesh.nodes.back()->x), mesh.nodes.back()->label));
			connect(mesh.verts.back(), mesh.nodes.back());
		} // end for v

		for (const auto& tri : tris)
		{
			mesh.add(new Face(mesh.verts[nOldVerts + tri[0]], mesh.verts[nOldVerts + tri[1]],
				mesh.verts[nOldVerts + tri[2]]));
		} // end for tri
	}
}