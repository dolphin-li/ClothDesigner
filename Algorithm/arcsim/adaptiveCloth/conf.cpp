/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "conf.hpp"

#include "io.hpp"
#include "magic.hpp"
#include "mot_parser.hpp"
#include "util.hpp"
#include <cassert>
#include <cfloat>
#include <json/json.h>
#include <fstream>
//#include <png.h>
#include "sstream"
using namespace std;

namespace arcsim
{


	void parse(bool&, const Json::Value&);
	void parse(int&, const Json::Value&);
	void parse(double&, const Json::Value&);
	void parse(string&, const Json::Value&);

	void complain(const Json::Value &json, const string &expected);

	template <int n> void parse(Vec<n> &v, const Json::Value &json)
	{
		if (!json.isArray()) complain(json, "array");
		assert(json.size() == n);
		for (int i = 0; i < n; i++)
			parse(v[i], json[i]);
	}

	template <typename T> void parse(T &x, const Json::Value &json, const T &x0)
	{
		if (json.isNull())
			x = x0;
		else
			parse(x, json);
	}

	template <typename T> void parse(vector<T> &v, const Json::Value &json)
	{
		if (!json.isArray()) complain(json, "array");
		v.resize(json.size());
		for (int i = 0; i < json.size(); i++)
			parse(v[i], json[i]);
	}

	void parse(Cloth&, const Json::Value&);
	void parse_motions(vector<Motion>&, const Json::Value&);
	void parse_handles(vector<Handle*>&, const Json::Value&,
		const vector<Cloth>&, const vector<Motion>&);
	void parse_obstacles(vector<Obstacle>&, const Json::Value&,
		const vector<Motion>&);
	void parse_morphs(vector<Morph>&, const Json::Value&, const vector<Cloth> &);
	void parse(Wind&, const Json::Value&);
	void parse(Magic&, const Json::Value&);

	void load_json(const string &configFilename, Simulation &sim)
	{
		Json::Value json;
		Json::Reader reader;
		ifstream file(configFilename.c_str());
		bool parsingSuccessful = reader.parse(file, json);
		if (!parsingSuccessful)
		{
			fprintf(stderr, "Error reading file: %s\n", configFilename.c_str());
			fprintf(stderr, "%s", reader.getFormatedErrorMessages().c_str());
			abort();
		}
		file.close();
		// Gather general data
		if (!json["frame_time"].empty())
		{
			parse(sim.frame_time, json["frame_time"]);
			parse(sim.frame_steps, json["frame_steps"], 1);
			sim.step_time = sim.frame_time / sim.frame_steps;
			parse(sim.end_time, json["end_time"], infinity);
			parse(sim.end_frame, json["end_frame"], infinity);
		}
		else if (!json["timestep"].empty())
		{
			parse(sim.step_time, json["timestep"]);
			parse(sim.frame_steps, json["save_frames"], 1);
			sim.frame_time = sim.step_time*sim.frame_steps;
			parse(sim.end_time, json["duration"], infinity);
			sim.end_frame = infinity;
		}
		sim.time = 0;
		parse(sim.cloths, json["cloths"]);
		parse_motions(sim.motions, json["motions"]);
		parse_handles(sim.handles, json["handles"], sim.cloths, sim.motions);
		parse_obstacles(sim.obstacles, json["obstacles"], sim.motions);
		parse_morphs(sim.morphs, json["morphs"], sim.cloths);
		parse(sim.gravity, json["gravity"], Vec3(0));
		parse(sim.wind, json["wind"]);
		parse(sim.friction, json["friction"], 0.6);
		parse(sim.obs_friction, json["obs_friction"], 0.3);
		string module_names[] = { "proximity", "physics", "strainlimiting",
			"collision", "remeshing", "separation",
			"popfilter", "plasticity" };
		for (int i = 0; i < Simulation::nModules; i++)
		{
			sim.enabled[i] = true;
			for (int j = 0; j < json["disable"].size(); j++)
			if (json["disable"][j] == module_names[i])
				sim.enabled[i] = false;
		}
		parse(arcsim::magic, json["magic"]);
		// disable strain limiting and plasticity if not needed
		bool has_strain_limits = false, has_plasticity = false;
		for (int c = 0; c < sim.cloths.size(); c++)
		for (int m = 0; m < sim.cloths[c].materials.size(); m++)
		{
			Cloth::Material *mat = sim.cloths[c].materials[m];
			if (finite(mat->strain_min) || finite(mat->strain_max))
				has_strain_limits = true;
			if (finite(mat->yield_curv))
				has_plasticity = true;
		}
		if (!has_strain_limits)
			sim.enabled[Simulation::StrainLimiting] = false;
		if (!has_plasticity)
			sim.enabled[Simulation::Plasticity] = false;
	}

	// Basic data types

	void complain(const Json::Value &json, const string &expected)
	{
		cout << "Expected " << expected << ", found " << json << " instead" << endl;
		abort();
	}

	void parse(bool &b, const Json::Value &json)
	{
		if (!json.isBool()) complain(json, "boolean");
		b = json.asBool();
	}
	void parse(int &n, const Json::Value &json)
	{
		if (!json.isIntegral()) complain(json, "integer");
		n = json.asInt();
	}
	void parse(double &x, const Json::Value &json)
	{
		if (!json.isNumeric()) complain(json, "real");
		x = json.asDouble();
	}
	void parse(string &s, const Json::Value &json)
	{
		if (!json.isString()) complain(json, "string");
		s = json.asString();
	}

	struct Range
	{
		double &min, &max;
		Range(double &min, double &max) : min(min), max(max) {}
	};

	void parse(Range range, const Json::Value &json, Vec2 range0)
	{
		if (json.isNull())
		{
			range.min = range0[0];
			range.max = range0[1];
			return;
		}
		assert(json.size() == 2);
		parse(range.min, json[0u]);
		parse(range.max, json[1]);
	}

	struct Box
	{
		Vec2 umin, umax;
		Box() {}
		Box(const Vec2 &umin, const Vec2 &umax) : umin(umin), umax(umax) {}
	};

	void parse(Box &box, const Json::Value &json, const Box &box0)
	{
		if (json.isNull())
		{
			box = box0;
			return;
		}
		assert(json.size() == 2);
		parse(box.umin, json[0u]);
		parse(box.umax, json[1]);
	}

	// Cloth

	void parse(Transformation&, const Json::Value&);
	void parse(Cloth::Material*&, const Json::Value&);
	void parse(Cloth::Remeshing&, const Json::Value&);

	struct Velocity { Vec3 v, w; Vec3 o; };
	void parse(Velocity &, const Json::Value &);
	void apply_velocity(Mesh &mesh, const Velocity &vel);

	void parse(Cloth &cloth, const Json::Value &json)
	{
		string filename;
		parse(filename, json["mesh"]);
		load_obj(cloth.mesh, filename);
		Transformation transform;
		parse(transform, json["transform"]);
		if (transform.scale != 1)
		for (int v = 0; v < cloth.mesh.verts.size(); v++)
			cloth.mesh.verts[v]->u *= transform.scale;
		compute_ms_data(cloth.mesh);
		apply_transformation(cloth.mesh, transform);
		Velocity velocity;
		parse(velocity, json["velocity"]);
		apply_velocity(cloth.mesh, velocity);
		parse(cloth.materials, json["materials"]);
		parse(cloth.remeshing, json["remeshing"]);
	}

	void parse(Transformation& transform, const Json::Value &json)
	{
		Vec<4> rot(0);
		parse(transform.translation, json["translate"], Vec3(0));
		parse(transform.scale, json["scale"], 1.);
		parse(rot, json["rotate"], Vec<4>(0));
		transform.rotation = Quaternion::from_axisangle(
			Vec3(rot[1], rot[2], rot[3]), rot[0] * M_PI / 180);
	}

	void parse(Velocity &velocity, const Json::Value &json)
	{
		parse(velocity.v, json["linear"], Vec3(0));
		parse(velocity.w, json["angular"], Vec3(0));
		parse(velocity.o, json["origin"], Vec3(0));
	}

	void apply_velocity(Mesh &mesh, const Velocity &vel)
	{
		for (int n = 0; n < mesh.nodes.size(); n++)
			mesh.nodes[n]->v = vel.v + cross(vel.w, mesh.nodes[n]->x - vel.o);
	}

	void load_material_data(Cloth::Material&, const string &filename);

	void parse(Cloth::Material *&material, const Json::Value &json)
	{
		string filename;
		parse(filename, json["data"]);
		material = new Cloth::Material;
		memset(material, 0, sizeof(Cloth::Material));
		load_material_data(*material, filename);
		double density_mult, stretching_mult, bending_mult, thicken;
		parse(density_mult, json["density_mult"], 1.);
		parse(stretching_mult, json["stretching_mult"], 1.);
		parse(bending_mult, json["bending_mult"], 1.);
		parse(thicken, json["thicken"], 1.);
		density_mult *= thicken;
		stretching_mult *= thicken;
		bending_mult *= thicken;
		material->density *= density_mult;
		for (int i = 0; i < sizeof(material->stretching.s) / sizeof(Vec4); i++)
			((Vec4*)&material->stretching.s)[i] *= stretching_mult;
		for (int i = 0; i < sizeof(material->bending.d) / sizeof(double); i++)
			((double*)&material->bending.d)[i] *= bending_mult;
		parse(material->damping, json["damping"], 0.);
		parse(Range(material->strain_min, material->strain_max),
			json["strain_limits"], Vec2(-infinity, infinity));
		parse(material->yield_curv, json["yield_curv"], infinity);
		parse(material->weakening, json["weakening"], 0.);
	}

	void parse(Cloth::Remeshing &remeshing, const Json::Value &json)
	{
		parse(remeshing.refine_angle, json["refine_angle"], infinity);
		parse(remeshing.refine_compression, json["refine_compression"], infinity);
		parse(remeshing.refine_velocity, json["refine_velocity"], infinity);
		parse(Range(remeshing.size_min, remeshing.size_max),
			json["size"], Vec2(-infinity, infinity));
		parse(remeshing.aspect_min, json["aspect_min"], -infinity);
	}

	// Other things

	void parse(Motion&, const Json::Value&);

	void parse_motions(vector<Motion> &motions, const Json::Value &json)
	{
		if (json.isObject() && !json.isNull())
		{
			string filename;
			double fps;
			Transformation trans;
			parse(filename, json["motfile"]);
			parse(fps, json["fps"]);
			parse(trans, json["transform"]);
			motions = load_mot(filename, fps);
			for (int m = 0; m < motions.size(); m++)
			{
				clean_up_quaternions(motions[m]);
				for (int p = 0; p < motions[m].points.size(); p++)
					motions[m].points[p].x = trans*motions[m].points[p].x;
				for (int p = 0; p < motions[m].points.size(); p++)
					fill_in_velocity(motions[m], p);
			}
		}
		else
			parse(motions, json);
	}

	void parse(Motion::Point&, const Json::Value&);

	void parse(Motion &motion, const Json::Value &json)
	{
		parse(motion.points, json);
		for (int p = 0; p < motion.points.size(); p++)
		if (motion.points[p].v.scale == infinity) // no velocity specified
			fill_in_velocity(motion, p);
	}

	void parse(Motion::Point &mp, const Json::Value &json)
	{
		parse(mp.t, json["time"]);
		parse(mp.x, json["transform"]);
		if (json["velocity"].isNull())
			mp.v.scale = infinity; // raise a flag
		else
		{
			parse(mp.v, json["velocity"]);
			if (mp.v.scale == 1)
				mp.v.scale = 0;
			if (mp.v.rotation.s == 1)
			{
				mp.v.rotation.s = 0;
				mp.v.rotation.v = Vec3(0);
			}
		}
	}

	void parse_handle(vector<Handle*> &, const Json::Value &,
		const vector<Cloth> &, const vector<Motion> &);

	void parse_handles(vector<Handle*> &hans, const Json::Value &jsons,
		const vector<Cloth> &cloths, const vector<Motion> &motions)
	{
		for (int j = 0; j < jsons.size(); j++)
			parse_handle(hans, jsons[j], cloths, motions);
	}

	void parse_node_handle(vector<Handle*> &hans, const Json::Value &json,
		const vector<Cloth> &cloths,
		const vector<Motion> &motions);

	void parse_circle_handle(vector<Handle*> &hans, const Json::Value &json,
		const vector<Cloth> &cloths,
		const vector<Motion> &motions);

	void parse_glue_handle(vector<Handle*> &hans, const Json::Value &json,
		const vector<Cloth> &cloths,
		const vector<Motion> &motions);

	void parse_handle(vector<Handle*> &hans, const Json::Value &json,
		const vector<Cloth> &cloths, const vector<Motion> &motions)
	{
		string type;
		parse(type, json["type"], string("node"));
		int nhans = hans.size();
		if (type == "node")
			parse_node_handle(hans, json, cloths, motions);
		else if (type == "circle")
			parse_circle_handle(hans, json, cloths, motions);
		else if (type == "glue")
			parse_glue_handle(hans, json, cloths, motions);
		else
		{
			cout << "Unknown handle type " << type << endl;
			abort();
		}
		double start_time, end_time, fade_time;
		parse(start_time, json["start_time"], 0.);
		parse(end_time, json["end_time"], infinity);
		parse(fade_time, json["fade_time"], 0.);
		for (int h = nhans; h < hans.size(); h++)
		{
			hans[h]->start_time = start_time;
			hans[h]->end_time = end_time;
			hans[h]->fade_time = fade_time;
		}
	}

	void parse_node_handle(vector<Handle*> &hans, const Json::Value &json,
		const vector<Cloth> &cloths,
		const vector<Motion> &motions)
	{
		int c, l, m;
		vector<int> ns;
		parse(c, json["cloth"], 0);
		parse(l, json["label"], -1);
		if (l == -1)
			parse(ns, json["nodes"]);
		parse(m, json["motion"], -1);
		const Mesh &mesh = cloths[c].mesh;
		const Motion *motion = (m != -1) ? &motions[m] : NULL;
		if (l != -1)
		{
			for (int n = 0; n < mesh.nodes.size(); n++)
			{
				if (mesh.nodes[n]->label != l)
					continue;
				NodeHandle *han = new NodeHandle;
				han->node = mesh.nodes[n];
				han->node->preserve = true;
				han->motion = motion;
				hans.push_back(han);
			}
		}
		if (!ns.empty())
		{
			for (int i = 0; i < ns.size(); i++)
			{
				NodeHandle *han = new NodeHandle;
				han->node = mesh.nodes[ns[i]];
				han->node->preserve = true;
				han->motion = motion;
				hans.push_back(han);
			}
		}
	}

	void parse_circle_handle(vector<Handle*> &hans, const Json::Value &json,
		const vector<Cloth> &cloths,
		const vector<Motion> &motions)
	{
		CircleHandle *han = new CircleHandle;
		int c, m;
		parse(c, json["cloth"], 0);
		han->mesh = (Mesh*)&cloths[c].mesh;
		parse(han->label, json["label"]);
		parse(m, json["motion"], -1);
		han->motion = (m != -1) ? &motions[m] : NULL;
		parse(han->c, json["circumference"]);
		parse(han->u, json["u"]);
		parse(han->xc, json["center"]);
		parse(han->dx0, json["axis0"]);
		parse(han->dx1, json["axis1"]);
		hans.push_back(han);
	}

	void parse_glue_handle(vector<Handle*> &hans, const Json::Value &json,
		const vector<Cloth> &cloths,
		const vector<Motion> &motions)
	{
		GlueHandle *han = new GlueHandle;
		int c;
		vector<int> ns;
		parse(c, json["cloth"], 0);
		parse(ns, json["nodes"]);
		if (ns.size() != 2)
		{
			cout << "Must glue exactly two nodes together" << endl;
			abort();
		}
		const Mesh &mesh = cloths[c].mesh;
		han->nodes[0] = (Node*)mesh.nodes[ns[0]];
		han->nodes[1] = (Node*)mesh.nodes[ns[1]];
		hans.push_back(han);
	}

	void parse_obstacle(Obstacle&, const Json::Value&, const vector<Motion>&);

	void parse_obstacles(vector<Obstacle> &obstacles, const Json::Value &json,
		const vector<Motion> &motions)
	{
		if (json.isString())
		{
			string fmt;
			parse(fmt, json);
			for (int i = 0; true; i++)
			{
				string filename = stringf(fmt, i);
				if (!fstream(filename.c_str(), ios::in))
					break;
				Obstacle obs;
				load_obj(obs.base_mesh, filename);
				obs.transform_spline = (i < motions.size()) ? &motions[i] : NULL;
				obs.start_time = 0;
				obs.end_time = infinity;
				obs.get_mesh(0);
				obstacles.push_back(obs);
			}
		}
		else
		{
			obstacles.resize(json.size());
			for (int j = 0; j < json.size(); j++)
				parse_obstacle(obstacles[j], json[j], motions);
		}
	}

	void parse_obstacle(Obstacle &obstacle, const Json::Value &json,
		const vector<Motion> &motions)
	{
		string filename;
		parse(filename, json["mesh"]);
		load_obj(obstacle.base_mesh, filename);
		Transformation transform;
		parse(transform, json["transform"]);
		apply_transformation(obstacle.base_mesh, transform);
		int m;
		parse(m, json["motion"], -1);
		obstacle.transform_spline = (m != -1) ? &motions[m] : NULL;
		parse(obstacle.start_time, json["start_time"], 0.);
		parse(obstacle.end_time, json["end_time"], infinity);
		obstacle.get_mesh(0);
	}

	void parse_morph(Morph&, const Json::Value&, const vector<Cloth>&);
	void parse(Spline<Morph::Weights>::Point &, const Json::Value &);

	void parse_morphs(vector<Morph> &morphs, const Json::Value &json,
		const vector<Cloth> &cloths)
	{
		morphs.resize(json.size());
		for (int j = 0; j < json.size(); j++)
			parse_morph(morphs[j], json[j], cloths);
	}

	void parse_morph(Morph &morph, const Json::Value &json,
		const vector<Cloth> &cloths)
	{
		int c;
		parse(c, json["cloth"], 0);
		morph.mesh = (Mesh*)&cloths[c].mesh;
		morph.targets.resize(json["targets"].size());
		for (int j = 0; j < json["targets"].size(); j++)
		{
			string filename;
			parse(filename, json["targets"][j]);
			load_obj(morph.targets[j], filename);
		}
		int nk = json["spline"].size();
		morph.weights.points.resize(nk);
		morph.log_stiffness.points.resize(nk);
		for (int k = 0; k < nk; k++)
		{
			const Json::Value &j = json["spline"][k];
			double t; parse(t, j["time"]);
			morph.weights.points[k].t = morph.log_stiffness.points[k].t = t;
			int m; parse(m, j["target"]);
			morph.weights.points[k].x.assign(morph.targets.size(), 0);
			morph.weights.points[k].x[m] = 1;
			double s; parse(s, j["stiffness"]);
			morph.log_stiffness.points[k].x = log(s);
		}
		for (int k = 0; k < nk; k++)
		{
			fill_in_velocity(morph.weights, k);
			fill_in_velocity(morph.log_stiffness, k);
		}
	}

	void parse(Wind &wind, const Json::Value &json)
	{
		parse(wind.density, json["density"], 1.);
		parse(wind.velocity, json["velocity"], Vec3(0));
		parse(wind.drag, json["drag"], 0.);
	}

	void parse(Magic &magic, const Json::Value &json)
	{
#define PARSE_MAGIC(param) parse(magic.param, json[#param], magic.param)
		PARSE_MAGIC(fixed_high_res_mesh);
		PARSE_MAGIC(handle_stiffness);
		PARSE_MAGIC(collision_stiffness);
		PARSE_MAGIC(repulsion_thickness);
		parse(magic.projection_thickness, json["projection_thickness"],
			0.1*magic.repulsion_thickness);
		PARSE_MAGIC(edge_flip_threshold);
		PARSE_MAGIC(rib_stiffening);
		PARSE_MAGIC(combine_tensors);
		PARSE_MAGIC(preserve_creases);
#undef PARSE_MAGIC
	}

	// JSON materials

	void parse(StretchingSamples&, const Json::Value&);
	void parse(BendingData&, const Json::Value&);

	void load_material_data(Cloth::Material &material, const string &filename)
	{
		Json::Value json;
		Json::Reader reader;
		ifstream file(filename.c_str());
		bool parsingSuccessful = reader.parse(file, json);
		if (!parsingSuccessful)
		{
			fprintf(stderr, "Error reading file: %s\n", filename.c_str());
			fprintf(stderr, "%s", reader.getFormatedErrorMessages().c_str());
			abort();
		}
		file.close();
		parse(material.density, json["density"]);
		parse(material.stretching, json["stretching"]);
		parse(material.bending, json["bending"]);
	}

	void parse(StretchingSamples &samples, const Json::Value &json)
	{
		StretchingData data;
		parse(data.d[0][0], json[0u]);
		for (int i = 1; i < 5; i++)
			data.d[0][i] = data.d[0][0];
		for (int i = 0; i < 5; i++)
			parse(data.d[1][i], json[i + 1]);
		evaluate_stretching_samples(samples, data);
	}

	void parse(BendingData &data, const Json::Value &json)
	{
		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 5; j++)
			parse(data.d[i][j], json[i][j]);
	}
}