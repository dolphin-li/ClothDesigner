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

#include "collision.hpp"

#include "collisionutil.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "optimization.hpp"
#include "simulation.hpp"
#include "timer.hpp"
#include <algorithm>
#include <fstream>
#include <omp.h>
using namespace std;

namespace arcsim
{
	static const int max_iter = 100;
	static const double &thickness = arcsim::magic.projection_thickness;

	static double obs_mass;
	static bool deform_obstacles;

	static vector<Vec3> xold;
	static vector<Vec3> xold_obs;

	double get_mass(const Node *node) { return is_free(node) ? node->m : obs_mass; }

	// returns pair of (i) is_free(vert), and
	// (ii) index of mesh in ::meshes or ::obs_meshes that contains vert
	pair<bool, int> find_in_meshes(const Node *node)
	{
		int m = find_mesh(node, *arcsim::meshes);
		if (m != -1)
			return make_pair(true, m);
		else
			return make_pair(false, find_mesh(node, *arcsim::obs_meshes));
	}

	struct Impact
	{
		enum Type { VF, EE } type;
		double t;
		Node *nodes[4];
		double w[4];
		Vec3 n;
		Impact() {}
		Impact(Type type, const Node *n0, const Node *n1, const Node *n2,
			const Node *n3) : type(type)
		{
			nodes[0] = (Node*)n0;
			nodes[1] = (Node*)n1;
			nodes[2] = (Node*)n2;
			nodes[3] = (Node*)n3;
		}
	};

	struct ImpactZone
	{
		vector<Node*> nodes;
		vector<Impact> impacts;
		bool active;
	};

	void update_active(const vector<AccelStruct*> &accs,
		const vector<AccelStruct*> &obs_accs,
		const vector<ImpactZone*> &zones);

	vector<Impact> find_impacts(const vector<AccelStruct*> &acc,
		const vector<AccelStruct*> &obs_accs);
	vector<Impact> independent_impacts(const vector<Impact> &impacts);

	void add_impacts(const vector<Impact> &impacts, vector<ImpactZone*> &zones);

	void apply_inelastic_projection(ImpactZone *zone,
		const vector<Constraint*> &cons);

	vector<Constraint> impact_constraints(const vector<ImpactZone*> &zones);

	ostream &operator<< (ostream &out, const Impact &imp);
	ostream &operator<< (ostream &out, const ImpactZone *zone);

	void collision_response(vector<Mesh*> &meshes, const vector<Constraint*> &cons,
		const vector<Mesh*> &obs_meshes)
	{
		arcsim::meshes = &meshes;
		arcsim::obs_meshes = &obs_meshes;
		arcsim::xold = node_positions(meshes);
		arcsim::xold_obs = node_positions(obs_meshes);
		vector<AccelStruct*> accs = create_accel_structs(meshes, true),
			obs_accs = create_accel_structs(obs_meshes, true);
		vector<ImpactZone*> zones;
		arcsim::obs_mass = 1e3;
		int iter;
		for (int deform = 0; deform <= 1; deform++)
		{
			arcsim::deform_obstacles = deform;
			zones.clear();
			for (iter = 0; iter < max_iter; iter++)
			{
				if (!zones.empty())
					update_active(accs, obs_accs, zones);
				vector<Impact> impacts = find_impacts(accs, obs_accs);
				impacts = independent_impacts(impacts);
				if (impacts.empty())
					break;
				add_impacts(impacts, zones);
				for (int z = 0; z < zones.size(); z++)
				{
					ImpactZone *zone = zones[z];
					apply_inelastic_projection(zone, cons);
				}
				for (int a = 0; a < accs.size(); a++)
					update_accel_struct(*accs[a]);
				for (int a = 0; a < obs_accs.size(); a++)
					update_accel_struct(*obs_accs[a]);
				if (deform_obstacles)
					arcsim::obs_mass /= 2;
			}
			if (iter < max_iter) // success!
				break;
		}
		if (iter == max_iter)
		{
			cerr << "Collision resolution failed to converge!" << endl;
			debug_save_meshes(meshes, "meshes");
			debug_save_meshes(obs_meshes, "obsmeshes");
			exit(1);
		}
		for (int m = 0; m < meshes.size(); m++)
		{
			compute_ws_data(*meshes[m]);
			update_x0(*meshes[m]);
		}
		for (int o = 0; o < obs_meshes.size(); o++)
		{
			compute_ws_data(*obs_meshes[o]);
			update_x0(*obs_meshes[o]);
		}
		for (int z = 0; z < zones.size(); z++)
			delete zones[z];
		destroy_accel_structs(accs);
		destroy_accel_structs(obs_accs);
	}

	void update_active(const vector<AccelStruct*> &accs,
		const vector<AccelStruct*> &obs_accs,
		const vector<ImpactZone*> &zones)
	{
		for (int a = 0; a < accs.size(); a++)
			mark_all_inactive(*accs[a]);
		for (int a = 0; a < obs_accs.size(); a++)
			mark_all_inactive(*obs_accs[a]);
		for (int z = 0; z < zones.size(); z++)
		{
			const ImpactZone *zone = zones[z];
			if (!zone->active)
				continue;
			for (int n = 0; n < zone->nodes.size(); n++)
			{
				const Node *node = zone->nodes[n];
				pair<bool, int> mi = find_in_meshes(node);
				AccelStruct *acc = (mi.first ? accs : obs_accs)[mi.second];
				for (int v = 0; v < node->verts.size(); v++)
				for (int f = 0; f < node->verts[v]->adjf.size(); f++)
					mark_active(*acc, node->verts[v]->adjf[f]);
			}
		}
	}

	// Impacts

	static int nthreads = 0;
	static vector<Impact> *impacts = NULL;

	void find_face_impacts(const Face *face0, const Face *face1);

	vector<Impact> find_impacts(const vector<AccelStruct*> &accs,
		const vector<AccelStruct*> &obs_accs)
	{
		if (!impacts)
		{
			arcsim::nthreads = omp_get_max_threads();
			arcsim::impacts = new vector<Impact>[arcsim::nthreads];
		}
		for (int t = 0; t < arcsim::nthreads; t++)
			arcsim::impacts[t].clear();
		for_overlapping_faces(accs, obs_accs, arcsim::thickness, find_face_impacts);
		vector<Impact> impacts;
		for (int t = 0; t < arcsim::nthreads; t++)
			append(impacts, arcsim::impacts[t]);
		return impacts;
	}

	bool vf_collision_test(const Vert *vert, const Face *face, Impact &impact);
	bool ee_collision_test(const Edge *edge0, const Edge *edge1, Impact &impact);

	void find_face_impacts(const Face *face0, const Face *face1)
	{
		int t = omp_get_thread_num();
		Impact impact;
		for (int v = 0; v < 3; v++)
		if (vf_collision_test(face0->v[v], face1, impact))
			arcsim::impacts[t].push_back(impact);
		for (int v = 0; v < 3; v++)
		if (vf_collision_test(face1->v[v], face0, impact))
			arcsim::impacts[t].push_back(impact);
		for (int e0 = 0; e0 < 3; e0++)
		for (int e1 = 0; e1 < 3; e1++)
		if (ee_collision_test(face0->adje[e0], face1->adje[e1], impact))
			arcsim::impacts[t].push_back(impact);
	}

	bool collision_test(Impact::Type type, const Node *node0, const Node *node1,
		const Node *node2, const Node *node3, Impact &impact);

	bool vf_collision_test(const Vert *vert, const Face *face, Impact &impact)
	{
		const Node *node = vert->node;
		if (node == face->v[0]->node
			|| node == face->v[1]->node
			|| node == face->v[2]->node)
			return false;
		if (!overlap(node_box(node, true), face_box(face, true), arcsim::thickness))
			return false;
		return collision_test(Impact::VF, node, face->v[0]->node, face->v[1]->node,
			face->v[2]->node, impact);
	}

	bool ee_collision_test(const Edge *edge0, const Edge *edge1, Impact &impact)
	{
		if (edge0->n[0] == edge1->n[0] || edge0->n[0] == edge1->n[1]
			|| edge0->n[1] == edge1->n[0] || edge0->n[1] == edge1->n[1])
			return false;
		if (!overlap(edge_box(edge0, true), edge_box(edge1, true), arcsim::thickness))
			return false;
		return collision_test(Impact::EE, edge0->n[0], edge0->n[1],
			edge1->n[0], edge1->n[1], impact);
	}

	int solve_cubic(double a3, double a2, double a1, double a0, double t[3]);

	Vec3 pos(const Node *node, double t);

	bool collision_test(Impact::Type type, const Node *node0, const Node *node1,
		const Node *node2, const Node *node3, Impact &impact)
	{
		impact.type = type;
		impact.nodes[0] = (Node*)node0;
		impact.nodes[1] = (Node*)node1;
		impact.nodes[2] = (Node*)node2;
		impact.nodes[3] = (Node*)node3;
		const Vec3 &x0 = node0->x0, v0 = node0->x - x0;
		Vec3 x1 = node1->x0 - x0, x2 = node2->x0 - x0, x3 = node3->x0 - x0;
		Vec3 v1 = (node1->x - node1->x0) - v0, v2 = (node2->x - node2->x0) - v0,
			v3 = (node3->x - node3->x0) - v0;
		double a0 = stp(x1, x2, x3),
			a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3),
			a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3),
			a3 = stp(v1, v2, v3);
		double t[4];
		int nsol = solve_cubic(a3, a2, a1, a0, t);
		t[nsol] = 1; // also check at end of timestep
		for (int i = 0; i < nsol; i++)
		{
			if (t[i] < 0 || t[i] > 1)
				continue;
			impact.t = t[i];
			Vec3 x0 = pos(node0, t[i]), x1 = pos(node1, t[i]),
				x2 = pos(node2, t[i]), x3 = pos(node3, t[i]);
			Vec3 &n = impact.n;
			double *w = impact.w;
			double d;
			bool inside;
			if (type == Impact::VF)
			{
				d = signed_vf_distance(x0, x1, x2, x3, &n, w);
				inside = (min(-w[1], -w[2], -w[3]) >= -1e-6);
			}
			else
			{// Impact::EE
				d = signed_ee_distance(x0, x1, x2, x3, &n, w);
				inside = (min(w[0], w[1], -w[2], -w[3]) >= -1e-6);
			}
			if (dot(n, w[1] * v1 + w[2] * v2 + w[3] * v3) > 0)
				n = -n;
			if (abs(d) < 1e-6 && inside)
				return true;
		}
		return false;
	}

	Vec3 pos(const Node *node, double t)
	{
		return node->x0 + t*(node->x - node->x0);
	}

	// Solving cubic equations

	double newtons_method(double a, double b, double c, double d, double x0,
		int init_dir);

	// solves a x^3 + b x^2 + c x + d == 0
	int solve_cubic(double a, double b, double c, double d, double x[3])
	{
		double xc[2];
		int ncrit = solve_quadratic(3 * a, 2 * b, c, xc);
		if (ncrit == 0)
		{
			x[0] = newtons_method(a, b, c, d, xc[0], 0);
			return 1;
		}
		else if (ncrit == 1)
		{// cubic is actually quadratic
			return solve_quadratic(b, c, d, x);
		}
		else
		{
			double yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
				d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
			int i = 0;
			if (yc[0] * a >= 0)
				x[i++] = newtons_method(a, b, c, d, xc[0], -1);
			if (yc[0] * yc[1] <= 0)
			{
				int closer = abs(yc[0]) < abs(yc[1]) ? 0 : 1;
				x[i++] = newtons_method(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
			}
			if (yc[1] * a <= 0)
				x[i++] = newtons_method(a, b, c, d, xc[1], 1);
			return i;
		}
	}

	double newtons_method(double a, double b, double c, double d, double x0,
		int init_dir)
	{
		if (init_dir != 0)
		{
			// quadratic approximation around x0, assuming y' = 0
			double y0 = d + x0*(c + x0*(b + x0*a)),
				ddy0 = 2 * b + x0*(6 * a);
			x0 += init_dir*sqrt(abs(2 * y0 / ddy0));
		}
		for (int iter = 0; iter < 100; iter++)
		{
			double y = d + x0*(c + x0*(b + x0*a));
			double dy = c + x0*(2 * b + x0 * 3 * a);
			if (dy == 0)
				return x0;
			double x1 = x0 - y / dy;
			if (abs(x0 - x1) < 1e-6)
				return x0;
			x0 = x1;
		}
		return x0;
	}

	// Independent impacts

	bool operator< (const Impact &impact0, const Impact &impact1)
	{
		return impact0.t < impact1.t;
	}

	bool conflict(const Impact &impact0, const Impact &impact1);

	vector<Impact> independent_impacts(const vector<Impact> &impacts)
	{
		vector<Impact> sorted = impacts;
		sort(sorted.begin(), sorted.end());
		vector<Impact> indep;
		for (int e = 0; e < sorted.size(); e++)
		{
			const Impact &impact = sorted[e];
			bool con = false;
			for (int e1 = 0; e1 < indep.size(); e1++)
			if (conflict(impact, indep[e1]))
				con = true;
			if (!con)
				indep.push_back(impact);
		}
		return indep;
	}

	bool conflict(const Impact &i0, const Impact &i1)
	{
		return (is_free(i0.nodes[0]) && is_in(i0.nodes[0], i1.nodes, 4))
			|| (is_free(i0.nodes[1]) && is_in(i0.nodes[1], i1.nodes, 4))
			|| (is_free(i0.nodes[2]) && is_in(i0.nodes[2], i1.nodes, 4))
			|| (is_free(i0.nodes[3]) && is_in(i0.nodes[3], i1.nodes, 4));
	}

	// Impact zones

	ImpactZone *find_or_create_zone(const Node *node, vector<ImpactZone*> &zones);
	void merge_zones(ImpactZone* zone0, ImpactZone *zone1,
		vector<ImpactZone*> &zones);

	void add_impacts(const vector<Impact> &impacts, vector<ImpactZone*> &zones)
	{
		for (int z = 0; z < zones.size(); z++)
			zones[z]->active = false;
		for (int i = 0; i < impacts.size(); i++)
		{
			const Impact &impact = impacts[i];
			Node *node = impact.nodes[is_free(impact.nodes[0]) ? 0 : 3];
			ImpactZone *zone = find_or_create_zone(node, zones);
			for (int n = 0; n < 4; n++)
			if (is_free(impact.nodes[n]) || arcsim::deform_obstacles)
				merge_zones(zone, find_or_create_zone(impact.nodes[n], zones),
				zones);
			zone->impacts.push_back(impact);
			zone->active = true;
		}
	}

	ImpactZone *find_or_create_zone(const Node *node, vector<ImpactZone*> &zones)
	{
		for (int z = 0; z < zones.size(); z++)
		if (is_in((Node*)node, zones[z]->nodes))
			return zones[z];
		ImpactZone *zone = new ImpactZone;
		zone->nodes.push_back((Node*)node);
		zones.push_back(zone);
		return zone;
	}

	void merge_zones(ImpactZone* zone0, ImpactZone *zone1,
		vector<ImpactZone*> &zones)
	{
		if (zone0 == zone1)
			return;
		append(zone0->nodes, zone1->nodes);
		append(zone0->impacts, zone1->impacts);
		exclude(zone1, zones);
		delete zone1;
	}

	// Response

	struct NormalOpt : public NLConOpt
	{
		ImpactZone *zone;
		double inv_m;
		NormalOpt() : zone(NULL), inv_m(0) { nvar = ncon = 0; }
		NormalOpt(ImpactZone *zone) : zone(zone), inv_m(0)
		{
			nvar = zone->nodes.size() * 3;
			ncon = zone->impacts.size();
			for (int n = 0; n < zone->nodes.size(); n++)
				inv_m += 1 / get_mass(zone->nodes[n]);
			inv_m /= zone->nodes.size();
		}
		void initialize(double *x) const;
		void precompute(const double *x) const;
		double objective(const double *x) const;
		void obj_grad(const double *x, double *grad) const;
		double constraint(const double *x, int i, int &sign) const;
		void con_grad(const double *x, int i, double factor, double *grad) const;
		void finalize(const double *x) const;
	};

	void apply_inelastic_projection(ImpactZone *zone,
		const vector<Constraint*> &cons)
	{
		if (!zone->active)
			return;
		augmented_lagrangian_method(NormalOpt(zone));
	}

	void NormalOpt::initialize(double *x) const
	{
		for (int n = 0; n < zone->nodes.size(); n++)
			set_subvec(x, n, zone->nodes[n]->x);
	}

	void NormalOpt::precompute(const double *x) const
	{
		for (int n = 0; n < zone->nodes.size(); n++)
			zone->nodes[n]->x = get_subvec(x, n);
	}

	const Vec3 &get_xold(const Node *node);

	double NormalOpt::objective(const double *x) const
	{
		double e = 0;
		for (int n = 0; n < zone->nodes.size(); n++)
		{
			const Node *node = zone->nodes[n];
			Vec3 dx = node->x - get_xold(node);
			e += inv_m*get_mass(node)*norm2(dx) / 2;
		}
		return e;
	}

	void NormalOpt::obj_grad(const double *x, double *grad) const
	{
		for (int n = 0; n < zone->nodes.size(); n++)
		{
			const Node *node = zone->nodes[n];
			Vec3 dx = node->x - get_xold(node);
			set_subvec(grad, n, inv_m*get_mass(node)*dx);
		}
	}

	double NormalOpt::constraint(const double *x, int j, int &sign) const
	{
		sign = 1;
		double c = -arcsim::thickness;
		const Impact &impact = zone->impacts[j];
		for (int n = 0; n < 4; n++)
			c += impact.w[n] * dot(impact.n, impact.nodes[n]->x);
		return c;
	}

	void NormalOpt::con_grad(const double *x, int j, double factor,
		double *grad) const
	{
		const Impact &impact = zone->impacts[j];
		for (int n = 0; n < 4; n++)
		{
			int i = find(impact.nodes[n], zone->nodes);
			if (i != -1)
				add_subvec(grad, i, factor*impact.w[n] * impact.n);
		}
	}

	void NormalOpt::finalize(const double *x) const
	{
		precompute(x);
	}

	const Vec3 &get_xold(const Node *node)
	{
		pair<bool, int> mi = find_in_meshes(node);
		int ni = get_index(node, mi.first ? *arcsim::meshes : *arcsim::obs_meshes);
		return (mi.first ? arcsim::xold : arcsim::xold_obs)[ni];
	}
}