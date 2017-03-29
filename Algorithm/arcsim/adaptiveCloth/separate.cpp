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

#include "separate.hpp"

#include "collisionutil.hpp"
#include "geometry.hpp"
#include "io.hpp"
#include "magic.hpp"
#include "optimization.hpp"
#include "simulation.hpp"
#include "util.hpp"
#include <omp.h>
using namespace std;

namespace arcsim
{

	static const int max_iter = 100;
	static const double &thickness = arcsim::magic.projection_thickness;

	static const vector<Mesh*> *old_meshes;
	static vector<Vec3> xold;

	typedef Vec3 Bary; // barycentric coordinates

	struct Ixn
	{// intersection
		Face *f0, *f1;
		Bary b0, b1;
		Vec3 n;
		Ixn() {}
		Ixn(const Face *f0, const Bary &b0, const Face *f1, const Bary &b1,
			const Vec3 &n) : f0((Face*)f0), f1((Face*)f1), b0(b0), b1(b1), n(n)
		{}
	};

	ostream &operator<< (ostream &out, const Ixn &ixn) { out << ixn.f0 << "@" << ixn.b0 << " " << ixn.f1 << "@" << ixn.b1 << " " << ixn.n; return out; }

	void update_active(const vector<AccelStruct*> &accs, const vector<Ixn> &ixns);

	vector<Ixn> find_intersections(const vector<AccelStruct*> &accs,
		const vector<AccelStruct*> &obs_accs);

	void solve_ixns(const vector<Ixn> &ixns);

	void separate(vector<Mesh*> &meshes, const vector<Mesh*> &old_meshes,
		const vector<Mesh*> &obs_meshes)
	{
		arcsim::meshes = &meshes;
		arcsim::old_meshes = &old_meshes;
		arcsim::obs_meshes = &obs_meshes;
		arcsim::xold = node_positions(meshes);
		vector<AccelStruct*> accs = create_accel_structs(meshes, false),
			obs_accs = create_accel_structs(obs_meshes, false);
		vector<Ixn> ixns;
		int iter;
		for (iter = 0; iter < max_iter; iter++)
		{
			if (!ixns.empty())
				update_active(accs, ixns);
			vector<Ixn> new_ixns = find_intersections(accs, obs_accs);
			if (new_ixns.empty())
				break;
			append(ixns, new_ixns);
			solve_ixns(ixns);
			for (int m = 0; m < meshes.size(); m++)
			{
				compute_ws_data(*meshes[m]);
				update_accel_struct(*accs[m]);
			}
		}
		if (iter == max_iter)
		{
			cerr << "Post-remeshing separation failed to converge!" << endl;
			debug_save_meshes(meshes, "meshes");
			debug_save_meshes(old_meshes, "oldmeshes");
			debug_save_meshes(obs_meshes, "obsmeshes");
			exit(1);
		}
		for (int m = 0; m < meshes.size(); m++)
		{
			compute_ws_data(*meshes[m]);
			update_x0(*meshes[m]);
		}
		destroy_accel_structs(accs);
		destroy_accel_structs(obs_accs);
	}

	Vec3 pos(const Face *face, const Bary &b)
	{
		return b[0] * face->v[0]->node->x
			+ b[1] * face->v[1]->node->x
			+ b[2] * face->v[2]->node->x;
	}

	Vec3 old_pos(const Face *face, const Bary &b)
	{
		if (!is_free(face))
			return pos(face, b);
		Vec2 u = b[0] * face->v[0]->u + b[1] * face->v[1]->u + b[2] * face->v[2]->u;
		int m;
		for (m = 0; m < arcsim::meshes->size(); m++)
		if ((*arcsim::meshes)[m]->faces[face->index] == face)
			break;
		Face *old_face = get_enclosing_face(*(*arcsim::old_meshes)[m], u);
		Bary old_b = get_barycentric_coords(u, old_face);
		return pos(old_face, old_b);
	}

	void update_active(const vector<AccelStruct*> &accs, const vector<Ixn> &ixns)
	{
		// for (int a = 0; a < accs.size(); a++)
		//     mark_all_inactive(*accs[a]);
		// for (int i = 0; i < ixns.size(); i++)
		//     for (int f = 0; f < 2; f++) {
		//         const Face *face = f==0 ? ixns[i].f0 : ixns[i].f1;
		//         int m = find(face, *::meshes);
		//         if (m == -1)
		//             continue;
		//         mark_active(*accs[m], face);
		//     }
	}

	static int nthreads = 0;
	static vector<Ixn> *ixns = NULL;

	void find_face_intersection(const Face *face0, const Face *face1);

	vector<Ixn> find_intersections(const vector<AccelStruct*> &accs,
		const vector<AccelStruct*> &obs_accs)
	{
		if (!arcsim::ixns)
		{
			arcsim::nthreads = omp_get_max_threads();
			arcsim::ixns = new vector<Ixn>[arcsim::nthreads];
		}
		for (int t = 0; t < arcsim::nthreads; t++)
			arcsim::ixns[t].clear();
		for_overlapping_faces(accs, obs_accs, arcsim::thickness, find_face_intersection);
		vector<Ixn> ixns;
		for (int t = 0; t < arcsim::nthreads; t++)
			append(ixns, arcsim::ixns[t]);
		return ixns;
	}

	bool adjacent(const Face *face0, const Face *face1);

	bool intersection_midpoint(const Face *face0, const Face *face1,
		Bary &b0, Bary &b1);
	bool farthest_points(const Face *face0, const Face *face1, const Vec3 &d,
		Bary &b0, Bary &b1);

	void find_face_intersection(const Face *face0, const Face *face1)
	{
		if (adjacent(face0, face1))
			return;
		int t = omp_get_thread_num();
		Bary b0, b1;
		bool is_ixn = intersection_midpoint(face0, face1, b0, b1);
		if (!is_ixn)
			return;
		Vec3 n = normalize(old_pos(face0, b0) - old_pos(face1, b1));
		farthest_points(face0, face1, n, b0, b1);
		arcsim::ixns[t].push_back(Ixn(face0, b0, face1, b1, n));
	}

	bool adjacent(const Face *face0, const Face *face1)
	{
		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		if (face0->v[i]->node == face1->v[j]->node)
			return true;
		return false;
	}

	bool face_plane_intersection(const Face *face, const Face *plane,
		Bary &b0, Bary &b1);
	int major_axis(const Vec3 &v);

	bool intersection_midpoint(const Face *face0, const Face *face1,
		Bary &b0, Bary &b1)
	{
		if (norm2(cross(face0->n, face1->n)) < 1e-12)
			return false;
		Bary b00, b01, b10, b11;
		bool ix0 = face_plane_intersection(face0, face1, b00, b01),
			ix1 = face_plane_intersection(face1, face0, b10, b11);
		if (!ix0 || !ix1)
			return false;
		int axis = major_axis(cross(face0->n, face1->n));
		double a00 = pos(face0, b00)[axis], a01 = pos(face0, b01)[axis],
			a10 = pos(face1, b10)[axis], a11 = pos(face1, b11)[axis];
		double amin = std::max(std::min(a00, a01), std::min(a10, a11)),
			amax = std::min(std::max(a00, a01), std::max(a10, a11)),
			amid = (amin + amax) / 2;
		if (amin > amax)
			return false;
		b0 = (a01 == a00) ? b00 : b00 + (amid - a00) / (a01 - a00)*(b01 - b00);
		b1 = (a11 == a10) ? b10 : b10 + (amid - a10) / (a11 - a10)*(b11 - b10);
		return true;
	}

	bool face_plane_intersection(const Face *face, const Face *plane,
		Bary &b0, Bary &b1)
	{
		const Vec3 &x0 = plane->v[0]->node->x, &n = plane->n;
		double h[3];
		int sign_sum = 0;
		for (int v = 0; v < 3; v++)
		{
			h[v] = dot(face->v[v]->node->x - x0, n);
			sign_sum += sgn(h[v]);
		}
		if (sign_sum == -3 || sign_sum == 3)
			return false;
		int v0 = -1;
		for (int v = 0; v < 3; v++)
		if (sgn(h[v]) == -sign_sum)
			v0 = v;
		double t0 = h[v0] / (h[v0] - h[NEXT(v0)]), t1 = h[v0] / (h[v0] - h[PREV(v0)]);
		b0[v0] = 1 - t0;
		b0[NEXT(v0)] = t0;
		b0[PREV(v0)] = 0;
		b1[v0] = 1 - t1;
		b1[PREV(v0)] = t1;
		b1[NEXT(v0)] = 0;
		return true;
	}

	int major_axis(const Vec3 &v)
	{
		return (abs(v[0]) > abs(v[1]) && abs(v[0]) > abs(v[2])) ? 0
			: (abs(v[1]) > abs(v[2])) ? 1 : 2;
	}

	double vf_clear_distance(const Face *face0, const Face *face1, const Vec3 &d,
		double last_dist, Bary &b0, Bary &b1);
	double ee_clear_distance(const Face *face0, const Face *face1, const Vec3 &d,
		double last_dist, Bary &b0, Bary &b1);

	bool farthest_points(const Face *face0, const Face *face1, const Vec3 &d,
		Bary &b0, Bary &b1)
	{
		double last_dist = 0;
		last_dist = vf_clear_distance(face0, face1, d, last_dist, b0, b1);
		last_dist = vf_clear_distance(face1, face0, -d, last_dist, b1, b0);
		last_dist = ee_clear_distance(face0, face1, d, last_dist, b0, b1);
		return last_dist > 0;
	}

	double vf_clear_distance(const Face *face0, const Face *face1, const Vec3 &d,
		double last_dist, Bary &b0, Bary &b1)
	{
		for (int v = 0; v < 3; v++)
		{
			const Vec3 &xv = face0->v[v]->node->x, &x0 = face1->v[0]->node->x,
				&x1 = face1->v[1]->node->x, &x2 = face1->v[2]->node->x;
			const Vec3 &n = face1->n;
			double h = dot(xv - x0, n), dh = dot(d, n);
			if (h*dh >= 0)
				continue;
			double a0 = stp(x2 - x1, xv - x1, d),
				a1 = stp(x0 - x2, xv - x2, d),
				a2 = stp(x1 - x0, xv - x0, d);
			if (a0 <= 0 || a1 <= 0 || a2 <= 0)
				continue;
			double dist = -h / dh;
			if (dist > last_dist)
			{
				last_dist = dist;
				b0 = Bary(0);
				b0[v] = 1;
				b1[0] = a0 / (a0 + a1 + a2);
				b1[1] = a1 / (a0 + a1 + a2);
				b1[2] = a2 / (a0 + a1 + a2);
			}
		}
		return last_dist;
	}

	double ee_clear_distance(const Face *face0, const Face *face1, const Vec3 &d,
		double last_dist, Bary &b0, Bary &b1)
	{
		for (int e0 = 0; e0 < 3; e0++)
		{
			for (int e1 = 0; e1 < 3; e1++)
			{
				const Vec3 &x00 = face0->v[e0]->node->x,
					&x01 = face0->v[NEXT(e0)]->node->x,
					&x10 = face1->v[e1]->node->x,
					&x11 = face1->v[NEXT(e1)]->node->x;
				Vec3 n = cross(normalize(x01 - x00), normalize(x11 - x10));
				double h = dot(x00 - x10, n), dh = dot(d, n);
				if (h*dh >= 0)
					continue;
				double a00 = stp(x01 - x10, x11 - x10, d),
					a01 = stp(x11 - x10, x00 - x10, d),
					a10 = stp(x01 - x00, x11 - x00, d),
					a11 = stp(x10 - x00, x01 - x00, d);
				if (a00*a01 <= 0 || a10*a11 <= 0)
					continue;
				double dist = -h / dh;
				if (dist > last_dist)
				{
					last_dist = dist;
					b0 = Bary(0);
					b0[e0] = a00 / (a00 + a01);
					b0[NEXT(e0)] = a01 / (a00 + a01);
					b1 = Bary(0);
					b1[e1] = a10 / (a10 + a11);
					b1[NEXT(e1)] = a11 / (a10 + a11);
				}
			}
		}
		return last_dist;
	}

	struct SeparationOpt : public NLConOpt
	{
		const vector<Ixn> &ixns;
		vector<Node*> nodes;
		double inv_m;
		SeparationOpt(const vector<Ixn> &ixns) : ixns(ixns), inv_m(0)
		{
			for (int i = 0; i < ixns.size(); i++)
			{
				if (is_free(ixns[i].f0))
				for (int v = 0; v < 3; v++)
					include(ixns[i].f0->v[v]->node, nodes);
				if (is_free(ixns[i].f1))
				for (int v = 0; v < 3; v++)
					include(ixns[i].f1->v[v]->node, nodes);
			}
			nvar = nodes.size() * 3;
			ncon = ixns.size();
			for (int n = 0; n < nodes.size(); n++)
				inv_m += 1 / nodes[n]->a;
		}
		void initialize(double *x) const;
		double objective(const double *x) const;
		void obj_grad(const double *x, double *grad) const;
		double constraint(const double *x, int j, int &sign) const;
		void con_grad(const double *x, int j, double factor, double *grad) const;
		void finalize(const double *x) const;
	};

	void solve_ixns(const vector<Ixn> &ixns)
	{
		augmented_lagrangian_method(SeparationOpt(ixns));
	}

	void SeparationOpt::initialize(double *x) const
	{
		for (int n = 0; n < nodes.size(); n++)
			set_subvec(x, n, nodes[n]->x);
	}

	double SeparationOpt::objective(const double *x) const
	{
		double f = 0;
		for (int n = 0; n < nodes.size(); n++)
		{
			const Node *node = nodes[n];
			Vec3 dx = get_subvec(x, n) - arcsim::xold[get_index(node, *arcsim::meshes)];
			f += inv_m*node->a*dot(dx, dx) / 2;
		}
		return f;
	}

	void SeparationOpt::obj_grad(const double *x, double *grad) const
	{
		for (int n = 0; n < nodes.size(); n++)
		{
			const Node *node = nodes[n];
			Vec3 dx = get_subvec(x, n) - arcsim::xold[get_index(node, *arcsim::meshes)];
			set_subvec(grad, n, inv_m*node->a*dx);
		}
	}

	double SeparationOpt::constraint(const double *x, int j, int &sign) const
	{
		const Ixn &ixn = ixns[j];
		sign = 1;
		double c = -arcsim::thickness;
		for (int v = 0; v < 3; v++)
		{
			int i0 = find(ixn.f0->v[v]->node, nodes),
				i1 = find(ixn.f1->v[v]->node, nodes);
			Vec3 x0 = (i0 != -1) ? get_subvec(x, i0) : ixn.f0->v[v]->node->x,
				x1 = (i1 != -1) ? get_subvec(x, i1) : ixn.f1->v[v]->node->x;
			c += ixn.b0[v] * dot(ixn.n, x0);
			c -= ixn.b1[v] * dot(ixn.n, x1);
		}
		return c;
	}

	void SeparationOpt::con_grad(const double *x, int j, double factor,
		double *grad) const
	{
		const Ixn &ixn = ixns[j];
		for (int v = 0; v < 3; v++)
		{
			int i0 = find(ixn.f0->v[v]->node, nodes),
				i1 = find(ixn.f1->v[v]->node, nodes);
			if (i0 != -1)
				add_subvec(grad, i0, factor*ixn.b0[v] * ixn.n);
			if (i1 != -1)
				add_subvec(grad, i1, -factor*ixn.b1[v] * ixn.n);
		}
	}

	void SeparationOpt::finalize(const double *x) const
	{
		for (int n = 0; n < nodes.size(); n++)
			nodes[n]->x = get_subvec(x, n);
	}
}