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

#include "plasticity.hpp"

#include "bah.hpp"
#include "geometry.hpp"
#include "optimization.hpp"
#include "physics.hpp"
#include <omp.h>

using namespace std;

namespace arcsim
{

	static const double mu = 1e-6;

	Mat2x2 edges_to_face(const Vec3 &theta, const Face *face);
	Vec3 face_to_edges(const Mat2x2 &S, const Face *face);

	void reset_plasticity(Cloth &cloth)
	{
		Mesh &mesh = cloth.mesh;
		for (int n = 0; n < mesh.nodes.size(); n++)
			mesh.nodes[n]->y = mesh.nodes[n]->x;
		for (int e = 0; e < mesh.edges.size(); e++)
		{
			Edge *edge = mesh.edges[e];
			edge->theta_ideal = edge->reference_angle = edge->theta;
			edge->damage = 0;
		}
		for (int f = 0; f < mesh.faces.size(); f++)
		{
			Face *face = mesh.faces[f];
			Vec3 theta = Vec3(face->adje[0]->theta,
				face->adje[1]->theta,
				face->adje[2]->theta);
			face->S_plastic = edges_to_face(theta, face);
			face->damage = 0;
		}
	}

	void recompute_edge_plasticity(Mesh &mesh);

	void optimize_plastic_embedding(Cloth &cloth);

	void plastic_update(Cloth &cloth)
	{
		Mesh &mesh = cloth.mesh;
		const vector<Cloth::Material*> &materials = cloth.materials;
		for (int f = 0; f < mesh.faces.size(); f++)
		{
			Face *face = mesh.faces[f];
			double S_yield = materials[face->label]->yield_curv;
			Vec3 theta = Vec3(face->adje[0]->theta,
				face->adje[1]->theta,
				face->adje[2]->theta);
			Mat2x2 S_total = edges_to_face(theta, face);
			Mat2x2 S_elastic = S_total - face->S_plastic;
			double dS = norm_F(S_elastic);
			if (dS > S_yield)
			{
				face->S_plastic += S_elastic / dS*(dS - S_yield);
				face->damage += dS / S_yield - 1;
			}
		}
		recompute_edge_plasticity(cloth.mesh);
	}

	// ------------------------------------------------------------------ //

	struct EmbedOpt : public NLOpt
	{
		Cloth &cloth;
		Mesh &mesh;
		vector<Vec3> y0;
		mutable vector<Vec3> f;
		mutable SpMat<Mat3x3> J;
		EmbedOpt(Cloth &cloth) : cloth(cloth), mesh(cloth.mesh)
		{
			int nn = mesh.nodes.size();
			nvar = nn * 3;
			y0.resize(nn);
			for (int n = 0; n < nn; n++)
				y0[n] = mesh.nodes[n]->y;
			// f.resize(nv);
			// J = SpMat<Mat3x3>(nv,nv);
		}
		void initialize(double *x) const;
		void precompute(const double *x) const;
		double objective(const double *x) const;
		void gradient(const double *x, double *g) const;
		bool hessian(const double *x, SpMat<double> &H) const;
		void finalize(const double *x) const;
	};

	void reduce_stretching_stiffnesses(vector<Cloth::Material*> &materials);
	void restore_stretching_stiffnesses(vector<Cloth::Material*> &materials);

	void optimize_plastic_embedding(Cloth &cloth)
	{
		// vector<Cloth::Material> materials = cloth.materials;
		reduce_stretching_stiffnesses(cloth.materials);
		line_search_newtons_method(EmbedOpt(cloth), OptOptions().max_iter(1));
		restore_stretching_stiffnesses(cloth.materials);
		// cloth.materials = materials;
	}

	void EmbedOpt::initialize(double *x) const
	{
		for (int n = 0; n < mesh.nodes.size(); n++)
			set_subvec(x, n, Vec3(0));
	}

	void EmbedOpt::precompute(const double *x) const
	{
		int nn = mesh.nodes.size();
		f.assign(nn, Vec3(0));
		J = SpMat<Mat3x3>(nn, nn);
		for (int n = 0; n < mesh.nodes.size(); n++)
			mesh.nodes[n]->y = y0[n] + get_subvec(x, n);
		add_internal_forces<PS>(cloth, J, f, 0);
	}

	double EmbedOpt::objective(const double *x) const
	{
		for (int n = 0; n < mesh.nodes.size(); n++)
			mesh.nodes[n]->y = y0[n] + get_subvec(x, n);
		return internal_energy<PS>(cloth);
	}

	void EmbedOpt::gradient(const double *x, double *g) const
	{
		for (int n = 0; n < mesh.nodes.size(); n++)
		{
			const Node *node = mesh.nodes[n];
			set_subvec(g, n, -f[n]);
		}
	}

	static Mat3x3 get_submat(SpMat<double> &A, int i, int j)
	{
		Mat3x3 Aij;
		for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++)
			Aij(ii, jj) = A(i * 3 + ii, j * 3 + jj);
		return Aij;
	}
	static void set_submat(SpMat<double> &A, int i, int j, const Mat3x3 &Aij)
	{
		for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++)
			A(i * 3 + ii, j * 3 + jj) = Aij(ii, jj);
	}
	static void add_submat(SpMat<double> &A, int i, int j, const Mat3x3 &Aij)
	{
		for (int ii = 0; ii < 3; ii++) for (int jj = 0; jj < 3; jj++)
			A(i * 3 + ii, j * 3 + jj) += Aij(ii, jj);
	}

	bool EmbedOpt::hessian(const double *x, SpMat<double> &H) const
	{
		for (int i = 0; i < mesh.nodes.size(); i++)
		{
			const SpVec<Mat3x3> &Ji = J.rows[i];
			for (int jj = 0; jj < Ji.indices.size(); jj++)
			{
				int j = Ji.indices[jj];
				const Mat3x3 &Jij = Ji.entries[jj];
				set_submat(H, i, j, Jij);
			}
			add_submat(H, i, i, Mat3x3(arcsim::mu));
		}
		return true;
	}

	void EmbedOpt::finalize(const double *x) const
	{
		for (int n = 0; n < mesh.nodes.size(); n++)
			mesh.nodes[n]->y = y0[n] + get_subvec(x, n);
	}

	void reduce_stretching_stiffnesses(vector<Cloth::Material*> &materials)
	{
		for (int m = 0; m < materials.size(); m++)
		for (int i = 0; i < 40; i++)
		for (int j = 0; j < 40; j++)
		for (int k = 0; k < 40; k++)
			materials[m]->stretching.s[i][j][k] *= 1e-2;
	}

	void restore_stretching_stiffnesses(vector<Cloth::Material*> &materials)
	{
		for (int m = 0; m < materials.size(); m++)
		for (int i = 0; i < 40; i++)
		for (int j = 0; j < 40; j++)
		for (int k = 0; k < 40; k++)
			materials[m]->stretching.s[i][j][k] *= 1e2;
	}

	// ------------------------------------------------------------------ //

	Mat2x2 edges_to_face(const Vec3 &theta, const Face *face)
	{
		Mat2x2 S;
		for (int e = 0; e < 3; e++)
		{
			const Edge *edge = face->adje[e];
			Vec2 e_mat = face->v[PREV(e)]->u - face->v[NEXT(e)]->u,
				t_mat = perp(normalize(e_mat));
			S -= 1 / 2.*theta[e] * norm(e_mat)*outer(t_mat, t_mat);
		}
		S /= face->a;
		return S;
	}

	Vec3 face_to_edges(const Mat2x2 &S, const Face *face)
	{
		Vec3 s = face->a*Vec3(S(0, 0), S(1, 1), S(0, 1));
		Mat3x3 A;
		for (int e = 0; e < 3; e++)
		{
			const Edge *edge = face->adje[e];
			Vec2 e_mat = face->v[PREV(e)]->u - face->v[NEXT(e)]->u,
				t_mat = perp(normalize(e_mat));
			Mat2x2 Se = -1 / 2.*norm(e_mat)*outer(t_mat, t_mat);
			A.col(e) = Vec3(Se(0, 0), Se(1, 1), Se(0, 1));
		}
		return A.inv()*s;
	}

	void recompute_edge_plasticity(Mesh &mesh)
	{
		for (int e = 0; e < mesh.edges.size(); e++)
		{
			mesh.edges[e]->theta_ideal = 0;
			mesh.edges[e]->damage = 0;
		}
		for (int f = 0; f < mesh.faces.size(); f++)
		{
			const Face *face = mesh.faces[f];
			Vec3 theta = face_to_edges(face->S_plastic, face);
			for (int e = 0; e < 3; e++)
			{
				face->adje[e]->theta_ideal += theta[e];
				face->adje[e]->damage += face->damage;
			}
		}
		for (int e = 0; e < mesh.edges.size(); e++)
		{
			Edge *edge = mesh.edges[e];
			if (edge->adjf[0] && edge->adjf[1])
			{// edge has two adjacent faces
				edge->theta_ideal /= 2;
				edge->damage /= 2;
			}
			edge->reference_angle = edge->theta_ideal;
		}
	}

	// ------------------------------------------------------------------ //

	const vector<Residual> *res_old;

	void resample_callback(Face *face_new, const Face *face_old);

	vector<Residual> back_up_residuals(Mesh &mesh)
	{
		vector<Residual> res(mesh.faces.size());
		for (int f = 0; f < mesh.faces.size(); f++)
		{
			const Face *face = mesh.faces[f];
			Vec3 theta;
			for (int e = 0; e < 3; e++)
			{
				const Edge *edge = face->adje[e];
				theta[e] = edge->theta_ideal - dihedral_angle<PS>(edge);
			}
			res[f].S_res = edges_to_face(theta, face);
			res[f].damage = face->damage;
		}
		return res;
	}

	void restore_residuals(Mesh &mesh, const Mesh &old_mesh,
		const vector<Residual> &res_old)
	{
		arcsim::res_old = &res_old;
		BahNode *tree = new_bah_tree(old_mesh);
#pragma omp parallel for
		for (int f = 0; f < mesh.faces.size(); f++)
		{
			Face *face = mesh.faces[f];
			Vec3 theta;
			for (int e = 0; e < 3; e++)
				theta[e] = dihedral_angle<PS>(face->adje[e]);
			face->S_plastic = edges_to_face(theta, face);
			face->damage = 0;
			for_overlapping_faces(face, tree, resample_callback);
		}
		delete_bah_tree(tree);
		recompute_edge_plasticity(mesh);
	}

	double overlap_area(const Face *face0, const Face *face1);

	void resample_callback(Face *face_new, const Face *face_old)
	{
		double a = overlap_area(face_new, face_old) / face_new->a;
		const Residual &res = (*arcsim::res_old)[face_old->index];
		face_new->S_plastic += a*res.S_res;
		face_new->damage += a*res.damage;
	}

	// ------------------------------------------------------------------ //

	vector<Vec2> sutherland_hodgman(const vector<Vec2> &poly0,
		const vector<Vec2> &poly1);
	double area(const vector<Vec2> &poly);

	double overlap_area(const Face *face0, const Face *face1)
	{
		vector<Vec2> u0(3), u1(3);
		Vec2 u0min(face0->v[0]->u), u0max(u0min),
			u1min(face1->v[0]->u), u1max(u1min);
		for (int i = 0; i < 3; i++)
		{
			u0[i] = face0->v[i]->u;
			u1[i] = face1->v[i]->u;
			u0min = vec_min(u0min, u0[i]);
			u0max = vec_max(u0max, u0[i]);
			u1min = vec_min(u1min, u1[i]);
			u1max = vec_max(u1max, u1[i]);
		}
		if (u0min[0] > u1max[0] || u0max[0] < u1min[0]
			|| u0min[1] > u1max[1] || u0max[1] < u1min[1])
		{
			return 0;
		}
		return area(sutherland_hodgman(u0, u1));
	}

	vector<Vec2> clip(const vector<Vec2> &poly, const Vec2 &clip0,
		const Vec2 &clip1);

	vector<Vec2> sutherland_hodgman(const vector<Vec2> &poly0,
		const vector<Vec2> &poly1)
	{
		vector<Vec2> out(poly0);
		for (int i = 0; i < 3; i++)
			out = clip(out, poly1[i], poly1[(i + 1) % poly1.size()]);
		return out;
	}

	double distance(const Vec2 &v, const Vec2 &v0, const Vec2 &v1)
	{
		return wedge(v1 - v0, v - v0);
	}
	Vec2 lerp(double t, const Vec2 &a, const Vec2 &b) { return a + t*(b - a); }

	vector<Vec2> clip(const vector<Vec2> &poly, const Vec2 &clip0,
		const Vec2 &clip1)
	{
		if (poly.empty())
			return poly;
		vector<Vec2> newpoly;
		for (int i = 0; i < poly.size(); i++)
		{
			const Vec2 &v0 = poly[i], &v1 = poly[(i + 1) % poly.size()];
			double d0 = distance(v0, clip0, clip1), d1 = distance(v1, clip0, clip1);
			if (d0 >= 0)
				newpoly.push_back(v0);
			if (!((d0<0 && d1<0) || (d0 == 0 && d1 == 0) || (d0>0 && d1>0)))
				newpoly.push_back(lerp(d0 / (d0 - d1), v0, v1));
		}
		return newpoly;
	}

	double area(const vector<Vec2> &poly)
	{
		if (poly.empty())
			return 0;
		double a = 0;
		for (int i = 1; i < poly.size() - 1; i++)
			a += wedge(poly[i] - poly[0], poly[i + 1] - poly[0]) / 2;
		return a;
	}
}