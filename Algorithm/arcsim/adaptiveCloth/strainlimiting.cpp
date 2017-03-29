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

#include "strainlimiting.hpp"

#include "optimization.hpp"
#include "simulation.hpp"
#include <omp.h>
#include "io.hpp"
using namespace std;

namespace arcsim
{

	vector<Vec2> get_strain_limits(const vector<Cloth> &cloths)
	{
		vector<Vec2> strain_limits;
		for (int c = 0; c < cloths.size(); c++)
		{
			const Cloth &cloth = cloths[c];
			const Mesh &mesh = cloth.mesh;
			int f0 = strain_limits.size();
			strain_limits.resize(strain_limits.size() + cloth.mesh.faces.size());
			for (int f = 0; f < mesh.faces.size(); f++)
			{
				const Cloth::Material *material =
					cloth.materials[mesh.faces[f]->label];
				strain_limits[f0 + f] = Vec2(material->strain_min,
					material->strain_max);
			}
		}
		return strain_limits;
	}

	struct SLOpt : public NLConOpt
	{
		vector<Mesh*> meshes;
		int nn, nf;
		const vector<Vec2> &strain_limits;
		const vector<Constraint*> &cons;
		vector<Vec3> xold;
		vector<double> conold;
		mutable vector<double> s;
		mutable vector<Mat3x3> sg;
		double inv_m;
		SLOpt(vector<Mesh*> &meshes, const vector<Vec2> &strain_limits,
			const vector<Constraint*> &cons) :
			meshes(meshes), nn(size<Node>(meshes)), nf(size<Face>(meshes)),
			strain_limits(strain_limits), cons(cons),
			xold(node_positions(meshes)), s(nf * 2), sg(nf * 2)
		{
			nvar = nn * 3;
			ncon = cons.size() + nf * 4;
			conold.resize(cons.size());
			for (int j = 0; j < cons.size(); j++)
				conold[j] = cons[j]->value();
			inv_m = 0;
			for (int n = 0; n < nn; n++)
				inv_m += 1 / get<Node>(n, meshes)->m;
			inv_m /= nn;
		}
		void initialize(double *x) const;
		double objective(const double *x) const;
		void obj_grad(const double *x, double *grad) const;
		void precompute(const double *x) const;
		double constraint(const double *x, int j, int &sign) const;
		void con_grad(const double *x, int j, double factor, double *grad) const;
		void finalize(const double *x) const;
	};

	void strain_limiting(vector<Mesh*> &meshes, const vector<Vec2> &strain_limits,
		const vector<Constraint*> &cons)
	{
		augmented_lagrangian_method(SLOpt(meshes, strain_limits, cons));
	}

	void SLOpt::initialize(double *x) const
	{
		for (int n = 0; n < nn; n++)
		{
			const Node *node = get<Node>(n, meshes);
			set_subvec(x, n, node->x);
		}
	}

	void SLOpt::precompute(const double *x) const
	{
#pragma omp parallel for
		for (int n = 0; n < nn; n++)
			get<Node>(n, meshes)->x = get_subvec(x, n);
#pragma omp parallel for
		for (int f = 0; f < nf; f++)
		{
			const Face *face = get<Face>(f, meshes);
			Mat3x2 F = derivative(face->v[0]->node->x, face->v[1]->node->x,
				face->v[2]->node->x, face);
			SVD<3, 2> svd = singular_value_decomposition(F);
			s[f * 2 + 0] = svd.s[0];
			s[f * 2 + 1] = svd.s[1];
			Mat3x2 Vt_ = Mat3x2(0);
			for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				Vt_(i, j) = svd.Vt(i, j);
			Mat3x2 Delta = Mat3x2(Vec3(-1, 1, 0), Vec3(-1, 0, 1));
			Mat3x3 &Left = svd.U;
			Mat3x3 Right = Delta*face->invDm*Vt_.t();
			sg[f * 2 + 0] = outer(Left.col(0), Right.col(0));
			sg[f * 2 + 1] = outer(Left.col(1), Right.col(1));
		}
	}

	double SLOpt::objective(const double *x) const
	{
		double f = 0;
#pragma omp parallel for reduction (+: f)
		for (int n = 0; n < nn; n++)
		{
			const Node *node = get<Node>(n, meshes);
			Vec3 dx = node->x - xold[n];
			f += inv_m*node->m*norm2(dx) / 2.;
		}
		return f;
	}

	void SLOpt::obj_grad(const double *x, double *grad) const
	{
#pragma omp parallel for
		for (int n = 0; n < nn; n++)
		{
			const Node *node = get<Node>(n, meshes);
			Vec3 dx = node->x - xold[n];
			set_subvec(grad, n, inv_m*node->m*dx);
		}
	}

	double strain_con(const SLOpt &sl, const double *x, int j, int &sign);
	void strain_con_grad(const SLOpt &sl, const double *x, int j, double factor,
		double *grad);

	double SLOpt::constraint(const double *x, int j, int &sign) const
	{
		if (j < cons.size())
			return cons[j]->value(&sign) - conold[j];
		else
			return strain_con(*this, x, j - cons.size(), sign);
	}

	void SLOpt::con_grad(const double *x, int j, double factor,
		double *grad) const
	{
		if (j < cons.size())
		{
			MeshGrad mgrad = cons[j]->gradient();
			for (MeshGrad::iterator it = mgrad.begin(); it != mgrad.end(); it++)
			{
				int n = get_index(it->first, meshes);
				if (n == -1)
					continue;
				const Vec3 &g = it->second;
				for (int i = 0; i < 3; i++)
					grad[n * 3 + i] += factor*g[i];
			}
		}
		else
			strain_con_grad(*this, x, j - cons.size(), factor, grad);
	}

	double strain_con(const SLOpt &sl, const double *x, int j, int &sign)
	{
		int f = j / 4;
		int a = j / 2; // index into s, sg
		const Face *face = get<Face>(f, sl.meshes);
		double strain_min = sl.strain_limits[f][0],
			strain_max = sl.strain_limits[f][1];
		double c;
		double w = sqrt(face->a);
		if (strain_min == strain_max)
		{
			sign = 0;
			c = (j % 2 == 0) ? w*(sl.s[a] - strain_min) : 0;
		}
		else
		{
			if (j % 2 == 0)
			{ // lower bound
				sign = 1;
				c = w*(sl.s[a] - strain_min);
			}
			else
			{ // upper bound
				sign = -1;
				c = w*(sl.s[a] - strain_max);
			}
		}
		return c;
	}

	void add_strain_row(const Mat3x3 &sg, const Face *face,
		const vector<Mesh*> &meshes, double factor, double *grad);

	void strain_con_grad(const SLOpt &sl, const double *x, int j, double factor,
		double *grad)
	{
		int f = j / 4;
		int a = j / 2; // index into s, sg
		const Face *face = get<Face>(f, sl.meshes);
		double strain_min = sl.strain_limits[f][0],
			strain_max = sl.strain_limits[f][1];
		double w = sqrt(face->a);
		if (strain_min == strain_max)
		{
			if (j % 2 == 0)
				add_strain_row(w*sl.sg[a], face, sl.meshes, factor, grad);
		}
		else
			add_strain_row(w*sl.sg[a], face, sl.meshes, factor, grad);
	}

	void add_strain_row(const Mat3x3 &sg, const Face *face,
		const vector<Mesh*> &meshes, double factor, double *grad)
	{
		for (int i = 0; i < 3; i++)
		{
			int n = get_index(face->v[i]->node, meshes);
			for (int j = 0; j < 3; j++)
				grad[n * 3 + j] += factor*sg(j, i);
		}
	}

	void SLOpt::finalize(const double *x) const
	{
		for (int n = 0; n < nn; n++)
			get<Node>(n, meshes)->x = get_subvec(x, n);
	}

	// DEBUG
	void debug(const vector<string> &args)
	{
		Mesh mesh;
		load_obj(mesh, "meshes/square35785.obj");
		Mat3x3 M = diag(Vec3(2., 0.5, 1.));
		for (int n = 0; n < mesh.nodes.size(); n++)
		{
			mesh.nodes[n]->x[0] *= 2;
			mesh.nodes[n]->x[1] *= 0.5;
			mesh.nodes[n]->m = mesh.nodes[n]->a;
		}
		vector<Mesh*> meshes(1, &mesh);
		vector<Vec2> strain_limits(mesh.faces.size(), Vec2(0.95, 1.05));
		Timer timer;
		strain_limiting(meshes, strain_limits, vector<Constraint*>());
		timer.tock();
		cout << "total time: " << timer.total << endl;
		save_obj(mesh, "tmp/debug.obj");
	}
}