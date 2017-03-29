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

#include "proximity.hpp"

#include "collisionutil.hpp"
#include "geometry.hpp"
#include "magic.hpp"
#include "simulation.hpp"
#include <vector>
using namespace std;

namespace arcsim
{

	template <typename T> struct Min
	{
		double key;
		T val;
		Min() : key(infinity), val() {}
		void add(double key, T val)
		{
			if (key < this->key)
			{
				this->key = key;
				this->val = val;
			}
		}
	};

	static vector< Min<Face*> > node_prox[2];
	static vector< Min<Edge*> > edge_prox[2];
	static vector< Min<Node*> > face_prox[2];

	void find_proximities(const Face *face0, const Face *face1);
	Constraint *make_constraint(const Node *node, const Face *face,
		double mu, double mu_obs);
	Constraint *make_constraint(const Edge *edge0, const Edge *edge1,
		double mu, double mu_obs);

	vector<Constraint*> proximity_constraints(const vector<Mesh*> &meshes,
		const vector<Mesh*> &obs_meshes,
		double mu, double mu_obs)
	{
		arcsim::meshes = &meshes;
		const double dmin = 2 * arcsim::magic.repulsion_thickness;
		vector<AccelStruct*> accs = create_accel_structs(meshes, false),
			obs_accs = create_accel_structs(obs_meshes, false);
		int nn = size<Node>(meshes),
			ne = size<Edge>(meshes),
			nf = size<Face>(meshes);
		for (int i = 0; i < 2; i++)
		{
			arcsim::node_prox[i].assign(nn, Min<Face*>());
			arcsim::edge_prox[i].assign(ne, Min<Edge*>());
			arcsim::face_prox[i].assign(nf, Min<Node*>());
		}
		for_overlapping_faces(accs, obs_accs, dmin, find_proximities);
		vector<Constraint*> cons;
		for (int n = 0; n < nn; n++)
		for (int i = 0; i < 2; i++)
		{
			Min<Face*> &m = arcsim::node_prox[i][n];
			if (m.key < dmin)
				cons.push_back(make_constraint(get<Node>(n, meshes), m.val,
				mu, mu_obs));
		}
		for (int e = 0; e < ne; e++)
		for (int i = 0; i < 2; i++)
		{
			Min<Edge*> &m = arcsim::edge_prox[i][e];
			if (m.key < dmin)
				cons.push_back(make_constraint(get<Edge>(e, meshes), m.val,
				mu, mu_obs));
		}
		for (int f = 0; f < nf; f++)
		for (int i = 0; i < 2; i++)
		{
			Min<Node*> &m = arcsim::face_prox[i][f];
			if (m.key < dmin)
				cons.push_back(make_constraint(m.val, get<Face>(f, meshes),
				mu, mu_obs));
		}
		destroy_accel_structs(accs);
		destroy_accel_structs(obs_accs);
		return cons;
	}

	void add_proximity(const Node *node, const Face *face);
	void add_proximity(const Edge *edge0, const Edge *edge1);

	void find_proximities(const Face *face0, const Face *face1)
	{
		for (int v = 0; v < 3; v++)
			add_proximity(face0->v[v]->node, face1);
		for (int v = 0; v < 3; v++)
			add_proximity(face1->v[v]->node, face0);
		for (int e0 = 0; e0 < 3; e0++)
		for (int e1 = 0; e1 < 3; e1++)
			add_proximity(face0->adje[e0], face1->adje[e1]);
	}

	void add_proximity(const Node *node, const Face *face)
	{
		if (node == face->v[0]->node
			|| node == face->v[1]->node
			|| node == face->v[2]->node)
			return;
		Vec3 n;
		double w[4];
		double d = signed_vf_distance(node->x, face->v[0]->node->x,
			face->v[1]->node->x, face->v[2]->node->x,
			&n, w);
		d = abs(d);
		bool inside = (min(-w[1], -w[2], -w[3]) >= -1e-6);
		if (!inside)
			return;
		if (is_free(node))
		{
			int side = dot(n, node->n) >= 0 ? 0 : 1;
			arcsim::node_prox[side][get_index(node, *arcsim::meshes)].add(d, (Face*)face);
		}
		if (is_free(face))
		{
			int side = dot(-n, face->n) >= 0 ? 0 : 1;
			arcsim::face_prox[side][get_index(face, *arcsim::meshes)].add(d, (Node*)node);
		}
	}

	bool in_wedge(double w, const Edge *edge0, const Edge *edge1)
	{
		Vec3 x = (1 - w)*edge0->n[0]->x + w*edge0->n[1]->x;
		bool in = true;
		for (int s = 0; s < 2; s++)
		{
			const Face *face = edge1->adjf[s];
			if (!face)
				continue;
			const Node *node0 = edge1->n[s], *node1 = edge1->n[1 - s];
			Vec3 e = node1->x - node0->x, n = face->n, r = x - node0->x;
			in &= (stp(e, n, r) >= 0);
		}
		return in;
	}

	void add_proximity(const Edge *edge0, const Edge *edge1)
	{
		if (edge0->n[0] == edge1->n[0] || edge0->n[0] == edge1->n[1]
			|| edge0->n[1] == edge1->n[0] || edge0->n[1] == edge1->n[1])
			return;
		Vec3 n;
		double w[4];
		double d = signed_ee_distance(edge0->n[0]->x, edge0->n[1]->x,
			edge1->n[0]->x, edge1->n[1]->x,
			&n, w);
		d = abs(d);
		bool inside = (min(w[0], w[1], -w[2], -w[3]) >= -1e-6
			&& in_wedge(w[1], edge0, edge1)
			&& in_wedge(-w[3], edge1, edge0));
		if (!inside)
			return;
		if (is_free(edge0))
		{
			Vec3 edge0n = edge0->n[0]->n + edge0->n[1]->n;
			int side = dot(n, edge0n) >= 0 ? 0 : 1;
			arcsim::edge_prox[side][get_index(edge0, *arcsim::meshes)].add(d, (Edge*)edge1);
		}
		if (is_free(edge1))
		{
			Vec3 edge1n = edge1->n[0]->n + edge1->n[1]->n;
			int side = dot(-n, edge1n) >= 0 ? 0 : 1;
			arcsim::edge_prox[side][get_index(edge1, *arcsim::meshes)].add(d, (Edge*)edge0);
		}
	}

	double area(const Node *node);
	double area(const Edge *edge);
	double area(const Face *face);

	Constraint *make_constraint(const Node *node, const Face *face,
		double mu, double mu_obs)
	{
		IneqCon *con = new IneqCon;
		con->nodes[0] = (Node*)node;
		con->nodes[1] = (Node*)face->v[0]->node;
		con->nodes[2] = (Node*)face->v[1]->node;
		con->nodes[3] = (Node*)face->v[2]->node;
		for (int n = 0; n < 4; n++)
			con->free[n] = is_free(con->nodes[n]);
		double a = std::min(area(node), area(face));
		con->stiff = arcsim::magic.collision_stiffness*a;
		double d = signed_vf_distance(con->nodes[0]->x, con->nodes[1]->x,
			con->nodes[2]->x, con->nodes[3]->x,
			&con->n, con->w);
		if (d < 0)
			con->n = -con->n;
		con->mu = (!is_free(node) || !is_free(face)) ? mu_obs : mu;
		return con;
	}

	Constraint *make_constraint(const Edge *edge0, const Edge *edge1,
		double mu, double mu_obs)
	{
		IneqCon *con = new IneqCon;
		con->nodes[0] = (Node*)edge0->n[0];
		con->nodes[1] = (Node*)edge0->n[1];
		con->nodes[2] = (Node*)edge1->n[0];
		con->nodes[3] = (Node*)edge1->n[1];
		for (int n = 0; n < 4; n++)
			con->free[n] = is_free(con->nodes[n]);
		double a = std::min(area(edge0), area(edge1));
		con->stiff = arcsim::magic.collision_stiffness*a;
		double d = signed_ee_distance(con->nodes[0]->x, con->nodes[1]->x,
			con->nodes[2]->x, con->nodes[3]->x,
			&con->n, con->w);
		if (d < 0)
			con->n = -con->n;
		con->mu = (!is_free(edge0) || !is_free(edge1)) ? mu_obs : mu;
		return con;
	}

	double area(const Node *node)
	{
		if (is_free(node))
			return node->a;
		double a = 0;
		for (int v = 0; v < node->verts.size(); v++)
		for (int f = 0; f < node->verts[v]->adjf.size(); f++)
			a += area(node->verts[v]->adjf[f]) / 3;
		return a;
	}

	double area(const Edge *edge)
	{
		double a = 0;
		if (edge->adjf[0])
			a += area(edge->adjf[0]) / 3;
		if (edge->adjf[1])
			a += area(edge->adjf[1]) / 3;
		return a;
	}

	double area(const Face *face)
	{
		if (is_free(face))
			return face->a;
		const Vec3 &x0 = face->v[0]->node->x, &x1 = face->v[1]->node->x,
			&x2 = face->v[2]->node->x;
		return norm(cross(x1 - x0, x2 - x0)) / 2;
	}
}