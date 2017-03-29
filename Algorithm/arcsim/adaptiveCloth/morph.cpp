#include "morph.hpp"

#include "geometry.hpp"
using namespace std;

namespace arcsim
{

	Vec3 blend(const vector<Mesh> &targets, const vector<double> &w,
		const Vec2 &u)
	{
		Vec3 x = Vec3(0);
		for (int m = 0; m < targets.size(); m++)
		{
			if (w[m] == 0)
				continue;
			Face *face = get_enclosing_face(targets[m], u);
			if (!face)
				continue;
			Vec3 b = get_barycentric_coords(u, face);
			x += w[m] * (b[0] * face->v[0]->node->x
				+ b[1] * face->v[1]->node->x
				+ b[2] * face->v[2]->node->x);
		}
		return x;
	}

	Vec3 Morph::pos(double t, const Vec2 &u) const
	{
		return blend(targets, weights.pos(t), u);
	}

	void apply(const Morph &morph, double t)
	{
		for (int n = 0; n < morph.mesh->nodes.size(); n++)
		{
			Node *node = morph.mesh->nodes[n];
			Vec3 x = Vec3(0);
			for (int v = 0; v < node->verts.size(); v++)
				x += morph.pos(t, node->verts[v]->u);
			node->x = x / (double)node->verts.size();
		}
	}
}