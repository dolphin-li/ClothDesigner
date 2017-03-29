#pragma once
#include "mesh.hpp"
namespace arcsim
{
	struct Morph
	{
		Mesh *mesh;
		std::vector<Mesh> targets;
		typedef std::vector<double> Weights;
		Spline<Weights> weights;
		Spline<double> log_stiffness;
		Vec3 pos(double t, const Vec2 &u) const;
	};

	void apply(const Morph &morph, double t);

}
