#pragma once

#include "ldpMat\ldp_basic_vec.h"
class ObjMesh;
namespace ldp
{
	struct SimulationParam
	{
		float rho;						// for chebshev accereration
		float under_relax;				// jacobi relax param
		int lap_damping;				// loops of laplacian damping
		float air_damping;				// damping of the air
		float bending_k;				// related to the thickness of the cloth
		float spring_k;					// related to the elasticity of the cloth
		float spring_k_raw;				// spring_k_raw / avgArea = spring_k
		float stitch_k;					// stiffness of stithed vertex, for sewing
		float stitch_k_raw;				// stitch_k_raw / avgArea = stitch_k
		float stitch_bending_k;			// bending of stitch sewings.
		float stitch_ratio;				// for each stitch, the length will -= ratio*time_step each update
		int out_iter;					// number of iterations
		int inner_iter;					// number of iterations
		float control_mag;				// for dragging, the stiffness of dragged point
		float time_step;				// simulation time step
		ldp::Float3 gravity;
		SimulationParam();
		void setDefaultParam();
	};

	struct ClothDesignParam
	{
		float pointMergeDistThre;				// ignore two close points, in meters
		float curveSampleStep;					// sample points on curves, in meters
		float curveSampleAngleThreCos;			// if the cos(angle) of the points corner is too small, the point should be sampled.
		float pointInsidePolyThre;				// in meters
		float curveFittingThre;					// fitting inputs into cubics, in meters
		float triangulateThre;					// size of triangle edges, in meters

		ClothDesignParam();
		void setDefaultParam();
	};

	extern ClothDesignParam g_designParam;

	struct DragInfo
	{
		ObjMesh* selected_cloth;
		int selected_vert_id;
		Float3 target;
		DragInfo()
		{
			selected_cloth = nullptr;
			selected_vert_id = -1;
			target = 0;
		}
	};

	struct StitchPoint
	{
		Int2 vids;
		float w;
	};
	typedef std::pair<StitchPoint, StitchPoint> StitchPointPair;

	enum SimulationMode
	{
		SimulationNotInit,
		SimulationOn,
		SimulationPause,
	};

	// pts: a closed polygon of n points, the last connects to the first.
	// decide whether p is inside pts, thresholded by g_designParam.pointInsidePolyThre
	// if nearestEdgeId != nullptr, return the nearest edge index from it.
	bool pointInPolygon(int n, const Float2* pts, Float2 p, 
		int* nearestEdgeId=nullptr, float* minDist = nullptr);

	inline float nearestPointOnSeg_getParam(Float2 p, Float2 a, Float2 b)
	{
		float t = (p - a).dot(b-a) / (b-a).sqrLength();
		return std::min(1.f, std::max(0.f, t));
	}

	// find the nearest point on ab to p
	inline ldp::Float2 nearestPointOnSeg(Float2 p, Float2 a, Float2 b)
	{
		return a + nearestPointOnSeg_getParam(p, a, b) * (b-a);
	}

	// calculate the distance of p to ab
	inline float pointSegDistance(Float2 p, Float2 a, Float2 b)
	{
		return (nearestPointOnSeg(p, a, b) - p).length();
	}
}