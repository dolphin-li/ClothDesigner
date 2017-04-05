#pragma once

#include "ldpMat\ldp_basic_vec.h"
class ObjMesh;
namespace ldp
{
	// param for a single cloth piece
	struct PieceParam
	{
		float bending_k_mult = 0.f;			// multiplied on SimulationParam.bending_k for each piece
		float piece_outgo_dist = 0.f;		// we want some pieces to be outside of others, this hack may be helpful		
		PieceParam();
		void setDefaultParam();
	};

	struct SimulationParam
	{
		float rho = 0.f;						// for chebshev accereration
		float under_relax = 0.f;				// jacobi relax param
		int lap_damping = 0;					// loops of laplacian damping
		float air_damping = 0.f;				// damping of the air
		float bending_k = 0.f;					// related to the thickness of the cloth
		float spring_k = 0.f;					// related to the elasticity of the cloth
		float spring_k_raw = 0.f;				// spring_k_raw / avgArea = spring_k
		float stitch_k = 0.f;					// stiffness of stithed vertex, for sewing
		float stitch_k_raw = 0.f;				// stitch_k_raw / avgArea = stitch_k
		float stitch_bending_k = 0.f;			// bending of stitch sewings.
		float stitch_ratio = 0.f;				// for each stitch, the length will -= ratio*time_step each update
		int out_iter = 0;						// number of iterations
		int inner_iter = 0;						// number of iterations
		float control_mag = 0.f;				// for dragging, the stiffness of dragged point
		float time_step = 0.f;					// simulation time step
		bool enable_self_collistion = false;
		ldp::Float3 gravity;
		SimulationParam();
		void setDefaultParam();
	};

	struct ClothDesignParam
	{
		float pointMergeDistThre = 0.f;					// ignore two close points, in meters
		float curveSampleStep = 0.f;					// sample points on curves, in meters
		float curveSampleAngleThreCos = 0.f;			// if the cos(angle) of the points corner is too small, the point should be sampled.
		float pointInsidePolyThre = 0.f;				// in meters
		float curveFittingThre = 0.f;					// fitting inputs into cubics, in meters
		float triangulateThre = 0.f;					// size of triangle edges, in meters

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

	typedef int StitchPoint;
	struct StitchPointPair
	{
		StitchPoint first = 0;
		StitchPoint second = 0;
		size_t type = 0;
		StitchPointPair(StitchPoint a, StitchPoint b, size_t t) :first(a), second(b), type(t){}
		bool operator < (const StitchPointPair& r)
		{
			return first < r.first || (first == r.first && second < r.second);
		}
	};

	enum SimulationMode
	{
		SimulationNotInit,
		SimulationOn,
		SimulationPause,
	};

	enum BatchSimulateMode
	{
		BatchSimNotInit,
		BatchSimOn,
		BatchSimPause,
		BatchSimFinished,
	};
	// pts: a closed polygon of n points, the last connects to the first.
	// decide whether p is inside pts, thresholded by g_designParam.pointInsidePolyThre
	// if nearestEdgeId != nullptr, return the nearest edge index from it.
	bool pointInPolygon(int n, const Float2* pts, Float2 p, 
		int* nearestEdgeId=nullptr, float* minDist = nullptr);

	template<typename T, int N>
	inline T nearestPointOnSeg_getParam(ldp::ldp_basic_vec<T, N> p, 
		ldp::ldp_basic_vec<T, N> a, ldp::ldp_basic_vec<T, N> b)
	{
		T t = (p - a).dot(b-a) / (b-a).sqrLength();
		return std::min(T(1), std::max(T(0), t));
	}

	// find the nearest point on ab to p
	template<typename T, int N>
	inline ldp::ldp_basic_vec<T, N> nearestPointOnSeg(ldp::ldp_basic_vec<T, N> p,
		ldp::ldp_basic_vec<T, N> a, ldp::ldp_basic_vec<T, N> b)
	{
		return a + nearestPointOnSeg_getParam(p, a, b) * (b-a);
	}

	// calculate the distance of p to ab
	template<typename T, int N>
	inline T pointSegDistance(ldp::ldp_basic_vec<T, N> p, 
		ldp::ldp_basic_vec<T, N> a, ldp::ldp_basic_vec<T, N> b)
	{
		return (nearestPointOnSeg(p, a, b) - p).length();
	}
}