#pragma once

#include "ldpMat\ldp_basic_vec.h"
class ObjMesh;
namespace ldp
{
	// param for a single cloth piece
	struct PieceParam
	{
		float bending_k_mult = 0.f;			// multiplied on SimulationParam.bending_k for each piece
		float spring_k_mult = 0.f;
		std::string material_name;
		static std::string default_material_folder;
		PieceParam();
		void setDefaultParam();
	};

	struct SimulationParam
	{
		float bending_k = 0.f;					// related to the thickness of the cloth
		float spring_k = 0.f;					// related to the elasticity of the cloth
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