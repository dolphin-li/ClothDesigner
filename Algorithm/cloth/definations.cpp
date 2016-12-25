#include "definations.h"

namespace ldp
{
	ClothDesignParam g_designParam;
	ClothDesignParam::ClothDesignParam()
	{
		setDefaultParam();
	}

	void ClothDesignParam::setDefaultParam()
	{
		pointMergeDistThre = 1e-4;									// in meters
		curveSampleStep = 1e-2;										// in meters
		curveSampleAngleThreCos = cos(15.f * ldp::PI_S / 180.f);	// 15 degree
		pointInsidePolyThre = 1e-2;									// in meters
		curveFittingThre = 2e-3;									// in meters
		triangulateThre = 2e-2;										// in meters
	}

	PieceParam::PieceParam()
	{
		setDefaultParam();
	}

	void PieceParam::setDefaultParam()
	{
		bending_k_mult = 1;
	}

	SimulationParam::SimulationParam()
	{
		setDefaultParam();
	}

	void SimulationParam::setDefaultParam()
	{
		rho = 0.996;
		under_relax = 0.5;
		lap_damping = 4;
		air_damping = 0.999;
		bending_k = 10;
		spring_k_raw = 1000;
		spring_k = 0;//will be updated after built topology
		stitch_k_raw = 90000;
		stitch_k = 0;//will be updated after built topology
		stitch_bending_k = 10;
		out_iter = 8;
		inner_iter = 40;
		time_step = 1.0 / 240.0;
		stitch_ratio = 5;
		control_mag = 400;
		gravity = ldp::Float3(0, 0, -9.8);
	}

	bool pointInPolygon(int n, const Float2* v, Float2 p, int* nearestEdgeId, float* minDistPtr)
	{
		float d = -1;
		const float x = p[0], y = p[1];
		float minDist = FLT_MAX;
		if (nearestEdgeId)
			*nearestEdgeId = -1;
		for (int i = 0, j = n - 1; i < n; j = i++)
		{
			const float xi = v[i][0], yi = v[i][1], xj = v[j][0], yj = v[j][1];
			const float inv_k = (xj - xi) / (yj - yi);
			if ((yi>y) != (yj > y) && x - xi < inv_k * (y - yi))
				d = -d;
			float dist = pointSegDistance(p, v[i], v[j]);
			if (dist < minDist)
			{
				minDist = dist;
				if (nearestEdgeId)
					*nearestEdgeId = i;
			}
		}
		if (minDistPtr)
			*minDistPtr = minDist;
		minDist *= d;
		return minDist >= -g_designParam.pointInsidePolyThre;
	}
}