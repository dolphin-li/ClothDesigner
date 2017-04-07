#ifndef SRBF_H_
#define SRBF_H_ 1

struct srbf{
#ifdef IN_BSGP
	float2	_center;																			//lattitude/longtitude coordinate of the SRBF centers
	float3	_weight;																			//srbf weight
#else
	float	_center[2];
	float	_weight[3];
#endif //IN_BSGP
	float	_lambda;																			//bandwidth coefficient		
};

#endif //SRBF_H_