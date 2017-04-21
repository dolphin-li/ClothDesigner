///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2015, Huamin Wang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////////////////
//  LEVEL_SET_COLLISION
//**************************************************************************************
#ifndef __LEVEL_SET_COLLISION_H__
#define __LEVEL_SET_COLLISION_H__

#define INDEX(i,j,k) (size_yz*(int)(i)+size_z*(int)(j)+(int)(k))

#ifdef	USE_CUDA		//GPU version

///////////////////////////////////////////////////////////////////////////////////////////
// trilinear_Value
///////////////////////////////////////////////////////////////////////////////////////////
__device__ float trilinear_Value(const float* phi, const float x, const float y, const float z, const int size_yz, const int size_z)
{
	int  i=(int)(x), j=(int)(y), k=(int)(z);
	float a=x-i, b=y-j, c=z-k;
	return	(1-a)*(1-b)*(1-c)*phi[INDEX(i,j  ,k  )]+a*(1-b)*(1-c)*phi[INDEX(i+1,j  ,k  )]+
			(1-a)*(  b)*(1-c)*phi[INDEX(i,j+1,k  )]+a*(  b)*(1-c)*phi[INDEX(i+1,j+1,k  )]+
			(1-a)*(1-b)*(  c)*phi[INDEX(i,j  ,k+1)]+a*(1-b)*(  c)*phi[INDEX(i+1,j  ,k+1)]+
			(1-a)*(  b)*(  c)*phi[INDEX(i,j+1,k+1)]+a*(  b)*(  c)*phi[INDEX(i+1,j+1,k+1)];
}

#ifdef __CUDACC__
__device__ __forceinline__ float trilinear_Value(cudaTextureObject_t phi, float x, float y, float z)
{
	return tex3D<float>(phi, x, y, z);
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// Gradient
///////////////////////////////////////////////////////////////////////////////////////////
__device__ void Gradient(const float* phi, const float x, const float y, const float z, float &gx, float &gy, float &gz, const int size_yz, const int size_z)
{
	gx=(trilinear_Value(phi, x+1, y, z, size_yz, size_z)-trilinear_Value(phi, x-1, y, z, size_yz, size_z))*0.5;
	gy=(trilinear_Value(phi, x, y+1, z, size_yz, size_z)-trilinear_Value(phi, x, y-1, z, size_yz, size_z))*0.5;
	gz=(trilinear_Value(phi, x, y, z+1, size_yz, size_z)-trilinear_Value(phi, x, y, z-1, size_yz, size_z))*0.5;
}

#ifdef __CUDACC__
__device__ __forceinline__ void Gradient(cudaTextureObject_t phi, float x, float y, float z, float &gx, float &gy, float &gz)
{
	gx = (trilinear_Value(phi, x + 1, y, z) - trilinear_Value(phi, x - 1, y, z))*0.5;
	gy = (trilinear_Value(phi, x, y + 1, z) - trilinear_Value(phi, x, y - 1, z))*0.5;
	gz = (trilinear_Value(phi, x, y, z + 1) - trilinear_Value(phi, x, y, z - 1))*0.5;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// Level_Set_Projection
///////////////////////////////////////////////////////////////////////////////////////////
__device__ float Level_Set_Projection(const float* phi, float &x, float &y, float &z, const float goal, const int size_x, const int size_y, const int size_z, const int size_yz)
{	
	float vp;
	float gx, gy, gz;
	float ret=9999999;
	for(int loop=0; loop<4; loop++)
	{
		//Out of range
		if(x<2 || y<2 || z<2 || x>size_x-3 || y>size_y-3 || z>size_z-3)	return ret;
		
		vp=trilinear_Value(phi, x, y, z, size_yz, size_z);
		if(loop==0)				ret=vp-goal;
		if(vp>goal && loop==0)	return ret;
		if(fabs(vp-goal)<0.001) return ret;		
			
		Gradient(phi, x, y, z, gx, gy, gz, size_yz, size_z);
		float factor=CLAMP(vp-goal, -1, 1)/sqrtf(gx*gx+gy*gy+gz*gz);
		x=x-gx*factor;
		y=y-gy*factor;
		z=z-gz*factor;
	}
	return ret;
}

#ifdef __CUDACC__
__device__ float Level_Set_Projection(cudaTextureObject_t phi, float &x, float &y, float &z, float goal)
{	
	float vp = 0.f;
	float gx = 0.f, gy = 0.f, gz = 0.f;
	float ret = 9999999;
	for (int loop = 0; loop<4; loop++)
	{
		vp = trilinear_Value(phi, x, y, z);
		if (loop == 0)				ret = vp - goal;
		if (vp>goal && loop == 0)	return ret;
		if (fabs(vp - goal)<0.001) return ret;

		Gradient(phi, x, y, z, gx, gy, gz);
		float factor = CLAMP(vp - goal, -1.f, 1.f) / sqrtf(gx*gx + gy*gy + gz*gz);
		x = x - gx*factor;
		y = y - gy*factor;
		z = z - gz*factor;
	}
	return ret;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// Level_Set_Depth
///////////////////////////////////////////////////////////////////////////////////////////
__device__ float Level_Set_Depth(const float* phi, float x, float y, float z, const float goal, const int size_x, const int size_y, const int size_z, const int size_yz)
{
	if(x<2 || y<2 || z<2 || x>size_x-3 || y>size_y-3 || z>size_z-3)	return 9999999;
	return trilinear_Value(phi, x, y, z, size_yz, size_z)-goal;
}

__device__ void Level_Set_Gradient(const float* phi, float x, float y, float z, float &gx, float &gy, float &gz, const int size_x, const int size_y, const int size_z, const int size_yz)
{
	gx=0;
	gy=0;
	gz=0;
	if(x<2 || y<2 || z<2 || x>size_x-3 || y>size_y-3 || z>size_z-3)	return;
	Gradient(phi, x, y, z, gx, gy, gz, size_yz, size_z);
}

#ifdef __CUDACC__
__device__ __forceinline__ float Level_Set_Depth(cudaTextureObject_t phi, float x, float y, float z, float goal)
{
	return trilinear_Value(phi, x, y, z)-goal;
}

__device__ __forceinline__ void Level_Set_Gradient(cudaTextureObject_t phi, float x, float y, float z, float &gx, float &gy, float &gz)
{
	gx = gy = gz = 0;
	Gradient(phi, x, y, z, gx, gy, gz);
}
#endif

#else					//CPU version

///////////////////////////////////////////////////////////////////////////////////////////
// trilinear_Value
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
TYPE trilinear_Value(const TYPE* phi, const TYPE x, const TYPE y, const TYPE z, const int size_yz, const int size_z)
{
	int  i=(int)(x), j=(int)(y), k=(int)(z);
	TYPE a=x-i, b=y-j, c=z-k;
	return	(1-a)*(1-b)*(1-c)*phi[INDEX(i,j  ,k  )]+a*(1-b)*(1-c)*phi[INDEX(i+1,j  ,k  )]+
			(1-a)*(  b)*(1-c)*phi[INDEX(i,j+1,k  )]+a*(  b)*(1-c)*phi[INDEX(i+1,j+1,k  )]+
			(1-a)*(1-b)*(  c)*phi[INDEX(i,j  ,k+1)]+a*(1-b)*(  c)*phi[INDEX(i+1,j  ,k+1)]+
			(1-a)*(  b)*(  c)*phi[INDEX(i,j+1,k+1)]+a*(  b)*(  c)*phi[INDEX(i+1,j+1,k+1)];
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Gradient
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
void Gradient(const TYPE* phi, const TYPE x, const TYPE y, const TYPE z, TYPE &gx, TYPE &gy, TYPE &gz, const int size_yz, const int size_z)
{
	gx=(trilinear_Value(phi, x+1, y, z, size_yz, size_z)-trilinear_Value(phi, x-1, y, z, size_yz, size_z))*0.5;
	gy=(trilinear_Value(phi, x, y+1, z, size_yz, size_z)-trilinear_Value(phi, x, y-1, z, size_yz, size_z))*0.5;
	gz=(trilinear_Value(phi, x, y, z+1, size_yz, size_z)-trilinear_Value(phi, x, y, z-1, size_yz, size_z))*0.5;
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Level_Set_Projection
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
TYPE Level_Set_Projection(TYPE* phi, TYPE &x, TYPE &y, TYPE &z, const TYPE goal, const int size_x, const int size_y, const int size_z, const int size_yz)
{	
	TYPE vp;
	TYPE gx, gy, gz;
	TYPE ret=9999999;
	for(int loop=0; loop<4; loop++)
	{
		//Out of range
		if(x<2 || y<2 || z<2 || x>size_x-3 || y>size_y-3 || z>size_z-3)	return ret;
		
		vp=trilinear_Value(phi, x, y, z, size_yz, size_z);
		if(loop==0)				ret=vp-goal;
		if(vp>goal && loop==0)	return ret;
		if(fabs(vp-goal)<0.001) return ret;
			
		Gradient(phi, x, y, z, gx, gy, gz, size_yz, size_z);
		TYPE factor=CLAMP(vp-goal, -1, 1)/sqrtf(gx*gx+gy*gy+gz*gz);
		x=x-gx*factor;
		y=y-gy*factor;
		z=z-gz*factor;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Level set depth value
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
TYPE Level_Set_Depth(const TYPE* phi, TYPE &x, TYPE &y, TYPE &z, const TYPE goal, const int size_x, const int size_y, const int size_z, const int size_yz)
{
	float ret=9999999;
	if(x<2 || y<2 || z<2 || x>size_x-3 || y>size_y-3 || z>size_z-3)	return 9999999;
	return trilinear_Value(phi, x, y, z, size_yz, size_z)-goal;
}

template <class TYPE>
void Level_Set_Gradient(const TYPE* phi, TYPE x, TYPE y, TYPE z, TYPE &gx, TYPE &gy, TYPE &gz, const int size_x, const int size_y, const int size_z, const int size_yz)
{
	gx=0;
	gy=0;
	gz=0;
	if(x<2 || y<2 || z<2 || x>size_x-3 || y>size_y-3 || z>size_z-3)	return;
	Gradient(phi, x, y, z, gx, gy, gz, size_yz, size_z);
}

#endif

#endif