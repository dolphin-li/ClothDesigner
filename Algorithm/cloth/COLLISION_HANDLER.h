///////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2014, Huamin Wang
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
//  Class COLLISION_HANDLER
//	dev stands for device
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_COLLISION_HANDLER_H__
#define __WHMIN_COLLISION_HANDLER_H__
#define	INDEX(i,j,k)	((i)*(size_yz)+(j)*(size_z)+(k))

#define	INT_X(x)		((int)(((x)-start_x)*inv_h))
#define	INT_Y(y)		((int)(((y)-start_y)*inv_h))
#define	INT_Z(z)		((int)(((z)-start_z)*inv_h))

//#define USE_CUDA

#ifdef USE_CUDA

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>


///////////////////////////////////////////////////////////////////////////////////////////
//  Vertex Grid kernel
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void Grid_0_Kernel(float* X, int* vertex_id, int* vertex_bucket, int number, float start_x, float start_y, float start_z, float inv_h, int size_yz, int size_z)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=number)	return;

	vertex_id		[i]=i;
	vertex_bucket	[i]=INDEX(INT_X(X[i*3+0]), INT_Y(X[i*3+1]), INT_Z(X[i*3+2]));
}

__global__ void Grid_1_Kernel(int* vertex_bucket, int* bucket_ranges, int number)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=number)	return;

	if(i==0			|| vertex_bucket[i]!=vertex_bucket[i-1])	bucket_ranges[vertex_bucket[i]*2+0]=i;		//begin at i
	if(i==number-1	|| vertex_bucket[i]!=vertex_bucket[i+1])	bucket_ranges[vertex_bucket[i]*2+1]=i+1;	//  end at i+1
}

///////////////////////////////////////////////////////////////////////////////////////////
//  Triangle Test kernel
///////////////////////////////////////////////////////////////////////////////////////////
__device__ int cudaMin(const int a, const int b, const int c)
{
	int     r=a;	if(b<r) r=b;	if(c<r) r=c; 	return r;
}

__device__ int cudaMax(const int a, const int b, const int c)	
{
	int     r=a;	if(b>r) r=b;	if(c>r) r=c; 	return r;
}

__device__ float cudaMin(const float a, const float b, const float c)
{
	float   r=a;	if(b<r) r=b;	if(c<r) r=c; 	return r;
}

__device__ float cudaMax(const float a, const float b, const float c)	
{
	float   r=a;	if(b>r) r=b;	if(c>r) r=c; 	return r;
}

///////////////////////////////////////////////////////////////////////////////////////////
////  Squared vertex-edge distance
////	r is the barycentric weight of the closest point: 1-r, r.
///////////////////////////////////////////////////////////////////////////////////////////
__device__ float cuda_Squared_VE_Distance(float xi[], float xa[], float xb[], float &r, float *N=0)
{
	float xia[3], xba[3];
	for(int n=0; n<3; n++)
	{
		xia[n]=xi[n]-xa[n];
		xba[n]=xb[n]-xa[n];
	}
	float xia_xba=DOT(xia, xba);
	float xba_xba=DOT(xba, xba);
	if(xia_xba<0)				r=0;
	else if(xia_xba>xba_xba)	r=1;
	else						r=xia_xba/xba_xba;
	float _N[3];
	if(N==0)	N=_N;
	N[0]=xi[0]-xa[0]*(1-r)-xb[0]*r;
	N[1]=xi[1]-xa[1]*(1-r)-xb[1]*r;
	N[2]=xi[2]-xa[2]*(1-r)-xb[2]*r;
	return DOT(N, N);
}

///////////////////////////////////////////////////////////////////////////////////////////
////  Squared vertex-triangle distance
////	bb and bc are the barycentric weights of the closest point: 1-bb-bc, bb, bc.
///////////////////////////////////////////////////////////////////////////////////////////
__device__ float cuda_Squared_VT_Distance(float gap, float xi[], float xa[], float xb[], float xc[], float &ba, float &bb, float &bc, float *N=0)
{
	float  xba[3],  xca[3],  xia[3];
	for(int n=0; n<3; n++)
	{
		xba[n]=xb[n]-xa[n];
		xca[n]=xc[n]-xa[n];
		xia[n]=xi[n]-xa[n];
	}

	float _N[3];
	if(N==0)	N=_N;
	CROSS(xba, xca, N);
	float nn=DOT(N, N);

	float temp[3];
	CROSS(xia, xca, temp);
	float weight_iaca=DOT(N, temp);
	CROSS(xba, xia, temp);
	float weight_baia=DOT(N, temp);

	if(nn>1e-16f && weight_iaca>=0 && weight_baia>=0 && nn-weight_iaca-weight_baia>=0)
	{
		bb=weight_iaca/nn;
		bc=weight_baia/nn;
		ba=1-bb-bc;
	}
	else
	{
		return 999999;
		/*float min_distance=MY_INFINITE;
		float r, distance;
		if(nn-weight_iaca-weight_baia<0 && ((distance=cuda_Squared_VE_Distance(xi, xb, xc, r))<min_distance))
		{
			min_distance=distance;
			bb=1-r;
			bc=r;
			ba=0;
		}
		if(weight_iaca<0 && ((distance=cuda_Squared_VE_Distance(xi, xa, xc, r))<min_distance))		
		{
			min_distance=distance;
			bb=0;
			bc=r;
			ba=1-bb-bc;
		}			
		if(weight_baia<0 && ((distance=cuda_Squared_VE_Distance(xi, xa, xb, r))<min_distance))
		{
			min_distance=distance;
			bb=r;
			bc=0;
			ba=1-bb-bc;
		}*/
	}

	N[0]=xi[0]-xa[0]*ba-xb[0]*bb-xc[0]*bc;
	N[1]=xi[1]-xa[1]*ba-xb[1]*bb-xc[1]*bc;
	N[2]=xi[2]-xa[2]*ba-xb[2]*bb-xc[2]*bc;
	return DOT(N, N);
}


__global__ void Triangle_Test_Kernel0(float* old_X, float* new_X, int number, int* T, int t_number, float* I, float* R, float* W, int* c_status, int* bucket_ranges, int*vertex_id, int l, float gap,
	float start_x, float start_y, float start_z, float inv_h, int size_yz, int size_x, int size_y, int size_z, int* t_key, int* t_idx)
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;
	if (t >= t_number)	return;

	int va = T[t * 3 + 0];
	int vb = T[t * 3 + 1];
	int vc = T[t * 3 + 2];
	int va_status = c_status[INDEX(INT_X(old_X[va * 3 + 0]), INT_Y(old_X[va * 3 + 1]), INT_Z(old_X[va * 3 + 2]))];
	int vb_status = c_status[INDEX(INT_X(old_X[vb * 3 + 0]), INT_Y(old_X[vb * 3 + 1]), INT_Z(old_X[vb * 3 + 2]))];
	int vc_status = c_status[INDEX(INT_X(old_X[vc * 3 + 0]), INT_Y(old_X[vc * 3 + 1]), INT_Z(old_X[vc * 3 + 2]))];

	int min_i = cudaMin(MIN(INT_X(old_X[va * 3 + 0]), INT_X(new_X[va * 3 + 0])), MIN(INT_X(old_X[vb * 3 + 0]), INT_X(new_X[vb * 3 + 0])), MIN(INT_X(old_X[vc * 3 + 0]), INT_X(new_X[vc * 3 + 0]))) - 1;
	int min_j = cudaMin(MIN(INT_Y(old_X[va * 3 + 1]), INT_Y(new_X[va * 3 + 1])), MIN(INT_Y(old_X[vb * 3 + 1]), INT_Y(new_X[vb * 3 + 1])), MIN(INT_Y(old_X[vc * 3 + 1]), INT_Y(new_X[vc * 3 + 1]))) - 1;
	int min_k = cudaMin(MIN(INT_Z(old_X[va * 3 + 2]), INT_Z(new_X[va * 3 + 2])), MIN(INT_Z(old_X[vb * 3 + 2]), INT_Z(new_X[vb * 3 + 2])), MIN(INT_Z(old_X[vc * 3 + 2]), INT_Z(new_X[vc * 3 + 2]))) - 1;
	int max_i = cudaMax(MAX(INT_X(old_X[va * 3 + 0]), INT_X(new_X[va * 3 + 0])), MAX(INT_X(old_X[vb * 3 + 0]), INT_X(new_X[vb * 3 + 0])), MAX(INT_X(old_X[vc * 3 + 0]), INT_X(new_X[vc * 3 + 0]))) + 1;
	int max_j = cudaMax(MAX(INT_Y(old_X[va * 3 + 1]), INT_Y(new_X[va * 3 + 1])), MAX(INT_Y(old_X[vb * 3 + 1]), INT_Y(new_X[vb * 3 + 1])), MAX(INT_Y(old_X[vc * 3 + 1]), INT_Y(new_X[vc * 3 + 1]))) + 1;
	int max_k = cudaMax(MAX(INT_Z(old_X[va * 3 + 2]), INT_Z(new_X[va * 3 + 2])), MAX(INT_Z(old_X[vb * 3 + 2]), INT_Z(new_X[vb * 3 + 2])), MAX(INT_Z(old_X[vc * 3 + 2]), INT_Z(new_X[vc * 3 + 2]))) + 1;

	min_i = MAX(MIN(min_i, size_x - 1), 0);
	min_j = MAX(MIN(min_j, size_y - 1), 0);
	min_k = MAX(MIN(min_k, size_z - 1), 0);
	max_i = MAX(MIN(max_i, size_x - 1), 0);
	max_j = MAX(MIN(max_j, size_y - 1), 0);
	max_k = MAX(MIN(max_k, size_z - 1), 0);

	t_idx[t]=t;
	t_key[t]=1;
	for (int i = min_i; i <= max_i; i++)
	for (int j = min_j; j <= max_j; j++)
	for (int k = min_k; k <= max_k; k++)
	{
		int vid = INDEX(i, j, k);

		//if(t==33200)	printf("value: %d, %d, %d, %d\n", c_status[vid], va_status, vb_status, vc_status);

		if (c_status[vid]<l && va_status<l && vb_status<l && vc_status<l)				continue;
		t_key[t]=0;
	}
}


__global__ void Triangle_Test_Kernel(float* old_X, float* new_X, int number, int* T, int t_number, float* I, float* R, float* W, int* c_status, int* bucket_ranges, int*vertex_id, int l, float gap,
	float start_x, float start_y, float start_z, float inv_h, int size_yz, int size_x, int size_y, int size_z, int* t_key, int* t_idx)
{
	int t = blockDim.x * blockIdx.x + threadIdx.x;
	if(t>=t_number)	return;

	//Convert
	if(t_key && t_idx)	t=t_idx[t];

	int va=T[t*3+0];
	int vb=T[t*3+1];
	int vc=T[t*3+2];
	int va_status=c_status[INDEX(INT_X(old_X[va*3+0]), INT_Y(old_X[va*3+1]), INT_Z(old_X[va*3+2]))];
	int vb_status=c_status[INDEX(INT_X(old_X[vb*3+0]), INT_Y(old_X[vb*3+1]), INT_Z(old_X[vb*3+2]))];
	int vc_status=c_status[INDEX(INT_X(old_X[vc*3+0]), INT_Y(old_X[vc*3+1]), INT_Z(old_X[vc*3+2]))];

	int min_i=cudaMin(MIN(INT_X(old_X[va*3+0]), INT_X(new_X[va*3+0])), MIN(INT_X(old_X[vb*3+0]), INT_X(new_X[vb*3+0])), MIN(INT_X(old_X[vc*3+0]), INT_X(new_X[vc*3+0])))-1;
	int min_j=cudaMin(MIN(INT_Y(old_X[va*3+1]), INT_Y(new_X[va*3+1])), MIN(INT_Y(old_X[vb*3+1]), INT_Y(new_X[vb*3+1])), MIN(INT_Y(old_X[vc*3+1]), INT_Y(new_X[vc*3+1])))-1;
	int min_k=cudaMin(MIN(INT_Z(old_X[va*3+2]), INT_Z(new_X[va*3+2])), MIN(INT_Z(old_X[vb*3+2]), INT_Z(new_X[vb*3+2])), MIN(INT_Z(old_X[vc*3+2]), INT_Z(new_X[vc*3+2])))-1;
	int max_i=cudaMax(MAX(INT_X(old_X[va*3+0]), INT_X(new_X[va*3+0])), MAX(INT_X(old_X[vb*3+0]), INT_X(new_X[vb*3+0])), MAX(INT_X(old_X[vc*3+0]), INT_X(new_X[vc*3+0])))+1;
	int max_j=cudaMax(MAX(INT_Y(old_X[va*3+1]), INT_Y(new_X[va*3+1])), MAX(INT_Y(old_X[vb*3+1]), INT_Y(new_X[vb*3+1])), MAX(INT_Y(old_X[vc*3+1]), INT_Y(new_X[vc*3+1])))+1;
	int max_k=cudaMax(MAX(INT_Z(old_X[va*3+2]), INT_Z(new_X[va*3+2])), MAX(INT_Z(old_X[vb*3+2]), INT_Z(new_X[vb*3+2])), MAX(INT_Z(old_X[vc*3+2]), INT_Z(new_X[vc*3+2])))+1;

	min_i=MAX(MIN(min_i, size_x-1), 0);
	min_j=MAX(MIN(min_j, size_y-1), 0);
	min_k=MAX(MIN(min_k, size_z-1), 0);
	max_i=MAX(MIN(max_i, size_x-1), 0);
	max_j=MAX(MIN(max_j, size_y-1), 0);
	max_k=MAX(MIN(max_k, size_z-1), 0);

	//if(t==10263)
	//{	
	//	printf("start: %f, %f, %f; %d, %d, %d\n", start_x, start_y, start_z, size_x, size_y, size_z);
	//	printf("new Xa: %d, %d, %d\n", INT_X(new_X[va*3+0]), INT_Y(new_X[va*3+1]), INT_Z(new_X[va*3+2]));
	//	printf("new Xb: %d, %d, %d\n", INT_X(new_X[vb*3+0]), INT_Y(new_X[vb*3+1]), INT_Z(new_X[vb*3+2]));
	//	printf("new Xc: %f, %f, %f\n", new_X[vc*3+0], new_X[vc*3+1], new_X[vc*3+2]);
	//	printf("vi: (%d, %d, %d) (%d, %d, %d)\n", min_i, min_j, min_k, max_i, max_j, max_k);
	//}


	for(int i=min_i; i<=max_i; i++)
	for(int j=min_j; j<=max_j; j++)
	for(int k=min_k; k<=max_k; k++)
	{
		//if(t==10263 && i==48 && j==35 && k==19)		printf("yes ok\n");

		int vid=INDEX(i, j, k);

		//if(t==33200)	printf("new value: %d, %d, %d, %d\n", c_status[vid], va_status, vb_status, vc_status);


		if(c_status[vid]<l && va_status<l && vb_status<l && vc_status<l)				continue;

		//if(t_key[t]!=0)	printf("what the fuck %d (%d) (%d)\n", t_key[t], l, t);

		for(int ptr=bucket_ranges[vid*2+0]; ptr<bucket_ranges[vid*2+1]; ptr++)
		{
			int vi=vertex_id[ptr];
			//if(t==10263 && vid==134230)	printf("ptr %d: %d to %d (%d)\n", vid, bucket_ranges[vid*2+0], bucket_ranges[vid*2+1], vertex_id[ptr]);
		
			if(vi==va || vi==vb || vi==vc)												continue;
			if(va==vb || vb==vc || vc==va)												continue;

			float ba, bb, bc, N[3];
			float d;

		//	int c, C=1;
		//	for(c=1; c<=C; c++)
		//	{
		//		float rate=((float)c)/C;

				float *x0_1=&new_X[vi*3];
				float *x1_1=&new_X[va*3];
				float *x2_1=&new_X[vb*3];
				float *x3_1=&new_X[vc*3];

			//	float x0_1[3];
			//	x0_1[0]=old_X[vi*3+0]*(1-rate)+new_X[vi*3+0]*rate;
			//	x0_1[1]=old_X[vi*3+1]*(1-rate)+new_X[vi*3+1]*rate;
			//	x0_1[2]=old_X[vi*3+2]*(1-rate)+new_X[vi*3+2]*rate;
			//	float x1_1[3];
			//	x1_1[0]=old_X[va*3+0]*(1-rate)+new_X[va*3+0]*rate;
			//	x1_1[1]=old_X[va*3+1]*(1-rate)+new_X[va*3+1]*rate;
			//	x1_1[2]=old_X[va*3+2]*(1-rate)+new_X[va*3+2]*rate;
			//	float x2_1[3];
			//	x2_1[0]=old_X[vb*3+0]*(1-rate)+new_X[vb*3+0]*rate;
			//	x2_1[1]=old_X[vb*3+1]*(1-rate)+new_X[vb*3+1]*rate;
			//	x2_1[2]=old_X[vb*3+2]*(1-rate)+new_X[vb*3+2]*rate;
			//	float x3_1[3];
			//	x3_1[0]=old_X[vc*3+0]*(1-rate)+new_X[vc*3+0]*rate;
			//	x3_1[1]=old_X[vc*3+1]*(1-rate)+new_X[vc*3+1]*rate;
			//	x3_1[2]=old_X[vc*3+2]*(1-rate)+new_X[vc*3+2]*rate;

				if(x0_1[0]+gap*2<cudaMin(x1_1[0], x2_1[0], x3_1[0]))									continue;
				if(x0_1[1]+gap*2<cudaMin(x1_1[1], x2_1[1], x3_1[1]))									continue;
				if(x0_1[2]+gap*2<cudaMin(x1_1[2], x2_1[2], x3_1[2]))									continue;
				if(x0_1[0]-gap*2>cudaMax(x1_1[0], x2_1[0], x3_1[0]))									continue;
				if(x0_1[1]-gap*2>cudaMax(x1_1[1], x2_1[1], x3_1[1]))									continue;
				if(x0_1[2]-gap*2>cudaMax(x1_1[2], x2_1[2], x3_1[2]))									continue;
				if(x0_1[0]+x0_1[1]+gap*3<cudaMin(x1_1[0]+x1_1[1], x2_1[0]+x2_1[1], x3_1[0]+x3_1[1]))	continue;
				if(x0_1[1]+x0_1[2]+gap*3<cudaMin(x1_1[1]+x1_1[2], x2_1[1]+x2_1[2], x3_1[1]+x3_1[2]))	continue;
				if(x0_1[2]+x0_1[0]+gap*3<cudaMin(x1_1[2]+x1_1[0], x2_1[2]+x2_1[0], x3_1[2]+x3_1[0]))	continue;
				if(x0_1[0]-x0_1[1]+gap*3<cudaMin(x1_1[0]-x1_1[1], x2_1[0]-x2_1[1], x3_1[0]-x3_1[1]))	continue;
				if(x0_1[1]-x0_1[2]+gap*3<cudaMin(x1_1[1]-x1_1[2], x2_1[1]-x2_1[2], x3_1[1]-x3_1[2]))	continue;
				if(x0_1[2]-x0_1[0]+gap*3<cudaMin(x1_1[2]-x1_1[0], x2_1[2]-x2_1[0], x3_1[2]-x3_1[0]))	continue;
				if(x0_1[0]+x0_1[1]-gap*3>cudaMax(x1_1[0]+x1_1[1], x2_1[0]+x2_1[1], x3_1[0]+x3_1[1]))	continue;
				if(x0_1[1]+x0_1[2]-gap*3>cudaMax(x1_1[1]+x1_1[2], x2_1[1]+x2_1[2], x3_1[1]+x3_1[2]))	continue;
				if(x0_1[2]+x0_1[0]-gap*3>cudaMax(x1_1[2]+x1_1[0], x2_1[2]+x2_1[0], x3_1[2]+x3_1[0]))	continue;
				if(x0_1[0]-x0_1[1]-gap*3>cudaMax(x1_1[0]-x1_1[1], x2_1[0]-x2_1[1], x3_1[0]-x3_1[1]))	continue;
				if(x0_1[1]-x0_1[2]-gap*3>cudaMax(x1_1[1]-x1_1[2], x2_1[1]-x2_1[2], x3_1[1]-x3_1[2]))	continue;
				if(x0_1[2]-x0_1[0]-gap*3>cudaMax(x1_1[2]-x1_1[0], x2_1[2]-x2_1[0], x3_1[2]-x3_1[0]))	continue;


				d=cuda_Squared_VT_Distance(gap, x0_1, x1_1, x2_1, x3_1, ba, bb, bc, N);
				if(d>gap*gap*4)	continue;		//proximity found!
		//	}


			//Take normal into consideration
			d=sqrt(d);
			float old_N[3];
			old_N[0]=old_X[vi*3+0]-ba*old_X[va*3+0]-bb*old_X[vb*3+0]-bc*old_X[vc*3+0];
			old_N[1]=old_X[vi*3+1]-ba*old_X[va*3+1]-bb*old_X[vb*3+1]-bc*old_X[vc*3+1];
			old_N[2]=old_X[vi*3+2]-ba*old_X[va*3+2]-bb*old_X[vb*3+2]-bc*old_X[vc*3+2];
			if(DOT(old_N, N)<0)	d=-d;

		
			if(d<gap*2 && l==0)	//apply repulsion force
			{
				float k=(gap*2-d)*5/d;
				for(int n=0; n<3; n++)
				{
					atomicAdd(&R[vi*3+n],  k*N[n]   );
					atomicAdd(&R[va*3+n], -k*N[n]*ba);
					atomicAdd(&R[vb*3+n], -k*N[n]*bb);
					atomicAdd(&R[vc*3+n], -k*N[n]*bc);
				}
				atomicAdd(&W[vi], 1);
				atomicAdd(&W[va], 1);
				atomicAdd(&W[vb], 1);
				atomicAdd(&W[vc], 1);
			}


			if(d>gap)	continue;

			//if(c>C)	continue;
			//printf("C: %d\n", c);
			//if(ba==0 || bb==0 || bc==0)	continue;

			float k=(gap*1.1-d)/(d*(1+ba*ba+bb*bb+bc*bc));

			for(int n=0; n<3; n++)
			{
				atomicAdd(&I[vi*3+n],  k*N[n]);
				atomicAdd(&I[va*3+n], -k*N[n]*ba);
				atomicAdd(&I[vb*3+n], -k*N[n]*bb);
				atomicAdd(&I[vc*3+n], -k*N[n]*bc);
			}

			c_status[vid]=l+1;
		/*	has_changed=true;*/

		}
	}
}

__global__ void Triangle_Test_2_Kernel(float* new_X, float* V, float* I, float* R, float* W, int number, int l, float inv_t)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=number)	return;

	
	if(l==0 && W[i]!=0)
	{
		V[i*3+0]+=R[i*3+0]*inv_t/W[i];
		V[i*3+1]+=R[i*3+1]*inv_t/W[i];
		V[i*3+2]+=R[i*3+2]*inv_t/W[i];
	}

	new_X[i*3+0]+=I[i*3+0]*0.4;
	new_X[i*3+1]+=I[i*3+1]*0.4;
	new_X[i*3+2]+=I[i*3+2]*0.4; 
}



class COLLISION_HANDLER
{
public:
	float*	dev_I;	//impulse
	float*	dev_R;	//repulsion
	float*	dev_W;
	int*	dev_vertex_id;
	int*	dev_vertex_bucket;
	int*	dev_bucket_ranges;
	int		dev_max_bucket_size;
	int*	dev_c_status;
	int*	dev_t_key;
	int*	dev_t_idx;

	int		max_bucket_size;

	COLLISION_HANDLER()
	{
		dev_I				= 0;
		dev_R				= 0;
		dev_W				= 0;
		dev_vertex_id		= 0;
		dev_vertex_bucket	= 0;
		dev_bucket_ranges	= 0;
		dev_c_status		= 0;
		dev_t_key			= 0;
		dev_t_idx			= 0;
	}

	void Initialize(int number, int t_number)
	{
		if(dev_I)				cudaFree(dev_I);
		if(dev_R)				cudaFree(dev_R);
		if(dev_W)				cudaFree(dev_W);
		if(dev_vertex_id)		cudaFree(dev_vertex_id);
		if(dev_vertex_bucket)	cudaFree(dev_vertex_bucket);
		if(dev_bucket_ranges)	cudaFree(dev_bucket_ranges);
		if(dev_c_status)		cudaFree(dev_c_status);
		if(dev_t_key)			cudaFree(dev_t_key);
		if(dev_t_idx)			cudaFree(dev_t_idx);
		
		max_bucket_size	= 41943040; //ldp: +0

		cudaMalloc((void**)&dev_I,				sizeof(float)*number*3);
		cudaMalloc((void**)&dev_R,				sizeof(float)*number*3);
		cudaMalloc((void**)&dev_W,				sizeof(float)*number  );
		cudaMalloc((void**)&dev_vertex_id,		sizeof(int  )*number  );
		cudaMalloc((void**)&dev_vertex_bucket,	sizeof(int  )*number  );
		cudaMalloc((void**)&dev_bucket_ranges,	sizeof(int  )*max_bucket_size);
		cudaMalloc((void**)&dev_c_status,		sizeof(int  )*max_bucket_size/2);
		cudaMalloc((void**)&dev_t_key,			sizeof(int  )*t_number);
		cudaMalloc((void**)&dev_t_idx,			sizeof(int  )*t_number);
	}

	~COLLISION_HANDLER()
	{
		if(dev_I)				cudaFree(dev_I);
		if(dev_R)				cudaFree(dev_R);
		if(dev_W)				cudaFree(dev_W);
		if(dev_vertex_id)		cudaFree(dev_vertex_id);
		if(dev_vertex_bucket)	cudaFree(dev_vertex_bucket);
		if(dev_bucket_ranges)	cudaFree(dev_bucket_ranges);
		if(dev_c_status)		cudaFree(dev_c_status);
		if(dev_t_key)			cudaFree(dev_t_key);
		if(dev_t_idx)			cudaFree(dev_t_idx);
	}


	void Run(float* dev_old_X, float* dev_new_X, float* dev_V, int number, int* dev_T, int t_number, float* X, float inv_t)	//X is in CPU
	{
		//TIMER timer;
		int threadsPerBlock = 256;
		int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;
		int t_threadsPerBlock = 256;
		int t_blocksPerGrid = (t_number + t_threadsPerBlock - 1) / t_threadsPerBlock;

		if(!dev_c_status)	Initialize(number, t_number);
		
		float	gap=0.003; //0.0015

		///////////////////////////////////////////////////////////////////////////
		//	Step 1: Initialization
		///////////////////////////////////////////////////////////////////////////
		float	h		= 0.006; //twice the max vertex velocity
		float	inv_h	= 1.0/h;

		float	min_x= MY_INFINITE, min_y= MY_INFINITE, min_z= MY_INFINITE;
		float	max_x=-MY_INFINITE, max_y=-MY_INFINITE, max_z=-MY_INFINITE;
		for(int i=0, i3=0; i<number; i++, i3+=3)
		{
			min_x=Min(min_x, X[i3+0]);
			max_x=Max(max_x, X[i3+0]);
			min_y=Min(min_y, X[i3+1]);
			max_y=Max(max_y, X[i3+1]);
			min_z=Min(min_z, X[i3+2]);
			max_z=Max(max_z, X[i3+2]);
		}


		// Initialize the culling grid sizes
		float	start_x	= min_x-1.5*h;
		float	start_y	= min_y-1.5*h;
		float	start_z	= min_z-1.5*h;
		int		size_x	= (int)((max_x-start_x)*inv_h+2);
		int		size_y	= (int)((max_y-start_y)*inv_h+2);
		int		size_z	= (int)((max_z-start_z)*inv_h+2);
		int		size_yz	= size_y*size_z;
		int		size	= size_x*size_y*size_z;

		//printf("size: %d; %d\n", size, max_bucket_size);

		///////////////////////////////////////////////////////////////////////////
		//	Step 2: Vertex Grid Construction (for VT tests)
		///////////////////////////////////////////////////////////////////////////
		

		if(size*2>max_bucket_size)	{printf("ERROR: out of buffer length in COLLISION_HANDLER %d, %d\n", size*2, max_bucket_size); getchar();}

		//assign vertex_id and vertex_bucket (let us use the new X for grid test)

		Grid_0_Kernel << <blocksPerGrid, threadsPerBlock>> >(dev_old_X, dev_vertex_id, dev_vertex_bucket, number, start_x, start_y, start_z, inv_h, size_yz, size_z);

		thrust::device_ptr<int> dev_data_ptr(dev_vertex_id);		
		thrust::device_ptr<int> dev_keys_ptr(dev_vertex_bucket);

		thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr+number, dev_data_ptr);

		int* sorted_dev_vertex_bucket	= thrust::raw_pointer_cast(dev_keys_ptr);
		int* sorted_dev_vertex_id		= thrust::raw_pointer_cast(dev_data_ptr);

		cudaMemset(dev_bucket_ranges, 0, sizeof(int)*size*2);		
		Grid_1_Kernel << <blocksPerGrid, threadsPerBlock>> >(dev_vertex_bucket, dev_bucket_ranges, number);
		

		cudaMemset(dev_c_status, 0, sizeof(int)*size);
		//cudaMemset(dev_t_key, 0, sizeof(int)*t_number);		
		//printf("time a: %f\n", timer.Get_Time());
		int l;
		for(l=0; l<64; l++)
		{
			cudaMemset(dev_I, 0, sizeof(float)*number*3);
			cudaMemset(dev_R, 0, sizeof(float)*number*3);
			cudaMemset(dev_W, 0, sizeof(float)*number  );
				
		/*	cudaMemset(dev_t_key, 0, sizeof(int)*t_number);
			if(l!=0)
			{
				Triangle_Test_Kernel0 << <t_blocksPerGrid, t_threadsPerBlock>> >(dev_old_X, dev_new_X, number, dev_T, t_number, dev_I, dev_R, dev_W, dev_c_status, dev_bucket_ranges, sorted_dev_vertex_id, l, gap,
					start_x, start_y, start_z, inv_h, size_yz, size_x, size_y, size_z, dev_t_key, dev_t_idx);

				thrust::device_ptr<int> dev_t_key_ptr(dev_t_key);		
				thrust::device_ptr<int> dev_t_idx_ptr(dev_t_idx);		
				thrust::sort_by_key(dev_t_key_ptr, dev_t_key_ptr+t_number, dev_t_idx_ptr);

				int sum=t_number-thrust::reduce(dev_t_key_ptr, dev_t_key_ptr+t_number);
				
				int t_threadsPerBlock = 256;
				int t_blocksPerGrid = (sum + t_threadsPerBlock - 1) / t_threadsPerBlock;
							

				Triangle_Test_Kernel << <t_blocksPerGrid, t_threadsPerBlock>> >(dev_old_X, dev_new_X, number, dev_T, sum, dev_I, dev_R, dev_W, dev_c_status, dev_bucket_ranges, sorted_dev_vertex_id, l, gap,
					start_x, start_y, start_z, inv_h, size_yz, size_x, size_y, size_z, dev_t_key, dev_t_idx);

			//	int t_key[65536];
			//	int t_idx[65536];
			//	cudaMemcpy(t_key, dev_t_key, sizeof(int)*t_number, cudaMemcpyDeviceToHost);
			//	cudaMemcpy(t_idx, dev_t_idx, sizeof(int)*t_number, cudaMemcpyDeviceToHost);
			//	for(int i=0; i<t_number; i++)
			//		printf("t_key %d: %d (%d)\n", i, t_key[i], t_idx[i]);
			//	getchar();

				//thrust::device_ptr<int> res = thrust::max_element(dev_c_status_ptr, dev_c_status_ptr+size);
			}
			
			if(l==0)*/
				Triangle_Test_Kernel << <t_blocksPerGrid, t_threadsPerBlock>> >(dev_old_X, dev_new_X, number, dev_T, t_number, dev_I, dev_R, dev_W, dev_c_status, dev_bucket_ranges, sorted_dev_vertex_id, l, gap,
					start_x, start_y, start_z, inv_h, size_yz, size_x, size_y, size_z, 0, 0);

			thrust::device_ptr<int> dev_c_status_ptr(dev_c_status);
			thrust::device_ptr<int> res = thrust::max_element(dev_c_status_ptr, dev_c_status_ptr+size);
			//cudaMemcpy(c_status, dev_c_status, sizeof(int)*size, cudaMemcpyDeviceToHost);
			//int max_c=0;
			//for(int i=0; i<size; i++)	max_c=MAX(max_c, c_status[i]);
			//printf("what: %d, %d\n", max_c, l);
			if(res[0]!=l+1)	break;

			Triangle_Test_2_Kernel << <blocksPerGrid, threadsPerBlock>> >(dev_new_X, dev_V, dev_I, dev_R, dev_W, number, l, inv_t);
		/*	printf("\n");						
			// Finally, apply impulse onto the vertices
			for(int v=0; v<number; v++)
			{
				//TYPE w=1.0f/weights[v];                
				new_X[v*3+0]+=I[v*3+0]*0.4;
				new_X[v*3+1]+=I[v*3+1]*0.4;
				new_X[v*3+2]+=I[v*3+2]*0.4;				
			}
			//printf("sum: %f\n", sum);
			printf("It %2d: %4d/%6d\n", l,  vt_test_counter, vt_collision_counter);
            if(l%2==1 || has_changed==false)  printf("\n");
			if(has_changed==false)	break;
			*/

		}

		if(l==64)	{printf("ERROR: collision still not converge.\n");}
		//printf("l %d\t", l);

		//printf("time b: %f\n", timer.Get_Time());
	}


};

#else

#include "DISTANCE.h"
template <class TYPE>
class COLLISION_HANDLER
{
public:
	//fast converter
	TYPE*	X;
	TYPE*	I;

	int*	vertex_id;
	int*	vertex_bucket;
	int*	bucket_ranges;
	int		max_bucket_size;

	int*	c_status;	//whether a vertex in the grid cell gets modified

	COLLISION_HANDLER()
	{
		X				= 0;
		I				= 0;
		vertex_id		= 0;
		vertex_bucket	= 0;
		bucket_ranges	= 0;
		c_status		= 0;
	}

	void Initialize(int number, int t_number)
	{
		if(X)				delete[] X;
		if(I)				delete[] I;
		if(vertex_id)		delete[] vertex_id;
		if(vertex_bucket)	delete[] vertex_bucket;
		if(bucket_ranges)	delete[] bucket_ranges;
		if(c_status)		delete[] c_status;
		
		max_bucket_size	= 4194304;
		X				= new TYPE[number*30];
		I				= new TYPE[number*3];
		vertex_id		= new int [number  ];
		vertex_bucket	= new int [number  ];
		bucket_ranges	= new int [max_bucket_size];
		c_status		= new int [max_bucket_size/2];
	}

	~COLLISION_HANDLER()
	{
		if(X)				delete[] X;
		if(I)				delete[] I;
		if(vertex_id)		delete[] vertex_id;
		if(vertex_bucket)	delete[] vertex_bucket;
		if(bucket_ranges)	delete[] bucket_ranges;
		if(c_status)		delete[] c_status;
	}

///////////////////////////////////////////////////////////////////////////////////////////
//  Call Run to process collisions.
/////////////////////////////////////////////////////////////////////////////////////////// 	
	void Run(TYPE* old_X, TYPE* new_X, int number, int* T, int t_number)
	{
		if(!c_status)	Initialize(number, t_number);
		
		TYPE	gap=0.0015; //0.0015

		///////////////////////////////////////////////////////////////////////////
		//	Step 1: Initialization
		///////////////////////////////////////////////////////////////////////////
		TYPE	h		= 0.01; //twice the max vertex velocity
		TYPE	inv_h	= 1.0/h;
		// Determine the mesh range
		TYPE	min_x= MY_INFINITE, min_y= MY_INFINITE, min_z= MY_INFINITE;
		TYPE	max_x=-MY_INFINITE, max_y=-MY_INFINITE, max_z=-MY_INFINITE;
		for(int i=0, i3=0; i<number; i++, i3+=3)
		{
			min_x=Min(min_x, old_X[i3+0]);
			max_x=Max(max_x, old_X[i3+0]);
			min_y=Min(min_y, old_X[i3+1]);
			max_y=Max(max_y, old_X[i3+1]);
			min_z=Min(min_z, old_X[i3+2]);
			max_z=Max(max_z, old_X[i3+2]);
		}
		// Initialize the culling grid sizes
		TYPE	start_x	= min_x-1.5*h;
		TYPE	start_y	= min_y-1.5*h;
		TYPE	start_z	= min_z-1.5*h;
		int		size_x	= (int)((max_x-start_x)*inv_h+2);
		int		size_y	= (int)((max_y-start_y)*inv_h+2);
		int		size_z	= (int)((max_z-start_z)*inv_h+2);
		int		size_yz	= size_y*size_z;
		int		size	= size_x*size_y*size_z;

		///////////////////////////////////////////////////////////////////////////
		//	Step 2: Vertex Grid Construction (for VT tests)
		///////////////////////////////////////////////////////////////////////////
		if(size*2>max_bucket_size)	{printf("ERROR: out of buffer length in COLLISION_HANDLER %d, %d\n", size*2, max_bucket_size); getchar();}

		//assign vertex_id and vertex_bucket (let us use the new X for grid test)
		for(int i=0; i<number; i++)
		{
			vertex_id		[i]=i;
			vertex_bucket	[i]=INDEX(INT_X(old_X[i*3+0]), INT_Y(old_X[i*3+1]), INT_Z(old_X[i*3+2]));
		}
		//sort vertex_id and vertex_bucket
		Quick_Sort_V(vertex_bucket, vertex_id, 0, number-1);
		//build bucket_ranges
		memset(bucket_ranges, 0, sizeof(int)*size*2);
		for(int i=0; i<number; i++)
		{
			if(i==0			|| vertex_bucket[i]!=vertex_bucket[i-1])	bucket_ranges[vertex_bucket[i]*2+0]=i;		//begin at i
			if(i==number-1	|| vertex_bucket[i]!=vertex_bucket[i+1])	bucket_ranges[vertex_bucket[i]*2+1]=i+1;	//  end at i+1
		}


		//Create X array
		/*for(int i=0; i<number; i++) for(int n=0; n<10; n++)
		{
			TYPE rate=(n+1)*0.1;
			X[(i*10+n)*3+0]=old_X[i*3+0]*(1-rate)+new_X[i*3+0]*rate;
			X[(i*10+n)*3+1]=old_X[i*3+1]*(1-rate)+new_X[i*3+1]*rate;
			X[(i*10+n)*3+2]=old_X[i*3+2]*(1-rate)+new_X[i*3+2]*rate;
		}*/
			

		///////////////////////////////////////////////////////////////////////////
		//	Step 3: collision test starts now
		///////////////////////////////////////////////////////////////////////////
		memset(c_status, 0, sizeof(int)*size);
		int l;
		int counter;
		TIMER timer, full_timer;
		for(l=0; l<64; l++)
		{
			counter=0;
			bool has_changed			= false;
			int  vt_test_counter		= 0;
			int  vt_collision_counter	= 0;

			memset(I, 0, sizeof(TYPE)*number*3);

			// Part 2: Vertex-triangle collision
			for(int t=0; t<t_number; t++)
			{
				//Calculate vertex bounding box and find a triangle t
				int va=T[t*3+0];
				int vb=T[t*3+1];
				int vc=T[t*3+2];
				int va_status=c_status[INDEX(INT_X(old_X[va*3+0]), INT_Y(old_X[va*3+1]), INT_Z(old_X[va*3+2]))];
				int vb_status=c_status[INDEX(INT_X(old_X[vb*3+0]), INT_Y(old_X[vb*3+1]), INT_Z(old_X[vb*3+2]))];
				int vc_status=c_status[INDEX(INT_X(old_X[vc*3+0]), INT_Y(old_X[vc*3+1]), INT_Z(old_X[vc*3+2]))];

				int min_i=Min(MIN(INT_X(old_X[va*3+0]), INT_X(new_X[va*3+0])), MIN(INT_X(old_X[vb*3+0]), INT_X(new_X[vb*3+0])), MIN(INT_X(old_X[vc*3+0]), INT_X(new_X[vc*3+0])))-1;
				int min_j=Min(MIN(INT_Y(old_X[va*3+1]), INT_Y(new_X[va*3+1])), MIN(INT_Y(old_X[vb*3+1]), INT_Y(new_X[vb*3+1])), MIN(INT_Y(old_X[vc*3+1]), INT_Y(new_X[vc*3+1])))-1;
				int min_k=Min(MIN(INT_Z(old_X[va*3+2]), INT_Z(new_X[va*3+2])), MIN(INT_Z(old_X[vb*3+2]), INT_Z(new_X[vb*3+2])), MIN(INT_Z(old_X[vc*3+2]), INT_Z(new_X[vc*3+2])))-1;
				int max_i=Max(MAX(INT_X(old_X[va*3+0]), INT_X(new_X[va*3+0])), MAX(INT_X(old_X[vb*3+0]), INT_X(new_X[vb*3+0])), MAX(INT_X(old_X[vc*3+0]), INT_X(new_X[vc*3+0])))+1;
				int max_j=Max(MAX(INT_Y(old_X[va*3+1]), INT_Y(new_X[va*3+1])), MAX(INT_Y(old_X[vb*3+1]), INT_Y(new_X[vb*3+1])), MAX(INT_Y(old_X[vc*3+1]), INT_Y(new_X[vc*3+1])))+1;
				int max_k=Max(MAX(INT_Z(old_X[va*3+2]), INT_Z(new_X[va*3+2])), MAX(INT_Z(old_X[vb*3+2]), INT_Z(new_X[vb*3+2])), MAX(INT_Z(old_X[vc*3+2]), INT_Z(new_X[vc*3+2])))+1;

				min_i=MAX(MIN(min_i, size_x-1), 0);
				min_j=MAX(MIN(min_j, size_y-1), 0);
				min_k=MAX(MIN(min_k, size_z-1), 0);
				max_i=MAX(MIN(max_i, size_x-1), 0);
				max_j=MAX(MIN(max_j, size_y-1), 0);
				max_k=MAX(MIN(max_k, size_z-1), 0);
				
				
				/*if(t==10263)
				{
					printf("start: %f, %f, %f; %d, %d, %d\n", start_x, start_y, start_z, size_x, size_y, size_z);
					printf("Xa: %f, %f, %f\n", old_X[va*3+0], old_X[va*3+1], old_X[va*3+2]);
					printf("Xb: %f, %f, %f\n", old_X[vb*3+0], old_X[vb*3+1], old_X[vb*3+2]);
					printf("Xc: %f, %f, %f\n", old_X[vc*3+0], old_X[vc*3+1], old_X[vc*3+2]);
					printf("Xa: %f, %f, %f\n", new_X[va*3+0], new_X[va*3+1], new_X[va*3+2]);
					printf("Xb: %f, %f, %f\n", new_X[vb*3+0], new_X[vb*3+1], new_X[vb*3+2]);
					printf("Xc: %f, %f, %f\n", new_X[vc*3+0], new_X[vc*3+1], new_X[vc*3+2]);

					
					printf("new Xb: %d, %d, %d\n", INT_X(new_X[vb*3+0]), INT_Y(new_X[vb*3+1]), INT_Z(new_X[vb*3+2]));

					printf("vi: (%d, %d, %d) (%d, %d, %d)\n", min_i, min_j, min_k, max_i, max_j, max_k);
					getchar();
				}*/

				for(int i=min_i; i<=max_i; i++)
				for(int j=min_j; j<=max_j; j++)
				for(int k=min_k; k<=max_k; k++)
				{
					//if(t==10263 && i==48 && j==35 && k==19)		printf("yes ok\n");

					int vid=INDEX(i, j, k);
					if(c_status[vid]<l && va_status<l && vb_status<l && vc_status<l)				continue;

					


					//if(bucket_ranges[vid*2+0]==-1)	continue;
					for(int ptr=bucket_ranges[vid*2+0]; ptr<bucket_ranges[vid*2+1]; ptr++)
					{				
						int vi=vertex_id[ptr];
						//if(t==10263)	printf("ptr %d (%d, %d, %d): %d to %d (%d)\n", vid, i, j, k, bucket_ranges[vid*2+0], bucket_ranges[vid*2+1], vertex_id[ptr]);
						
						if(vi==va || vi==vb || vi==vc)												continue;
						if(va==vb || vb==vc || vc==va)												continue;

						// Perform planar culling (both axial and non-axial)
						TYPE *x0_1=&new_X[vi*3];
						TYPE *x1_1=&new_X[va*3];
						TYPE *x2_1=&new_X[vb*3];
						TYPE *x3_1=&new_X[vc*3];
						if(x0_1[0]+gap<Min(x1_1[0], x2_1[0], x3_1[0]))	continue;
						if(x0_1[1]+gap<Min(x1_1[1], x2_1[1], x3_1[1]))	continue;
						if(x0_1[2]+gap<Min(x1_1[2], x2_1[2], x3_1[2]))	continue;
						if(x0_1[0]-gap>Max(x1_1[0], x2_1[0], x3_1[0]))	continue;
						if(x0_1[1]-gap>Max(x1_1[1], x2_1[1], x3_1[1]))	continue;
						if(x0_1[2]-gap>Max(x1_1[2], x2_1[2], x3_1[2]))	continue;
						if(x0_1[0]+x0_1[1]+gap*1.5<Min(x1_1[0]+x1_1[1], x2_1[0]+x2_1[1], x3_1[0]+x3_1[1]))	continue;
						if(x0_1[1]+x0_1[2]+gap*1.5<Min(x1_1[1]+x1_1[2], x2_1[1]+x2_1[2], x3_1[1]+x3_1[2]))	continue;
						if(x0_1[2]+x0_1[0]+gap*1.5<Min(x1_1[2]+x1_1[0], x2_1[2]+x2_1[0], x3_1[2]+x3_1[0]))	continue;
						if(x0_1[0]-x0_1[1]+gap*1.5<Min(x1_1[0]-x1_1[1], x2_1[0]-x2_1[1], x3_1[0]-x3_1[1]))	continue;
						if(x0_1[1]-x0_1[2]+gap*1.5<Min(x1_1[1]-x1_1[2], x2_1[1]-x2_1[2], x3_1[1]-x3_1[2]))	continue;
						if(x0_1[2]-x0_1[0]+gap*1.5<Min(x1_1[2]-x1_1[0], x2_1[2]-x2_1[0], x3_1[2]-x3_1[0]))	continue;
						if(x0_1[0]+x0_1[1]-gap*1.5>Max(x1_1[0]+x1_1[1], x2_1[0]+x2_1[1], x3_1[0]+x3_1[1]))	continue;
						if(x0_1[1]+x0_1[2]-gap*1.5>Max(x1_1[1]+x1_1[2], x2_1[1]+x2_1[2], x3_1[1]+x3_1[2]))	continue;
						if(x0_1[2]+x0_1[0]-gap*1.5>Max(x1_1[2]+x1_1[0], x2_1[2]+x2_1[0], x3_1[2]+x3_1[0]))	continue;
						if(x0_1[0]-x0_1[1]-gap*1.5>Max(x1_1[0]-x1_1[1], x2_1[0]-x2_1[1], x3_1[0]-x3_1[1]))	continue;
						if(x0_1[1]-x0_1[2]-gap*1.5>Max(x1_1[1]-x1_1[2], x2_1[1]-x2_1[2], x3_1[1]-x3_1[2]))	continue;
						if(x0_1[2]-x0_1[0]-gap*1.5>Max(x1_1[2]-x1_1[0], x2_1[2]-x2_1[0], x3_1[2]-x3_1[0]))	continue;
						

						// Perform actual detection
						vt_test_counter++;

						TYPE ba, bb, bc, N[3];
						TYPE d=Squared_VT_Distance(x0_1, x1_1, x2_1, x3_1, ba, bb, bc, N);
						//printf("d is: %f\n", );
						if(d>gap*gap)					continue;

						//printf("here: %d, %d\n", vi, t);

						//if(ba==0 || bb==0 || bc==0)		continue;

						d=sqrt(d);
						TYPE k=(gap*1.1-d)/(d*(1+ba*ba+bb*bb+bc*bc));

						for(int n=0; n<3; n++)
						{
							I[vi*3+n]+=k*N[n];
							I[va*3+n]-=k*N[n]*ba;
							I[vb*3+n]-=k*N[n]*bb;
							I[vc*3+n]-=k*N[n]*bc;
						}						

						//test again
						//TYPE x0[3], x1[3], x2[3], x3[3];
						//for(int n=0; n<3; n++)
						//{
						//	x0[n]=new_X[vi*3+n]+I[vi*3+n];
						//	x1[n]=new_X[va*3+n]+I[va*3+n];
						//	x2[n]=new_X[vb*3+n]+I[vb*3+n];
						//	x3[n]=new_X[vc*3+n]+I[vc*3+n];
						//}


						//if(vi==12791)
						{
						//	printf("hit %d to %d (%d, %d, %d) %f, %f, %f, %d\n", vi, t, va, vb, vc, bb, bc, ba, ba==0);
						//	printf("N: %f, %f, %f\n", N[0], N[1], N[2]);
						//	printf("dist: %f\n", sqrtf(Squared_VT_Distance(x0_0, x1_0, x2_0, x3_0, bb, bc, N)));
						//	printf("N: %f, %f, %f\n", N[0], N[1], N[2]);
						}


						//The cell has been changed
						c_status[vid]=l+1;

						has_changed=true;
						vt_collision_counter++;
					}
				}
			}
			
			// Finally, apply impulse onto the vertices
			for(int v=0; v<number; v++)
			{
				//TYPE w=1.0f/weights[v];                
				new_X[v*3+0]+=I[v*3+0]*0.4;
				new_X[v*3+1]+=I[v*3+1]*0.4;
				new_X[v*3+2]+=I[v*3+2]*0.4;				
			}

			//printf("sum: %f\n", sum);
			printf("It %2d: %4d/%6d\n", l,  vt_test_counter, vt_collision_counter);
            if(l%2==1 || has_changed==false)  printf("\n");
			if(has_changed==false)	break;

			if(l==0)	timer.Start();
			//getchar();
			//if(l==1)	getchar();
		}

		printf("good timer: %f (%f)\n", timer.Get_Time(), full_timer.Get_Time());
		if(l==64)	{printf("ERROR: collision still not converge.\n");getchar();}
	}
	
	void Quick_Sort_V(int a[], int b[], int l, int r)
	{		
		if(l<r)
		{
			int j=Quick_Sort_Partition_V(a, b, l, r);	
			Quick_Sort_V(a, b, l, j-1);
			Quick_Sort_V(a, b, j+1, r);
		}
	}
	
	int Quick_Sort_Partition_V(int a[], int b[], int l, int r) 
	{
		int pivot, i, j;
		pivot = a[l];
		i = l; j = r+1;		
		while(1)
		{
			do ++i; while( a[i]<=pivot && i <= r );
			do --j; while( a[j]> pivot );
			if( i >= j ) break;
			//Swap i and j
			Swap(a[i], a[j]);
			Swap(b[i], b[j]);
		}
		//Swap l and j
		Swap(a[l], a[j]);
		Swap(b[l], b[j]);
		return j;
	}

};
#endif

#undef INDEX

#endif

