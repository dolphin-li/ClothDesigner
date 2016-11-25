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
//  Intersection functions
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __WHMIN_INTERSECTION_H__
#define __WHMIN_INTERSECTION_H__
#include "MY_MATH.h"


///////////////////////////////////////////////////////////////////////////////////////////
////  Test whether a ray intersects a triangle
///////////////////////////////////////////////////////////////////////////////////////////
template <class TYPE>
bool Ray_Triangle_Intersection(TYPE x0[], TYPE x1[], TYPE x2[], TYPE p0[], TYPE dir[], TYPE &min_t)
{
	TYPE e1[3], e2[3], s1[3];
	e1[0]=x1[0]-x0[0];
	e1[1]=x1[1]-x0[1];
	e1[2]=x1[2]-x0[2];
	e2[0]=x2[0]-x0[0];
	e2[1]=x2[1]-x0[1];
	e2[2]=x2[2]-x0[2];
	CROSS(dir, e2, s1);
	TYPE divisor=DOT(s1, e1);
	if(divisor==0) return false;
	// Test the first barycentric coordinate
	TYPE tt[3];
	tt[0]=p0[0]-x0[0];
	tt[1]=p0[1]-x0[1];
	tt[2]=p0[2]-x0[2];
	TYPE b1=DOT(tt, s1);
	if(divisor>0 && (b1<0 || b1>divisor))		return false;
	if(divisor<0 && (b1>0 || b1<divisor))		return false;	
	// Test the second barycentric coordinate
	TYPE s2[3];
	CROSS(tt, e1, s2);
	TYPE b2=DOT(dir, s2);
	if(divisor>0 && (b2<0 || b1+b2>divisor))	return false;
	if(divisor<0 && (b2>0 || b1+b2<divisor))	return false;		
	// Compute t to intersection point
	min_t=DOT(e2, s2)/divisor;
	return min_t>0;
}

template <class TYPE>
int Triangle_Edges_Intersection(TYPE x0[], TYPE x1[], TYPE x2[], TYPE xi[], TYPE xj[], TYPE xk[], TYPE data0[], TYPE data1[])
{
	TYPE e10[3], e20[3], n[3];
	e10[0] = x1[0] - x0[0];
	e10[1] = x1[1] - x0[1];
	e10[2] = x1[2] - x0[2];
	e20[0] = x2[0] - x0[0];
	e20[1] = x2[1] - x0[1];
	e20[2] = x2[2] - x0[2];
	CROSS(e10, e20, n);
	TYPE nn = DOT(n, n);
	if(nn <1e-20f)		return 0;

	TYPE ei0[3];
	ei0[0] = xi[0] - x0[0];
	ei0[1] = xi[1] - x0[1];
	ei0[2] = xi[2] - x0[2];
	TYPE hi = DOT(ei0, n);

	TYPE ej0[3];
	ej0[0] = xj[0] - x0[0];
	ej0[1] = xj[1] - x0[1];
	ej0[2] = xj[2] - x0[2];
	TYPE hj = DOT(ej0, n);

	TYPE ek0[3];
	ek0[0] = xk[0] - x0[0];
	ek0[1] = xk[1] - x0[1];
	ek0[2] = xk[2] - x0[2];
	TYPE hk = DOT(ek0, n);

	int ret=0;

	// Test edge ij
	if(hi<=0 && hj>=0 || hi>=0 && hj<=0)
	{
	//	TODO here

	//	if(hj>=0)
	//	{
			data1[ret * 3 + 0] = hj / (hj - hi);
			data1[ret * 3 + 1] = 1 - data1[ret * 3 + 0];
	//	}
	//	else
	//	{
	//		data1[ret * 3 + 1] = hi / (hi - hj);
	//		data1[ret * 3 + 0] = 1 - data1[ret * 3 + 1];
	//	}

		data1[ret*3+2] = 0;

		TYPE e[3];
		e[0] = xi[0] * data1[ret * 3 + 0] + xj[0] * data1[ret * 3 + 1] + xk[0] * data1[ret * 3 + 2] - x0[0];
		e[1] = xi[1] * data1[ret * 3 + 0] + xj[1] * data1[ret * 3 + 1] + xk[1] * data1[ret * 3 + 2] - x0[1];
		e[2] = xi[2] * data1[ret * 3 + 0] + xj[2] * data1[ret * 3 + 1] + xk[2] * data1[ret * 3 + 2] - x0[2];
		TYPE b0, b1, b2;
		TYPE temp[3];
		CROSS(e10, e, temp);
		b2=DOT(temp, n);
		CROSS(e, e20, temp);
		b1=DOT(temp, n);
		b0=nn-b1-b2;
		if(b0>0 && b1>0 && b2>0)
		{
			data0[ret * 3 + 0] = b0 / nn;
			data0[ret * 3 + 1] = b1 / nn;
			data0[ret * 3 + 2] = b2 / nn;
			ret++;
		}
	}

	// Test edge ik
	if (hi <= 0 && hk >= 0 || hi >= 0 && hk <= 0)
	{
		data1[ret * 3 + 0] = hk / (hk - hi);
		data1[ret * 3 + 1] = 0;
		data1[ret * 3 + 2] = 1 - data1[ret * 3 + 0];

		TYPE e[3];
		e[0] = xi[0] * data1[ret * 3 + 0] + xj[0] * data1[ret * 3 + 1] + xk[0] * data1[ret * 3 + 2] - x0[0];
		e[1] = xi[1] * data1[ret * 3 + 0] + xj[1] * data1[ret * 3 + 1] + xk[1] * data1[ret * 3 + 2] - x0[1];
		e[2] = xi[2] * data1[ret * 3 + 0] + xj[2] * data1[ret * 3 + 1] + xk[2] * data1[ret * 3 + 2] - x0[2];
		TYPE b0, b1, b2;
		TYPE temp[3];
		CROSS(e10, e, temp);
		b2 = DOT(temp, n);
		CROSS(e, e20, temp);
		b1 = DOT(temp, n);
		b0 = nn - b1 - b2;
		if (b0 > 0 && b1 > 0 && b2 > 0)
		{
			data0[ret * 3 + 0] = b0 / nn;
			data0[ret * 3 + 1] = b1 / nn;
			data0[ret * 3 + 2] = b2 / nn;
			ret++;

		}
	}

	// Test edge jk
	if (hk <= 0 && hj >= 0 || hk >= 0 && hj <= 0)
	{
		data1[ret * 3 + 0] = 0;
		data1[ret * 3 + 1] = hk / (hk - hj);
		data1[ret * 3 + 2] = 1 - data1[ret * 3 + 1];

		TYPE e[3];
		e[0] = xi[0] * data1[ret * 3 + 0] + xj[0] * data1[ret * 3 + 1] + xk[0] * data1[ret * 3 + 2] - x0[0];
		e[1] = xi[1] * data1[ret * 3 + 0] + xj[1] * data1[ret * 3 + 1] + xk[1] * data1[ret * 3 + 2] - x0[1];
		e[2] = xi[2] * data1[ret * 3 + 0] + xj[2] * data1[ret * 3 + 1] + xk[2] * data1[ret * 3 + 2] - x0[2];
		TYPE b0, b1, b2;
		TYPE temp[3];
		CROSS(e10, e, temp);
		b2 = DOT(temp, n);
		CROSS(e, e20, temp);
		b1 = DOT(temp, n);
		b0 = nn - b1 - b2;
		if (b0 > 0 && b1 > 0 && b2 > 0)
		{
			data0[ret * 3 + 0] = b0 / nn;
			data0[ret * 3 + 1] = b1 / nn;
			data0[ret * 3 + 2] = b2 / nn;
			ret++;

			/*printf("xj: %f, %f, %f\n", xj[0], xj[1], xj[2]);
			printf("xk: %f, %f, %f\n", xk[0], xk[1], xk[2]);
			printf("x: %f, %f, %f\n",
				xi[0] * data1[ret * 3 + 0] + xj[0] * data1[ret * 3 + 1] + xk[0] * data1[ret * 3 + 2],
				xi[1] * data1[ret * 3 + 0] + xj[1] * data1[ret * 3 + 1] + xk[1] * data1[ret * 3 + 2],
				xi[2] * data1[ret * 3 + 0] + xj[2] * data1[ret * 3 + 1] + xk[2] * data1[ret * 3 + 2]);

			printf("data: %f, %f, %f\n", data1[ret * 3 + 0], data1[ret * 3 + 1], data1[ret * 3 + 2]);

			printf("xa: %f, %f, %f\n", x0[0], x0[1], x0[2]);
			printf("xb: %f, %f, %f\n", x1[0], x1[1], x1[2]);
			printf("xc: %f, %f, %f\n", x2[0], x2[1], x2[2]);

			printf("value: %f, %f; %f, %f, %f\n", hj, hk, b0, b1, b2);*/
		}
	}

	return ret;
}


template <class TYPE>
bool Triangle_Edge_Intersection(const TYPE x0[], const TYPE x1[], const TYPE x2[], const TYPE xi[], const TYPE xj[], TYPE x[])
{
	TYPE e10[3], e20[3], n[3];
	e10[0] = x1[0] - x0[0];
	e10[1] = x1[1] - x0[1];
	e10[2] = x1[2] - x0[2];
	e20[0] = x2[0] - x0[0];
	e20[1] = x2[1] - x0[1];
	e20[2] = x2[2] - x0[2];
	
	CROSS(e10, e20, n);
	TYPE nn = DOT(n, n);
	if(nn <1e-20f)		return false;

	TYPE ei0[3];
	ei0[0] = xi[0] - x0[0];
	ei0[1] = xi[1] - x0[1];
	ei0[2] = xi[2] - x0[2];
	TYPE hi = DOT(ei0, n);

	TYPE ej0[3];
	ej0[0] = xj[0] - x0[0];
	ej0[1] = xj[1] - x0[1];
	ej0[2] = xj[2] - x0[2];
	TYPE hj = DOT(ej0, n);
	
	// Test edge ij
	if(hi<=0 && hj>=0 || hi>=0 && hj<=0)
	{
		TYPE min_t = hj / (hj - hi);

		x[0] = xi[0] * min_t + xj[0] * (1 - min_t);
		x[1] = xi[1] * min_t + xj[1] * (1 - min_t);
		x[2] = xi[2] * min_t + xj[2] * (1 - min_t);

		TYPE e[3];
		e[0] = x[0] - x0[0];
		e[1] = x[1] - x0[1];
		e[2] = x[2] - x0[2];
		TYPE b0, b1, b2;
		TYPE temp[3];
		CROSS(e10, e, temp);
		b2=DOT(temp, n);
		CROSS(e, e20, temp);
		b1=DOT(temp, n);
		b0=nn-b1-b2;
		if(b0>0 && b1>0 && b2>0)	return true;
		
	}
	return false;
}


#endif

