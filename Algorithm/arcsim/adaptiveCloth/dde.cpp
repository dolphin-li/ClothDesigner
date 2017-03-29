/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "dde.hpp"

#include "cloth.hpp"
#include "util.hpp"

namespace arcsim
{

	using namespace std;

	static const int nsamples = 30;

	Vec4 evaluate_stretching_sample(const Mat2x2 &G, const StretchingData &data);

	void evaluate_stretching_samples(StretchingSamples &samples,
		const StretchingData &data)
	{
		for (int i = 0; i < arcsim::nsamples; i++)
		for (int j = 0; j < arcsim::nsamples; j++)
		for (int k = 0; k < arcsim::nsamples; k++)
		{
			Mat2x2 G;
			G(0, 0) = -0.25 + i / (arcsim::nsamples*1.0);
			G(1, 1) = -0.25 + j / (arcsim::nsamples*1.0);
			G(0, 1) = G(1, 0) = k / (arcsim::nsamples*1.0);
			samples.s[i][j][k] = evaluate_stretching_sample(G, data);
		}
	}

	Vec4 evaluate_stretching_sample(const Mat2x2 &_G, const StretchingData &data)
	{
		Mat2x2 G = _G;
		G = G*2. + Mat2x2(1);
		Eig<2> eig = eigen_decomposition(G);
		Vec2 w = Vec2(sqrt(eig.l[0]), sqrt(eig.l[1]));
		Vec2 V = eig.Q.col(0);
		double angle_weight = fabsf(atan2f(V[1], V[0]) / M_PI) * 8;
		if (angle_weight<0)                      angle_weight = 0;
		if (angle_weight>4 - 1e-6)        angle_weight = 4 - 1e-6;
		int   angle_id = (int)angle_weight;
		angle_weight = angle_weight - angle_id;
		double strain_value = (w[0] - 1) * 6;
		if (strain_value<0)                      strain_value = 0;
		if (strain_value>1 - 1e-6)        strain_value = 1 - 1e-6;
		int   strain_id = (int)strain_value;
		//if(strain_id>1)               strain_id=1;
		strain_id = 0;
		double strain_weight = strain_value - strain_id;
		Vec4 real_elastic;
		real_elastic =
			data.d[strain_id][angle_id] * (1 - strain_weight)*(1 - angle_weight) +
			data.d[strain_id + 1][angle_id] * (strain_weight)*(1 - angle_weight) +
			data.d[strain_id][angle_id + 1] * (1 - strain_weight)*(angle_weight)+
			data.d[strain_id + 1][angle_id + 1] * (strain_weight)*(angle_weight);
		if (real_elastic[0] < 0)   real_elastic[0] = 0;
		if (real_elastic[1] < 0)   real_elastic[1] = 0;
		if (real_elastic[2] < 0)   real_elastic[2] = 0;
		if (real_elastic[3] < 0)   real_elastic[3] = 0;
		real_elastic[0] *= 2;
		real_elastic[1] *= 2;
		real_elastic[2] *= 2;
		real_elastic[3] *= 2;
		return real_elastic;
	}

	Vec4 stretching_stiffness(const Mat2x2 &G, const StretchingSamples &samples)
	{
		double a = (G(0, 0) + 0.25)*nsamples;
		double b = (G(1, 1) + 0.25)*nsamples;
		double c = fabsf(G(0, 1))*nsamples;
		a = clamp(a, 0.0, nsamples - 1 - 1e-5);
		b = clamp(b, 0.0, nsamples - 1 - 1e-5);
		c = clamp(c, 0.0, nsamples - 1 - 1e-5);
		int ai = (int)floor(a);
		int bi = (int)floor(b);
		int ci = (int)floor(c);
		if (ai<0)        ai = 0;
		if (bi<0)        bi = 0;
		if (ci<0)        ci = 0;
		if (ai>nsamples - 2)        ai = nsamples - 2;
		if (bi>nsamples - 2)        bi = nsamples - 2;
		if (ci>nsamples - 2)        ci = nsamples - 2;
		a = a - ai;
		b = b - bi;
		c = c - ci;
		double weight[2][2][2];
		weight[0][0][0] = (1 - a)*(1 - b)*(1 - c);
		weight[0][0][1] = (1 - a)*(1 - b)*(c);
		weight[0][1][0] = (1 - a)*(b)*(1 - c);
		weight[0][1][1] = (1 - a)*(b)*(c);
		weight[1][0][0] = (a)*(1 - b)*(1 - c);
		weight[1][0][1] = (a)*(1 - b)*(c);
		weight[1][1][0] = (a)*(b)*(1 - c);
		weight[1][1][1] = (a)*(b)*(c);
		Vec4 stiffness = Vec4(0);
		for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
		for (int k = 0; k < 2; k++)
		for (int l = 0; l < 4; l++)
		{
			stiffness[l] += samples.s[ai + i][bi + j][ci + k][l] * weight[i][j][k];
		}
		return stiffness;
	}

	double bending_stiffness(const Edge *edge, int side,
		const BendingData &data, double initial_angle)
	{
		double curv = edge->theta*edge->l / (edge->adjf[0]->a + edge->adjf[1]->a);
		double alpha = curv / 2;
		double value = alpha*0.2; // because samples are per 0.05 cm^-1 = 5 m^-1
		if (value > 4) value = 4;
		int             value_i = (int)value;
		if (value_i<0)   value_i = 0;
		if (value_i>3)   value_i = 3;
		value -= value_i;
		Vec2 du = edge_vert(edge, side, 1)->u - edge_vert(edge, side, 0)->u;
		double    bias_angle = (atan2f(du[1], du[0]) + initial_angle) * 4 / M_PI;
		if (bias_angle<0)        bias_angle = -bias_angle;
		if (bias_angle>4)        bias_angle = 8 - bias_angle;
		if (bias_angle > 2)        bias_angle = 4 - bias_angle;
		int             bias_id = (int)bias_angle;
		if (bias_id<0)   bias_id = 0;
		if (bias_id>1)   bias_id = 1;
		bias_angle -= bias_id;
		double actual_ke = data.d[bias_id][value_i] * (1 - bias_angle)*(1 - value)
			+ data.d[bias_id + 1][value_i] * (bias_angle)*(1 - value)
			+ data.d[bias_id][value_i + 1] * (1 - bias_angle)*(value)
			+data.d[bias_id + 1][value_i + 1] * (bias_angle)*(value);
		if (actual_ke < 0) actual_ke = 0;
		return actual_ke;
	}
}