#include "ImageData.h"
#include <iostream>
#include <fstream>
namespace ldp
{
	//==========================================================================
	/// bwdist: the same with matlab's
	//		using the linear-time Euclidean distance transform method

	////////// Functions F and Sep for the SDT labelling
	inline int bwdist_sqr(int u)
	{
		return u*u;
	}
	inline int bwdist_F(int u, int i, int gi2)
	{
		return (u - i)*(u - i) + gi2;
	}
	inline int bwdist_Sep(int i, int u, int gi2, int gu2)
	{
		return (u*u - i*i + gu2 - gi2) / (2 * (u - i));
	}
	/////////

	void bwdist(const MaskImage& mask, FloatImage& distMap)
	{
		typedef ImageTemplate<int> VolumeInt;
		VolumeInt tmpXVolume, tmpXYVolume;
		const ldp::Int2 resolution = mask.getResolution();
		const int inf = resolution[0] + resolution[1];

		// phase x-----------------------------------------------------
		tmpXVolume.resize(mask.getResolution());

#pragma omp parallel for
		for (int y = 0; y < resolution[1]; y++)
		{
			if (mask(0, y) == 0)
				tmpXVolume(0, y) = 0;
			else
				tmpXVolume(0, y) = inf;

			// Forward scan
			for (int x = 1; x < resolution[0]; x++)
			{
				if (mask(x, y) == 0)
					tmpXVolume(x, y) = 0;
				else
					tmpXVolume(x, y) = 1 + tmpXVolume(x - 1, y);
			}

			//Backward scan
			for (int x = resolution[0] - 2; x >= 0; x--)
			if (tmpXVolume(x + 1, y) < tmpXVolume(x, y))
				tmpXVolume(x, y) = 1 + tmpXVolume(x + 1, y);
		}// end for y

		// phase y-----------------------------------------------------
		tmpXYVolume.resize(mask.getResolution());
		{
			std::vector<int> s(resolution[1]), t(resolution[1]);
			for (int x = 0; x < resolution[0]; x++)
			{
				int q = 0, w = 0;
				s[0] = 0;
				t[0] = 0;

				//Forward Scan
				for (int u = 1; u < resolution[1]; u++)
				{
					while (q >= 0 && (bwdist_F(t[q], s[q], bwdist_sqr(tmpXVolume(x, s[q]))) >
						bwdist_F(t[q], u, bwdist_sqr(tmpXVolume(x, u)))))
						q--;

					if (q < 0)
					{
						q = 0;
						s[0] = u;
					}
					else
					{
						w = 1 + bwdist_Sep(s[q], u, bwdist_sqr(tmpXVolume(x, s[q])),
							bwdist_sqr(tmpXVolume(x, u)));
						if (w < resolution[1])
						{
							q++;
							s[q] = u;
							t[q] = w;
						}
					}
				}

				//Backward Scan
				for (int u = resolution[1] - 1; u >= 0; --u)
				{
					tmpXYVolume(x, u) = bwdist_F(u, s[q], bwdist_sqr(tmpXVolume(x, s[q])));
					if (u == t[q])
						q--;
				}
			}// end for x
		}// end for z

		distMap.resize(tmpXYVolume.getResolution());
		for (int y = 0; y < resolution[1]; y++)
		for (int x = 0; x < resolution[0]; x++)
			distMap(x, y) = sqrt((float)tmpXYVolume(x, y));
	}
}// namespace mpu