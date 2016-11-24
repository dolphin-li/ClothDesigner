#include "ConvolutionPyramid.h"
namespace ldp
{
#undef min
#undef max
	const static int g_zero_padding = 5;

	ConvolutionPyramid::ConvolutionPyramid()
	{
	}

	ConvolutionPyramid::~ConvolutionPyramid()
	{
	}

	void ConvolutionPyramid::convolve_boundary(FloatImage& srcDst)
	{
		const float kernel5x5[5] = { 0.1507f, 0.6836f, 1.0334f, 0.6836f, 0.1507f };
		const float kernel3x3[3] = { 0.0312f, 0.7753f, 0.0312f };
		const float mul = sqrt(0.0270f);
		const float kernel5x5up[5] = { mul * 0.1507f, mul * 0.6836f, mul * 1.0334f, mul * 0.6836f, mul * 0.1507f };

		PyramidConvolve(srcDst, kernel5x5, kernel3x3, kernel5x5up);
	}

	void ConvolutionPyramid::solve_poisson(FloatImage& srcDiv_dstVolume)
	{
		throw std::exception("error: result of this method is incorrect");
		const float kernel5x5[5] = { 0.15f, 0.5f, 0.7f, 0.5f, 0.15f };
		const float kernel3x3[3] = { 0.175f, 0.547f, 0.175f };

		PyramidConvolve(srcDiv_dstVolume, kernel5x5, kernel3x3, kernel5x5);
	}

	void ConvolutionPyramid::PyramidConvolve(FloatImage& srcDst, const float* kernel5x5,
		const float* kernel3x3, const float* kernel5x5up)
	{
		const ldp::Int2 res = srcDst.getResolution();
		const int nMaxLevel = (int)ceil(log((float)std::max(res[0], std::max(res[1], res[2]))) / log(2.0f));

		std::vector<FloatImage*> pyramid;
		pyramid.resize(nMaxLevel);
		pyramid[0] = &srcDst;
		for (int k = 1; k < pyramid.size(); k++)
			pyramid[k] = new FloatImage();

		/// down---------------------------------
		FloatImage imConv;
		for (int i = 1; i < nMaxLevel; i++)
		{
			ZeroPadding5x5(imConv, *pyramid[i - 1]);
			conv_helper::conv2<float, 5>(imConv.data(), kernel5x5, imConv.getResolution());
			DownSamplex2(*pyramid[i], imConv);
		}//i
		imConv.clear();

		/// up------------------------------------

		// on the coarse level
		// imCurrent = conv(pad(imCurrent), g_3x3)
		FloatImage imCurrent;
		ZeroPadding5x5(imCurrent, *pyramid[nMaxLevel - 1]);
		conv_helper::conv2<float, 3>(imCurrent.data(), kernel3x3, imCurrent.getResolution());

		FloatImage imTmpDown, imTmpUp;
		for (int i = nMaxLevel - 2; i >= 0; i--)
		{
			// imTmpDown = unpad(imCurrent)
			imCurrent.subImageTo(imTmpDown, g_zero_padding, ldp::Int2(imCurrent.getResolution()) 
				- g_zero_padding);

			// imTmpUp = conv(upscale(imTmpDown), h2_5x5)
			imTmpUp.resize(pyramid[i]->getResolution() + g_zero_padding * 2);
			VolumeUpscalex2_ZeroHalf(imTmpUp, imTmpDown);
			conv_helper::conv2<float, 5>(imTmpUp.data(), kernel5x5up, imTmpUp.getResolution());

			// imCurrent = conv(pad(imCurrent), g_3x3)
			ZeroPadding5x5(imCurrent, *pyramid[i]);
			conv_helper::conv2<float, 3>(imCurrent.data(), kernel3x3, imCurrent.getResolution());

			// imCurrent += imTmpUp
			AddImage(imCurrent, imTmpUp);
		}

		// unpad
		imCurrent.subImageTo(srcDst, g_zero_padding, imCurrent.getResolution() - g_zero_padding);

		/// free
		for (int k = 1; k < pyramid.size(); k++)
			delete pyramid[k];
	}

	// A = A + B
	void ConvolutionPyramid::AddImage(FloatImage& A, const FloatImage& B)
	{
		if (A.getResolution() != B.getResolution())
			throw std::exception("AddImage: size mis-matched");

		const unsigned int sz = A.getResolution()[0] * A.getResolution()[1];
		float* dst = A.data();
		const float* src = B.data();
		for (int i = 0; i < sz; i++)
			dst[i] += src[i];
	}

	// C = unpad(A+B)
	void ConvolutionPyramid::AddImageUnpad5x5(FloatImage& C, const FloatImage& A, const FloatImage& B)
	{
		ldp::Int2 res = A.getResolution();
		if (res != B.getResolution())
			throw std::exception("AddImageUnpad: size mis-matched");
		ldp::Int2 dstRes = res - 5 * 2;
		if (dstRes[0] < 0 || dstRes[1] < 0 || dstRes[2] < 0)
			throw std::exception("AddImageUnpad: too small to unpad");

		C.resize(dstRes);

		for (int y = 0; y < dstRes[1]; y++)
		{
			const float* A_ptr_y = A.data() + (y + 5) * A.stride_Y() + 5;
			const float* B_ptr_y = B.data() + (y + 5) * B.stride_Y() + 5;
			float* C_ptr_y = C.data() + y * C.stride_Y();
			for (int x = 0; x < dstRes[0]; x++)
			{
				C_ptr_y[x] = A_ptr_y[x] + B_ptr_y[x];
			}// x
		}// y
	}

	void ConvolutionPyramid::ZeroPadding5x5(FloatImage& dst, const FloatImage& src)
	{
		ldp::Int2 srcRes = src.getResolution();
		dst.resize(srcRes + 5 * 2);
		conv_helper::zero_padding2<float, 5>(dst.data(), src.data(), src.getResolution());
	}

	void ConvolutionPyramid::DownSamplex2(FloatImage& imDst, const FloatImage& imSrc)
	{
		if (imDst.data() == imSrc.data())
			throw std::exception("VolumeDecimate: does not support inplace operation");

		const ldp::Int2 low_res = imSrc.getResolution() / 2;
		imDst.resize(low_res);

		for (int y = 0; y < low_res[1]; y++)
		{
			const float* src_y_ptr = imSrc.data() + imSrc.stride_Y() * y * 2;
			float* dst_y_ptr = imDst.data() + imDst.stride_Y() * y;
			for (int x = 0, xs=0; x < low_res[0]; x++, xs+=2)
				dst_y_ptr[x] = src_y_ptr[xs];
		}// end for y
	}

	void ConvolutionPyramid::VolumeUpscalex2_ZeroHalf(FloatImage& imDst, const FloatImage& imSrc)
	{
		const ldp::Int2 src_res = imSrc.getResolution();
		const ldp::Int2 dst_res = imDst.getResolution();
		
		for (int k = 0; k < 2; k++)
		if (src_res[k] * 2 != dst_res[k] && src_res[k] * 2 + 1 != dst_res[k])
			throw std::exception("VolumeUpscalex2: illegal size");

		for (int y = 0; y < src_res[1]; y++)
		{
			const float* src_y_ptr = imSrc.data() + imSrc.stride_Y() * y;
			float* dst_y_ptr = imDst.data() + imDst.stride_Y() * y * 2;
			memset(dst_y_ptr + imDst.stride_Y(), 0, sizeof(float)*imDst.stride_Y());
			for (int x = 0; x < src_res[0]; x++)
			{
				int x2 = (x << 1);
				dst_y_ptr[x2] = src_y_ptr[x];
				dst_y_ptr[x2 + 1] = 0;
			}// end for x
		}// end for y
	}

}