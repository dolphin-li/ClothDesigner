#pragma once

#include "conv\Convolution_Helper.h"
#include "ldpMat\ldp_basic_mat.h"
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
namespace ldp
{
	template<typename T>
	class ImageTemplate
	{
	public:
		ImageTemplate(){}
		~ImageTemplate(){ clear(); }

		//=================================================================
		void clear()
		{
			m_data.clear();
			m_resolution = 0;
		}
		void resize(unsigned short  x, unsigned short  y)
		{
			m_resolution = ldp::UShort2(x, y);
			m_data.resize(x*y);
		}
		void resize(ldp::UShort2 p)
		{
			resize(p[0], p[1]);
		}
		void fill(T v)
		{
			std::fill(m_data.begin(), m_data.end(), v);
		}

		ldp::UShort2 size()const
		{
			return m_resolution;
		}

		int width()const
		{
			return m_resolution[0];
		}

		int height()const
		{
			return m_resolution[1];
		}

		//=================================================================
		/// data access methods
		ldp::UShort2 getResolution()const{ return m_resolution; }
		int stride_X()const{ return 1; }
		int stride_Y()const{ return m_resolution[0]; }

		bool contains(ldp::Int2 idx)const
		{
			return idx[0] < m_resolution[0] && idx[0] >= 0
				&& idx[1] < m_resolution[1] && idx[1] >= 0;
		}

		// get the value from the data array
		T& operator()(ldp::UShort2 p){ return (*this)(p[0], p[1]); }
		const T& operator()(ldp::UShort2 p)const { return (*this)(p[0], p[1]); }
		T& operator()(unsigned short  x, unsigned short  y)
		{
			return m_data[y*m_resolution[0] + x];
		}
		const T& operator()(unsigned short x, unsigned short  y)const
		{
			return m_data[y*m_resolution[0] + x];
		}

		T bilinear_at(ldp::Float2 p)const
		{
			ldp::Int2 p0 = p;
			ldp::Float2 d = p - ldp::Float2(p0);
			int inc[2] = { p0[0]<m_resolution[0] - 1, p0[1]<m_resolution[1] - 1 };

			const int y_stride = stride_Y();

			const T* data_y0 = data() + y_stride * p0[1];
			const T* data_y1 = data_y0 + y_stride * inc[1];
			const int x0 = p0[0];
			const int x1 = x0 + inc[0];

			// first we interpolate along x-direction
			float c0 = data_y0[x0] * (1 - d[0]) + data_y0[x1] * d[0];
			float c1 = data_y1[x0] * (1 - d[0]) + data_y1[x1] * d[0];

			// finally along y
			float c = c0 * (1 - d[1]) + c1 * d[1];

			return c;
		}

		T *data(){ return m_data.data(); }
		const T *data()const{ return m_data.data(); }

		T *data_XY(ldp::UShort2 p){ return &(*this)(p); }
		const T *data_XY(ldp::UShort2 p)const{ return &(*this)(p); }
		T *data_XY(unsigned short x, unsigned short y){ return &(*this)(x, y); }
		const T *data_XY(unsigned short x, unsigned short y)const{ return &(*this)(x, y); }

		// fill in the patch with given data memory and given size/pos
		// dst must be of size patchSize*patchSize*patchSize
		template <int patchSize>
		void getPatch(T* dst, const ldp::UShort2& patch_x0y0)const
		{
			const T* src = data_XY(patch_x0y0);
			const int stride_y = stride_Y();
			for (int y = 0; y < patchSize; y++)
			{
				for (int x = 0; x < patchSize; x++)
					*dst++ = src[x];
				src += stride_y;
			}
		}
		void getPatch(T* dst, const ldp::UShort2& patch_x0y0, int patchSize)const
		{
			switch (patchSize)
			{
			default:
				throw std::exception("too big patch size");
				break;
			case 1:
				getPatch<1>(dst, patch_x0y0);
				break;
			case 2:
				getPatch<2>(dst, patch_x0y0);
				break;
			case 3:
				getPatch<3>(dst, patch_x0y0);
				break;
			case 4:
				getPatch<4>(dst, patch_x0y0);
				break;
			case 5:
				getPatch<5>(dst, patch_x0y0);
				break;
			case 6:
				getPatch<6>(dst, patch_x0y0);
				break;
			case 7:
				getPatch<7>(dst, patch_x0y0);
				break;
			case 8:
				getPatch<8>(dst, patch_x0y0);
				break;
			case 9:
				getPatch<9>(dst, patch_x0y0);
				break;
			case 10:
				getPatch<10>(dst, patch_x0y0);
				break;
			case 11:
				getPatch<11>(dst, patch_x0y0);
				break;
			}
		}
		void setPatch(const T* src, const ldp::UShort2& patch_x0y0, int patchSize)
		{
			T* dst = data_XY(patch_x0y0);
			const int stride_y = stride_Y();

			for (int y = 0; y < patchSize; y++)
			{
				for (int x = 0; x < patchSize; x++)
					dst[x] = *src++;
				dst += stride_y;
			}
		}
		void fillPatch(T v, const ldp::UShort2& patch_x0y0, int patchSize)
		{
			T* dst = data_XY(patch_x0y0);
			const int stride_y = stride_Y();
			for (int y = 0; y < patchSize; y++)
			{
				for (int x = 0; x < patchSize; x++)
					dst[x] = v;
				dst += stride_y;
			}
		}

		/// convolutions
		void convolve_max(int radius)
		{
			switch (radius)
			{
			default:
				throw std::exception("non-supported convolve-max radius!");
				break;
			case 0:
				return;
			case 1:
				return conv_helper::max_filter2<T, 3>(data(), m_resolution);
			case 2:
				return conv_helper::max_filter2<T, 5>(data(), m_resolution);
			case 3:
				return conv_helper::max_filter2<T, 7>(data(), m_resolution);
			case 4:
				return conv_helper::max_filter2<T, 9>(data(), m_resolution);
			case 5:
				return conv_helper::max_filter2<T, 11>(data(), m_resolution);
			case 6:
				return conv_helper::max_filter2<T, 13>(data(), m_resolution);
			case 7:
				return conv_helper::max_filter2<T, 15>(data(), m_resolution);
			case 8:
				return conv_helper::max_filter2<T, 17>(data(), m_resolution);
			case 9:
				return conv_helper::max_filter2<T, 19>(data(), m_resolution);
			case 10:
				return conv_helper::max_filter2<T, 21>(data(), m_resolution);
			case 11:
				return conv_helper::max_filter2<T, 23>(data(), m_resolution);
			case 12:
				return conv_helper::max_filter2<T, 25>(data(), m_resolution);
			case 13:
				return conv_helper::max_filter2<T, 27>(data(), m_resolution);
			case 14:
				return conv_helper::max_filter2<T, 29>(data(), m_resolution);
			case 15:
				return conv_helper::max_filter2<T, 31>(data(), m_resolution);
			case 16:
				return conv_helper::max_filter2<T, 33>(data(), m_resolution);
			case 17:
				return conv_helper::max_filter2<T, 35>(data(), m_resolution);
			case 18:
				return conv_helper::max_filter2<T, 37>(data(), m_resolution);
			case 19:
				return conv_helper::max_filter2<T, 39>(data(), m_resolution);
			case 20:
				return conv_helper::max_filter2<T, 41>(data(), m_resolution);
			case 21:
				return conv_helper::max_filter2<T, 43>(data(), m_resolution);
			case 22:
				return conv_helper::max_filter2<T, 45>(data(), m_resolution);
			case 23:
				return conv_helper::max_filter2<T, 47>(data(), m_resolution);
			case 24:
				return conv_helper::max_filter2<T, 49>(data(), m_resolution);
			}
		}
		void convolve_min(int radius)
		{
			switch (radius)
			{
			default:
				throw std::exception("non-supported convolve-max radius!");
				break;
			case 0:
				return;
			case 1:
				return conv_helper::min_filter2<T, 3>(data(), m_resolution);
			case 2:
				return conv_helper::min_filter2<T, 5>(data(), m_resolution);
			case 3:
				return conv_helper::min_filter2<T, 7>(data(), m_resolution);
			case 4:
				return conv_helper::min_filter2<T, 9>(data(), m_resolution);
			case 5:
				return conv_helper::min_filter2<T, 11>(data(), m_resolution);
			case 6:
				return conv_helper::min_filter2<T, 13>(data(), m_resolution);
			case 7:
				return conv_helper::min_filter2<T, 15>(data(), m_resolution);
			case 8:
				return conv_helper::min_filter2<T, 17>(data(), m_resolution);
			case 9:
				return conv_helper::min_filter2<T, 19>(data(), m_resolution);
			}
		}

		// the same with matlab conv(...,'same')
		void convolve(const float* kernel, int kernelSize)
		{
			switch (kernelSize)
			{
			default:
				throw std::exception("non-supported convolve kernelsize!");
				break;
			case 1:
				return conv_helper::conv2<T, 1>(data(), kernel, m_resolution);
			case 2:										
				return conv_helper::conv2<T, 2>(data(), kernel, m_resolution);
			case 3:										
				return conv_helper::conv2<T, 3>(data(), kernel, m_resolution);
			case 4:										
				return conv_helper::conv2<T, 4>(data(), kernel, m_resolution);
			case 5:									
				return conv_helper::conv2<T, 5>(data(), kernel, m_resolution);
			case 6:										
				return conv_helper::conv2<T, 6>(data(), kernel, m_resolution);
			case 7:										
				return conv_helper::conv2<T, 7>(data(), kernel, m_resolution);
			case 8:										
				return conv_helper::conv2<T, 8>(data(), kernel, m_resolution);
			case 9:										
				return conv_helper::conv2<T, 9>(data(), kernel, m_resolution);
			case 10:
				return conv_helper::conv2<T, 10>(data(), kernel, m_resolution);
			case 11:
				return conv_helper::conv2<T, 11>(data(), kernel, m_resolution);
			case 12:
				return conv_helper::conv2<T, 12>(data(), kernel, m_resolution);
			case 13:
				return conv_helper::conv2<T, 13>(data(), kernel, m_resolution);
			}
		}

		void mirrorExtendTo(ImageTemplate<T>& rhs, int radius)const
		{
			rhs.resize(m_resolution[0] + radius * 2, m_resolution[1] + radius * 2);

			for (int y = 0; y < rhs.m_resolution[1]; y++)
			{
				int src_y = abs(y - radius);
				if (src_y >= m_resolution[1])
					src_y = std::max(0, 2 * (int)m_resolution[1] - 2 - src_y);
				const T* src_y_ptr = data() + stride_Y() * src_y;
				T* dst_y_ptr = rhs.data() + rhs.stride_Y() * y;
				for (int x = 0; x < rhs.m_resolution[0]; x++)
				{
					int src_x = abs(x - radius);
					if (src_x >= m_resolution[0])
						src_x = std::max(0, 2 * (int)m_resolution[0] - 2 - src_x);
					dst_y_ptr[x] = src_y_ptr[src_x];
				}
			}// y
		}

		void zeroPaddingTo(ImageTemplate<T>& rhs, int radius)const
		{
			rhs.resize(m_resolution[0] + radius * 2, m_resolution[1] + radius * 2);

			for (int y = 0; y < rhs.m_resolution[1]; y++)
			{
				T* dst_y_ptr = rhs.data() + rhs.stride_Y() * y;

				int src_y = y - radius;
				if (src_y >= m_resolution[1] || src_y < 0)
				{
					memset(dst_y_ptr, 0, sizeof(T) * m_resolution[0]);
					continue;
				}

				const T* src_y_ptr = data() + stride_Y() * src_y;
				for (int x = 0; x < rhs.m_resolution[0]; x++)
				{
					int src_x = x - radius;
					if (src_x >= m_resolution[0] || src_x < 0)
						dst_y_ptr[x] = 0;
					else
						dst_y_ptr[x] = src_y_ptr[src_x];
				}
			}// y
		}

		void subImageTo(ImageTemplate<T>& rhs, ldp::Int2 begin, ldp::Int2 end)const
		{
			for (int k = 0; k < 2; k++)
			{
				begin[k] = std::max(begin[k], 0);
				end[k] = std::min(end[k], (int)m_resolution[k]);
				if (end[k] <= begin[k])
				{
					rhs.clear();
					return;
				}
			}
			rhs.resize(ldp::Int2(end)-ldp::Int2(begin));

			for (int y = 0; y < rhs.m_resolution[1]; y++)
			{
				const T* src_y_ptr = stride_Y() * (y + begin[1]) + data() + begin[0];
				T* dst_y_ptr = rhs.stride_Y() * y + rhs.data();
				for (int x = 0; x < rhs.m_resolution[0]; x++)
					dst_y_ptr[x] = src_y_ptr[x];
			}// y
		}

		void subImageFrom(ImageTemplate<T>& rhs, ldp::Int2 begin, ldp::Int2 end)
		{
			for (int k = 0; k < 2; k++)
			{
				begin[k] = std::max(begin[k], 0);
				end[k] = std::min(end[k], (int)m_resolution[k]);
				if (end[k] <= begin[k])
					return;
			}

			for (int y = 0; y < rhs.m_resolution[1]; y++)
			{
				T* dst_y_ptr = stride_Y() * (y + begin[1]) + rhs.data() + begin[0];
				const T* src_y_ptr = rhs.stride_Y() * y + data();
				for (int x = 0; x < rhs.m_resolution[0]; x++)
					dst_y_ptr[x] = src_y_ptr[x];
			}// y
		}

		ImageTemplate<T>& operator += (const ImageTemplate<T>& rhs)
		{
			assert(width() == rhs.width() && height() == rhs.height());
			for (size_t i = 0; i < m_data.size(); i++)
				m_data[i] += rhs.m_data[i];
			return *this;
		}
		ImageTemplate<T>& operator -= (const ImageTemplate<T>& rhs)
		{
			assert(width() == rhs.width() && height() == rhs.height());
			for (size_t i = 0; i < m_data.size(); i++)
				m_data[i] -= rhs.m_data[i];
			return *this;
		}
		typename type_promote<T, T>::type sum()const
		{
			typename type_promote<T, T>::type s = 0;
			for (size_t i = 0; i < m_data.size(); i++)
				s += m_data[i];
			return s;
		}
	protected:
		ldp::UShort2 m_resolution;
		std::vector<T> m_data;
	};

	typedef ImageTemplate<unsigned char> MaskImage;
	typedef ImageTemplate<int> IntImage;
	typedef ImageTemplate<unsigned short> KinectDepthImage;
	typedef ImageTemplate<float> FloatImage;
	typedef ImageTemplate<double> DoubleImage;
	typedef ImageTemplate<ldp::UChar3> RgbImage;
	typedef ImageTemplate<ldp::UChar4> RgbaImage;
	typedef ImageTemplate<ldp::Float3> RgbFloatImage;
	typedef ImageTemplate<ldp::Float4> RgbaFloatImage;

	/// bwdist: the same with matlab's
	//		using the linear-time Euclidean distance transform method
	void bwdist(const MaskImage& mask, FloatImage& distMap);
}
#pragma pop_macro("max")
#pragma pop_macro("min")