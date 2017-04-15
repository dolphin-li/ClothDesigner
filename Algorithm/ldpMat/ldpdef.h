#pragma once

#include <iostream>
#include <iomanip>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <hash_map>
#include <hash_set>
#include <complex>
#include <assert.h>
#include "half.hpp"
// determine which OS
#if defined(_WIN32)
	#define LDP_OS_WIN
#elif defined(__APPLE__) && defined(__MACH__)
	#define LDP_OS_MAC
#else
	#define LDP_OS_LNX
#endif

// define internal datatype
#if defined(LDP_OS_WIN)
	#include <windows.h>
#undef min
#undef max
	typedef LARGE_INTEGER gtime_t;
#elif defined(LDP_OS_MAC)
	// http://developer.apple.com/qa/qa2004/qa1398.html
	#include <mach/mach_time.h>
	typedef uint64_t gtime_t;
#elif defined(LDP_OS_LNX)
	#include <sys/time.h>
	typedef struct timeval gtime_t;
#endif
namespace ldp
{
	/** *********************************************************************************
	* Release assert
	* **********************************************************************************/
#ifdef NDEBUG
	#ifdef  __cplusplus
	extern "C" {
	#endif
	_CRTIMP void __cdecl _wassert(_In_z_ const wchar_t * _Message, _In_z_ const wchar_t *_File, _In_ unsigned _Line);
	#ifdef  __cplusplus
	}
	#endif
#endif
	#define release_assert(_Expression) (void)( (!!(_Expression)) || (_wassert(_CRT_WIDE(#_Expression), _CRT_WIDE(__FILE__), __LINE__), 0) )

	/** *********************************************************************************
	*cuda wrappers
	* **********************************************************************************/
#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

	template<class T> __device__ __host__ void ldp_swap(T& a, T& b)
	{
		T c = a;
		a = b;
		b = c;
	}

	template<class T> __device__ __host__ T numeric_limits_eps();

	template<>  __device__ __host__ inline float numeric_limits_eps<float>()
	{
		return FLT_EPSILON;
	}
	template<>  __device__ __host__ inline double numeric_limits_eps<double>()
	{
		return DBL_EPSILON;
	}

	template<class T> __device__ __host__ T numeric_limits_max();

	template<>  __device__ __host__ inline float numeric_limits_max<float>()
	{
		return FLT_MAX;
	}
	template<>  __device__ __host__ inline double numeric_limits_max<double>()
	{
		return DBL_MAX;
	}

	template<class T> __device__ __host__ T numeric_limits_min();

	template<>  __device__ __host__ inline float numeric_limits_min<float>()
	{
		return FLT_MIN;
	}
	template<>  __device__ __host__ inline double numeric_limits_min<double>()
	{
		return DBL_MIN;
	}

	template<class T> __device__ __host__ T numeric_limits_inf();
	template<>  __device__ __host__ inline float numeric_limits_inf<float>()
	{
		return HUGE_VALF;
	}
	template<>  __device__ __host__ inline double numeric_limits_inf<double>()
	{
		return HUGE_VALD;
	}



	/** *********************************************************************************
	* Basic Type Declaration
	* **********************************************************************************/
	typedef std::complex<double> ComplexD;
	typedef std::complex<float> ComplexF;
	template<typename T, size_t N> class ldp_basic_vec;
	template<class T, size_t N, size_t M> class ldp_basic_mat;

	//is int
	template<typename T>
	struct is_int
	{
		static const bool value=false;
	};
	template<>
	struct is_int<int>
	{
		static const bool value=true;
	};

	//is float
	template<typename T>
	struct is_float
	{
		static const bool value=false;
	};
	template<>
	struct is_float<float>
	{
		static const bool value=true;
	};

	//is double
	template<typename T>
	struct is_double
	{
		static const bool value=false;
	};
	template<>
	struct is_double<double>
	{
		static const bool value=true;
	};

	//is float complex
	template<typename T>
	struct is_complexf
	{
		static const bool value=false;
	};
	template<>
	struct is_complexf<ComplexF>
	{
		static const bool value=true;
	};

	//is double complex
	template<typename T>
	struct is_complexd
	{
		static const bool value=false;
	};
	template<>
	struct is_complexd<ComplexD>
	{
		static const bool value=true;
	};

	//is basic type: char, short, int, long long, float, double
	template<typename T>
	struct is_basic_type
	{
		static const bool value=false;
	};
	template<>
	struct is_basic_type<char>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<unsigned char>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<short>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<unsigned short>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<int>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<unsigned int>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<long long>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<unsigned long long>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<float>
	{
		static const bool value=true;
	};
	template<>
	struct is_basic_type<double>
	{
		static const bool value=true;
	};

	//type of norm of different types
	template<class T>
	struct norm_type
	{
		typedef int type;
	};

	template<>
	struct norm_type<float>
	{
		typedef float type;
	};
	template<>
	struct norm_type<double>
	{
		typedef double type;
	};
	template<>
	struct norm_type<ComplexF>
	{
		typedef float type;
	};
	template<>
	struct norm_type<ComplexD>
	{
		typedef double type;
	};

	/** *********************************************************************************
	* Basic Constant and Functions
	* **********************************************************************************/
	const double DOUBLE_EPS = 2.220446049250313e-016;
	const float  SINGLE_EPS = 1.1920929e-007f;
	const double DOUBLE_MIN = 2.225073858507200e-308;
	const double DOUBLE_MAX = 1.797693134862315e+308;
	const float  SINGLE_MIN = 1.1754944e-038f;
	const float  SINGLE_MAX = 3.4028234e+038f;
	const double LOG_2D = 0.693147180559945;
	const float  LOG_2S = 0.693147180559945f;
	const double PI_D = 3.141592653589793;
	const float PI_S = 3.141592653589793f;
	const double SQRT_TWO_D = 1.414213562373095;
	const float SQRT_TWO_S = 1.414213562373095f;

	template<class T>
	__device__ __host__ inline T eps(T x = T(1.0))
	{
		if(x == T(1.0)) return numeric_limits_eps<T>();
		if (x > numeric_limits_max<T>()) return numeric_limits_max<T>();
		if (x < numeric_limits_min<T>()) return numeric_limits_min<T>();
		int n = (int)(log(abs(x))/log(T(2)));
		return numeric_limits_eps<T>() * pow(T(2.0), n);
	}

	template<class T> 
	__device__ __host__ int sign(const T& v)
	{
		return (v>0) - (v<0);
	}

	template<class T> 
	__device__ __host__ inline T sqr(T x)
	{
		return x*x;
	}

	template<class T>
	__device__ __host__ inline bool isInf(T x)
	{
		return x > numeric_limits_max<T>() || x < -numeric_limits_max<T>();
	}

	template<class T>
	__device__ __host__ inline bool isNan(T x)
	{
		return x!=x;
	}

	/** *********************************************************************************
	* Type Promotion
	* Rules:
	* char				< int
	* unsigned char		< int
	* short				< int
	* unsinged short	< int
	* int < unsigned int < long long < unsigned long long < float < double
	* **********************************************************************************/
	template<class T, class E> struct type_promote 
	{
		typedef typename T type;
	};

#define LDP_TYPE_PROMOTION_RULES_1(A, B)				\
	template<> struct type_promote<A, A>				\
	{													\
		typedef B type;									\
	};		

#define LDP_TYPE_PROMOTION_RULES_2(A, B, C)				\
	template<> struct type_promote<A, B>				\
	{													\
		typedef C type;									\
	};													\
	template<> struct type_promote<B, A>				\
	{													\
		typedef C type;									\
	};				

	LDP_TYPE_PROMOTION_RULES_1(char,				int								); 
	LDP_TYPE_PROMOTION_RULES_1(unsigned char,		int								); 
	LDP_TYPE_PROMOTION_RULES_1(short,				int								); 
	LDP_TYPE_PROMOTION_RULES_1(unsigned short,		int								); 
	LDP_TYPE_PROMOTION_RULES_2(char,				unsigned char,				int	);
	LDP_TYPE_PROMOTION_RULES_2(char,				short,						int	);
	LDP_TYPE_PROMOTION_RULES_2(char,				unsigned short,				int	);
	LDP_TYPE_PROMOTION_RULES_2(char,				int,						int	);
	LDP_TYPE_PROMOTION_RULES_2(char,				unsigned int,				unsigned int);
	LDP_TYPE_PROMOTION_RULES_2(char,				long long,					long long	);
	LDP_TYPE_PROMOTION_RULES_2(char,				unsigned long long,			unsigned long long	);
	LDP_TYPE_PROMOTION_RULES_2(char,				float,						float	);
	LDP_TYPE_PROMOTION_RULES_2(char,				double,						double	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned char,		short,						int	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned char,		unsigned short,				int	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned char,		int,						int	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned char,		unsigned int,				unsigned int	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned char,		long long,					long long	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned char,		unsigned long long,			unsigned long long	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned char,		float,						float	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned char,		double,						double	);
	LDP_TYPE_PROMOTION_RULES_2(short,				unsigned short,				int	);
	LDP_TYPE_PROMOTION_RULES_2(short,				int,						int	);
	LDP_TYPE_PROMOTION_RULES_2(short,				unsigned int,				unsigned int	);
	LDP_TYPE_PROMOTION_RULES_2(short,				long long,					long long	);
	LDP_TYPE_PROMOTION_RULES_2(short,				unsigned long long,			unsigned long long	);
	LDP_TYPE_PROMOTION_RULES_2(short,				float,						float	);
	LDP_TYPE_PROMOTION_RULES_2(short,				double,						double	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned short,		int,						int	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned short,		unsigned int,				unsigned int	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned short,		long long,					long long	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned short,		unsigned long long,			unsigned long long	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned short,		float,						float	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned short,		double,						double	);
	LDP_TYPE_PROMOTION_RULES_2(int,					unsigned int,				unsigned int	);
	LDP_TYPE_PROMOTION_RULES_2(int,					long long,					long long	);
	LDP_TYPE_PROMOTION_RULES_2(int,					unsigned long long,			unsigned long long	);
	LDP_TYPE_PROMOTION_RULES_2(int,					float,						float	);
	LDP_TYPE_PROMOTION_RULES_2(int,					double,						double	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned int,		long long,					long long	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned int,		unsigned long long,			unsigned long long	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned int,		float,						float	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned int,		double,						double	);
	LDP_TYPE_PROMOTION_RULES_2(long long,			unsigned long long,			unsigned long long	);
	LDP_TYPE_PROMOTION_RULES_2(long long,			float,						float	);
	LDP_TYPE_PROMOTION_RULES_2(long long,			double,						double	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned long long,	float,						float	);
	LDP_TYPE_PROMOTION_RULES_2(unsigned long long,	double,						double	);
	LDP_TYPE_PROMOTION_RULES_2(float,				double,						double	);

#undef LDP_TYPE_PROMOTION_RULES_1
#undef LDP_TYPE_PROMOTION_RULES_2


	/** *********************************************************************************************************
	* Timing Function
	* **********************************************************************************************************/

	// get current time
	static inline gtime_t gtime_now(void)
	{
	#if defined(LDP_OS_WIN)
		gtime_t time;
		QueryPerformanceCounter(&time);
		return time;
	#elif defined(LDP_OS_MAC)
		return mach_absolute_time();
	#elif defined(LDP_OS_LNX)
		gtime_t time;
		gettimeofday(&time, NULL);
		return time;
	#endif
	}

	// absolute difference between two times (in seconds)
	static inline double gtime_seconds(gtime_t start, gtime_t end)
	{
	#if defined(LDP_OS_WIN)
		if (start.QuadPart > end.QuadPart) {
			gtime_t temp = end;
			end = start;
			start = temp;
		}
		gtime_t system_freq;
		QueryPerformanceFrequency(&system_freq);
		return (double)(end.QuadPart - start.QuadPart) / system_freq.QuadPart;
	#elif defined(LDP_OS_MAC)
		if (start > end) {
			uint64_t temp = start;
			start = end;
			end   = temp;
		}
		// calculate platform timing epoch
		static mach_timebase_info_data_t info;
		mach_timebase_info(&info);
		double nano = (double)info.numer / (double)info.denom;
		return (end - start) * nano * 1e-9;
	#elif defined(LDP_OS_LNX)
		struct timeval elapsed;
		timersub(&start, &end, &elapsed);
		long sec = elapsed.tv_sec;
		long usec = elapsed.tv_usec;
		double t = sec + usec * 1e-6;
		return t >= 0 ? t : -t;
	#endif
	}

	// Return current time in seconds
	static inline double seconds()
	{
		gtime_t now_sd = gtime_now();
	#if defined(LDP_OS_WIN)
		gtime_t system_freq;
		QueryPerformanceFrequency(&system_freq);
		return (double)(now_sd.QuadPart) / system_freq.QuadPart;
	#elif defined(LDP_OS_MAC)
		// calculate platform timing epoch
		static mach_timebase_info_data_t info;
		mach_timebase_info(&info);
		double nano = (double)info.numer / (double)info.denom;
		return now_sd * nano * 1e-9;
	#elif defined(LDP_OS_LNX)
		struct timeval elapsed;
		struct timeval start = 0;
		timersub(&start, &now_sd, &elapsed);
		long sec = elapsed.tv_sec;
		long usec = elapsed.tv_usec;
		double t = sec + usec * 1e-6;
		return t >= 0 ? t : -t;
	#endif
	}

	static gtime_t __ldp_static_tic_recorder_;
	static inline void tic(){
		__ldp_static_tic_recorder_ = gtime_now();
	}
	static inline double toc(const char* label = "TimeCost", bool isInfoShow = true){
		double sc = gtime_seconds(__ldp_static_tic_recorder_, gtime_now());
		if (isInfoShow){
			printf("%s:\t%f\n", label, sc);
		}
		return sc;
	}
}//namespace ldp