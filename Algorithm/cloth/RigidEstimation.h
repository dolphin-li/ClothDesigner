#include <eigen\Dense>

namespace rigid_estimation
{
	typedef float real;
	typedef Eigen::Matrix<real, 4, 4, 0, 4, 4> Mat4;
	typedef Eigen::Matrix<real, 3, 3, 0, 3, 3> Mat3;
	typedef Eigen::Matrix<real, 3, 1, 0, 3, 1> Point;
	typedef Eigen::Matrix<real, 2, 1, 0, 2, 1> Point2;
	typedef Eigen::Quaternion<real> Quaternion;

	// estimate the rigid transformation given two sets of corresponding points:
	// \argmin{\sum_i{||R*src[i] + T - dst[i]||^2}}
	// that is, each src point, after a rotation and a translation, should be the dst point
	// NOTE: n >= 3 is required;
	//			to get stable results, n should be not too small
	void estimate_rigid(int n, const Point* src, const Point* dst, Mat3& R, Point& T);

	// estimate the rigid+uniform_scale transformation given two sets of corresponding points:
	// \argmin{\sum_i{||R*S*src[i] + T - dst[i]||^2}}
	// that is, each src point, after a rotation, a scaling and a translation, should be the dst point
	// NOTE: n >= 3 is required;
	//			to get stable results, n should be not too small
	void estimate_rigid_with_scale(int n, const Point* src, const Point* dst, Mat3& R, Point& T, real& S);

	// estimate the rigid+scale transformation given two sets of corresponding points:
	// \argmin{\sum_i{||R*S*src[i] + T - dst[i]||^2}}
	// that is, each src point, after a rotation, a scaling and a translation, should be the dst point
	// NOTE: n >= 3 is required;
	//			to get stable results, n should be not too small
	void estimate_rigid_with_scale(int n, const Point* src, const Point* dst, Mat3& R, Point& T, Point& S);
	
	// similar with xyz scale, but scale of x = y
	void estimate_rigid_with_scale_xz(int n, const Point* src, const Point* dst, Mat3& R, Point& T, Point2& Sxz);

	// a test function to debug the method above
	// also, look at this function to find the usage of the method above
	void debug_estimation();
}