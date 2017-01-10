#include "RigidEstimation.h"
namespace rigid_estimation
{
	void estimate_rigid(int n, const Point* src, const Point* dst, Mat3& R, Point& T)
	{
		// translate both src and dst to O
		Point mean_src = Point::Zero();
		Point mean_dst = Point::Zero();
		for (int i = 0; i < n; i++)
		{
			mean_src += src[i];
			mean_dst += dst[i];
		}
		mean_src /= n;
		mean_dst /= n;

		// compute the covariance matrix
		Mat3 H = Mat3::Zero();
		for (int i = 0; i < n; i++)
		{
			Point s = src[i] - mean_src;
			Point t = dst[i] - mean_dst;
			for (int y = 0; y < 3; y++)
			for (int x = 0; x < 3; x++)
				H(y, x) += s[y] * t[x];
		}

		// SVD decomposition on the covariance matrix
		Eigen::JacobiSVD<Mat3> svd = H.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
		svd.compute(H);
		Mat3 U = svd.matrixU();
		Mat3 V = svd.matrixV();
		Mat3 D = Mat3::Identity();
		D(2, 2) = (U*V.transpose()).determinant();

		// rotation
		R = V * D * U.transpose();

		// translation
		T = mean_dst - R * mean_src;
	}

	void estimate_rigid_with_scale(int n, const Point* src, const Point* dst, Mat3& R, Point& T, real& S)
	{
		// translate both src and dst to O
		Point mean_src = Point::Zero();
		Point mean_dst = Point::Zero();
		for (int i = 0; i < n; i++)
		{
			mean_src += src[i];
			mean_dst += dst[i];
		}
		mean_src /= n;
		mean_dst /= n;

		// compute the covariance matrix
		Mat3 H = Mat3::Zero();
		for (int i = 0; i < n; i++)
		{
			Point s = src[i] - mean_src;
			Point t = dst[i] - mean_dst;
			for (int y = 0; y < 3; y++)
			for (int x = 0; x < 3; x++)
				H(y, x) += s[y] * t[x];
		}

		// SVD decomposition on the covariance matrix
		Eigen::JacobiSVD<Mat3> svd = H.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
		svd.compute(H);
		Mat3 U = svd.matrixU();
		Mat3 V = svd.matrixV();
		Mat3 D = Mat3::Identity();
		D(2, 2) = (U*V.transpose()).determinant();

		// rotation
		R = V * D * U.transpose();

		// scaling
		real XtX = 0, YtRX = 0;
		for (int i = 0; i < n; i++)
		{
			Point s = src[i] - mean_src;
			Point t = dst[i] - mean_dst;
			XtX += s.dot(s);
			YtRX += t.dot(R*s);
		}
		S = YtRX / XtX;

		// translation
		T = mean_dst - S * R * mean_src;
	}

	void estimate_rigid_with_scale(int n, const Point* src, const Point* dst, Mat3& R, Point& T, Point& S)
	{
		// translate both src and dst to O
		Point mean_src = Point::Zero();
		Point mean_dst = Point::Zero();
		for (int i = 0; i < n; i++)
		{
			mean_src += src[i];
			mean_dst += dst[i];
		}
		mean_src /= n;
		mean_dst /= n;

		// iterative solving for S and R
		S.setOnes();
		Point lastS = S;
		for (int iter = 0; iter < 100; iter++)
		{
			// compute the covariance matrix
			Mat3 H = Mat3::Zero();
			for (int i = 0; i < n; i++)
			{
				Point s = S.cwiseProduct(src[i] - mean_src);
				Point t = dst[i] - mean_dst;
				for (int y = 0; y < 3; y++)
				for (int x = 0; x < 3; x++)
					H(y, x) += s[y] * t[x];
			}

			// SVD decomposition on the covariance matrix
			Eigen::JacobiSVD<Mat3> svd = H.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
			svd.compute(H);
			Mat3 U = svd.matrixU();
			Mat3 V = svd.matrixV();
			Mat3 D = Mat3::Identity();
			D(2, 2) = (U*V.transpose()).determinant();

			// rotation
			R = V * D * U.transpose();

			// scaling
			Point sumS = Point::Zero(), sumT = Point::Zero();
			for (int i = 0; i < n; i++)
			{
				Point s = src[i] - mean_src;
				Point t = dst[i] - mean_dst;
				for (int k = 0; k < 3; k++)
				{
					sumS[k] += s[k] * s[k];
					sumT[k] += t.dot(s[k] * R.col(k));
				}
			}
			S = sumT.cwiseQuotient(sumS);

			// termination
			real err = (lastS - S).norm();
			lastS = S;
			if (err < std::numeric_limits<real>::epsilon())
				break;
		}

		// translation
		T = mean_dst - R * S.cwiseProduct(mean_src);
	}

	void estimate_rigid_with_scale_xz(int n, const Point* src, const Point* dst, Mat3& R, Point& T, Point2& Sxy)
	{
		// translate both src and dst to O
		Point mean_src = Point::Zero();
		Point mean_dst = Point::Zero();
		for (int i = 0; i < n; i++)
		{
			mean_src += src[i];
			mean_dst += dst[i];
		}
		mean_src /= n;
		mean_dst /= n;

		// iterative solving for S and R
		Sxy.setOnes();
		Point2 lastSxy = Sxy;
		for (int iter = 0; iter < 100; iter++)
		{
			// compute the covariance matrix
			Mat3 H = Mat3::Zero();
			Point Sxyz(Sxy[0], Sxy[0], Sxy[1]);
			for (int i = 0; i < n; i++)
			{
				Point s = Sxyz.cwiseProduct(src[i] - mean_src);
				Point t = dst[i] - mean_dst;
				for (int y = 0; y < 3; y++)
				for (int x = 0; x < 3; x++)
					H(y, x) += s[y] * t[x];
			}

			// SVD decomposition on the covariance matrix
			Eigen::JacobiSVD<Mat3> svd = H.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
			svd.compute(H);
			Mat3 U = svd.matrixU();
			Mat3 V = svd.matrixV();
			Mat3 D = Mat3::Identity();
			D(2, 2) = (U*V.transpose()).determinant();

			// rotation
			R = V * D * U.transpose();

			// scaling
			Point2 sumS = Point2::Zero(), sumT = Point2::Zero();
			for (int i = 0; i < n; i++)
			{
				Point s = src[i] - mean_src;
				Point t = dst[i] - mean_dst;
				sumS[0] += s[0] * s[0] + s[1] * s[1];
				sumS[1] += s[2] * s[2];
				sumT[0] += t.dot(s[0] * R.col(0)) + t.dot(s[1] * R.col(1));
				sumT[1] += t.dot(s[2] * R.col(2));
			}
			Sxy = sumT.cwiseQuotient(sumS);

			// termination
			real err = (lastSxy - Sxy).norm();
			lastSxy = Sxy;
			if (err < std::numeric_limits<real>::epsilon())
				break;
		}

		// translation
		Point Sxyz(Sxy[0], Sxy[0], Sxy[1]);
		T = mean_dst - R * Sxyz.cwiseProduct(mean_src);
	}

	void debug_estimation()
	{
		const static int n = 10;
		
		Point src[n], dst[n];
		Mat3 ground_truth_R, estimated_R;
		Point ground_truth_T, estimated_T;
		Point ground_truth_S, estimated_S;
		Point2 ground_truth_Sxz, estimated_Sxz;

		// perform testing 10000 times
		for (int test_iter = 0; test_iter < 1000; test_iter++)
		{
			// 1. generate some random source points
			for (int i = 0; i < n; i++)
				src[i] = Point::Random();

			// 2. generate a random rotation and a random translation
			Quaternion q;
			q.setFromTwoVectors(Point::Ones(), Point::Random());
			q.normalize();
			ground_truth_R = q.toRotationMatrix();
			ground_truth_T = Point::Random();
			ground_truth_S = Point::Random().cwiseAbs();
			ground_truth_S[1] = ground_truth_S[0];
			ground_truth_Sxz = Point2(ground_truth_S[0], ground_truth_S[2]);
			estimated_S = ground_truth_S;
			estimated_Sxz = ground_truth_Sxz;

			// 3. transform the source points to dst points according to the given S, R and T
			for (int i = 0; i < n; i++)
				dst[i] = ground_truth_R * ground_truth_S.cwiseProduct(src[i]) + ground_truth_T;

			// 4. Estimate R and T by our method
			//estimate_rigid_with_scale(n, src, dst, estimated_R, estimated_T, estimated_S);
			estimate_rigid_with_scale_xz(n, src, dst, estimated_R, estimated_T, estimated_Sxz);

			// 5. check whether the estimated is OK
			real err_R = (ground_truth_R - estimated_R).norm();
			real err_T = (ground_truth_T - estimated_T).norm();
			real err_S = (ground_truth_S - estimated_S).norm() / ground_truth_S.norm();
			real err_Sxz = (ground_truth_Sxz - estimated_Sxz).norm() / ground_truth_Sxz.norm();
			if (err_R > 1e-4 || err_T > 1e-4 || err_S > 1e-4 || err_Sxz > 1e-4)
			{
				printf("error estimation %d: %f %f %f %f\n", test_iter, err_R, err_T, err_S, err_Sxz);
			}
		}// end for test_iter

	}
}