#include "SmplManager.h"
#include "matio\matio.h"
#include "renderable\ObjMesh.h"
#include <gl\glut.h>
#include "ldpmat\Quaternion.h"
#include "camera\camera.h"
#include "RigidEstimation.h"
#include "BFGS\BFGSFit.h"
#include "LMSolver.h"
#include "TinyXML\tinyxml.h"
#include <fstream>

#pragma region --mat_utils

#define CHECK_THROW_EXCPT(cond)\
if (!(cond)) { throw std::exception(("!!!!Exception: " + std::string(#cond)).c_str()); }

typedef SmplManager::real real;
typedef SmplManager::Face Face;
typedef SmplManager::DMat DMat;
typedef SmplManager::SpMat SpMat;
typedef SmplManager::DVec DVec;
typedef SmplManager::Vec2 Vec2;
typedef SmplManager::Vec3 Vec3;
typedef SmplManager::Vec4 Vec4;
typedef SmplManager::Mat2 Mat2;
typedef SmplManager::Mat3 Mat3;
typedef SmplManager::Mat4 Mat4;

static std::string load_string(mat_t* file, const char* var_name)
{
	matvar_t* matvar = Mat_VarRead(file, var_name);

	if (matvar == nullptr)
		return std::string();

	CHECK_THROW_EXCPT(matvar->class_type == MAT_C_CHAR && matvar->data_type == MAT_T_UINT8);

	if (matvar == nullptr)
		return std::string();

	return (const char*)matvar->data;
}

static void load_array(std::vector<Face>& out, mat_t* file, const char* var_name)
{
	out.clear();

	matvar_t* matvar = Mat_VarRead(file, var_name);

	CHECK_THROW_EXCPT(matvar->data_size == 4 && matvar->data_type == MAT_T_UINT32);
	CHECK_THROW_EXCPT(matvar->rank == 2);
	CHECK_THROW_EXCPT(matvar->dims[1] == 3);

	if (matvar == nullptr)
		return;

	out.resize(matvar->dims[0]);

	unsigned int* in_fx = (unsigned int*)matvar->data;
	unsigned int* in_fy = in_fx + matvar->dims[0];
	unsigned int* in_fz = in_fy + matvar->dims[0];

	for (size_t i = 0; i < out.size(); i++)
	{
		out[i][0] = in_fx[i];
		out[i][1] = in_fy[i];
		out[i][2] = in_fz[i];
	}
}

static void load_vert_sym_idx(std::vector<int>& out, mat_t* file, const char* var_name)
{
	out.clear();

	matvar_t* matvar = Mat_VarRead(file, var_name);

	CHECK_THROW_EXCPT(matvar->data_size == 8 && matvar->data_type == MAT_T_INT64);
	CHECK_THROW_EXCPT(matvar->rank == 2 && matvar->dims[0] == 1);

	if (matvar == nullptr)
		return;

	out.resize(matvar->dims[1]);

	__int64* in_f = (__int64*)matvar->data;

	for (size_t i = 0; i < out.size(); i++)
	{
		out[i] = in_f[i];
	}
}

static void load_kintree_table(std::vector<int>& out, mat_t* file, const char* var_name)
{
	out.clear();

	matvar_t* matvar = Mat_VarRead(file, var_name);

	CHECK_THROW_EXCPT(matvar->data_size == 4 && matvar->data_type == MAT_T_UINT32);
	CHECK_THROW_EXCPT(matvar->rank == 2);
	CHECK_THROW_EXCPT(matvar->dims[0] == 2);

	if (matvar == nullptr)
		return;

	out.resize(matvar->dims[1]);

	unsigned int* in_f = (unsigned int*)matvar->data;

	for (size_t i = 0; i < out.size(); i++)
	{
		out[i] = in_f[i * 2];
		CHECK_THROW_EXCPT(in_f[i * 2 + 1] == i);
	}
}

static void load_array(DMat& out, mat_t* file, const char* var_name)
{
	matvar_t* matvar = Mat_VarRead(file, var_name);

	CHECK_THROW_EXCPT(matvar->data_size == 8 && matvar->data_type == MAT_T_DOUBLE);

	if (matvar == nullptr)
		return;

	int cnt = matvar->dims[0];
	for (int i = 1; i < matvar->rank; i++)
		cnt *= matvar->dims[i];

	out.resize(cnt / matvar->dims[matvar->rank - 1], matvar->dims[matvar->rank - 1]);

	for (int x = 0; x < out.cols(); x++)
	{
		const double* in_f = (const double*)matvar->data + out.rows() * x;
		for (int y = 0; y < out.rows(); y++)
			out(y, x) = in_f[y];
	}
}

static void load_array(SpMat& out, mat_t* file, const char* var_name)
{
	matvar_t* matvar = Mat_VarRead(file, var_name);

	CHECK_THROW_EXCPT(matvar->data_type == MAT_T_DOUBLE);
	CHECK_THROW_EXCPT(matvar->rank == 2);

	if (matvar == nullptr)
		return;

	out.resize(matvar->dims[0], matvar->dims[1]);
	std::vector<Eigen::Triplet<real>> cooSys;

	if (matvar->class_type == MAT_C_SPARSE)
	{
		mat_sparse_t* spdata = (mat_sparse_t*)matvar->data;
		for (int x = 0; x < spdata->njc - 1; x++)
		{
			int cb = spdata->jc[x];
			int ce = spdata->jc[x + 1];
			for (int c = cb; c < ce; c++)
			{
				int y = spdata->ir[c];
				double v = ((const double*)spdata->data)[c];
				cooSys.push_back(Eigen::Triplet<real>(y, x, v));
			}
		}
	}
	else
	{
		for (int x = 0; x < out.cols(); x++)
		{
			const double* in_f = (const double*)matvar->data + out.rows() * x;
			for (int y = 0; y < out.rows(); y++)
			{
				if (in_f[y])
					cooSys.push_back(Eigen::Triplet<real>(y, x, in_f[y]));
			}
		}
	}

	if (cooSys.size())
		out.setFromTriplets(cooSys.begin(), cooSys.end());
}

inline Mat3 convert(ldp::Mat3d A)
{
	Mat3 B;
	for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		B(i, j) = A(i, j);
	return B;
}

inline ldp::Mat3d convert(Mat3 A)
{
	ldp::Mat3d B;
	for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		B(i, j) = A(i, j);
	return B;
}

inline Vec3 convert(ldp::Double3 v)
{
	return Vec3(v[0], v[1], v[2]);
}

inline ldp::Double3 convert(Vec3 v)
{
	return ldp::Double3(v[0], v[1], v[2]);
}

inline Mat3 angles2rot(Vec3 v)
{
	real theta = v.norm();
	if (theta == 0)
		return Mat3::Identity();
	v /= theta;
	return convert(ldp::QuaternionD().fromAngleAxis(theta, ldp::Double3(v[0], v[1], v[2])).toRotationMatrix3());
}

inline Vec3 rot2angles(Mat3 R)
{
	ldp::Mat3d A = convert(R);
	ldp::QuaternionD q;
	q.fromRotationMatrix(A);
	ldp::Double3 v;
	double ag = 0;
	q.toAngleAxis(v, ag);
	v *= ag;
	return Vec3(v[0], v[1], v[2]);
}

static GLUquadric* get_quadric()
{
	static GLUquadric* q = gluNewQuadric();
	return q;
}

static ldp::Mat4f get_z2x_rot()
{
	static ldp::Mat4f R = ldp::QuaternionF().fromRotationVecs(ldp::Float3(0, 0, 1),
		ldp::Float3(1, 0, 0)).toRotationMatrix();
	return R;
}

static ldp::Mat4f get_z2y_rot()
{
	static ldp::Mat4f R = ldp::QuaternionF().fromRotationVecs(ldp::Float3(0, 0, 1),
		ldp::Float3(0, 1, 0)).toRotationMatrix();
	return R;
}

static void solid_axis(float base, float length)
{
	GLUquadric* q = get_quadric();
	gluCylinder(q, base, base, length, 32, 32);
	glTranslatef(0, 0, length);
	gluCylinder(q, base*2.5f, 0.f, length* 0.2f, 32, 32);
	glTranslatef(0, 0, -length);
}

inline int colorToSelectId(ldp::Float4 c)
{
	ldp::UInt4 cl = c*255.f;
	return (cl[0] << 24) + (cl[1] << 16) + (cl[2] << 8) + cl[3];
}

inline ldp::Float4 selectIdToColor(unsigned int id)
{
	int r = (id >> 24) & 0xff;
	int g = (id >> 16) & 0xff;
	int b = (id >> 8) & 0xff;
	int a = id & 0xff;
	return ldp::Float4(r, g, b, a) / 255.f;
}

const float g_node_size = 0.01f;
const float g_skel_size = 0.003f;
const float g_axis_size = 0.001f;
const float g_axis_length = 0.05f;

#pragma endregion


typedef SmplManager::DMat DMat;

#pragma region --JointRotSolver
class JointRotSolver : public CDenseLMSolver
{
public:
	const real m_reg_weight_0 = 1e0;
public:
	void init(SmplManager* smpl)
	{
		m_smpl = smpl;

		// apply shape vec
		m_v_shaped = m_smpl->m_shapedirs * m_smpl->m_curShapes;
		m_v_shaped.resize(m_smpl->m_v_template.rows(), m_smpl->m_v_template.cols());
		m_v_shaped += m_smpl->m_v_template;
		m_joints_pos_0 = m_smpl->m_J_regressor * m_v_shaped;

		M = m_smpl->numPoses() * 3 + m_smpl->numPoses() * m_smpl->numVarEachPose(); // +reg term
		N = m_smpl->numPoses() * m_smpl->numVarEachPose() + 3; // + rigid translation
	}

	void solve(const std::vector<Vec3>& targetJoints)
	{
		m_targetJoints = targetJoints;

		m_jointW.clear();
		m_jointW.resize(m_smpl->numPoses(), 1);

		DVec X(m_smpl->m_curPoses.rows() * m_smpl->m_curPoses.cols() + 3);
		X.setZero();
		for (int r = 0; r < m_smpl->m_curPoses.rows(); r++)
		for (int c = 0; c < m_smpl->m_curPoses.cols(); c++)
			X[r*m_smpl->m_curPoses.cols() + c] = m_smpl->m_curPoses(r, c);
		Optimize(X, 100, false);
		for (int r = 0; r < m_smpl->m_curPoses.rows(); r++)
		for (int c = 0; c < m_smpl->m_curPoses.cols(); c++)
			m_smpl->m_curPoses(r, c) = X[r*m_smpl->m_curPoses.cols() + c];
		m_smpl->calcGlobalTrans();
	}

	void solveOneNode(int iJoint, Vec3 target)
	{
		m_enable_rigid_t = false;
		m_jointW.clear();
		m_jointW.resize(m_smpl->numPoses(), 0);

		m_targetJoints.resize(m_smpl->numPoses());
		for (int i = 0; i < m_smpl->numPoses(); i++)
			m_targetJoints[i] = m_smpl->getCurNodeCenter(i);
		m_targetJoints[iJoint] = target;
		m_jointW[iJoint] = 1e8;

		DVec X(m_smpl->m_curPoses.rows() * m_smpl->m_curPoses.cols() + 3);
		X.setZero();
		for (int r = 0; r < m_smpl->m_curPoses.rows(); r++)
		for (int c = 0; c < m_smpl->m_curPoses.cols(); c++)
			X[r*m_smpl->m_curPoses.cols() + c] = m_smpl->m_curPoses(r, c);
		Optimize(X, 100, false);
		for (int r = 0; r < m_smpl->m_curPoses.rows(); r++)
		for (int c = 0; c < m_smpl->m_curPoses.cols(); c++)
			m_smpl->m_curPoses(r, c) = X[r*m_smpl->m_curPoses.cols() + c];
		m_smpl->calcGlobalTrans();
	}

protected:
	virtual real CalcEnergyFunc(const DVec& x, DVec& fx)
	{
		const int nJoints = m_smpl->numPoses();
		const int nVarEachJoint = m_smpl->numVarEachPose();

		// compute rotations
		calcGlobalTrans(x);

		Vec3 rigid_T(x[x.size() - 3], x[x.size() - 2], x[x.size() - 1]);
		if (!m_enable_rigid_t)
			rigid_T = Vec3::Zero();

		// data term
		for (int iJoint = 0; iJoint < nJoints; iJoint++)
		{
			Vec3 v(m_joints_pos_0(iJoint, 0), m_joints_pos_0(iJoint, 1), m_joints_pos_0(iJoint, 2));
			Mat3 R = Mat3::Identity();
			if (m_smpl->m_kintree_table[iJoint] >= 0)
				R = m_jointsR[m_smpl->m_kintree_table[iJoint]];
			Vec3 v_transformed = R * v + m_jointsT[iJoint] + rigid_T;
			for (int k = 0; k < 3; k++)
				fx[iJoint * 3 + k] = m_jointW[iJoint] * (v_transformed[k] - m_targetJoints[iJoint][k]);
		} // iVerts

		// reg term
		const float w0 = m_reg_weight_0;
		for (int iReg = 0; iReg < nJoints * nVarEachJoint; iReg++)
			fx[nJoints * 3 + iReg] = x[iReg] * w0;

		return fx.dot(fx);
	}

	void calcGlobalTrans(const DVec& jointsPoses)
	{
		m_jointsR.resize(m_smpl->m_curJ.rows());
		m_jointsT.resize(m_smpl->m_curJ.rows());
		for (size_t iJoints = 0; iJoints < m_smpl->m_curJ.rows(); iJoints++)
		{
			Vec3 r(jointsPoses[iJoints * 3], jointsPoses[iJoints * 3 + 1], jointsPoses[iJoints * 3 + 2]);

			Mat3 R = angles2rot(r);
			Vec3 v(m_joints_pos_0(iJoints, 0), m_joints_pos_0(iJoints, 1), m_joints_pos_0(iJoints, 2));
			int iParent = m_smpl->m_kintree_table[iJoints];
			if (iParent < 0)
			{
				m_jointsR[iJoints] = R;
				m_jointsT[iJoints] = v - m_jointsR[iJoints] * v;
			}
			else
			{
				m_jointsR[iJoints] = m_jointsR[iParent] * R;
				auto v1 = m_jointsR[iParent] * v + m_jointsT[iParent];
				m_jointsT[iJoints] = v1 - m_jointsR[iJoints] * v;
			}
		} // iJoints
	}
private:
	SmplManager* m_smpl = nullptr;
	DMat m_v_shaped, m_poseVec;
	DMat m_joints_pos_0;
	std::vector<Vec3> m_targetJoints;
	std::vector<Mat3> m_jointsR;
	std::vector<Vec3> m_jointsT;
	std::vector<real> m_jointW;
	bool m_enable_rigid_t = true;
};
#pragma endregion

#pragma region --PoseSolver
class PoseSolver : public CDenseLMSolver
{
public:
	const float m_reg_weight_0 = 1e-3f;
public:
	void init(SmplManager* smpl, std::vector<ldp::Float3>& targetVerts)
	{
		m_smpl = smpl;
		m_targetVerts = &targetVerts;

		// apply shape vec
		m_v_shaped = m_smpl->m_shapedirs * m_smpl->m_curShapes;
		m_v_shaped.resize(m_smpl->m_v_template.rows(), m_smpl->m_v_template.cols());
		m_v_shaped += m_smpl->m_v_template;
		m_smpl->m_curJ = m_smpl->m_J_regressor * m_v_shaped;

		// apply pose vec
		m_smpl->calcPoseVector207(m_smpl->m_curPoses, m_poseVec);
		m_v_posed = m_smpl->m_posedirs * m_poseVec;
		m_v_posed.resize(m_smpl->m_v_template.rows(), m_smpl->m_v_template.cols());
		m_v_posed += m_v_shaped;

		M = m_v_posed.rows() * 3 + m_smpl->numPoses() * m_smpl->numVarEachPose(); // +reg term
		N = m_smpl->numPoses() * m_smpl->numVarEachPose() + 3 + 1; // + rigid translation + scale
	}

	void solve(int nMaxIter, bool showInfo)
	{
		DVec X(N);
		X.setZero();
		for (int r = 0; r < m_smpl->m_curPoses.rows(); r++)
		for (int c = 0; c < m_smpl->m_curPoses.cols(); c++)
			X[r*m_smpl->m_curPoses.cols() + c] = m_smpl->m_curPoses(r, c);
		X[X.size() - 1] = 1;
		Optimize(X, nMaxIter, showInfo);
		for (int r = 0; r < m_smpl->m_curPoses.rows(); r++)
		for (int c = 0; c < m_smpl->m_curPoses.cols(); c++)
			m_smpl->m_curPoses(r, c) = X[r*m_smpl->m_curPoses.cols() + c];

		Float3 rigid_T(X[X.size() - 4], X[X.size() - 3], X[X.size() - 2]);
		float rigid_S = X[X.size() - 1];
		for (auto& v : *m_targetVerts)
			v = (v - rigid_T) / rigid_S;
	}

protected:
	virtual real CalcEnergyFunc(const DVec& x, DVec& fx)
	{
		const int nVerts = m_v_posed.rows();
		const int nJoints = m_smpl->numPoses();
		const int nVarEachJoint = m_smpl->numVarEachPose();

		// compute rotations
		calcGlobalTrans(x);
		computeVertexRT();

		Vec3 rigid_T(x[x.size() - 4], x[x.size() - 3], x[x.size() - 2]);
		real rigid_S = x[x.size() - 1];

		// data term
		for (int iVerts = 0; iVerts < nVerts; iVerts++)
		{
			Vec3 v(m_v_posed(iVerts, 0), m_v_posed(iVerts, 1), m_v_posed(iVerts, 2));
			Vec3 v_transformed = rigid_S * (m_vertsR[iVerts] * v + m_vertsT[iVerts]) + rigid_T;
			for (int k = 0; k < 3; k++)
				fx[iVerts * 3 + k] = v_transformed[k] - (*m_targetVerts)[iVerts][k];
		} // iVerts

		// reg term
		const float w0 = m_reg_weight_0 * nVerts / nJoints;
		for (int iReg = 0; iReg < nJoints * nVarEachJoint; iReg++)
			fx[nVerts * 3 + iReg] = x[iReg] * w0;

		return fx.dot(fx);
	}

	void calcGlobalTrans(const DVec& jointsPoses)
	{
		m_jointsR.resize(m_smpl->m_curJ.rows());
		m_jointsT.resize(m_smpl->m_curJ.rows());
		for (size_t iJoints = 0; iJoints < m_smpl->m_curJ.rows(); iJoints++)
		{
			Vec3 r(jointsPoses[iJoints * 3], jointsPoses[iJoints * 3 + 1], jointsPoses[iJoints * 3 + 2]);

			Mat3 R = angles2rot(r);
			Vec3 v(m_smpl->m_curJ(iJoints, 0), m_smpl->m_curJ(iJoints, 1), m_smpl->m_curJ(iJoints, 2));
			int iParent = m_smpl->m_kintree_table[iJoints];
			if (iParent < 0)
			{
				m_jointsR[iJoints] = R;
				m_jointsT[iJoints] = v - m_jointsR[iJoints] * v;
			}
			else
			{
				m_jointsR[iJoints] = m_jointsR[iParent] * R;
				auto v1 = m_jointsR[iParent] * v + m_jointsT[iParent];
				m_jointsT[iJoints] = v1 - m_jointsR[iJoints] * v;
			}
		} // iJoints
	}

	void computeVertexRT()
	{
		// compute verts R and T
		m_vertsR.resize(m_v_posed.rows());
		m_vertsT.resize(m_v_posed.rows());

		for (int iVerts = 0; iVerts < (int)m_vertsR.size(); iVerts++)
		{
			int wb = m_smpl->m_weights.outerIndexPtr()[iVerts];
			int we = m_smpl->m_weights.outerIndexPtr()[iVerts + 1];
			Vec3 sumT = Vec3::Zero();
			Mat3 sumR = Mat3::Zero();
			for (int iw = wb; iw < we; iw++)
			{
				int jointId = m_smpl->m_weights.innerIndexPtr()[iw];
				real jointW = m_smpl->m_weights.valuePtr()[iw];
				sumR += jointW * m_jointsR[jointId];
				sumT += jointW * m_jointsT[jointId];
			}
			m_vertsR[iVerts] = sumR;
			m_vertsT[iVerts] = sumT;
		} // iVerts
	}
private:
	SmplManager* m_smpl = nullptr;
	DMat m_v_shaped, m_v_posed, m_poseVec;
	std::vector<ldp::Float3>* m_targetVerts = nullptr;
	std::vector<Mat3> m_vertsR, m_jointsR;
	std::vector<Vec3> m_vertsT, m_jointsT;
};
#pragma endregion

#pragma region --ShapeSolver
class ShapeSolver : public Program
{
public:
	// Constructor, initialize
	ShapeSolver(int n, double* x, double* lb, double* ub, long int* btype,
		int m = defaultm, int maxiter = defaultmaxiter,
		double factr = defaultfactr, double pgtol = defaultpgtol)
		: Program(n, x, lb, ub, btype, defaultm, maxiter, factr, pgtol)
	{

	}
	void initialize(SmplManager* smpl, const std::vector<ldp::Float3>& tartgetVerts)
	{
		m_smpl = smpl;
		m_tarV.resize(tartgetVerts.size(), 3);
		for (size_t i = 0; i < tartgetVerts.size(); i++)
		{
			auto& v = tartgetVerts[i];
			m_tarV(i, 0) = v[0];
			m_tarV(i, 1) = v[1];
			m_tarV(i, 2) = v[2];
		} // end for v

		// compute verts R and T
		m_vertsR.resize(m_tarV.rows());
		m_vertsT.resize(m_tarV.rows());
		for (int iVerts = 0; iVerts < (int)m_vertsR.size(); iVerts++)
		{
			int wb = m_smpl->m_weights.outerIndexPtr()[iVerts];
			int we = m_smpl->m_weights.outerIndexPtr()[iVerts + 1];
			Vec3 sumT = Vec3::Zero();
			Mat3 sumR = Mat3::Zero();
			for (int iw = wb; iw < we; iw++)
			{
				int jointId = m_smpl->m_weights.innerIndexPtr()[iw];
				real jointW = m_smpl->m_weights.valuePtr()[iw];
				sumR += jointW * m_smpl->m_curJrots[jointId];
				sumT += jointW * m_smpl->m_curJtrans[jointId];
			}
			m_vertsR[iVerts] = sumR;
			m_vertsT[iVerts] = sumT;
		} // iVerts
	}
protected:
	virtual double computeObjective(int n, double* x)
	{
		Eigen::Map<DMat> X(x, n, 1);

		// apply shape coef
		v_shaped = m_smpl->m_shapedirs * X;
		v_shaped.resize(m_tarV.rows(), m_tarV.cols());
		v_shaped += m_smpl->m_v_template;

		// apply joints
		v_posed = v_shaped;
		for (int iVerts = 0; iVerts < v_shaped.rows(); iVerts++)
		{
			Vec3 v(v_shaped(iVerts, 0), v_shaped(iVerts, 1), v_shaped(iVerts, 2));
			Vec3 v_transformed = m_vertsR[iVerts] * v + m_vertsT[iVerts];
			v_posed(iVerts, 0) = v_transformed[0];
			v_posed(iVerts, 1) = v_transformed[1];
			v_posed(iVerts, 2) = v_transformed[2];
		} // iVerts
		v_shaped = v_posed;

		// compute the dif
		v_shaped -= m_tarV;
		return (v_shaped.transpose() * v_shaped).squaredNorm();
	}

	virtual void computeGradient(int n, double* x, double* g)
	{
		Eigen::Map<DMat> X(x, n, 1);
		Eigen::Map<DMat> G(g, n, 1);

		// apply shape coef
		v_shaped = m_smpl->m_shapedirs * X;
		v_shaped.resize(m_tarV.rows(), m_tarV.cols());
		v_shaped += m_smpl->m_v_template;

		// apply joints
		v_posed = v_shaped;
		for (int iVerts = 0; iVerts < v_shaped.rows(); iVerts++)
		{
			Vec3 v(v_shaped(iVerts, 0), v_shaped(iVerts, 1), v_shaped(iVerts, 2));
			Vec3 v_transformed = m_vertsR[iVerts] * v + m_vertsT[iVerts];
			v_posed(iVerts, 0) = v_transformed[0];
			v_posed(iVerts, 1) = v_transformed[1];
			v_posed(iVerts, 2) = v_transformed[2];
		} // iVerts
		v_shaped = v_posed;

		// compute the dif
		v_shaped -= m_tarV;

		// reverse apply the joints
		v_posed = v_shaped;
		for (int iVerts = 0; iVerts < v_shaped.rows(); iVerts++)
		{
			Vec3 v(v_shaped(iVerts, 0), v_shaped(iVerts, 1), v_shaped(iVerts, 2));
			Vec3 v_transformed = m_vertsR[iVerts].transpose() * v;
			v_posed(iVerts, 0) = v_transformed[0];
			v_posed(iVerts, 1) = v_transformed[1];
			v_posed(iVerts, 2) = v_transformed[2];
		} // iVerts
		v_shaped = v_posed;

		// reverse apply the shapes
		v_shaped.resize(v_shaped.size(), 1);
		G = m_smpl->m_shapedirs.transpose() * v_shaped;
	}

	virtual void iterCallback(int t, double* x, double f)
	{

	}
protected:
	SmplManager* m_smpl = nullptr;
	DMat m_tarV;
	DMat v_shaped, v_posed;
	std::vector<Mat3> m_vertsR;
	std::vector<Vec3> m_vertsT;
};
#pragma endregion


SmplManager::SmplManager()
{
	m_bsStyle = BsStyle::LBS;
	m_bsType = BsType::LROTMIN;
	m_inited = false;
	m_selectedNode = 0;
	m_selectedNode_mouseMove = 0;
	m_selectIdStart = 0;
	m_selectedNodeAxis = -1;
	m_selectedNodeAxis_mouseMove = -1;
	m_axisRenderMode = AxisTrans;
	m_select_mousePressed = false;
	m_renderCam = nullptr;
	m_jointRotSolver.reset(new JointRotSolver());
}

SmplManager::~SmplManager()
{}

void SmplManager::clear()
{
	m_bsStyle = BsStyle::LBS;
	m_bsType = BsType::LROTMIN;
	m_inited = false;
	m_faces.clear();
	m_J.resize(0, 0);
	m_J_regressor.resize(0, 0);
	m_J_regressor_prior.resize(0, 0);
	m_kintree_table.clear();
	m_posedirs.resize(0, 0);
	m_shapedirs.resize(0, 0);
	m_v_template.resize(0, 0);
	m_vert_sym_idxs.clear();
	m_weights.resize(0, 0);
	m_weights_prior.resize(0, 0);

	m_curPoses.resize(0, 0);
	m_curShapes.resize(0, 0);
	m_curJ.resize(0, 0);
	m_curJrots.clear();
	m_curJtrans.clear();
	m_curVerts.clear();
	m_curFNormals.clear();
	m_curVNormals.clear();
	m_bbox[0] = 0;
	m_bbox[1] = 0;
	m_selectedNode = 0;
	m_selectedNode_mouseMove = 0;
	m_selectedNodeAxis = -1;
	m_selectedNodeAxis_mouseMove = -1;
	m_selectIdStart = 0;
	m_axisRenderMode = AxisTrans;
	m_select_mousePressed = false;
	m_renderCam = nullptr;
}

void SmplManager::loadFromMat(const char* filename)
{
	clear();

	mat_t* file = Mat_Open(filename, MAT_ACC_RDONLY);
	if (file == nullptr)
		throw std::exception(("loadFromMat failed:" + std::string(filename)).c_str());
	printf("loaded file: %s\n", filename);

	mat_ft version = Mat_GetVersion(file);
	printf("loaded version: %x\n", version);

	// bsstyle
	std::string bs_style = load_string(file, "bs_style");
	if (!bs_style.empty())
	{
		if (bs_style == "lbs")
			m_bsStyle = BsStyle::LBS;
		else if (bs_style == "dqbs")
			m_bsStyle = BsStyle::DQBS;
		else
			throw std::exception(("non supported bs_style" + bs_style).c_str());
	}

	// bstype
	std::string bs_type = load_string(file, "bs_type");
	if (!bs_type.empty())
	{
		if (bs_type == "lrotmin")
			m_bsType = BsType::LROTMIN;
		else
			throw std::exception(("non supported bs_type" + bs_type).c_str());
	}

	// faces
	load_array(m_faces, file, "f");

	// Joint positions
	load_array(m_J, file, "J");
	m_curPoses.resize(m_J.rows(), 3);
	m_curPoses.setZero();

	// Joint regressor
	load_array(m_J_regressor, file, "J_regressor");
	load_array(m_J_regressor_prior, file, "J_regressor_prior");

	// Kintree, joints hierachy
	load_kintree_table(m_kintree_table, file, "kintree_table");

	// pose dirs
	load_array(m_posedirs, file, "posedirs");

	// shape dirs
	load_array(m_shapedirs, file, "shapedirs");
	m_curShapes.resize(m_shapedirs.cols(), 1);
	m_curShapes.setZero();

	// mean shape
	load_array(m_v_template, file, "v_template");
	m_curVerts.resize(m_v_template.rows());
	for (size_t i = 0; i < m_curVerts.size(); i++)
	for (int k = 0; k < 3; k++)
		m_curVerts[i][k] = m_v_template(i, k);

	// symmetry
	load_vert_sym_idx(m_vert_sym_idxs, file, "vert_sym_idxs");

	// weights
	load_array(m_weights, file, "weights");
	load_array(m_weights_prior, file, "weights_prior");
	m_weights = m_weights.transpose();
	m_weights_prior = m_weights_prior.transpose();

	// close
	Mat_Close(file);
	m_inited = true;

	updateCurMesh();
}

void SmplManager::updateCurMesh()
{
	// 1. update vertices-------------------------------------------------

	// apply shape vec
	DMat v_shaped = m_shapedirs * m_curShapes;
	v_shaped.resize(m_v_template.rows(), m_v_template.cols());
	v_shaped += m_v_template;
	m_curJ = m_J_regressor * v_shaped;

	// apply pose vec
	DMat v_posed, poseVec;
	calcPoseVector207(m_curPoses, poseVec);
	v_posed = m_posedirs * poseVec;
	v_posed.resize(m_v_template.rows(), m_v_template.cols());
	v_posed += v_shaped;

	// updating to the vertex buffer
	m_curVerts.resize(m_v_template.rows());
	for (size_t i = 0; i < m_curVerts.size(); i++)
	for (int k = 0; k < 3; k++)
		m_curVerts[i][k] = v_posed(i, k);

	// apply joint rotations
	calcGlobalTrans();
#pragma omp parallel for
	for (int iVerts = 0; iVerts < (int)m_curVerts.size(); iVerts++)
	{
		Vec3 v = m_curVerts[iVerts];
		int wb = m_weights.outerIndexPtr()[iVerts];
		int we = m_weights.outerIndexPtr()[iVerts + 1];
		Vec3 v_transformed = Vec3::Zero();
		for (int iw = wb; iw < we; iw++)
		{
			int jointId = m_weights.innerIndexPtr()[iw];
			real jointW = m_weights.valuePtr()[iw];
			v_transformed += jointW * (m_curJrots[jointId] * v + m_curJtrans[jointId]);
		}
		m_curVerts[iVerts] = v_transformed;
	} // iVerts

	// 2. update normals and bounds ------------------------------------
	m_curFNormals.resize(m_faces.size());
	m_curVNormals.resize(m_curVerts.size());
	std::fill(m_curVNormals.begin(), m_curVNormals.end(), Vec3::Zero());
	m_bbox[0] = FLT_MAX;
	m_bbox[1] = FLT_MIN;
	for (size_t i = 0; i < m_faces.size(); i++)
	{
		const Face &f = m_faces[i];
		Vec3 nm = (m_curVerts[f[1]] - m_curVerts[f[0]]).cross(m_curVerts[f[2]] - m_curVerts[f[0]]);
		for (int k = 0; k < 3; k++)
			m_curVNormals[f[k]] += nm;
		m_curFNormals[i] = nm.normalized();
	}
	for (size_t i = 0; i < m_curVNormals.size(); i++)
	{
		m_curVNormals[i].normalize();
		for (int k = 0; k < 3; k++)
		{
			m_bbox[0][k] = std::min(m_bbox[0][k], (float)m_curVerts[i][k]);
			m_bbox[1][k] = std::max(m_bbox[1][k], (float)m_curVerts[i][k]);
		}
	} // i
}

void SmplManager::setMaxShapeCoef(real c)
{
	m_maxShapeCoef = c;
	m_minShapeCoef = std::min(m_minShapeCoef, c);
	if (!m_inited)
		return;
	for (int i = 0; i < m_curShapes.rows(); i++)
		m_curShapes(i, 0) = std::min(m_maxShapeCoef, std::max(m_minShapeCoef, m_curShapes(i, 0)));
	updateCurMesh();
}

void SmplManager::setMinShapeCoef(real c)
{
	m_minShapeCoef = c;
	m_maxShapeCoef = std::max(m_maxShapeCoef, c);
	if (!m_inited)
		return;
	for (int i = 0; i < m_curShapes.rows(); i++)
		m_curShapes(i, 0) = std::min(m_maxShapeCoef, std::max(m_minShapeCoef, m_curShapes(i, 0)));
	updateCurMesh();
}

void  SmplManager::rigidFitting(std::vector<ldp::Float3>& newVertices)
{
	if (newVertices.size() != m_curVerts.size())
		throw std::exception("rigidFitting: given vertices size not matched!");

	using rigid_estimation::Point;
	rigid_estimation::Mat3 rigid_R = rigid_estimation::Mat3::Identity();
	Point rigid_T = Point::Zero();
	rigid_estimation::real rigid_S = 1;
	std::vector<Point> newV(m_curVerts.size()), curV(m_curVerts.size());
	for (size_t i = 0; i < m_curVerts.size(); i++)
	{
		for (int k = 0; k < 3; k++)
		{
			newV[i][k] = newVertices[i][k];
			curV[i][k] = m_curVerts[i][k];
		}
	}
	rigid_estimation::estimate_rigid_with_scale((int)newV.size(), newV.data(), curV.data(), rigid_R, rigid_T, rigid_S);
	for (size_t i = 0; i < newVertices.size(); i++)
	{
		auto& v = newVertices[i];
		Point p(v[0], v[1], v[2]);
		p = rigid_R * rigid_S * p + rigid_T;
		v = ldp::Float3(p[0], p[1], p[2]);
	} // end for v
}

void SmplManager::fittingShapes(const std::vector<ldp::Float3>& newVertices, bool showInfo)
{
	if (newVertices.size() != m_curVerts.size())
		throw std::exception("fittingShapes: given vertices size not matched!");

	// solve the fitting problem
	std::vector<real> minb(m_curShapes.size(), m_minShapeCoef), maxb(m_curShapes.size(), m_maxShapeCoef);
	std::vector<long> btype(m_curShapes.size(), 2);
	ShapeSolver bfit(m_curShapes.size(), m_curShapes.data(), minb.data(), maxb.data(), btype.data());
	bfit.initialize(this, newVertices);
	auto status = bfit.runSolver();
	if (status != SolverExitStatus::success)
		printf("warning: bfgs solver not converged %d\n", status);
}

void SmplManager::fittingPoses(std::vector<ldp::Float3>& newVertices, bool showInfo)
{
	if (newVertices.size() != m_curVerts.size())
		throw std::exception("fittingPoses: given vertices size not matched!");

	PoseSolver solver;

	solver.init(this, newVertices);
	solver.solve(50, showInfo);
}

void SmplManager::fittingShapePoses(std::vector<ldp::Float3>& newVertices, bool showInfo)
{
	std::vector<real> minb(m_curShapes.size(), m_minShapeCoef), maxb(m_curShapes.size(), m_maxShapeCoef);
	std::vector<long> btype(m_curShapes.size(), 2);

	PoseSolver poseSolver;
	ShapeSolver shapeSolver(m_curShapes.size(), m_curShapes.data(), minb.data(), maxb.data(), btype.data());

	DMat oldPose, oldShape;
	for (int iter = 0; iter < 10; iter++)
	{
		// fitting poses
		oldPose = m_curPoses;
		poseSolver.init(this, newVertices);
		poseSolver.solve(20 + iter * 5, showInfo);
		updateCurMesh();

		// fitting shapes
		oldShape = m_curShapes;
		shapeSolver.initialize(this, newVertices);
		auto status = shapeSolver.runSolver();
		if (status != SolverExitStatus::success)
			printf("warning: bfgs solver not converged %d\n", status);

		// compute the diff
		double poseDif = (oldPose - m_curPoses).norm() / (1e-5 + m_curPoses.norm());
		double shapeDif = (oldShape - m_curShapes).norm() / (1e-5 + m_curShapes.norm());
		if (showInfo)
			printf("iter %d, poseDif %f, shapeDif %f\n", iter, poseDif, shapeDif);
		if (poseDif < 1e-5 && shapeDif < 1e-5)
			break;
	} // end for iter
}

void SmplManager::calcGlobalTrans()
{
	m_curJrots.resize(m_curJ.rows());
	m_curJtrans.resize(m_curJ.rows());
	for (size_t iJoints = 0; iJoints < m_curJ.rows(); iJoints++)
	{
		Vec3 r(m_curPoses(iJoints, 0), m_curPoses(iJoints, 1), m_curPoses(iJoints, 2));
		Mat3 R = angles2rot(r);
		Vec3 v(m_curJ(iJoints, 0), m_curJ(iJoints, 1), m_curJ(iJoints, 2));
		int iParent = m_kintree_table[iJoints];
		if (iParent < 0)
		{
			m_curJrots[iJoints] = R;
			m_curJtrans[iJoints] = v - m_curJrots[iJoints] * v;
		}
		else
		{
			m_curJrots[iJoints] = m_curJrots[iParent] * R;
			auto v1 = m_curJrots[iParent] * v + m_curJtrans[iParent];
			m_curJtrans[iJoints] = v1 - m_curJrots[iJoints] * v;
		}
	} // iJoints
}

void SmplManager::saveShapeCoeffs(std::string filename)const
{
	if (!m_inited)
		return;
	FILE* pFile = fopen(filename.c_str(), "w");
	if (!pFile)
		throw std::exception(("io error: " + filename).c_str());

	for (int i = 0; i < m_curShapes.rows(); i++)
	{
		for (int j = 0; j < m_curShapes.cols(); j++)
			fprintf(pFile, "%f ", m_curShapes(i, j));
		fprintf(pFile, "\n");
	}

	fclose(pFile);
}

void SmplManager::loadShapeCoeffs(std::string filename)
{
	if (!m_inited)
		return;
	std::ifstream stm(filename);
	if (stm.fail())
		throw std::exception(("io error: " + filename).c_str());
	for (int i = 0; i < m_curShapes.rows(); i++)
	for (int j = 0; j < m_curShapes.cols(); j++)
		stm >> m_curShapes(i, j);
	stm.close();
	updateCurMesh();
}

void SmplManager::savePoseCoeffs(std::string filename)const
{
	if (!m_inited)
		return;
	FILE* pFile = fopen(filename.c_str(), "w");
	if (!pFile)
		throw std::exception(("io error: " + filename).c_str());

	for (int i = 0; i < m_curPoses.rows(); i++)
	{
		for (int j = 0; j < m_curPoses.cols(); j++)
			fprintf(pFile, "%f ", m_curPoses(i, j));
		fprintf(pFile, "\n");
	}

	fclose(pFile);
}

void SmplManager::loadPoseCoeffs(std::string filename)
{
	if (!m_inited)
		return;
	std::ifstream stm(filename);
	if (stm.fail())
		throw std::exception(("io error: " + filename).c_str());
	for (int i = 0; i < m_curPoses.rows(); i++)
	for (int j = 0; j < m_curPoses.cols(); j++)
		stm >> m_curPoses(i, j);
	stm.close();
	updateCurMesh();
}

void SmplManager::saveCoeffsToXml(TiXmlElement* ele, bool saveShape, bool savePose)const
{
	if (!m_inited)
		return;

	if (saveShape)
	{
		TiXmlElement* sEle = new TiXmlElement("shape");
		ele->LinkEndChild(sEle);
		std::string str;
		for (int i = 0; i < m_curShapes.size(); i++)
		{
			str += std::to_string(m_curShapes(i, 0));
			if (i + 1 < m_curShapes.size())
				str += " ";
		}
		sEle->SetAttribute("value", str.c_str());
	} // end if saveShape

	if (savePose)
	{
		TiXmlElement* sEle = new TiXmlElement("pose");
		ele->LinkEndChild(sEle);
		std::string str;
		for (int r = 0; r < m_curPoses.rows(); r++)
		{
			for (int c = 0; c < m_curPoses.cols(); c++)
			{
				str += std::to_string(m_curPoses(r, c));
				if (c + 1 < m_curPoses.cols())
					str += " ";
			}
			if (r + 1 < m_curPoses.rows())
				str += "  ";
		}
		sEle->SetAttribute("value", str.c_str());
	} // end if savePose
}

void SmplManager::loadCoeffsFromXml(TiXmlElement* ele, bool loadShape, bool loadPose)
{
	if (!m_inited)
		return;

	for (auto sEle = ele->FirstChildElement(); sEle; sEle = sEle->NextSiblingElement())
	{
		if (sEle->Value() == std::string("shape") && loadShape)
		{
			if (!sEle->Attribute("value"))
				throw std::exception("xmlError: attribute \"value\" for \"shape\" not found");
			std::stringstream stm(sEle->Attribute("value"));
			for (int r = 0; r < m_curShapes.rows(); r++)
			for (int c = 0; c < m_curShapes.cols(); c++)
				stm >> m_curShapes(r, c);
		} // end if shape
		else if (sEle->Value() == std::string("pose") && loadPose)
		{
			if (!sEle->Attribute("value"))
				throw std::exception("xmlError: attribute \"value\" for \"pose\" not found");
			std::stringstream stm(sEle->Attribute("value"));
			for (int r = 0; r < m_curPoses.rows(); r++)
			for (int c = 0; c < m_curPoses.cols(); c++)
				stm >> m_curPoses(r, c);
		} // end if pose
	} // end for sEle

	updateCurMesh();
}

void SmplManager::setAxisRenderMode(AxisRenderMode mode)
{
	m_axisRenderMode = mode;
}

void SmplManager::setRenderCamera(const ldp::Camera* cam)
{
	m_renderCam = cam;
}

void SmplManager::render(int showType, int frameIndex)
{
	if (!_isEnabled || !m_inited)
		return;
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	if (typeid(real) == typeid(float))
	{
		glVertexPointer(3, GL_FLOAT, 0, m_curVerts.data());
		glNormalPointer(GL_FLOAT, 0, m_curVNormals.data());
	}
	else if (typeid(real) == typeid(double))
	{
		glVertexPointer(3, GL_DOUBLE, 0, m_curVerts.data());
		glNormalPointer(GL_DOUBLE, 0, m_curVNormals.data());
	}

	if (showType & SW_LIGHTING)
		glEnable(GL_LIGHTING);
	else
		glDisable(GL_LIGHTING);

	if (showType & SW_E || showType & SW_V)
	{
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1., 1.);
	}

	if (showType & SW_F)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glColor3f(0.8, 0.8, 0.8);
		glDrawElements(GL_TRIANGLES, m_faces.size() * 3, GL_UNSIGNED_INT, m_faces.data());
	}

	if (showType & SW_E)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDisable(GL_LIGHTING);
		glDisable(GL_TEXTURE_2D);
		glColor3f(0.2, 0.3, 0.4);
		glLineWidth(1.0f);
		glDrawElements(GL_TRIANGLES, m_faces.size() * 3, GL_UNSIGNED_INT, m_faces.data());
	}

	if (showType & SW_V)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
		glDisable(GL_TEXTURE_2D);
		glPointSize(1);
		glColor3f(1, 1, 0);
		glDrawElements(GL_TRIANGLES, m_faces.size() * 3, GL_UNSIGNED_INT, m_faces.data());
	}

	if (showType & SW_SKELETON)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glEnable(GL_LIGHTING);
		glDisable(GL_TEXTURE_2D);

		float scale = (m_bbox[1] - m_bbox[0]).length();

		// 1. draw nodes as cubes
		for (size_t i = 0; i < m_curJ.rows(); i++)
		{
			if (m_selectedNode_mouseMove == i)
				glColor3f(1, 1, 1);
			else
				glColor3f(0.8, 0.6, 0.4);
			ldp::Float3 p = convert(getCurNodeCenter(i));
			ldp::Mat4f M = ldp::Mat4f().eye();
			M.setRotationPart(convert(m_curJrots[i]));
			glPushMatrix();
			glTranslatef(p[0], p[1], p[2]);
			glMultMatrixf(M.ptr());
			glutSolidCube(scale * g_node_size);
			glPopMatrix();
		}

		// 2. draw lines between nodes
		glColor3f(0.8, 0.2, 0.8);
		for (size_t i = 0; i < m_kintree_table.size(); i++)
		{
			int iparent = m_kintree_table[i];
			if (iparent < 0 || iparent >= m_curJ.rows())
				continue;
			ldp::Float3 p = convert(getCurNodeCenter(i));
			ldp::Float3 pp = convert(getCurNodeCenter(iparent));
			ldp::QuaternionF Q = ldp::QuaternionF().fromRotationVecs(ldp::Float3(0, 0, 1), pp - p);
			glPushMatrix();
			glTranslatef(p[0], p[1], p[2]);
			glMultMatrixf(Q.toRotationMatrix().ptr());
			glutSolidCone(scale * g_skel_size, (p - pp).length(), 32, 8);
			glPopMatrix();
		}

		// 3. render selected node
		switch (m_axisRenderMode)
		{
		case Renderable::AxisTrans:
			renderSelectedNode_Trans(scale, false);
			break;
		case Renderable::AxisRot:
			renderSelectedNode_Rot(scale, false);
			break;
		default:
			break;
		}
	}

	glPopClientAttrib();
	glPopAttrib();
}

void SmplManager::renderSelectedNode_Trans(float scale, bool indexMode)
{
	if (m_selectedNode < 0 || m_selectedNode >= m_curJ.rows())
		return;

	static GLUquadric* quadratic = gluNewQuadric();
	ldp::Float3 p = convert(getCurNodeCenter(m_selectedNode));
	ldp::Mat4f M = ldp::Mat4f().eye();
	M.setRotationPart(convert(m_curJrots[m_selectedNode]));
	glPushMatrix();
	glTranslatef(p[0], p[1], p[2]);
	glMultMatrixf(M.ptr());

	// x axis
	glColor3f(1, 0, 0);
	if (m_selectedNodeAxis_mouseMove == 0)
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(numPoses() + m_selectIdStart).ptr());
	glMultMatrixf(get_z2x_rot().ptr());
	solid_axis(scale * g_axis_size, scale * g_axis_length);
	glMultMatrixf(get_z2x_rot().trans().ptr());

	// y axis
	glColor3f(0, 1, 0);
	if (m_selectedNodeAxis_mouseMove == 1)
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(numPoses() + m_selectIdStart + 1).ptr());
	glMultMatrixf(get_z2y_rot().ptr());
	solid_axis(scale * g_axis_size, scale * g_axis_length);
	glMultMatrixf(get_z2y_rot().trans().ptr());

	// z axis
	glColor3f(0, 0, 1);
	if (m_selectedNodeAxis_mouseMove == 2)
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(numPoses() + m_selectIdStart + 2).ptr());
	solid_axis(scale * g_axis_size, scale * g_axis_length);

	glPopMatrix();

}

void SmplManager::renderSelectedNode_Rot(float scale, bool indexMode)
{
	if (m_selectedNode < 0 || m_selectedNode >= m_curJ.rows())
		return;
	ldp::Float3 p = convert(getCurNodeCenter(m_selectedNode));
	ldp::Mat4f M = ldp::Mat4f().eye();
	M.setRotationPart(convert(m_curJrots[m_selectedNode]));
	glPushMatrix();
	glTranslatef(p[0], p[1], p[2]);
	glMultMatrixf(M.ptr());

	// x axis
	glColor3f(1, 0, 0);
	if (m_selectedNodeAxis_mouseMove == 0)
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(numPoses() + m_selectIdStart).ptr());
	glMultMatrixf(get_z2x_rot().ptr());
	glutSolidTorus(scale * g_axis_length * 0.03, scale * g_axis_length, 16, 128);
	glMultMatrixf(get_z2x_rot().trans().ptr());

	// y axis
	glColor3f(0, 1, 0);
	if (m_selectedNodeAxis_mouseMove == 1)
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(numPoses() + m_selectIdStart + 1).ptr());
	glMultMatrixf(get_z2y_rot().ptr());
	glutSolidTorus(scale * g_axis_length * 0.03, scale * g_axis_length, 16, 128);
	glMultMatrixf(get_z2y_rot().trans().ptr());

	// z axis
	glColor3f(0, 0, 1);
	if (m_selectedNodeAxis_mouseMove == 2)
		glColor3f(1, 1, 1);
	if (indexMode)
		glColor4fv(selectIdToColor(numPoses() + m_selectIdStart + 2).ptr());
	glutSolidTorus(scale * g_axis_length * 0.03, scale * g_axis_length, 4, 128);

	// sphere
	if (!indexMode)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glColor4f(0.6, 0.6, 0.6, 0.5);
		glutSolidSphere(scale * g_axis_length, 32, 32);
		glDisable(GL_BLEND);
	}

	glPopMatrix();
}

void SmplManager::renderForSelection(int showType, int idStart)
{
	if (!_isEnabled || !m_inited)
		return;
	m_selectIdStart = idStart + 1;
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	if (typeid(real) == typeid(float))
	{
		glVertexPointer(3, GL_FLOAT, 0, m_curVerts.data());
		glNormalPointer(GL_FLOAT, 0, m_curVNormals.data());
	}
	else if (typeid(real) == typeid(double))
	{
		glVertexPointer(3, GL_DOUBLE, 0, m_curVerts.data());
		glNormalPointer(GL_DOUBLE, 0, m_curVNormals.data());
	}
	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);

	if (showType & SW_SKELETON)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		float scale = (m_bbox[1] - m_bbox[0]).length();

		// 1. draw nodes as cubes
		for (size_t i = 0; i < m_curJ.rows(); i++)
		{
			ldp::Float3 p = convert(getCurNodeCenter(i));
			ldp::Mat4f M = ldp::Mat4f().eye();
			M.setRotationPart(convert(m_curJrots[i]));
			glColor4fv(selectIdToColor(i + m_selectIdStart).ptr());
			glPushMatrix();
			glTranslatef(p[0], p[1], p[2]);
			glMultMatrixf(M.ptr());
			glutSolidCube(scale * g_node_size);
			glPopMatrix();
		}

		// 3. render selected node
		switch (m_axisRenderMode)
		{
		case Renderable::AxisTrans:
			renderSelectedNode_Trans(scale, true);
			break;
		case Renderable::AxisRot:
			renderSelectedNode_Rot(scale, true);
			break;
		default:
			break;
		}
	}

	// finally, render faces as mask
	if (showType & SW_F)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glColor4fv(selectIdToColor(numPoses() + m_selectIdStart + 10).ptr());
		glDrawElements(GL_TRIANGLES, m_faces.size() * 3, GL_UNSIGNED_INT, m_faces.data());
	}

	glPopClientAttrib();
	glPopAttrib();
}

void SmplManager::selectAction(ldp::Float4 selectedColor,
	SelectActionType actionType, int actionCode)
{
	int selectedId = colorToSelectId(selectedColor) - m_selectIdStart;

	switch (actionType)
	{
	case Renderable::MOUSE_MOVE:
		selectAction_mouseMove(selectedId);
		break;
	case Renderable::MOUSE_L_PRESS:
		selectAction_mousePress(selectedId);
		break;
	case Renderable::MOUSE_L_RELEASE:
		selectAction_mouseRelease(selectedId);
		break;
	case Renderable::MOUSE_R_PRESS:
		break;
	case Renderable::MOUSE_R_RELEASE:
		break;
	default:
		break;
	}
}

void SmplManager::selectAction_mouseMove(int id)
{
	if (!m_select_mousePressed)
	{
		m_selectedNode_mouseMove = -1;
		m_selectedNodeAxis_mouseMove = -1;
		if (id < numPoses())
		{
			m_selectedNode_mouseMove = id;
		}
		else if (id < numPoses() + 3)
		{
			m_selectedNodeAxis_mouseMove = id - numPoses();
		}
	}
}

void SmplManager::selectAction_mousePress(int id)
{
	m_select_mousePressed = true;

	if (id < 0)
	{
		m_selectedNodeAxis = -1;
	}
	else if (id < numPoses())
	{
		m_selectedNode = id;
		printf("selected node: %d\n", m_selectedNode);
	}
	else if (id < numPoses() + 3)
	{
		m_selectedNodeAxis = id - numPoses();
	}
}

void SmplManager::selectAction_mouseRelease(int id)
{
	m_select_mousePressed = false;
	m_selectedNode_mouseMove = -1;
	m_selectedNodeAxis_mouseMove = -1;
}

void SmplManager::transformAction(TransformActionType act, ldp::Float2 p)
{
	if (m_renderCam == nullptr)
		return;

	switch (act)
	{
	case Renderable::TRANSLATE_BEGIN:
		return transformAction_translate_begin(p);
	case Renderable::TRANSLATE_MOVE:
		return transformAction_translate_move(p);
	case Renderable::ROTATE_BEGIN:
		return transformAction_rotate_begin(p);
	case Renderable::ROTATE_MOVE:
		return transformAction_rotate_move(p);
	default:
		return;
	}
}

void SmplManager::transformAction_translate_begin(ldp::Float2 mousePos)
{
	if (m_selectedNode < 0 || m_selectedNode >= numPoses())
		return;
	if (m_selectedNodeAxis < 0 || m_selectedNodeAxis >= 3)
		return;
	m_transform_startMousePos = mousePos;
	m_transform_joint_c_kept = convert(getCurNodeCenter(m_selectedNode));
	m_jointRotSolver->init(this);
}

void SmplManager::transformAction_translate_move(ldp::Float2 mousePos)
{
	if (m_selectedNode < 0 || m_selectedNode >= numPoses())
		return;
	if (m_selectedNodeAxis < 0 || m_selectedNodeAxis >= 3)
		return;

	ldp::Float3 axis = 0;
	axis[m_selectedNodeAxis] = 1;

	ldp::Float3 c = m_transform_joint_c_kept;
	ldp::Float3 c_uvd = m_renderCam->getScreenCoords(c);

	const int h = m_renderCam->getViewPortBottom();
	ldp::Double3 wp = m_renderCam->getWorldCoords(ldp::Float3(mousePos[0], h - 1 - mousePos[1], c_uvd[2]));
	ldp::Double3 wlp = m_renderCam->getWorldCoords(ldp::Float3(m_transform_startMousePos[0],
		h - 1 - m_transform_startMousePos[1], c_uvd[2]));
	ldp::Double3 dir = (wp - wlp)*axis;

	// solve for inverse kinematics
	m_jointRotSolver->solveOneNode(m_selectedNode, convert(dir + m_transform_joint_c_kept));
}

void SmplManager::transformAction_translate_end(ldp::Float2 mousePos)
{
	updateCurMesh();
}

void SmplManager::transformAction_rotate_begin(ldp::Float2 mousePos)
{
	if (m_selectedNode < 0 || m_selectedNode >= numPoses())
		return;
	if (m_selectedNodeAxis < 0 || m_selectedNodeAxis >= 3)
		return;
	m_transform_startMousePos = mousePos;
	m_transform_joint_r_kept = ldp::Float3(m_curPoses(m_selectedNode, 0),
		m_curPoses(m_selectedNode, 1), m_curPoses(m_selectedNode, 2));
}

void SmplManager::transformAction_rotate_move(ldp::Float2 mousePos)
{
	if (m_selectedNode < 0 || m_selectedNode >= numPoses())
		return;
	if (m_selectedNodeAxis < 0 || m_selectedNodeAxis >= 3)
		return;

	ldp::Mat3f R = convert(angles2rot(convert(m_transform_joint_r_kept)));

	ldp::Float3 c = convert(getCurNodeCenter(m_selectedNode));
	ldp::Mat3f A = convert(m_curJrots[m_selectedNode]);
	if (m_kintree_table[m_selectedNode] >= 0)
		A = convert(m_curJrots[m_kintree_table[m_selectedNode]]).trans() * A;
	ldp::Float3 axis, gaxis;
	for (int k = 0; k < 3; k++)
	{
		axis[k] = A(k, m_selectedNodeAxis);
		gaxis[k] = m_curJrots[m_selectedNode](k, m_selectedNodeAxis);
	}
	ldp::Float3 c1 = c + gaxis;
	ldp::Float3 c_uvd = m_renderCam->getScreenCoords(c);
	ldp::Float3 c1_uvd = m_renderCam->getScreenCoords(c1);
	ldp::Float2 c_uv(c_uvd[0] / c_uvd[2], c_uvd[1] / c_uvd[2]);
	c_uv[1] = m_renderCam->getViewPortBottom() - c_uv[1];
	ldp::Float2 d1 = (mousePos - c_uv).normalize();
	ldp::Float2 d2 = (m_transform_startMousePos - c_uv).normalize();
	float ag = asin(d1.cross(d2));
	if (c_uvd[2] < c1_uvd[2])
		ag = -ag;
	R = ldp::QuaternionF().fromAngleAxis(ag, axis).toRotationMatrix3() * R;

	ldp::Float3 r = convert(rot2angles(convert(R)));
	for (int k = 0; k < 3; k++)
		m_curPoses(m_selectedNode, k) = r[k];

	updateCurMesh();
}

void SmplManager::transformAction_rotate_end(ldp::Float2 mousePos)
{

}

void SmplManager::setPoseShapeVals(const std::vector<float>* poses,
	const std::vector<float>* shapes)
{
	if (poses)
	{
		CHECK_THROW_EXCPT(poses->size() == m_curPoses.size());
		int pos = 0;
		for (int r = 0; r < m_curPoses.rows(); r++)
		for (int c = 0; c < m_curPoses.cols(); c++)
			m_curPoses(r, c) = poses->at(pos++);
	}
	if (shapes)
	{
		CHECK_THROW_EXCPT(shapes->size() == m_curShapes.size());
		int pos = 0;
		for (int r = 0; r < m_curShapes.rows(); r++)
		for (int c = 0; c < m_curShapes.cols(); c++)
			m_curShapes(r, c) = shapes->at(pos++);
	}

	updateCurMesh();
}

void SmplManager::calcPoseVector207(const DMat& poses_24x3, DMat& poses_207)
{
	poses_207.resize((poses_24x3.rows() - 1) * 9, 1);
	for (int y = 1; y < poses_24x3.rows(); y++)
	{
		Vec3 v(poses_24x3(y, 0), poses_24x3(y, 1), poses_24x3(y, 2));
		Mat3 R = angles2rot(v) - Mat3::Identity();
		for (int k = 0; k < 9; k++)
			poses_207((y - 1) * 9 + k, 0) = R.data()[k];
	}
}

Vec3 SmplManager::getCurNodeCenter(int i)const
{
	Vec3 v(m_curJ(i, 0), m_curJ(i, 1), m_curJ(i, 2));
	int iparent = m_kintree_table[i];
	if (iparent < 0)
		return  v;
	else
		return m_curJrots[iparent] * v + m_curJtrans[iparent];
}

void SmplManager::toObjMesh(ObjMesh& mesh)const
{
	CHECK_THROW_EXCPT(m_inited);

	mesh.clear();
	mesh.vertex_list.resize(m_curVerts.size());
	for (size_t i = 0; i < m_curVerts.size(); i++)
		mesh.vertex_list[i] = convert(m_curVerts[i]);
	mesh.face_list.resize(m_faces.size());

	// copy faces
	for (size_t i = 0; i < mesh.face_list.size(); i++)
	{
		ObjMesh::obj_face& f = mesh.face_list[i];
		f.vertex_count = 3;
		for (int k = 0; k < 3; k++)
			f.vertex_index[k] = m_faces[i][k];
	}

	// update normals and bounds
	mesh.updateNormals();
	mesh.updateBoundingBox();
}

