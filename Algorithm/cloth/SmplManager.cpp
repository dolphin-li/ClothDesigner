#include "SmplManager.h"
#include "matio\matio.h"
#include "Renderable\ObjMesh.h"
#include <GL\glut.h>
#include "ldpMat\Quaternion.h"
#include "Camera\camera.h"

#pragma region --mat_utils

#define CHECK_THROW_EXCPT(cond)\
if (!(cond)) { throw std::exception(("!!!!Exception: " + std::string(#cond)).c_str()); }

typedef SmplManager::real real;
typedef SmplManager::Face Face;
typedef SmplManager::DMat DMat;
typedef SmplManager::SpMat SpMat;

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
		out[i] = in_f[i*2];
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

inline ldp::Mat3f angles2rot(ldp::Float3 v)
{
	float theta = v.length();
	if (theta == 0)
		return ldp::Mat3f().eye();
	v /= theta;
	return ldp::QuaternionF().fromAngleAxis(theta, v).toRotationMatrix3();
}

inline ldp::Float3 rot2angles(ldp::Mat3f R)
{
	ldp::QuaternionF q;
	q.fromRotationMatrix(R);
	ldp::Float3 v;
	float ag;
	q.toAngleAxis(v, ag);
	v *= ag;
	return v;
}

static GLUquadric* get_quadric()
{
	static GLUquadric* q = gluNewQuadric();
	return q;
}

static ldp::Mat4f get_z2x_rot()
{
	static ldp::Mat4f R = ldp::QuaternionF().fromRotationVecs(ldp::Float3(0,0,1), 
		ldp::Float3(1,0,0)).toRotationMatrix();
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
	return (cl[0]<<24) + (cl[1]<<16) + (cl[2]<<8) + cl[3];
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
}

SmplManager::~SmplManager()
{
}

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
	for (size_t iVerts = 0; iVerts < m_curVerts.size(); iVerts++)
	{
		ldp::Float3 v = m_curVerts[iVerts];
		int wb = m_weights.outerIndexPtr()[iVerts];
		int we = m_weights.outerIndexPtr()[iVerts + 1];
		ldp::Float3 v_transformed = 0.f;
		for (int iw = wb; iw < we; iw++)
		{
			int jointId = m_weights.innerIndexPtr()[iw];
			real jointW = m_weights.valuePtr()[iw];
			ldp::Float3 c(m_curJ(jointId, 0), m_curJ(jointId, 1), m_curJ(jointId, 2));
			v_transformed += jointW * (m_curJrots[jointId] * (v - c) + m_curJtrans[jointId] + c);
		}
		m_curVerts[iVerts] = v_transformed;
	} // iVerts
	
	// 2. update normals and bounds ------------------------------------
	m_curFNormals.resize(m_faces.size());
	m_curVNormals.resize(m_curVerts.size());
	std::fill(m_curVNormals.begin(), m_curVNormals.end(), 0.f);
	m_bbox[0] = FLT_MAX;
	m_bbox[1] = FLT_MIN;
	for (size_t i = 0; i < m_faces.size(); i++)
	{
		const Face &f = m_faces[i];
		ldp::Float3 nm = ldp::Float3(m_curVerts[f[1]] - m_curVerts[f[0]]).cross(
			m_curVerts[f[2]] - m_curVerts[f[0]]);
		for (int k = 0; k < 3; k++)
			m_curVNormals[f[k]] += nm;
		m_curFNormals[i] = nm.normalize();
	}
	for (size_t i = 0; i < m_curVNormals.size(); i++)
	{
		m_curVNormals[i].normalizeLocal();
		for (int k = 0; k < 3; k++)
		{
			m_bbox[0][k] = std::min(m_bbox[0][k], m_curVerts[i][k]);
			m_bbox[1][k] = std::max(m_bbox[1][k], m_curVerts[i][k]);
		}
	} // i
}

void SmplManager::calcGlobalTrans()
{
	m_curJrots.resize(m_curJ.rows());
	m_curJtrans.resize(m_curJ.rows());
	for (size_t iJoints = 0; iJoints < m_curJ.rows(); iJoints++)
	{
		ldp::Float3 r(m_curPoses(iJoints, 0), m_curPoses(iJoints, 1), m_curPoses(iJoints, 2));
		ldp::Mat3f R = angles2rot(r);
		int iParent = m_kintree_table[iJoints];
		if (iParent < 0)
		{
			m_curJrots[iJoints] = R;
			m_curJtrans[iJoints] = 0.f;
		}
		else
		{
			m_curJrots[iJoints] = m_curJrots[iParent] * R;
			ldp::Float3 v(m_curJ(iJoints, 0), m_curJ(iJoints, 1), m_curJ(iJoints, 2));
			ldp::Float3 vp(m_curJ(iParent, 0), m_curJ(iParent, 1), m_curJ(iParent, 2));
			m_curJtrans[iJoints] = m_curJrots[iParent] * (v - vp) + vp + m_curJtrans[iParent] - v;
		}
	} // iJoints
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
	glVertexPointer(3, GL_FLOAT, 0, m_curVerts.data());
	glNormalPointer(GL_FLOAT, 0, m_curVNormals.data());

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
		glDrawElements(GL_TRIANGLES, m_faces.size()*3, GL_UNSIGNED_INT, m_faces.data());
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
			ldp::Float3 p = getCurNodeCenter(i);
			ldp::Mat4f M = ldp::Mat4f().eye();
			M.setRotationPart(m_curJrots[i]);
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
			ldp::Float3 p = getCurNodeCenter(i);
			ldp::Float3 pp = getCurNodeCenter(iparent);
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
	ldp::Float3 p = ldp::Float3(m_curJ(m_selectedNode, 0),
		m_curJ(m_selectedNode, 1), m_curJ(m_selectedNode, 2)) + m_curJtrans[m_selectedNode];
	ldp::Mat4f M = ldp::Mat4f().eye();
	M.setRotationPart(m_curJrots[m_selectedNode]);
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
	ldp::Float3 p = ldp::Float3(m_curJ(m_selectedNode, 0),
		m_curJ(m_selectedNode, 1), m_curJ(m_selectedNode, 2)) + m_curJtrans[m_selectedNode];
	ldp::Mat4f M = ldp::Mat4f().eye();
	M.setRotationPart(m_curJrots[m_selectedNode]);
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
	m_selectIdStart = idStart+1;
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, m_curVerts.data());
	glNormalPointer(GL_FLOAT, 0, m_curVNormals.data());
	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);

	if (showType & SW_SKELETON)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		float scale = (m_bbox[1] - m_bbox[0]).length();

		// 1. draw nodes as cubes
		for (size_t i = 0; i < m_curJ.rows(); i++)
		{
			ldp::Float3 p = ldp::Float3(m_curJ(i, 0), m_curJ(i, 1), m_curJ(i, 2)) + m_curJtrans[i];
			ldp::Mat4f M = ldp::Mat4f().eye();
			M.setRotationPart(m_curJrots[i]);
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
		transformAction_translate_begin(p);
		break;
	case Renderable::TRANSLATE_MOVE:
		transformAction_translate_move(p);
		break;
	case Renderable::ROTATE_BEGIN:
		transformAction_rotate_begin(p);
		break;
	case Renderable::ROTATE_MOVE:
		transformAction_rotate_move(p);
		break;
	default:
		break;
	}
}

void SmplManager::transformAction_translate_begin(ldp::Float2 mousePos)
{

}

void SmplManager::transformAction_translate_move(ldp::Float2 mousePos)
{

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

	ldp::Mat3f R = angles2rot(m_transform_joint_r_kept);

	ldp::Float3 c = getCurNodeCenter(m_selectedNode);
	ldp::Mat3f A = m_curJrots[m_selectedNode];
	if (m_kintree_table[m_selectedNode] >= 0)
		A = m_curJrots[m_kintree_table[m_selectedNode]].trans() * A;
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

	ldp::Float3 r = rot2angles(R);
	for (int k = 0; k < 3; k++)
		m_curPoses(m_selectedNode, k) = r[k];

	updateCurMesh();
}

void SmplManager::setPoseShapeVals(const std::vector<float>* poses, 
	const std::vector<float>* shapes)
{
	if (poses)
	{
		CHECK_THROW_EXCPT(poses->size() == m_curPoses.size());
		for (size_t i = 0; i < poses->size(); i++)
			m_curPoses.data()[i] = poses->at(i);
	}
	if (shapes)
	{
		CHECK_THROW_EXCPT(shapes->size() == m_curShapes.size());
		for (size_t i = 0; i < shapes->size(); i++)
			m_curShapes.data()[i] = shapes->at(i);
	}

	updateCurMesh();
}

void SmplManager::calcPoseVector207(const DMat& poses_24x3, DMat& poses_207)
{
	poses_207.resize((poses_24x3.rows()-1)*9, 1);
	for (int y = 1; y < poses_24x3.rows(); y++)
	{
		ldp::Float3 v(poses_24x3(y, 0), poses_24x3(y, 1), poses_24x3(y, 2));
		ldp::Mat3f R = angles2rot(v) - ldp::Mat3f().eye();
		for (int k = 0; k < 9; k++)
			poses_207((y - 1) * 9 + k, 0) = R.ptr()[k];
	}
}

ldp::Float3 SmplManager::getCurNodeCenter(int i)const
{
	return ldp::Float3(m_curJ(i, 0), m_curJ(i, 1), m_curJ(i, 2)) + m_curJtrans[i];
}

void SmplManager::toObjMesh(ObjMesh& mesh)const
{
	CHECK_THROW_EXCPT(m_inited);

	mesh.clear();
	mesh.vertex_list = m_curVerts;
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
	FILE* pFile = fopen(filename.c_str(), "r");
	if (!pFile)
		throw std::exception(("io error: " + filename).c_str());

	for (int i = 0; i < m_curShapes.rows(); i++)
	{
		float tmp = 0;
		for (int j = 0; j < m_curShapes.cols(); j++)
		{
			fscanf(pFile, "%f", &tmp);
			m_curShapes(i, j) = tmp;
		}
	}

	fclose(pFile);

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
	FILE* pFile = fopen(filename.c_str(), "r");
	if (!pFile)
		throw std::exception(("io error: " + filename).c_str());

	for (int i = 0; i < m_curPoses.rows(); i++)
	{
		float tmp = 0;
		for (int j = 0; j < m_curPoses.cols(); j++)
		{
			fscanf(pFile, "%f", &tmp);
			m_curPoses(i, j) = tmp;
		}
	}

	fclose(pFile);

	updateCurMesh();
}
