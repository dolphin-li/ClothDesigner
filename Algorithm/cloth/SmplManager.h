#pragma once

#include "ldpMat\ldp_basic_mat.h"
#include <eigen\Dense>
#include <eigen\Sparse>
#include "Renderable\Renderable.h"

class ObjMesh;
class TiXmlElement;
class SmplManager : public Renderable
{
public:
	enum BsStyle
	{
		LBS = 0,
		DQBS
	};
	enum BsType
	{
		LROTMIN = 0,
	};
	typedef double real;
	typedef ldp::Int3 Face;
	typedef Eigen::SparseMatrix<real> SpMat;
	typedef Eigen::Matrix<real, -1, -1> DMat;
	typedef Eigen::Matrix<real, -1, 1> DVec;
	typedef Eigen::Matrix<real, 3, 1> Vec3;
	typedef Eigen::Matrix<real, 2, 1> Vec2;
	typedef Eigen::Matrix<real, 4, 1> Vec4;
	typedef Eigen::Matrix<real, 3, 3> Mat3;
	typedef Eigen::Matrix<real, 2, 2> Mat2;
	typedef Eigen::Matrix<real, 4, 4> Mat4;
	friend class PoseSolver;
	friend class ShapeSolver;
	friend class JointRotSolver;
public:
	SmplManager();
	~SmplManager();

	virtual void render(int showType, int frameIndex = 0);
	virtual void renderConstColor(Float3 color)const {}
	virtual void renderForSelection(int showType, int idStart);
	virtual void selectAction(ldp::Float4 selectedColor, SelectActionType actionType, int actionCode);
	virtual void transformAction(TransformActionType act, ldp::Float2 mousePos);
	virtual void setAxisRenderMode(AxisRenderMode mode);
	virtual void setRenderCamera(const ldp::Camera* cam);
	virtual int getMeshType()const { return TYPE_SMPL; }
	virtual ldp::Float3 getCenter()const { return (m_bbox[0] + m_bbox[1])*0.5f; }
	virtual ldp::Float3 getBoundingBox(int i)const { return m_bbox[i]; }
	virtual void clear();

	bool isInitialized()const { return m_inited; }
	int selectedJointId()const { return m_selectedNode; }

	void loadFromMat(const char* filename);
	void toObjMesh(ObjMesh& mesh)const;
	int numShapes()const { return m_curShapes.size(); }
	int numVarEachPose()const { return m_curPoses.cols(); }
	int numPoses()const { return m_curPoses.rows(); }
	void setPoseShapeVals(const std::vector<float>* poses = nullptr, const std::vector<float>* shapes = nullptr);

	real getMaxShapeCoef()const { return m_maxShapeCoef; }
	void setMaxShapeCoef(real c);
	real getMinShapeCoef()const { return m_minShapeCoef; }
	void setMinShapeCoef(real c);

	// rigid-fitting the given vertex to self
	void rigidFitting(std::vector<ldp::Float3>& newVertices);

	// fitting shape coeffs from given vertices
	void fittingShapes(const std::vector<ldp::Float3>& newVertices, bool showInfo = false);

	// fitting pose coeffs from given vertices
	void fittingPoses(std::vector<ldp::Float3>& newVertices, bool showInfo = false);

	// fitting both shape and pose coeffs
	void fittingShapePoses(std::vector<ldp::Float3>& newVertices, bool showInfo = false);

	// compute the mesh data based on the current shapes and rots
	void updateCurMesh();

	// update the global rotation/translation from the local ones
	void calcGlobalTrans();

	void saveShapeCoeffs(std::string filename)const;
	void loadShapeCoeffs(std::string filename);
	void savePoseCoeffs(std::string filename)const;
	void loadPoseCoeffs(std::string filename);

	void saveCoeffsToXml(TiXmlElement* ele, bool saveShape, bool savePose)const;
	void loadCoeffsFromXml(TiXmlElement* ele, bool loadShape, bool loadPose);
public:
	Vec3 getCurNodeCenter(int idx)const;
	int getNodeParent(int idx)const { return m_kintree_table[idx]; }
	const Mat3& getCurNodeRots(int idx)const { return m_curJrots[idx]; }
	const Vec3& getCurNodeTrans(int idx)const { return m_curJtrans[idx]; }
	real getCurShapeCoef(int idx)const { return m_curShapes(idx, 0); }
	void setCurShapeCoef(int idx, real val) { m_curShapes(idx, 0) = val; }
	real getCurPoseCoef(int idx, int axis)const { return m_curPoses(idx, axis); }
	void setCurPoseCoef(int idx, int axis, real val) { m_curPoses(idx, axis) = val; }
protected:
	void calcPoseVector207(const DMat& poses_24x3, DMat& poses_207);
	void selectAction_mouseMove(int selectedId);
	void selectAction_mousePress(int selectedId);
	void selectAction_mouseRelease(int selectedId);

	void transformAction_translate_begin(ldp::Float2 mousePos);
	void transformAction_translate_move(ldp::Float2 mousePos);
	void transformAction_translate_end(ldp::Float2 mousePos);
	void transformAction_rotate_begin(ldp::Float2 mousePos);
	void transformAction_rotate_move(ldp::Float2 mousePos);
	void transformAction_rotate_end(ldp::Float2 mousePos);

	void renderSelectedNode_Trans(float scale, bool indexMode);
	void renderSelectedNode_Rot(float scale, bool indexMode);
private:
	bool m_inited;

	// params
	real m_maxShapeCoef = 5;
	real m_minShapeCoef = -5;

	// model data -----------------------------------

	BsStyle m_bsStyle;
	BsType m_bsType;
	std::vector<Face> m_faces;
	DMat m_J;
	SpMat m_J_regressor;
	SpMat m_J_regressor_prior;
	std::vector<int> m_kintree_table;
	DMat m_posedirs;
	DMat m_shapedirs;
	DMat m_v_template;
	std::vector<int> m_vert_sym_idxs;
	SpMat m_weights;
	SpMat m_weights_prior;

	// specific data -----------------------------------

	DMat m_curPoses;
	DMat m_curShapes;
	DMat m_curJ;
	std::vector<Mat3> m_curJrots;
	std::vector<Vec3> m_curJtrans;
	std::vector<Vec3> m_curVerts;
	std::vector<Vec3> m_curVNormals;
	std::vector<Vec3> m_curFNormals;
	ldp::Float3 m_bbox[2];

	// ui related ---------------------------------------
	const ldp::Camera* m_renderCam = nullptr;
	AxisRenderMode m_axisRenderMode;
	int m_selectedNode = 0;
	int m_selectedNodeAxis = 0;
	int m_selectedNode_mouseMove = 0;
	int m_selectedNodeAxis_mouseMove = 0;
	int m_selectIdStart = 0;
	bool m_select_mousePressed = 0;
	ldp::Float2 m_transform_startMousePos;
	ldp::Float3 m_transform_joint_r_kept;
	ldp::Float3 m_transform_joint_c_kept;

	std::shared_ptr<JointRotSolver> m_jointRotSolver;
};
