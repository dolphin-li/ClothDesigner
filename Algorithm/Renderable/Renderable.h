#ifndef __RENDERABLE_H__
#define __RENDERABLE_H__

#include "assert.h"
#include "ldpMat\ldp_basic_mat.h"
#include <string>
using ldp::Float3;
namespace ldp
{
	class Camera;
}
class Renderable
{
public:
	Renderable()
	{
		_isEnabled = true;
		_isSelected = false;
		_name = "Renderable";
	}
	virtual ~Renderable()
	{

	}
public:
	const static int TYPE_GENERAL = 0x10000000;
	const static int TYPE_ABSTRACT = 0x10000001;
	const static int TYPE_TRIMESH = 0x10000002;
	const static int TYPE_QUADMESH = 0x10000003;
	const static int TYPE_POINTCLOUD = 0x10000004;
	const static int TYPE_OBJMESH = 0x10000005;
	const static int TYPE_NODE = 0x10000006;
	const static int TYPE_DEPTHIMAGE = 0x10000007;
	const static int TYPE_COLORIMAGE = 0x10000008;
	const static int TYPE_BMESH = 0x10000009;
	const static int TYPE_BONE_MESH = 0x10000010;
	const static int TYPE_TRIMESH_GROUP = 0x10000011;
	const static int TYPE_SMPL = 0x10000012;

	//show type
	const static int SW_V = 0x00000001;
	const static int SW_E = 0x00000002;
	const static int SW_F = 0x00000004;
	const static int SW_N = 0x00000008;
	const static int SW_FLAT = 0x00000010;
	const static int SW_SMOOTH = 0x00000020;
	const static int SW_TEXTURE = 0x00000040;
	const static int SW_LIGHTING = 0x00000080;
	const static int SW_COLOR = 0x00000100;
	const static int SW_OUTLINE = 0x00000200;
	const static int SW_SKELETON = 0x00000400;

	//
	enum SelectActionType
	{
		MOUSE_MOVE = 0,
		MOUSE_L_PRESS,
		MOUSE_L_RELEASE,
		MOUSE_R_PRESS,
		MOUSE_R_RELEASE
	};
	enum AxisRenderMode
	{
		AxisTrans,
		AxisRot
	};
	enum TransformActionType
	{
		TRANSLATE_BEGIN = 0,
		TRANSLATE_MOVE,
		ROTATE_BEGIN,
		ROTATE_MOVE
	};
public:
	virtual void render(int showType, int frameIndex = 0) = 0;

	virtual void renderConstColor(Float3 color)const = 0;

	virtual void renderForSelection(int showType, int idStart = 0) {}

	virtual void selectAction(ldp::Float4 selectedColor, SelectActionType actionType = MOUSE_MOVE, int actionCode = 0) {}

	virtual void transformAction(TransformActionType act, ldp::Float2 mousePos) {}

	virtual void setAxisRenderMode(AxisRenderMode mode) {}

	virtual void setRenderCamera(const ldp::Camera* cam) {}

	virtual int getMeshType()const { return TYPE_GENERAL; }

	virtual ldp::Float3 getCenter()const { return 0.f; }
	virtual ldp::Float3 getBoundingBox(int i)const { return 0.f; }

	virtual void clear() { assert(0 && "your child class should overload clear()"); }

	bool isEnabled()const
	{
		return _isEnabled;
	}

	void setEnabled(bool enable)
	{
		_isEnabled = enable;
	}

	bool isSelected()const
	{
		return _isSelected;
	}

	void setSelected(bool enable)
	{
		_isSelected = enable;
	}

	void setName(const char* name)
	{
		_name = name;
	}

	const char* getName()const
	{
		return _name.c_str();
	}
protected:
	bool _isEnabled;
	bool _isSelected;
	std::string _name;
};





#endif