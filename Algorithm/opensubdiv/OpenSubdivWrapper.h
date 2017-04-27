#pragma once
#include "osd/opengl.h"
#include "osd/cpuEvaluator.h"
#include "osd/cpuGLVertexBuffer.h"
#include "osd/glComputeEvaluator.h"
#include "osd/glVertexBuffer.h"
#include "osd/glXFBEvaluator.h"
#include "osd/glVertexBuffer.h"
#include "osd/glMesh.h"
#include "osd/glLegacyGregoryPatchTable.h"
#include "utils\glControlMeshDisplay.h"
#include <memory>

class ObjMesh;
class OpenSubdivWrapper
{
	typedef OpenSubdiv::Far::TopologyRefiner Refiner;
	typedef OpenSubdiv::Osd::GLMeshInterface MeshInterface;
	typedef OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::GLVertexBuffer, OpenSubdiv::Osd::GLStencilTableSSBO,
		OpenSubdiv::Osd::GLComputeEvaluator, OpenSubdiv::Osd::GLPatchTable> GlslMesh;
	typedef OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer, OpenSubdiv::Far::StencilTable,
		OpenSubdiv::Osd::CpuEvaluator, OpenSubdiv::Osd::GLPatchTable> CpuMesh;
	typedef OpenSubdiv::Osd::GLLegacyGregoryPatchTable PatchTable;
public:
	enum Scheme
	{
		Catmark,
		Linear,
		Loop
	};
	enum ComputeMode
	{
		Cpu,
		Glsl
	};
	struct Param
	{
		ComputeMode mode = Glsl;
		Scheme scheme = Catmark;
		int level = 1;
		bool doAdaptive = false;
		bool doSmoothCornerPatch = false;
		bool doSingleCreasePatch = false;
		bool doInfSharpPatch = false;
	};
public:
	OpenSubdivWrapper();
	~OpenSubdivWrapper();
	void create(const ObjMesh* objMesh, Param param);
	void release();
	void updateLevel(int level);
	void updateGeometry();
	void display(int showType);
protected:
	void bindBuffers();
	void bindTextures();
private:
	GLuint m_vao = 0;
	const ObjMesh* m_objMesh = nullptr;
	int m_showType = 0;
	Param m_param;
	std::shared_ptr<MeshInterface> m_subdivMesh;
	std::shared_ptr<PatchTable> m_legacyGregoryPatchTable;
	std::shared_ptr<Refiner> m_refiner;
	std::shared_ptr<GLControlMeshDisplay> m_controlMeshDisplay;
	std::vector<float> m_orgPositions;
};