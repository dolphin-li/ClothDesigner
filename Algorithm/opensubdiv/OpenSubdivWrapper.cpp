#include "GL\glew.h"
#include "OpenSubdivWrapper.h"
#include "utils\shape_utils.h"
#include "far/error.h"
#include "far\topologyRefinerFactory.h"

#pragma comment(lib, "osdGPU.lib")
#pragma comment(lib, "osdCPU.lib")

using namespace OpenSubdiv;
static void shapeFromObjMesh(const ObjMesh& obj, Shape& shape);
static Sdc::SchemeType GetSdcType(Shape const & shape);
static void updateUniformBlocks();

OpenSubdivWrapper::OpenSubdivWrapper()
{
	m_controlMeshDisplay.reset(new GLControlMeshDisplay);
}

OpenSubdivWrapper::~OpenSubdivWrapper()
{
	release();
}

void OpenSubdivWrapper::create(const ObjMesh* objMesh, Param param)
{
	release();

	glGenVertexArrays(1, &m_vao);
	if (m_vao == 0)
		throw std::exception("OpenSubdivWrapper::create, vao create failed!");

	m_param = param;
	Shape shape;
	shapeFromObjMesh(*objMesh, shape);

	// create Far mesh (topology)
	Sdc::SchemeType sdctype = GetSdcType(shape);
	Sdc::Options sdcoptions = GetSdcOptions(shape);
	m_refiner.reset(Far::TopologyRefinerFactory<Shape>::Create(shape,
		Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions)));

	// save coarse topology (used for coarse mesh drawing)
	m_controlMeshDisplay->SetTopology(m_refiner->GetLevel(0));
	m_orgPositions = shape.verts;

	// Adaptive refinement currently supported only for catmull-clark scheme
	bool doAdaptive = (m_param.doAdaptive && m_param.scheme == Scheme::Catmark);
	bool interleaveVarying = false;
	bool doSmoothCornerPatch = (m_param.doSmoothCornerPatch && m_param.scheme == Scheme::Catmark);
	bool doSingleCreasePatch = (m_param.doSingleCreasePatch && m_param.scheme == Scheme::Catmark);
	bool doInfSharpPatch = (m_param.doInfSharpPatch && m_param.scheme == Scheme::Catmark);

	Osd::MeshBitset bits;
	bits.set(Osd::MeshAdaptive, doAdaptive);
	bits.set(Osd::MeshUseSmoothCornerPatch, doSmoothCornerPatch);
	bits.set(Osd::MeshUseSingleCreasePatch, doSingleCreasePatch);
	bits.set(Osd::MeshUseInfSharpPatch, doInfSharpPatch);
	bits.set(Osd::MeshInterleaveVarying, interleaveVarying);
	bits.set(Osd::MeshFVarData, false);
	bits.set(Osd::MeshEndCapBSplineBasis, false);
	bits.set(Osd::MeshEndCapGregoryBasis, false);
	bits.set(Osd::MeshEndCapLegacyGregory, false);

	const int numVertexElements = 3;
	const int numVaryingElements = 0;

	if (m_param.mode == ComputeMode::Cpu)
	{
		m_subdivMesh.reset(new CpuMesh(m_refiner.get(), numVertexElements, 
			numVaryingElements, m_param.level, bits));
	} // end if cpu mode
	else if (m_param.mode == ComputeMode::Glsl)
	{
		static Osd::EvaluatorCacheT<Osd::GLComputeEvaluator> glComputeEvaluatorCache;
		m_subdivMesh.reset(new GlslMesh(m_refiner.get(), numVertexElements, numVaryingElements,
			m_param.level, bits, &glComputeEvaluatorCache));
	} // end if glsl mode
}

void OpenSubdivWrapper::release()
{
	if (m_vao)
		glDeleteVertexArrays(1, &m_vao);
	m_vao = 0;
	ObjMesh* m_objMesh = nullptr;
	m_showType = 0;
	m_controlMeshDisplay.reset((GLControlMeshDisplay*)nullptr);
	m_subdivMesh.reset((MeshInterface*)nullptr);
	m_legacyGregoryPatchTable.reset((PatchTable*)nullptr);
}

void OpenSubdivWrapper::updateLevel(int level)
{

}

void OpenSubdivWrapper::updateGeometry()
{
	const int nverts = (int)m_orgPositions.size() / 3;
	m_subdivMesh->UpdateVertexBuffer(m_orgPositions.data(), 0, nverts);
	m_subdivMesh->Refine();
}

void OpenSubdivWrapper::bindBuffers()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_subdivMesh->GetPatchTable()->GetPatchIndexBuffer());
	glBindBuffer(GL_ARRAY_BUFFER, m_subdivMesh->BindVertexBuffer());
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat)* 3, 0);

	glDisableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void OpenSubdivWrapper::bindTextures()
{
	// bind patch textures
	if (m_subdivMesh->GetPatchTable()->GetPatchParamTextureBuffer()) 
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER,
			m_subdivMesh->GetPatchTable()->GetPatchParamTextureBuffer());
	}

	// legacy gregory
	if (m_legacyGregoryPatchTable) 
	{
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_BUFFER,
			m_legacyGregoryPatchTable->GetVertexTextureBuffer());
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_BUFFER,
			m_legacyGregoryPatchTable->GetVertexValenceTextureBuffer());
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_BUFFER,
			m_legacyGregoryPatchTable->GetQuadOffsetsTextureBuffer());
	}
	glActiveTexture(GL_TEXTURE0);
}

void OpenSubdivWrapper::display(int showType)
{
	// make sure that the vertex buffer is interoped back as a GL resource.
	GLuint vbo = m_subdivMesh->BindVertexBuffer();

	// vertex texture update for legacy gregory drawing
	if (m_legacyGregoryPatchTable) 
	{
		glActiveTexture(GL_TEXTURE1);
		m_legacyGregoryPatchTable->UpdateVertexBuffer(vbo);
	}

	// update transform and lighting uniform blocks
	updateUniformBlocks();

	// also bind patch related textures
	bindTextures();

	if (g_displayStyle == kDisplayStyleWire)
		glDisable(GL_CULL_FACE);

	glEnable(GL_DEPTH_TEST);

	glBindVertexArray(g_vao);

	OpenSubdiv::Osd::PatchArrayVector const & patches =
		g_mesh->GetPatchTable()->GetPatchArrays();

	// patch drawing
	int patchCount[13]; // [Type] (see far/patchTable.h)
	int numTotalPatches = 0;
	int numDrawCalls = 0;
	memset(patchCount, 0, sizeof(patchCount));

	// primitive counting
	glBeginQuery(GL_PRIMITIVES_GENERATED, g_queries[0]);
#if defined(GL_VERSION_3_3)
	glBeginQuery(GL_TIME_ELAPSED, g_queries[1]);
#endif

	// core draw-calls
	for (int i = 0; i<(int)patches.size(); ++i) {
		OpenSubdiv::Osd::PatchArray const & patch = patches[i];

		OpenSubdiv::Far::PatchDescriptor desc = patch.GetDescriptor();
		OpenSubdiv::Far::PatchDescriptor::Type patchType = desc.GetType();

		patchCount[patchType] += patch.GetNumPatches();
		numTotalPatches += patch.GetNumPatches();

		GLenum primType = bindProgram(GetEffect(), patch);


		glDrawElements(primType,
			patch.GetNumPatches() * desc.GetNumControlVertices(),
			GL_UNSIGNED_INT,
			(void *)(patch.GetIndexBase() * sizeof(unsigned int)));
		++numDrawCalls;
	}

	s.Stop();
	float drawCpuTime = float(s.GetElapsed() * 1000.0f);

	glEndQuery(GL_PRIMITIVES_GENERATED);
#if defined(GL_VERSION_3_3)
	glEndQuery(GL_TIME_ELAPSED);
#endif

	glBindVertexArray(0);

	glUseProgram(0);

	if (g_displayStyle == kDisplayStyleWire)
		glEnable(GL_CULL_FACE);

	// draw the control mesh
	int stride = g_shadingMode == kShadingInterleavedVaryingColor ? 7 : 3;
	g_controlMeshDisplay.Draw(vbo, stride*sizeof(float),
		g_transformData.ModelViewProjectionMatrix);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	GLuint numPrimsGenerated = 0;
	GLuint timeElapsed = 0;
	glGetQueryObjectuiv(g_queries[0], GL_QUERY_RESULT, &numPrimsGenerated);
#if defined(GL_VERSION_3_3)
	glGetQueryObjectuiv(g_queries[1], GL_QUERY_RESULT, &timeElapsed);
#endif

	float drawGpuTime = timeElapsed / 1000.0f / 1000.0f;

	g_fpsTimer.Stop();
	float elapsed = (float)g_fpsTimer.GetElapsed();
	if (!g_freeze) {
		g_animTime += elapsed;
	}
	g_fpsTimer.Start();

	if (g_hud.IsVisible()) {

		typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

		double fps = 1.0 / elapsed;

		if (g_displayPatchCounts) {
			int x = -280;
			int y = -180;
			g_hud.DrawString(x, y, "NonPatch         : %d",
				patchCount[Descriptor::QUADS]); y += 20;
			g_hud.DrawString(x, y, "Regular          : %d",
				patchCount[Descriptor::REGULAR]); y += 20;
			g_hud.DrawString(x, y, "Gregory          : %d",
				patchCount[Descriptor::GREGORY]); y += 20;
			g_hud.DrawString(x, y, "Boundary Gregory : %d",
				patchCount[Descriptor::GREGORY_BOUNDARY]); y += 20;
			g_hud.DrawString(x, y, "Gregory Basis    : %d",
				patchCount[Descriptor::GREGORY_BASIS]); y += 20;
		}

		int y = -220;
		g_hud.DrawString(10, y, "Tess level : %d", g_tessLevel); y += 20;
		g_hud.DrawString(10, y, "Patches    : %d", numTotalPatches); y += 20;
		g_hud.DrawString(10, y, "Draw calls : %d", numDrawCalls); y += 20;
		g_hud.DrawString(10, y, "Primitives : %d", numPrimsGenerated); y += 20;
		g_hud.DrawString(10, y, "Vertices   : %d", g_mesh->GetNumVertices()); y += 20;
		g_hud.DrawString(10, y, "GPU Kernel : %.3f ms", g_gpuTime); y += 20;
		g_hud.DrawString(10, y, "CPU Kernel : %.3f ms", g_cpuTime); y += 20;
		g_hud.DrawString(10, y, "GPU Draw   : %.3f ms", drawGpuTime); y += 20;
		g_hud.DrawString(10, y, "CPU Draw   : %.3f ms", drawCpuTime); y += 20;
		g_hud.DrawString(10, y, "FPS        : %3.1f", fps); y += 20;

		g_hud.Flush();
	}

	glFinish();

	GLUtils::CheckGLErrors("display leave\n");
}

////////////////////////////////////////////////////////////////////////
void shapeFromObjMesh(const ObjMesh& obj, Shape& shape)
{

}

OpenSubdiv::Sdc::SchemeType GetSdcType(Shape const & shape)
{
	OpenSubdiv::Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

	switch (shape.scheme)
	{
	case kBilinear:
		type = OpenSubdiv::Sdc::SCHEME_BILINEAR;
		break;
	case kCatmark:
		type = OpenSubdiv::Sdc::SCHEME_CATMARK;
		break;
	case kLoop:
		type = OpenSubdiv::Sdc::SCHEME_LOOP;
		break;
	}
	return type;
}

OpenSubdiv::Sdc::Options GetSdcOptions(Shape const & shape) 
{
	typedef OpenSubdiv::Sdc::Options Options;
	Options result;
	result.SetVtxBoundaryInterpolation(Options::VTX_BOUNDARY_EDGE_ONLY);
	result.SetCreasingMethod(Options::CREASE_UNIFORM);
	result.SetTriangleSubdivision(Options::TRI_SUB_CATMARK);
	for (int i = 0; i < (int)shape.tags.size(); ++i)
	{
		Shape::tag * t = shape.tags[i];
		if (t->name == "interpolateboundary")
		{
			if ((int)t->intargs.size() != 1)
			{
				printf("expecting 1 integer for \"interpolateboundary\" tag n. %d\n", i);
				continue;
			}
			switch (t->intargs[0])
			{
			case 0: result.SetVtxBoundaryInterpolation(Options::VTX_BOUNDARY_NONE); break;
			case 1: result.SetVtxBoundaryInterpolation(Options::VTX_BOUNDARY_EDGE_AND_CORNER); break;
			case 2: result.SetVtxBoundaryInterpolation(Options::VTX_BOUNDARY_EDGE_ONLY); break;
			default: printf("unknown interpolate boundary : %d\n", t->intargs[0]); break;
			}
		}
		else if (t->name == "facevaryinginterpolateboundary")
		{
			if ((int)t->intargs.size() != 1)
			{
				printf("expecting 1 integer for \"facevaryinginterpolateboundary\" tag n. %d\n", i);
				continue;
			}
			switch (t->intargs[0])
			{
			case 0: result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_NONE); break;
			case 1: result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_CORNERS_ONLY); break;
			case 2: result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_CORNERS_PLUS1); break;
			case 3: result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_CORNERS_PLUS2); break;
			case 4: result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_BOUNDARIES); break;
			case 5: result.SetFVarLinearInterpolation(Options::FVAR_LINEAR_ALL); break;
			default: printf("unknown interpolate boundary : %d\n", t->intargs[0]); break;
			}
		}
		else if (t->name == "facevaryingpropagatecorners")
		{
			if ((int)t->intargs.size() == 1)
			{
				// XXXX no propagate corners in Options
				assert(0);
			}
			else
				printf("expecting single int argument for \"facevaryingpropagatecorners\"\n");
		}
		else if (t->name == "creasemethod")
		{
			if ((int)t->stringargs.size() == 0)
			{
				printf("the \"creasemethod\" tag expects a string argument\n");
				continue;
			}
			if (t->stringargs[0] == "normal")
			{
				result.SetCreasingMethod(Options::CREASE_UNIFORM);
			}
			else if (t->stringargs[0] == "chaikin")
			{
				result.SetCreasingMethod(Options::CREASE_CHAIKIN);
			}
			else
			{
				printf("the \"creasemethod\" tag only accepts \"normal\" or \"chaikin\" as value (%s)\n",
					t->stringargs[0].c_str());
			}
		}
		else if (t->name == "smoothtriangles")
		{
			if (shape.scheme != kCatmark)
			{
				printf("the \"smoothtriangles\" tag can only be applied to Catmark meshes\n");
				continue;
			}
			if (t->stringargs[0] == "catmark")
			{
				result.SetTriangleSubdivision(Options::TRI_SUB_CATMARK);
			}
			else if (t->stringargs[0] == "smooth")
			{
				result.SetTriangleSubdivision(Options::TRI_SUB_SMOOTH);
			}
			else
			{
				printf("the \"smoothtriangles\" tag only accepts \"catmark\" or \"smooth\" as value (%s)\n",
					t->stringargs[0].c_str());
			}
		} // end t->name
	} // end for shape.tags

	return result;
}