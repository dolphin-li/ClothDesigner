#pragma once

#include "utils\glShaderCache.h"
#include "far\patchDescriptor.h"
#include "utils\glUtils.h"
#include <sstream>
namespace OpenSubdiv
{
	union Effect 
	{
		Effect(int displayStyle_, int shadingMode_, int screenSpaceTess_,
		int fractionalSpacing_, int patchCull_, int singleCreasePatch_)
		: value(0) 
		{
			displayStyle = displayStyle_;
			shadingMode = shadingMode_;
			screenSpaceTess = screenSpaceTess_;
			fractionalSpacing = fractionalSpacing_;
			patchCull = patchCull_;
			singleCreasePatch = singleCreasePatch_;
		}

		struct 
		{
			unsigned int displayStyle : 2;
			unsigned int shadingMode : 4;
			unsigned int screenSpaceTess : 1;
			unsigned int fractionalSpacing : 1;
			unsigned int patchCull : 1;
			unsigned int singleCreasePatch : 1;
		};
		int value;

		bool operator < (const Effect &e) const 
		{
			return value < e.value;
		}
	};

	static Effect GetEffect()
	{
		return Effect(g_displayStyle,
			g_shadingMode,
			g_screenSpaceTess,
			g_fractionalSpacing,
			g_patchCull,
			g_singleCreasePatch);
	}

	// ---------------------------------------------------------------------------
	struct EffectDesc 
	{
		EffectDesc(OpenSubdiv::Far::PatchDescriptor desc,
		Effect effect) : desc(desc), effect(effect),
		maxValence(0), numElements(0) { }

		OpenSubdiv::Far::PatchDescriptor desc;
		Effect effect;
		int maxValence;
		int numElements;

		bool operator < (const EffectDesc &e) const 
		{
			return
				(desc < e.desc || ((desc == e.desc &&
				(maxValence < e.maxValence || ((maxValence == e.maxValence) &&
				(numElements < e.numElements || ((numElements == e.numElements) &&
				(effect < e.effect))))))));
		}
	};

	// ---------------------------------------------------------------------------

	class ShaderCache : public GLShaderCache<EffectDesc> 
	{
	public:
		virtual GLDrawConfig *CreateDrawConfig(EffectDesc const &effectDesc) 
		{

			using namespace OpenSubdiv;

			// compile shader program

			GLDrawConfig *config = new GLDrawConfig(GLUtils::GetShaderVersionInclude().c_str());

			Far::PatchDescriptor::Type type = effectDesc.desc.GetType();

			// common defines
			std::stringstream ss;

			if (type == Far::PatchDescriptor::QUADS) {
				ss << "#define PRIM_QUAD\n";
			}
			else {
				ss << "#define PRIM_TRI\n";
			}

			// OSD tessellation controls
			if (effectDesc.effect.screenSpaceTess) {
				ss << "#define OSD_ENABLE_SCREENSPACE_TESSELLATION\n";
			}
			if (effectDesc.effect.fractionalSpacing) {
				ss << "#define OSD_FRACTIONAL_ODD_SPACING\n";
			}
			if (effectDesc.effect.patchCull) {
				ss << "#define OSD_ENABLE_PATCH_CULL\n";
			}
			if (effectDesc.effect.singleCreasePatch) {
				ss << "#define OSD_PATCH_ENABLE_SINGLE_CREASE\n";
			}
			// for legacy gregory
			ss << "#define OSD_MAX_VALENCE " << effectDesc.maxValence << "\n";
			ss << "#define OSD_NUM_ELEMENTS " << effectDesc.numElements << "\n";

			// display styles
			switch (effectDesc.effect.displayStyle) {
			case kDisplayStyleWire:
				ss << "#define GEOMETRY_OUT_WIRE\n";
				break;
			case kDisplayStyleWireOnShaded:
				ss << "#define GEOMETRY_OUT_LINE\n";
				break;
			case kDisplayStyleShaded:
				ss << "#define GEOMETRY_OUT_FILL\n";
				break;
			}

			// shading mode
			switch (effectDesc.effect.shadingMode) {
			case kShadingMaterial:
				ss << "#define SHADING_MATERIAL\n";
				break;
			case kShadingVaryingColor:
				ss << "#define SHADING_VARYING_COLOR\n";
				break;
			case kShadingInterleavedVaryingColor:
				ss << "#define SHADING_VARYING_COLOR\n";
				break;
			case kShadingFaceVaryingColor:
				ss << "#define OSD_FVAR_WIDTH 2\n";
				ss << "#define SHADING_FACEVARYING_COLOR\n";
				if (!effectDesc.desc.IsAdaptive()) {
					ss << "#define SHADING_FACEVARYING_UNIFORM_SUBDIVISION\n";
				}
				break;
			case kShadingPatchType:
				ss << "#define SHADING_PATCH_TYPE\n";
				break;
			case kShadingPatchCoord:
				ss << "#define SHADING_PATCH_COORD\n";
				break;
			case kShadingNormal:
				ss << "#define SHADING_NORMAL\n";
				break;
			}

			if (type == Far::PatchDescriptor::TRIANGLES) {
				ss << "#define LOOP\n";
			}
			else if (type == Far::PatchDescriptor::QUADS) {
			}
			else {
				ss << "#define SMOOTH_NORMALS\n";
			}

			// need for patch color-coding : we need these defines in the fragment shader
			if (type == Far::PatchDescriptor::GREGORY) {
				ss << "#define OSD_PATCH_GREGORY\n";
			}
			else if (type == Far::PatchDescriptor::GREGORY_BOUNDARY) {
				ss << "#define OSD_PATCH_GREGORY_BOUNDARY\n";
			}
			else if (type == Far::PatchDescriptor::GREGORY_BASIS) {
				ss << "#define OSD_PATCH_GREGORY_BASIS\n";
			}

			// include osd PatchCommon
			ss << "#define OSD_PATCH_BASIS_GLSL\n";
			ss << Osd::GLSLPatchShaderSource::GetPatchBasisShaderSource();
			ss << Osd::GLSLPatchShaderSource::GetCommonShaderSource();
			std::string common = ss.str();
			ss.str("");

			// vertex shader
			ss << common
				// enable local vertex shader
				<< (effectDesc.desc.IsAdaptive() ? "" : "#define VERTEX_SHADER\n")
				<< shaderSource()
				<< Osd::GLSLPatchShaderSource::GetVertexShaderSource(type);
			config->CompileAndAttachShader(GL_VERTEX_SHADER, ss.str());
			ss.str("");

			if (effectDesc.desc.IsAdaptive()) {
				// tess control shader
				ss << common
					<< shaderSource()
					<< Osd::GLSLPatchShaderSource::GetTessControlShaderSource(type);
				config->CompileAndAttachShader(GL_TESS_CONTROL_SHADER, ss.str());
				ss.str("");

				// tess eval shader
				ss << common
					<< shaderSource()
					<< Osd::GLSLPatchShaderSource::GetTessEvalShaderSource(type);
				config->CompileAndAttachShader(GL_TESS_EVALUATION_SHADER, ss.str());
				ss.str("");
			}

			// geometry shader
			ss << common
				<< "#define GEOMETRY_SHADER\n"
				<< shaderSource();
			config->CompileAndAttachShader(GL_GEOMETRY_SHADER, ss.str());
			ss.str("");

			// fragment shader
			ss << common
				<< "#define FRAGMENT_SHADER\n"
				<< shaderSource();
			config->CompileAndAttachShader(GL_FRAGMENT_SHADER, ss.str());
			ss.str("");

			if (!config->Link()) {
				delete config;
				return NULL;
			}

			// assign uniform locations
			GLuint uboIndex;
			GLuint program = config->GetProgram();
			uboIndex = glGetUniformBlockIndex(program, "Transform");
			if (uboIndex != GL_INVALID_INDEX)
				glUniformBlockBinding(program, uboIndex, g_transformBinding);

			uboIndex = glGetUniformBlockIndex(program, "Tessellation");
			if (uboIndex != GL_INVALID_INDEX)
				glUniformBlockBinding(program, uboIndex, g_tessellationBinding);

			uboIndex = glGetUniformBlockIndex(program, "Lighting");
			if (uboIndex != GL_INVALID_INDEX)
				glUniformBlockBinding(program, uboIndex, g_lightingBinding);

			// assign texture locations
			GLint loc;
			glUseProgram(program);
			if ((loc = glGetUniformLocation(program, "OsdPatchParamBuffer")) != -1) {
				glUniform1i(loc, 0); // GL_TEXTURE0
			}
			if ((loc = glGetUniformLocation(program, "OsdFVarDataBuffer")) != -1) {
				glUniform1i(loc, 1); // GL_TEXTURE1
			}
			// for legacy gregory patches
			if ((loc = glGetUniformLocation(program, "OsdVertexBuffer")) != -1) {
				glUniform1i(loc, 2); // GL_TEXTURE2
			}
			if ((loc = glGetUniformLocation(program, "OsdValenceBuffer")) != -1) {
				glUniform1i(loc, 3); // GL_TEXTURE3
			}
			if ((loc = glGetUniformLocation(program, "OsdQuadOffsetBuffer")) != -1) {
				glUniform1i(loc, 4); // GL_TEXTURE4
			}
			glUseProgram(0);

			return config;
		}
	};

}