
/* renderer_nvpr_shader.cpp - shaders for use with NV_path_rendering path rendering */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include "nvpr_svg_config.h"

#if USE_NVPR

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <GL/glew.h>

#include <Cg/vector/xyzw.hpp>

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

#include "renderer_nvpr.hpp"

#include <Cg/mul.hpp>

// Release builds shouldn't have verbose conditions.
#ifdef NDEBUG
#define verbose (0)
#else
extern int verbose;
#endif

static const char *gradientProgramFileName = "gradient.cg";

static const char *myProgramName; // XXX put in header

void NVprRenderer::checkForCgError(const char *situation)
{
    CGerror error;
    const char *string = cgGetLastErrorString(&error);

    if (error != CG_NO_ERROR) {
        printf("%s: %s: %s\n",
            myProgramName, situation, string);
        if (error == CG_COMPILER_ERROR) {
            printf("%s\n", cgGetLastListing(myCgContext));
        }
        printf("Hit return to exit...");
        fflush(stdout);
        (void) getchar();
        exit(1);
    }
}

#ifdef NDEBUG
#define checkForCgError(str) /**/
#endif

void NVprRenderer::load_gradient_shaders()
{
    if (radial_center_gradient_program) {
        cgDestroyProgram(radial_center_gradient_program);
    }
    radial_center_gradient_program =
        cgCreateProgramFromFile(
        myCgContext,                /* Cg runtime context */
        CG_SOURCE,                  /* Program in human-readable form */
        gradientProgramFileName,    /* Name of file containing program */
        myCgFragmentProfile,        /* Profile: OpenGL ARB fragment program */
        "radial_center_gradient",          /* Entry function name */
        NULL);                      /* No extra compiler options */
    checkForCgError("creating radial center gradient fragment program from file");
    cgGLLoadProgram(radial_center_gradient_program);
    checkForCgError("loading radial center gradient fragment program");

    if (radial_focal_gradient_program) {
        cgDestroyProgram(radial_focal_gradient_program);
    }
    radial_focal_gradient_program =
        cgCreateProgramFromFile(
        myCgContext,                /* Cg runtime context */
        CG_SOURCE,                  /* Program in human-readable form */
        gradientProgramFileName,    /* Name of file containing program */
        myCgFragmentProfile,        /* Profile: OpenGL ARB fragment program */
        "radial_focal_gradient",          /* Entry function name */
        NULL);                      /* No extra compiler options */
    checkForCgError("creating radial focal gradient fragment program from file");
    cgGLLoadProgram(radial_focal_gradient_program);
    checkForCgError("loading radial focal gradient fragment program");
}

void NVprRenderer::init_cg(const char *vertex_profile_name,
                           const char *fragment_profile_name)
{
    // Have we already initialized a Cg context?
    if (!myCgContext) {
        myCgContext = cgCreateContext();
        cgGLSetDebugMode(CG_FALSE);  // don't call glGetError from Cg
        //cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);
        checkForCgError("creating context");
    }

    if (fragment_profile_name) {
        myCgFragmentProfile = cgGetProfile(fragment_profile_name);
    } else {
        myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    }
    cgGLSetOptimalOptions(myCgFragmentProfile);
    checkForCgError("selecting fragment profile");

    if (vertex_profile_name) {
        myCgVertexProfile = cgGetProfile(vertex_profile_name);
    } else {
        myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
    }
    cgGLSetOptimalOptions(myCgVertexProfile);
    checkForCgError("selecting vertex profile");
}

void NVprRenderer::report_cg_profiles()
{
    printf("NVpr: Cg vertex profile = %s\n", cgGetProfileString(myCgVertexProfile));
    printf("NVpr: Cg fragment profile = %s\n", cgGetProfileString(myCgFragmentProfile));
}

void NVprRenderer::load_shaders()
{
    load_gradient_shaders();
}

static void destroyProgram(CGprogram &program)
{
    if (program) {
        cgDestroyProgram(program);
    }
    program = 0;
}

void NVprRenderer::shutdown_shaders()
{
    destroyProgram(radial_center_gradient_program);
    destroyProgram(radial_focal_gradient_program);

    cgDestroyContext(myCgContext);
}

void NVprRenderer::enableFragmentShading()
{
    cgGLEnableProfile(myCgFragmentProfile);
    checkForCgError("enabling fragment profile");
}

void NVprRenderer::disableFragmentShading()
{
    cgGLDisableProfile(myCgFragmentProfile);
    checkForCgError("disabling fragment profile");
}

const char *NVprRenderer::getWindowTitle()
{
    return "NV_path_rendering path rendering";
}

const char *NVprRenderer::getName()
{
    return "NVpr";
}

bool NVprRenderer::configureShading(ShadingMode mode)
{
    switch (mode) {
    default:
        assert(!"invalid discard program mode");

    case RADIAL_CENTER_GRADIENT:
        if (radial_center_gradient_program) {
            cgGLBindProgram(radial_center_gradient_program);
            checkForCgError("binding radial center gradient fragment program");
            return true;
        }
        break;
    case RADIAL_FOCAL_GRADIENT:
        if (radial_focal_gradient_program) {
            cgGLBindProgram(radial_focal_gradient_program);
            checkForCgError("binding radial focal gradient fragment program");
            return true;
        }
        break;

    }

    return false;  // configuration unavailable
}

#endif // USE_NVPR
