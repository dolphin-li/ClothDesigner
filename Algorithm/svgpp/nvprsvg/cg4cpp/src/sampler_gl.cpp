/* 
 * Copyright 2008 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "sampler2D_state.hpp"

namespace Cg {

class __CGsampler_GL_factory {
    typedef struct {
        GLint swapbytes, rowlength, imageheight;
        GLint skiprows, skippixels, skipimages, alignment;
    } PixelStoreState;

    // Static helper methods for initialization
    static void initExtensionFuncs();
    static void savePackPixelStoreState(PixelStoreState &state);
    static void restorePackPixelStoreState(PixelStoreState &state);

protected:
    GLenum target;

    __CGsampler_GL_factory(GLenum target_) : target(target_) { initExtensionFuncs(); }

    void initSamplerState(__CGsampler_state *state);
};

class __CGsampler2D_GL_factory : public __CGsampler2D_factory, __CGsampler_GL_factory {
protected:
    __CGsampler2D_GL_factory() : __CGsampler_GL_factory(GL_TEXTURE_2D) { }
    void doFetch(__CGsampler2D_state *state);
};

void __CGsampler2D_GL_factory::doFetch(__CGsampler2D_state *state)
{
    initSamplerState(state);
}

class __CGsampler_GLTextureUnit_factory : __CGsampler_GL_factory {
    GLuint texUnit;
    GLint savedActiveTexture;  // save/restore temporary used by beginFetch & endFetch

protected:
    __CGsampler_GLTextureUnit_factory(GLenum target_, GLuint _texUnit)
        : __CGsampler_GL_factory(target_), texUnit(_texUnit) {};

    void beginFetch();
    void endFetch();
};

#if defined(__APPLE__) || defined(__ADM__)
   // No "GetProcAddress" required.
#else

#ifndef _WIN32
#include <GL/glx.h>  // for glXGetProcAddress
#endif

extern "C" {
# ifdef _WIN32  // Win32/WGL
    typedef int (__stdcall *PROC)();
    __declspec(dllimport) PROC __stdcall wglGetProcAddress(const char *);
#  define GET_PROC_ADDRESS(x) wglGetProcAddress(x)
#  define GL_STDCALL __stdcall
# else  // X11/GLX
#  define GET_PROC_ADDRESS(x) glXGetProcAddress((const GLubyte *)x)  // for Linux
#  define GL_STDCALL
# endif

    typedef void (GL_STDCALL * PFNGLACTIVETEXTUREARBPROC) (GLenum target);
    typedef void (GL_STDCALL * PFNGLGETBUFFERSUBDATAPROC) (GLenum target, GLintptr offset, GLsizeiptr size, GLvoid *data);
    typedef void (GL_STDCALL * PFNGLBINDBUFFERARBPROC) (GLenum target, GLuint buffer);

    static PFNGLACTIVETEXTUREARBPROC glActiveTexture;
    static PFNGLGETBUFFERSUBDATAPROC glGetBufferSubData;
    static PFNGLBINDBUFFERPROC glBindBuffer;
}

#endif

void __CGsampler_GL_factory::initExtensionFuncs()
{
# if defined(__APPLE__) || defined(__ADM__)
   // No "GetProcAddress" required.
#else
    static int funcsInited = 0;

    if (funcsInited) {
        // Already called
        return;
    }

    glActiveTexture = (PFNGLACTIVETEXTUREARBPROC) GET_PROC_ADDRESS("glActiveTexture");
    if (0 == glActiveTexture) {
        glActiveTexture = (PFNGLACTIVETEXTUREARBPROC) GET_PROC_ADDRESS("glActiveTextureARB");
    }

    glGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC) GET_PROC_ADDRESS("glGetBufferSubData");
    if (0 == glGetBufferSubData) {
        glGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC) GET_PROC_ADDRESS("glGetBufferSubDataARB");
    }

    glBindBuffer = (PFNGLBINDBUFFERPROC) GET_PROC_ADDRESS("glBindBuffer");
    if (0 == glBindBuffer) {
        glBindBuffer = (PFNGLBINDBUFFERPROC) GET_PROC_ADDRESS("glBindBufferARB");
    }

    funcsInited = 1;
#endif
}

void __CGsampler_GL_factory::savePackPixelStoreState(PixelStoreState &state)
{
    glGetIntegerv(GL_PACK_SWAP_BYTES, &state.swapbytes);
    glGetIntegerv(GL_PACK_ROW_LENGTH, &state.rowlength);
    glGetIntegerv(GL_PACK_IMAGE_HEIGHT, &state.imageheight);
    glGetIntegerv(GL_PACK_SKIP_ROWS, &state.skiprows);
    glGetIntegerv(GL_PACK_SKIP_PIXELS, &state.skippixels);
    glGetIntegerv(GL_PACK_SKIP_IMAGES, &state.skipimages);
    glGetIntegerv(GL_PACK_ALIGNMENT, &state.alignment);

    glPixelStorei(GL_PACK_SWAP_BYTES, GL_FALSE);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_IMAGE_HEIGHT, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SKIP_IMAGES, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
}

void __CGsampler_GL_factory::restorePackPixelStoreState(PixelStoreState &state)
{
    glPixelStorei(GL_PACK_SWAP_BYTES, state.swapbytes);
    glPixelStorei(GL_PACK_ROW_LENGTH, state.rowlength);
    glPixelStorei(GL_PACK_IMAGE_HEIGHT, state.imageheight);
    glPixelStorei(GL_PACK_SKIP_ROWS, state.skiprows);
    glPixelStorei(GL_PACK_SKIP_PIXELS, state.skippixels);
    glPixelStorei(GL_PACK_SKIP_IMAGES, state.skipimages);
    glPixelStorei(GL_PACK_ALIGNMENT, state.alignment);
}

void __CGsampler_GL_factory::initSamplerState(__CGsampler_state *state)
{
    glGetTexParameteriv(target, GL_TEXTURE_WRAP_S, (GLint*) &state->wrapS);
    glGetTexParameteriv(target, GL_TEXTURE_WRAP_T, (GLint*) &state->wrapT);
    glGetTexParameteriv(target, GL_TEXTURE_WRAP_R, (GLint*) &state->wrapR);

    glGetTexParameteriv(target, GL_TEXTURE_BASE_LEVEL, &state->baseLevel);
    glGetTexParameteriv(target, GL_TEXTURE_MAX_LEVEL, &state->maxLevel);

    glGetTexParameterfv(target, GL_TEXTURE_MIN_LOD, &state->minLod);
    glGetTexParameterfv(target, GL_TEXTURE_MAX_LOD, &state->maxLod);

    glGetTexParameterfv(target, GL_TEXTURE_LOD_BIAS, &state->lodBias);

    glGetTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, (GLfloat*) &state->borderValues);

    glGetTexParameteriv(target, GL_TEXTURE_MIN_FILTER, (GLint*) &state->minFilter);
    glGetTexParameteriv(target, GL_TEXTURE_MAG_FILTER, (GLint*) &state->magFilter);

    glGetTexParameteriv(target, GL_TEXTURE_COMPARE_MODE, (GLint*) &state->compareMode);
    glGetTexParameteriv(target, GL_TEXTURE_COMPARE_FUNC, (GLint*) &state->compareFunc);
    glGetTexParameteriv(target, GL_DEPTH_TEXTURE_MODE, (GLint*) &state->depthTextureMode);

    if ((state->magFilter == GL_LINEAR) &&
        ((state->minFilter == GL_NEAREST_MIPMAP_NEAREST) ||
         (state->minFilter == GL_LINEAR_MIPMAP_NEAREST))) {
        state->magnifyTransition = 0.5f;
    } else {
        state->magnifyTransition = 0.0f;
    }

    state->clampedLodBias = state->lodBias;

    state->trueBaseLevel = state->baseLevel;  // needs clamping
    state->effectiveMaxLevel = state->maxLevels;  // needs clamping

    state->clampedBorderValues = float4(state->borderValues[0],
                                       state->borderValues[1],
                                       state->borderValues[2],
                                       state->borderValues[3]);  // needs clamping based on format

    state->prefilterMode = GL_NONE;
}

void __CGsampler_GLTextureUnit_factory::beginFetch()
{
    glGetIntegerv(GL_ACTIVE_TEXTURE, &savedActiveTexture);
    if (savedActiveTexture != texUnit) {
        glActiveTexture(texUnit);
    }
}

void __CGsampler_GLTextureUnit_factory::endFetch()
{
    if (savedActiveTexture != texUnit) {
        glActiveTexture(savedActiveTexture);
    }
}

class __CGsampler_GLTextureObject_factory : __CGsampler_GL_factory {
    GLuint texobj;
    GLint savedTexobj;  // save/restore temporary used by beginFetch & endFetch

protected:
    __CGsampler_GLTextureObject_factory(GLenum target_, GLuint texobj_)
        : __CGsampler_GL_factory(target_), texobj(texobj_) {};

    void beginFetch();
    void endFetch();
};

void __CGsampler_GLTextureObject_factory::beginFetch()
{
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTexobj);
    if (savedTexobj != texobj) {
        glBindTexture(target, texobj);
    }
}

void __CGsampler_GLTextureObject_factory::endFetch()
{
    if (savedTexobj != texobj) {
        glBindTexture(target, savedTexobj);
    }
}

class __CGsampler2D_GLTextureUnit_factory : public __CGsampler2D_GL_factory, __CGsampler_GLTextureUnit_factory {
public:
    __CGsampler2D_GLTextureUnit_factory(GLuint texUnit)
        : __CGsampler_GLTextureUnit_factory(GL_TEXTURE_2D, texUnit) {}

    virtual __CGsampler2D_state *construct();
};

__CGsampler2D_state *__CGsampler2D_GLTextureUnit_factory::construct()
{
    __CGsampler2D_state *state = new __CGsampler2D_state;

    beginFetch();
    doFetch(state);
    endFetch();
    return state;
}

class __CGsampler2D_GLTextureObject_factory : public __CGsampler2D_GL_factory, __CGsampler_GLTextureObject_factory {
public:
    __CGsampler2D_GLTextureObject_factory(GLuint texobj_)
        : __CGsampler_GLTextureObject_factory(GL_TEXTURE_2D, texobj_) {}

    virtual __CGsampler2D_state *construct();
};

__CGsampler2D_state *__CGsampler2D_GLTextureObject_factory::construct()
{
    __CGsampler2D_state *state = new __CGsampler2D_state;

    beginFetch();
    doFetch(state);
    endFetch();
    return state;
}

sampler2D Sampler2DFromGLTextureUnit(GLenum texUnit)
{
    __CGsampler2D_GLTextureUnit_factory factory(texUnit);

    return sampler2D(factory);
}

sampler2D Sampler2DFromGLTextureObject(GLuint texobj)
{
    __CGsampler2D_GLTextureObject_factory factory(texobj);

    return sampler2D(factory);
}

sampler2D Sampler2DFromGLActiveTexture();

} // namespace Cg
