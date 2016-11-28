/* 
 * Copyright 2006 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
/* 
 * Copyright 2005 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "sampler_state.hpp"

namespace Cg {

int __CGsampler_state::modulo(int numer, int denom)
{
    int remainder = numer % denom;
    return remainder < 0 ? denom + remainder : remainder;
}

int1 __CGsampler_state::mirrorNearest(int numer, int denom)
{
    int div = numer / denom;
    int remainder = numer % denom;

    /* Compensate for the fact that C rounds integer division to zero
       when, in this case, we want to round to negative infinity. */
    if (remainder < 0) {
        return (div & 1) ? denom+remainder  /* even */
                         : -1-remainder;    /* odd */
    } else {
        return (div & 1) ? denom-1 - remainder  /* odd */
                         : remainder;           /* even */
    }
}

/* Like "mirrorNearest" but returns not only the mirror(f) but also mirror(f+1) as int2. */
int2 __CGsampler_state::mirrorLinear(int numer, int denom)
{
    int div = numer / denom;
    int remainder = numer % denom;
    int edge, direction;
    int2 loc;

    /* Compensate for the fact that C rounds integer division to zero
       when, in this case, we want to round to negative infinity. */
    if (remainder < 0) {
        if (div & 1) {
            /* even */
            loc[0] = denom+remainder;
            edge = denom-1;
            direction = +1;
        } else {
            /* odd */
            loc[0] = -1-remainder;
            edge = 0;
            direction = -1;
        }
    } else {
        if (div & 1) {
            /* odd */
            loc[0] = denom-1 - remainder;
            edge = 0;
            direction = -1;
        } else {
            /* even */
            loc[0] = remainder;
            edge = denom-1;
            direction = +1;
        }
    }
    if (loc[0] == edge) {
        loc[1] = edge;
    } else {
        loc[1] = loc[0] + direction;
    }
    return loc;
}

float4 __CGimage::fetch(int1 u, int1 v, int1 p, const float4 &borderValues) const
{
    int offset;
    float hemi;

    u += borderSize.x;
    v += borderSize.y;
    p += borderSize.z;

    int useBorderValues = (u < 0 || v < 0 || p < 0 || u >= width || v >= height || p >= depth);
    
    offset  = p;
    offset *= height;
    offset += v;
    offset *= width;
    offset += u;
    offset *= components;

    float4 result;
    switch (format) {
    case GL_DSDT_NV:
    case GL_HILO_NV:
        if (useBorderValues) {
            result.rg = borderValues.rg;
        } else {
            result.r = data[offset + 0];
            result.g = data[offset + 1];
        }
        result.b = 0.0;
        result.a = 1.0f;
        break;
    case GL_SIGNED_HILO_NV:
        if (useBorderValues) {
            result.rgb = borderValues.rgb;
        } else {
            result.r = data[offset + 0];
            result.g = data[offset + 1];
            hemi = 1 - result.r*result.r - result.g*result.g;
            if (hemi < 0) {
                hemi = 0;
            }
            result.b = sqrt(hemi);
        }
        result.a = 1.0f;
        break;
    case GL_LUMINANCE:
#ifdef GL_EXT_texture_sRGB
    case GL_SLUMINANCE_EXT:
#endif
        if (useBorderValues) {
            result.r = borderValues.r;
        } else {
            result.r = data[offset + 0];
        }
        result.g = result.r;
        result.b = result.r;
        result.a = 1.0f;
        break;
    case GL_INTENSITY:
        if (useBorderValues) {
            result.r = borderValues.r;
        } else {
            result.r = data[offset + 0];
        }
        result.g = result.r;
        result.b = result.r;
        result.a = result.r;
        break;
    case GL_ALPHA:
        result.r = 0;
        result.g = 0;
        result.b = 0;
        if (useBorderValues) {
            result.a = borderValues.a;
        } else {
            result.a = data[offset + 0];
        }
        break;
    case GL_LUMINANCE_ALPHA:
#ifdef GL_EXT_texture_sRGB
    case GL_SLUMINANCE_ALPHA_EXT:
#endif
        if (useBorderValues) {
            result.r = borderValues.r;
            result.a = borderValues.a;
        } else {
            result.r = data[offset + 0];
            result.a = data[offset + 1];
        }
        result.g = result.r;
        result.b = result.r;
        break;
#ifdef GL_NV_float_buffer
    case GL_FLOAT_R_NV:
        if (useBorderValues) {
            result.r = borderValues.r;
        } else {
            result.r = data[offset + 0];
        }
        result.g = 0.0;
        result.b = 0.0;
        result.a = 1.0f;
        break;
    case GL_FLOAT_RG_NV:
        if (useBorderValues) {
            result.rg = borderValues.rg;
        } else {
            result.r = data[offset + 0];
            result.g = data[offset + 1];
        }
        result.b = 0.0;
        result.a = 1.0f;
        break;
#endif
    case GL_RGB:
    case GL_DSDT_MAG_NV:
#ifdef GL_EXT_texture_sRGB
    case GL_SRGB_EXT:
#endif
        if (useBorderValues) {
            result.rgb = borderValues.rgb;
        } else {
            result.r = data[offset + 0];
            result.g = data[offset + 1];
            result.b = data[offset + 2];
        }
        result.a = 1.0f;
        break;
    case GL_RGBA:
    case GL_DSDT_MAG_INTENSITY_NV:
#ifdef GL_EXT_texture_sRGB
    case GL_SRGB_ALPHA_EXT:
#endif
        if (useBorderValues) {
            result = borderValues;
        } else {
            result.r = data[offset + 0];
            result.g = data[offset + 1];
            result.b = data[offset + 2];
            result.a = data[offset + 3];
        }
        break;
    case GL_DEPTH_COMPONENT:
        // Behave like intensity for now; see prefilter.
        if (useBorderValues) {
            result.r = borderValues.r;
        } else {
            result.r = data[offset + 0];
        }
        result.g = result.r;
        result.b = result.r;
        result.a = result.r;
        break;

#ifdef GL_EXT_texture_compression_rgtc
    case GL_COMPRESSED_RED_RGTC1_EXT:
    case GL_COMPRESSED_SIGNED_RED_RGTC1_EXT:
        if (useBorderValues) {
            result.r = borderValues.r;
        } else {
            result.r = data[offset + 0];
        }
        result.g = 0.0;
        result.b = 0.0;
        result.a = 1.0f;
        break;
    case GL_COMPRESSED_RED_GREEN_RGTC2_EXT:
    case GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT:
        if (useBorderValues) {
            result.rg = borderValues.rg;
        } else {
            result.r = data[offset + 0];
            result.g = data[offset + 1];
        }
        result.b = 0.0;
        result.a = 1.0f;
        break;
#endif

    default:
        assert(!"unexpected format");
        break;
    }
    return result;
}

void __CGimage::deriveFormatBasedState()
{
    // Initial assumption about clamping.
    clamped = false;
    clampMin = 0;
    clampMax = 1;

    switch (internalFormat) {
    case GL_DSDT_NV:
    case GL_DSDT8_NV:
        components = 2;
        format = GL_DSDT_NV;
        clampMin = -1;
        break;
    case GL_DSDT_MAG_NV:
    case GL_DSDT8_MAG8_NV:
        components = 3;
        format = GL_DSDT_MAG_NV;
        clampMin.xy = -1;  // mag is [0,1]
        break;
    case GL_DSDT_MAG_INTENSITY_NV:
    case GL_DSDT8_MAG8_INTENSITY8_NV:
        components = 4;
        format = GL_DSDT_MAG_INTENSITY_NV;
        clampMin.xy = -1;  // mag and intensity are [0,1]
        break;
    case GL_HILO_NV:
    case GL_HILO8_NV:
    case GL_HILO16_NV:
        components = 2;
        format = GL_HILO_NV;
        break;
    case GL_SIGNED_HILO_NV:
    case GL_SIGNED_HILO8_NV:
    case GL_SIGNED_HILO16_NV:
        components = 2;
        format = GL_SIGNED_HILO_NV;
        clampMin = -1;
        break;
    case 1:
    case GL_LUMINANCE4:
    case GL_LUMINANCE8:
    case GL_LUMINANCE12:
    case GL_LUMINANCE16:
    case GL_COMPRESSED_LUMINANCE:
        components = 1;
        format = GL_LUMINANCE;
        break;
    case GL_SIGNED_LUMINANCE_NV:
    case GL_SIGNED_LUMINANCE8_NV:
        components = 1;
        format = GL_LUMINANCE;
        clampMin = -1;
        break;
#ifdef GL_ARB_texture_float
    case GL_LUMINANCE16F_ARB:
    case GL_LUMINANCE32F_ARB:
        components = 1;
        format = GL_LUMINANCE;
        clamped = false;
        break;
#endif
    case 2:
    case GL_LUMINANCE4_ALPHA4:
    case GL_LUMINANCE8_ALPHA8:
    case GL_LUMINANCE12_ALPHA12:
    case GL_LUMINANCE16_ALPHA16:
    case GL_COMPRESSED_LUMINANCE_ALPHA:
        components = 2;
        format = GL_LUMINANCE_ALPHA;
        break;
#ifdef GL_ARB_texture_float
    case GL_LUMINANCE_ALPHA16F_ARB:
    case GL_LUMINANCE_ALPHA32F_ARB:
        components = 2;
        format = GL_LUMINANCE_ALPHA;
        clamped = false;
        break;
#endif
    case GL_ALPHA:
    case GL_ALPHA4:
    case GL_ALPHA8:
    case GL_ALPHA12:
    case GL_ALPHA16:
    case GL_COMPRESSED_ALPHA:
        components = 1;
        format = GL_ALPHA;
        break;
    case GL_SIGNED_ALPHA_NV:
    case GL_SIGNED_ALPHA8_NV:
        components = 1;
        format = GL_ALPHA;
        clampMin = -1;
        break;
#ifdef GL_ARB_texture_float
    case GL_ALPHA16F_ARB:
    case GL_ALPHA32F_ARB:
        components = 1;
        format = GL_ALPHA;
        clamped = false;
        break;
#endif
    case GL_INTENSITY:
    case GL_INTENSITY4:
    case GL_INTENSITY8:
    case GL_INTENSITY12:
    case GL_INTENSITY16:
    case GL_COMPRESSED_INTENSITY:
        components = 1;
        format = GL_INTENSITY;
        break;
    case GL_SIGNED_INTENSITY_NV:
    case GL_SIGNED_INTENSITY8_NV:
        components = 1;
        format = GL_INTENSITY;
        clampMin = -1;
        break;
#ifdef GL_ARB_texture_float
    case GL_INTENSITY16F_ARB:
    case GL_INTENSITY32F_ARB:
        components = 1;
        format = GL_INTENSITY;
        clamped = false;
        break;
#endif
#ifdef GL_NV_float_buffer
    case GL_FLOAT_R_NV:
    case GL_FLOAT_R16_NV:
    case GL_FLOAT_R32_NV:
        components = 1;
        format = GL_FLOAT_R_NV;
        clamped = false;
        break;
    case GL_FLOAT_RG_NV:
    case GL_FLOAT_RG16_NV:
    case GL_FLOAT_RG32_NV:
        components = 2;
        format = GL_FLOAT_RG_NV;
        clamped = false;
        break;
#endif
    case 3:
    case GL_RGB:
    case GL_RGB4:
    case GL_RGB5:
    case GL_RGB8:
    case GL_RGB10:
    case GL_RGB12:
    case GL_RGB16:
    case GL_COMPRESSED_RGB:
    case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        components = 3;
        format = GL_RGB;
        break;
    case GL_SIGNED_RGB_NV:
    case GL_SIGNED_RGB8_NV:
        components = 3;
        format = GL_RGB;
        clampMin = -1;
        break;
    case GL_SIGNED_RGB_UNSIGNED_ALPHA_NV:
    case GL_SIGNED_RGB8_UNSIGNED_ALPHA8_NV:
        components = 4;
        format = GL_RGB;
        clampMin.rgb = -1;  // alpha is [0,1]
        break;
#ifdef GL_NV_float_buffer
    case GL_FLOAT_RGB_NV:
    case GL_FLOAT_RGB16_NV:
    case GL_FLOAT_RGB32_NV:
        components = 3;
        format = GL_RGB;
        clamped = false;
        break;
#endif
#ifdef GL_ARB_texture_float
    case GL_RGB16F_ARB:
    case GL_RGB32F_ARB:
        components = 3;
        format = GL_RGB;
        clamped = false;
        break;
#endif
    case 4:
    case GL_RGBA:
    case GL_RGBA4:
    case GL_RGBA8:
    case GL_RGB10_A2:
    case GL_RGBA12:
    case GL_RGBA16:
    case GL_COMPRESSED_RGBA:
    case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:
    case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:
    case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
        components = 4;
        format = GL_RGBA;
        break;
    case GL_SIGNED_RGBA_NV:
    case GL_SIGNED_RGBA8_NV:
        components = 4;
        format = GL_RGBA;
        clampMin = -1;
        break;
#ifdef GL_NV_float_buffer
    case GL_FLOAT_RGBA_NV:
    case GL_FLOAT_RGBA16_NV:
    case GL_FLOAT_RGBA32_NV:
        components = 4;
        format = GL_RGBA;
        clamped = false;
        break;
#endif
#ifdef GL_ARB_texture_float
    case GL_RGBA16F_ARB:
    case GL_RGBA32F_ARB:
        components = 4;
        format = GL_RGBA;
        clamped = false;
        break;
#endif

#ifdef GL_EXT_texture_sRGB
    // sRGB formats
    case GL_SLUMINANCE_EXT:
    case GL_SLUMINANCE8_EXT:
    case GL_COMPRESSED_SLUMINANCE_EXT:
        components = 1;
        format = GL_SLUMINANCE_EXT;
        break;
    case GL_SLUMINANCE_ALPHA_EXT:
    case GL_SLUMINANCE8_ALPHA8_EXT:
    case GL_COMPRESSED_SLUMINANCE_ALPHA_EXT:
        components = 2;
        format = GL_SLUMINANCE_ALPHA_EXT;
        break;
    case GL_SRGB_EXT:
    case GL_SRGB8_EXT:
    case GL_COMPRESSED_SRGB_EXT:
    case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
        components = 3;
        format = GL_SRGB_EXT;
        break;
    case GL_SRGB_ALPHA_EXT:
    case GL_SRGB8_ALPHA8_EXT:
    case GL_COMPRESSED_SRGB_ALPHA_EXT:
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
        components = 4;
        format = GL_SRGB_ALPHA_EXT;
        break;
#endif

#ifdef GL_EXT_texture_shared_exponent
    case GL_RGB9_E5_EXT:
        components = 3;
        format = GL_RGB;
        clamped = false;
        break;
#endif

#ifdef GL_EXT_packed_float
    case GL_R11F_G11F_B10F_EXT:
        components = 3;
        format = GL_RGB;
        clamped = false;
        break;
#endif

#ifdef GL_EXT_texture_compression_latc
    case GL_COMPRESSED_LUMINANCE_LATC1_EXT:
        components = 1;
        format = GL_LUMINANCE;
        break;
    case GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT:
        components = 1;
        format = GL_LUMINANCE;
        clampMin = -1;
        break;
    case GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT:
        components = 2;
        format = GL_LUMINANCE_ALPHA;
        break;
    case GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT:
        components = 2;
        format = GL_LUMINANCE_ALPHA;
        clampMin = -1;
        break;
#endif

#ifdef GL_EXT_texture_compression_rgtc
    case GL_COMPRESSED_RED_RGTC1_EXT:
        components = 1;
        format = GL_COMPRESSED_RED_RGTC1_EXT;
        break;
    case GL_COMPRESSED_SIGNED_RED_RGTC1_EXT:
        components = 1;
        format = GL_COMPRESSED_SIGNED_RED_RGTC1_EXT;
        clampMin = -1;
        break;
    case GL_COMPRESSED_RED_GREEN_RGTC2_EXT:
        components = 2;
        format = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;
        break;
    case GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT:
        components = 2;
        format = GL_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
        clampMin = -1;
        break;
#endif

    // Depth formats
    case GL_DEPTH_COMPONENT:
    case GL_DEPTH_COMPONENT16:
    case GL_DEPTH_COMPONENT24:
    case GL_DEPTH_COMPONENT32:
        components = 1;
        format = GL_DEPTH_COMPONENT;
        break;

    default:
        assert(!"unexpected internal format");
        components = 4;
        format = GL_RGBA;
        break;
    }
}

#if defined(__APPLE__) || defined(__ADM__)
   // No "GetProcAddress" required.
#else

#ifndef _WIN32
# include <GL/glx.h>  // for glXGetProcAddress
#endif

extern "C" {
# ifdef _WIN32  // Win32/WGL
    typedef int (__stdcall *PROC)();
    __declspec(dllimport) PROC __stdcall wglGetProcAddress(const char *);
#  define GET_PROC_ADDRESS(x) wglGetProcAddress(x)
#  define GL_STDCALL __stdcall
# else  // X11/GLX
#  define GET_PROC_ADDRESS(x) glXGetProcAddress((const GLubyte*)x)  // for Linux
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

void __CGsampler_state::initExtensionFuncs()
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

void __CGimage::initImage(GLenum target, GLint level, int3 targetBorderSupport)
{
    // Query API state
    glGetTexLevelParameteriv(target, level, GL_TEXTURE_INTERNAL_FORMAT, (GLint*) &internalFormat);
    glGetTexLevelParameteriv(target, level, GL_TEXTURE_WIDTH, (GLint*) &width);
    glGetTexLevelParameteriv(target, level, GL_TEXTURE_HEIGHT, (GLint*) &height);
    glGetTexLevelParameteriv(target, level, GL_TEXTURE_DEPTH, (GLint*) &depth);
    glGetTexLevelParameteriv(target, level, GL_TEXTURE_BORDER, (GLint*) &border);

    // Now compute API-derived state
    deriveFormatBasedState();

    // Target border support is 1 if a the dimension for the target supports borders, and 0 if not
    borderSize = border * targetBorderSupport;

    int3 size = int3(width, height, depth);
    borderlessSize = size - 2*borderSize;
    borderlessSizeF = float3(borderlessSize);

    // Delete any prior image data.
    delete [] data;

    int totalComponents = width*height*depth * components;
    if (totalComponents > 0) {
        // Allocate space for texture image
        data = new float[totalComponents];

        if (target == GL_TEXTURE_BUFFER_EXT) {
            // Query the buffer object data.
            GLint currentBuffer, buffer;

            glGetIntegerv(GL_TEXTURE_BINDING_BUFFER_EXT, &currentBuffer);
            glGetIntegerv(GL_TEXTURE_BUFFER_DATA_STORE_BINDING_EXT, &buffer);
#ifndef NDEBUG
            // Format of texture buffer should match texture's internal format.
            GLint format;
            glGetIntegerv(GL_TEXTURE_BUFFER_FORMAT_EXT, &format);
            assert(format == internalFormat);
#endif
            glBindBuffer(GL_TEXTURE_BUFFER_EXT, buffer);
            GLsizeiptr bytes = sizeof(GLfloat) * totalComponents;
            const GLintptrARB theBeginning = 0;
            // XXXmjk Need to convert format!
            glGetBufferSubData(GL_TEXTURE_BUFFER_EXT, theBeginning, bytes, data);
            glBindBuffer(GL_TEXTURE_BUFFER_EXT, currentBuffer);
        } else {
            // Query the texture image
            glGetTexImage(target, level, format, GL_FLOAT, data);
        }
    } else {
        data = NULL;
    }
}

void __CGsampler_state::initSampler()
{
    glGetTexParameteriv(texTarget, GL_TEXTURE_WRAP_S, (GLint*) &wrapS);
    glGetTexParameteriv(texTarget, GL_TEXTURE_WRAP_T, (GLint*) &wrapT);
    glGetTexParameteriv(texTarget, GL_TEXTURE_WRAP_R, (GLint*) &wrapR);

    glGetTexParameteriv(texTarget, GL_TEXTURE_BASE_LEVEL, &baseLevel);
    glGetTexParameteriv(texTarget, GL_TEXTURE_MAX_LEVEL, &maxLevel);

    glGetTexParameterfv(texTarget, GL_TEXTURE_MIN_LOD, &minLod);
    glGetTexParameterfv(texTarget, GL_TEXTURE_MAX_LOD, &maxLod);

    glGetTexParameterfv(texTarget, GL_TEXTURE_LOD_BIAS, &lodBias);

    glGetTexParameterfv(texTarget, GL_TEXTURE_BORDER_COLOR, (GLfloat*) &borderValues);

    glGetTexParameteriv(texTarget, GL_TEXTURE_MIN_FILTER, (GLint*) &minFilter);
    glGetTexParameteriv(texTarget, GL_TEXTURE_MAG_FILTER, (GLint*) &magFilter);

    glGetTexParameteriv(texTarget, GL_TEXTURE_COMPARE_MODE, (GLint*) &compareMode);
    glGetTexParameteriv(texTarget, GL_TEXTURE_COMPARE_FUNC, (GLint*) &compareFunc);
    glGetTexParameteriv(texTarget, GL_DEPTH_TEXTURE_MODE, (GLint*) &depthTextureMode);

    if ((magFilter == GL_LINEAR) &&
        ((minFilter == GL_NEAREST_MIPMAP_NEAREST) ||
         (minFilter == GL_LINEAR_MIPMAP_NEAREST))) {
        magnifyTransition = 0.5f;
    } else {
        magnifyTransition = 0.0f;
    }

    clampedLodBias = lodBias;

    trueBaseLevel = baseLevel;  // needs clamping
    effectiveMaxLevel = maxLevels;  // needs clamping

    clampedBorderValues = float4(borderValues[0],
                                 borderValues[1],
                                 borderValues[2],
                                 borderValues[3]);  // needs clamping based on format

    prefilterMode = GL_NONE;
}

// Operations on texels prior to filtering (shadow mapping, paletted textures)
float4 __CGsampler_state::prefilter(const float4 &texel, float1 r)
{
    switch (prefilterMode) {
    case GL_DEPTH_COMPONENT:
        {
            float4 newTexel;
            float1 value;

            switch (compareMode) {
            case GL_NONE:
                // When compareMode is GL_NONE or texture format not depth component
                value = texel.r;
                break;
            case GL_COMPARE_R_TO_TEXTURE:
                switch (compareMode) {
                case GL_EQUAL:
                    value = (r == texel.r);
                    break;
                }
                break;
            }
            switch (depthTextureMode) {
            case GL_LUMINANCE:
                newTexel.r = value;
                newTexel.g = value;
                newTexel.b = value;
                newTexel.a = 1;
                break;
            case GL_ALPHA:
                newTexel.r = 0;
                newTexel.g = 0;
                newTexel.b = 0;
                newTexel.a = value;
                break;
            case GL_INTENSITY:
                newTexel.r = value;
                newTexel.g = value;
                newTexel.b = value;
                newTexel.a = value;
                break;
            }
            return newTexel;
        }
#ifdef GL_EXT_texture_sRGB
    case GL_SRGB_EXT:
        {
            float4 newTexel;
            // Apply sRGB-to-linear conversion to red, green, and blue
            for (int i=0; i<3; i++) {
                if (texel[i] <= 0.04045) {
                    newTexel[i] = texel[i] / 12.92f;
                } else {
                    newTexel[i] = pow((texel[i] + 0.055f)/1.055f, 2.4f);
                }
            }
            newTexel.a = texel.a;  // Alpha is not sRGB-to-linear converted
            return newTexel;
        }
#endif
    case GL_NONE:
        return texel;
    default:
        assert(!"unexpected prefilter mode");
        return texel;
    }
}

void __CGsampler_state::savePackPixelStoreState(PixelStoreState &state)
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

void __CGsampler_state::restorePackPixelStoreState(PixelStoreState &state)
{
    glPixelStorei(GL_PACK_SWAP_BYTES, state.swapbytes);
    glPixelStorei(GL_PACK_ROW_LENGTH, state.rowlength);
    glPixelStorei(GL_PACK_IMAGE_HEIGHT, state.imageheight);
    glPixelStorei(GL_PACK_SKIP_ROWS, state.skiprows);
    glPixelStorei(GL_PACK_SKIP_PIXELS, state.skippixels);
    glPixelStorei(GL_PACK_SKIP_IMAGES, state.skipimages);
    glPixelStorei(GL_PACK_ALIGNMENT, state.alignment);
}

void __CGsampler_state::initDerivedSampler(GLenum target)
{
    glGetIntegerv(GL_ACTIVE_TEXTURE, (GLint*) &texUnit);
    texTarget = target;

    initSampler();
    {
        PixelStoreState state;
        savePackPixelStoreState(state);
        {
            initImages();
        }
        restorePackPixelStoreState(state);
    }
}

void __CGsampler_state::initDerivedSampler(GLenum target, int unit)
{
    initExtensionFuncs();

    texUnit = unit + GL_TEXTURE0;
    texTarget = target;
    {
        GLint activeTexture;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
        {
            glActiveTexture(texUnit);
            initSampler();
            {
                PixelStoreState state;
                savePackPixelStoreState(state);
                {
                    initImages();
                }
                restorePackPixelStoreState(state);
            }
        }
        glActiveTexture(activeTexture);
    }
}

int1 __CGsampler_state::wrapNearest(GLenum wrapMode, int1 size, float1 coord)
{
    int pos = int(floor(coord));

    switch (wrapMode) {
    case GL_REPEAT:
        return modulo(pos, size);
    case GL_CLAMP_TO_BORDER:
        return clamp(pos, -1, size);
    case GL_CLAMP:
    case GL_CLAMP_TO_EDGE:
        return clamp(pos, 0, size-1);
    case GL_MIRRORED_REPEAT:
        return mirrorNearest(pos, size);
    case GL_MIRROR_CLAMP_EXT:
    case GL_MIRROR_CLAMP_TO_EDGE_EXT:
        pos = int(floor(abs(coord)));
        if (pos >= size) pos = size - 1;
        return pos;
    case GL_MIRROR_CLAMP_TO_BORDER_EXT:
        pos = int(floor(abs(coord)));
        if (pos > size) pos = size;
        return pos;
    default:
        assert(!"unexpected wrapMode");
        return 0;
    }
}

int2 __CGsampler_state::wrapLinear(GLenum wrapMode, int1 size, float1 coord, float &wrappedCoord)
{
    int2 pos;
    
    switch (wrapMode) {
    case GL_REPEAT:
        wrappedCoord = coord;
        pos[0] = int(floor(coord - 0.5f));
        pos[0] = modulo(pos[0], size);
        pos[1] = (pos[0] == size-1) ? 0 : pos[0]+1;
        // Early return!
        return pos;
    case GL_CLAMP_TO_BORDER:
        wrappedCoord = clamp(coord, -0.5f, size+0.5f);
        break;
    case GL_CLAMP:
        wrappedCoord = clamp(coord, 0, float(size));
        break;
    case GL_CLAMP_TO_EDGE:
        wrappedCoord = clamp(coord, 0.5f, size-0.5f);
        break;
    case GL_MIRRORED_REPEAT:
        wrappedCoord = coord;
        pos = mirrorLinear(int(floor(coord - 0.5f)), size);
        // Early return!
        return pos;
    case GL_MIRROR_CLAMP_EXT:
        wrappedCoord = clamp(abs(coord), 0.5f, float(size));
        return pos;
    case GL_MIRROR_CLAMP_TO_EDGE_EXT:
        wrappedCoord = clamp(abs(coord), 0.5f, size-0.5f);
        break;
    case GL_MIRROR_CLAMP_TO_BORDER_EXT:
        wrappedCoord = clamp(abs(coord), 0.5f, size+0.5f);
        break;
    default:
        assert(!"unexpected wrapMode");
        return int2(0,0);
    }
    pos[0] = int(std::floor(wrappedCoord - 0.5f));
    pos[1] = pos[0]+1;
    return pos;  // REPEAT and MIRRORED_REPEAT early return!
}

float4 __CGsampler_state::linear1D(float4 tex2[2], float1 weight)
{
    return lerp(tex2[0], tex2[1], weight);
}

float4 __CGsampler_state::linear2D(float4 tex2x2[2][2], float2 weight)
{
    float4 tex2[2];

    for (int i=0; i<2; i++)
        tex2[i] = linear1D(tex2x2[i], weight.s);

    return linear1D(tex2, weight.t);
}

float4 __CGsampler_state::linear3D(float4 tex2x2x2[2][2][2], float3 weight)
{
    float4 tex2x2[2];

    for (int i=0; i<2; i++)
        tex2x2[i] = linear2D(tex2x2x2[i], weight.st);

    return linear1D(tex2x2, weight.p);
}

float4 __CGsampler_state::magnify(int ndx, float4 strq)
{
    switch (magFilter) {
    default:
        assert(!"unexpected magnification filter");
    case GL_NEAREST:
        return nearestFilter(trueBaseLevel, strq);
    case GL_LINEAR:
        return linearFilter(trueBaseLevel, strq);
    }
}

float4 __CGsampler_state::minify(int ndx, float4 strq, float1 lod)
{
    assert(lod > 0);  // Otherwise should be in minify
    switch (minFilter) {
    default:
        assert(!"unexpected minification filter");
    case GL_NEAREST:
        return nearestFilter(trueBaseLevel, strq);
    case GL_NEAREST_MIPMAP_NEAREST:
        {
            int level = clamp(int1(trueBaseLevel + round(lod)), trueBaseLevel, effectiveMaxLevel);
            return nearestFilter(level, strq);
        }
    case GL_LINEAR_MIPMAP_NEAREST:
        {
            int level = clamp(int1(trueBaseLevel + round(lod)), trueBaseLevel, effectiveMaxLevel);
            return linearFilter(level, strq);
        }
    case GL_NEAREST_MIPMAP_LINEAR:
        {
            int level0 = max(trueBaseLevel + int(floor(lod)), effectiveMaxLevel);
            int level1 = max(level0 + 1, effectiveMaxLevel);
            float4 tex0 = nearestFilter(level0, strq);
            float4 tex1 = nearestFilter(level1, strq);
            return lerp(tex0, tex1, frac(lod));
        }
    case GL_LINEAR_MIPMAP_LINEAR:
        {
            int level0 = max(trueBaseLevel + int(floor(lod)), effectiveMaxLevel);
            int level1 = max(level0 + 1, effectiveMaxLevel);
            float4 tex0 = linearFilter(level0, strq);
            float4 tex1 = linearFilter(level1, strq);
            return lerp(tex0, tex1, frac(lod));
        }
    case GL_LINEAR:
        return linearFilter(trueBaseLevel, strq);
    }
}

} // namespace Cg

