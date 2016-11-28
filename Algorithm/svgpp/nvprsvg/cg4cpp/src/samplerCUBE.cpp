
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

class __CGsamplerCUBE_state : public __CGsampler_state {
    static const int numFaces = 6;

    __CGimage image[numFaces*maxLevels];

    virtual void initImages();

    virtual ~__CGsamplerCUBE_state() { }

    static void faceSelection(InOut<float3> str, int &face);

    virtual float4 minify(int ndx, float4 strq, float1 lod);
    virtual float4 magnify(int ndx, float4 strq);

    virtual float4 linearFilter(int level, float4 strq);
    virtual float4 nearestFilter(int level, float4 strq);

    virtual float4 nearest(const __CGimage &image, float4 strq);
    virtual float4 linear(const __CGimage &image, float4 strq);

public:
    __CGsamplerCUBE_state() {
        initDerivedSampler(GL_TEXTURE_CUBE_MAP);
    }
    __CGsamplerCUBE_state(int unit) {
        initDerivedSampler(GL_TEXTURE_CUBE_MAP, unit);
    }
    float4 sample(float4 strq, float lod);
    int3 size(int1 lod);
};

void __CGsamplerCUBE_state::initImages()
{
    // Width and height dimensions support borders.
    const int3 borderSupport = int3(1, 1, 0);

    for (int face=0; face<numFaces; face++) {
        for (int level=0; level<maxLevels; level++) {
            GLenum faceTarget = GL_TEXTURE_CUBE_MAP_POSITIVE_X + face;

            image[maxLevels*face+level].initImage(faceTarget, level, borderSupport);
        }
    }
}

float4 __CGsamplerCUBE_state::nearest(const __CGimage &image, float4 strq)
{
    int1 u = wrapNearest(wrapS, image.borderlessSize.x, strq.s);
    int1 v = wrapNearest(wrapT, image.borderlessSize.y, strq.t);
    float4 texel = image.fetch(u, v, 0, clampedBorderValues);

    return prefilter(texel, strq.q);
}

float4 __CGsamplerCUBE_state::linear(const __CGimage &image, float4 strq)
{
    int2 u, v;
    float4 tex[2][2];
    float s, t;

    u = wrapLinear(wrapS, image.borderlessSize.x, strq.s, s);
    v = wrapLinear(wrapT, image.borderlessSize.y, strq.t, t);
    // Fetch 2x2 cluster of texels
    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            float4 texel = image.fetch(u[j], v[i], 0, clampedBorderValues);

            tex[i][j] = prefilter(texel, strq.q);
        }
    }

    return linear2D(tex, frac(float2(s,t) - 0.5f));
}

float4 __CGsamplerCUBE_state::nearestFilter(int level, float4 strq)
{
    strq.st *= image[level].borderlessSizeF.xy;
    return nearest(image[level], strq);
}

float4 __CGsamplerCUBE_state::linearFilter(int level, float4 strq)
{
    strq.st *= image[level].borderlessSizeF.xy;
    return linear(image[level], strq);
}

float4 __CGsamplerCUBE_state::magnify(int ndx, float4 strq)
{
    int level = maxLevels*ndx+trueBaseLevel;

    switch (magFilter) {
    default:
        assert(!"unexpected magnification filter");
    case GL_NEAREST:
        return nearestFilter(level, strq);
    case GL_LINEAR:
        return linearFilter(level, strq);
    }
}

float4 __CGsamplerCUBE_state::minify(int ndx, float4 strq, float1 lod)
{
    int level = maxLevels*ndx+trueBaseLevel;

    switch (magFilter) {
    case GL_NEAREST:
        return nearestFilter(level, strq);
    default:
    case GL_LINEAR:
        return linearFilter(level, strq);
    }
}

void __CGsamplerCUBE_state::faceSelection(InOut<float3> str, int &face)
{
    float3 r = str;
    float3 absR = abs(r);
    float maxMag = max(max(absR.x, absR.y), absR.z);
    float oneOverMaxMag = 1 / maxMag;

    if (absR.x >= absR.y && absR.x >= absR.z) {
        if (r.s >= 0) {
            str.s = -r.z;
            str.t = -r.y;
            face = 0;
        } else {
            str.s =  r.z;
            str.t = -r.y;
            face = 1;
        }
    } else if (absR.y >= absR.x && absR.y >= absR.z) {
        if (r.t >= 0) {
            str.s =  r.x;
            str.t =  r.z;
            face = 2;
        } else {
            str.s =  r.x;
            str.t = -r.z;
            face = 3;
        }
    } else {
        if (r.p >= 0) {
            str.s =  r.x;
            str.t = -r.y;
            face = 4;
        } else {
            str.s = -r.x;
            str.t = -r.y;
            face = 5;
        }
    }
    str.st *= oneOverMaxMag;
    str.st = (str.st + 1.0f) / 2.0f;
    str.p = 0.0;  // or perhapse str.r = scale * oneOverMaxMag + bias
}

float4 __CGsamplerCUBE_state::sample(float4 strq, float lod)
{
    int face;

    faceSelection(strq.stp, face);

    lod += clampedLodBias;

    if (lod < minLod) {
        lod = minLod;
    } else if (lod > maxLod) {
        lod = maxLod;
    } else {
        // lod not clamped.
    }

    if (lod <= magnifyTransition) {
        return magnify(face, strq);
    } else {
        return minify(face, strq, lod);
    }
}

int3 __CGsamplerCUBE_state::size(int1 lod)
{
    lod += trueBaseLevel;

    if (lod <= effectiveMaxLevel) {
        return int3(image[lod].borderlessSize.xy, 0);
    } else {
        return int3(0);
    }
}

samplerCUBE::samplerCUBE() {
    state = new __CGsamplerCUBE_state;
}

samplerCUBE::samplerCUBE(int texUnit) {
    state = new __CGsamplerCUBE_state(texUnit);
}

samplerCUBE::samplerCUBE(const samplerCUBE &src) {
    state = src.state;
    state->ref();
}

samplerCUBE::~samplerCUBE() {
    state->unref();
}

samplerCUBE & samplerCUBE::operator = (const samplerCUBE &rhs) {
    if (this != &rhs) {
        state->unref();
        state = rhs.state;
        state->ref();
    }
    return *this;
}

float4 samplerCUBE::sample(float4 strq, float lod) {
    return state->sample(strq, lod);
}

float4 texCUBE(samplerCUBE s, float3 str)
{
    return s.sample(float4(str,0), 0.0f);
}

float4 texCUBEproj(samplerCUBE s, float3 str)
{
    return s.sample(float4(str,0), 0.0f);
}

float4 texCUBE(samplerCUBE s, float4 strq)
{
    return s.sample(strq, 0);  // q component used for shadow compare
}

float4 texCUBEproj(samplerCUBE s, float4 strq)
{
    return s.sample(strq, 0);  // q component used for shadow compare
}

} // namespace Cg
