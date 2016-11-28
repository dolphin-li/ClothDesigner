
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

class __CGsamplerRECT_state : public __CGsampler_state {

    __CGimage image[1];

    virtual void initImages();

    virtual ~__CGsamplerRECT_state() { }

    virtual float4 linearFilter(int level, float4 strq);
    virtual float4 nearestFilter(int level, float4 strq);

    virtual float4 nearest(const __CGimage &image, float4 strq);
    virtual float4 linear(const __CGimage &image, float4 strq);

public:
    __CGsamplerRECT_state() {
        initDerivedSampler(GL_TEXTURE_RECTANGLE_ARB);
    }
    __CGsamplerRECT_state(int unit) {
        initDerivedSampler(GL_TEXTURE_RECTANGLE_ARB, unit);
    }
    float4 sample(float4 strq, float lod);
    int3 size(int1 lod);
};

void __CGsamplerRECT_state::initImages()
{
    // No border support.
    const int3 borderSupport = int3(1, 0, 0);

    // Texture rectangles have just one level.
    image[0].initImage(GL_TEXTURE_RECTANGLE_ARB, 0, borderSupport);
}

float4 __CGsamplerRECT_state::nearest(const __CGimage &image, float4 strq)
{
    int1 u = wrapNearest(wrapS, image.borderlessSize.x, strq.s);
    int1 v = wrapNearest(wrapT, image.borderlessSize.y, strq.t);
    float4 texel = image.fetch(u, v, 0, clampedBorderValues);

    return prefilter(texel, strq.p);
}

float4 __CGsamplerRECT_state::linear(const __CGimage &image, float4 strq)
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

            tex[i][j] = prefilter(texel, strq.p);
        }
    }

    return linear2D(tex, frac(float2(s,t) - 0.5f));
}

float4 __CGsamplerRECT_state::nearestFilter(int level, float4 strq)
{
    // No scaling by level size.
    return nearest(image[level], strq);
}

float4 __CGsamplerRECT_state::linearFilter(int level, float4 strq)
{
    // No scaling by level size.
    return linear(image[level], strq);
}

float4 __CGsamplerRECT_state::sample(float4 strq, float lod)
{
    if (lod <= magnifyTransition) {
        return magnify(0, strq);
    } else {
        return minify(0, strq, 0);
    }
}

int3 __CGsamplerRECT_state::size(int1 lod)
{
    lod += trueBaseLevel;

    if (lod <= effectiveMaxLevel) {
        return int3(image[lod].borderlessSize.xy, 0);
    } else {
        return int3(0);
    }
}

samplerRECT::samplerRECT() {
    state = new __CGsamplerRECT_state;
}

samplerRECT::samplerRECT(int texUnit) {
    state = new __CGsamplerRECT_state(texUnit);
}

samplerRECT::samplerRECT(const samplerRECT &src) {
    state = src.state;
    state->ref();
}

samplerRECT::~samplerRECT() {
    state->unref();
}

samplerRECT & samplerRECT::operator = (const samplerRECT &rhs) {
    if (this != &rhs) {
        state->unref();
        state = rhs.state;
        state->ref();
    }
    return *this;
}

float4 samplerRECT::sample(float4 strq, float lod) {
    return state->sample(strq, lod);
}

float4 texRECT(samplerRECT s, float2 st)
{
    return s.sample(float4(st,0,0), 0.0f);
}

float4 texRECT(samplerRECT s, float2 st, float2 dx, float2 dy)
{
    return s.sample(float4(st,0,0), 0.0f);
}

float4 texRECT(samplerRECT s, float3 str)
{
    return s.sample(float4(str,0), 0.0f);
}

float4 texRECT(samplerRECT s, float3 str, float3 dx, float3 dy)
{
    return s.sample(float4(str,0), 0.0f);
}

float4 texRECTproj(samplerRECT s, float3 stq)
{
    float3 str = float3(stq.st / stq.p, 0);

    return s.sample(float4(str,0), 0.0f);
}

float4 texRECTproj(samplerRECT s, float4 strq)
{
    float3 str = strq.stp / strq.q;

    return s.sample(float4(str,0), 0.0f);
}

float4 texRECT(samplerRECT s, float4 strq, float4 dx, float4 dy)
{
    return s.sample(strq, 0.0f);
}

} // namespace Cg
