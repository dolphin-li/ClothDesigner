
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

#include "sampler2D_state.hpp"

namespace Cg {

void __CGsampler2D_state::initImages()
{
    // Width and height dimensions support borders.
    const int3 borderSupport = int3(1, 1, 0);

    for (int level=0; level<maxLevels; level++) {
        image[level].initImage(texTarget, level, borderSupport);
    }
}

float4 __CGsampler2D_state::nearest(const __CGimage &image, float4 strq)
{
    int1 u = wrapNearest(wrapS, image.borderlessSize.x, strq.s);
    int1 v = wrapNearest(wrapT, image.borderlessSize.y, strq.t);
    float4 texel = image.fetch(u, v, 0, clampedBorderValues);

    return prefilter(texel, strq.p);
}

float4 __CGsampler2D_state::linear(const __CGimage &image, float4 strq)
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

float4 __CGsampler2D_state::nearestFilter(int level, float4 strq)
{
    strq.st *= image[level].borderlessSizeF.xy;
    return nearest(image[level], strq);
}

float4 __CGsampler2D_state::linearFilter(int level, float4 strq)
{
    strq.st *= image[level].borderlessSizeF.xy;
    return linear(image[level], strq);
}

float4 __CGsampler2D_state::sample(float4 strq, float lod)
{
    lod += clampedLodBias;

    if (lod < minLod) {
        lod = minLod;
    } else if (lod > maxLod) {
        lod = maxLod;
    } else {
        // lod not clamped.
    }

    if (lod <= magnifyTransition) {
        return magnify(0, strq);
    } else {
        return minify(0, strq, lod);
    }
}

int3 __CGsampler2D_state::size(int1 lod)
{
    lod += trueBaseLevel;

    if (lod <= effectiveMaxLevel) {
        return int3(image[lod].borderlessSize.xy, 0);
    } else {
        return int3(0);
    }
}

sampler2D::sampler2D() {
    state = new __CGsampler2D_state;
}

sampler2D::sampler2D( __CGsampler2D_factory &from) {
    state = from.construct();
}

sampler2D::sampler2D(const sampler2D &src) {
    state = src.state;
    state->ref();
}

sampler2D::~sampler2D() {
    state->unref();
}

sampler2D & sampler2D::operator = (const sampler2D &rhs) {
    if (this != &rhs) {
        state->unref();
        state = rhs.state;
        state->ref();
    }
    return *this;
}

float4 sampler2D::sample(float4 strq, float lod) {
    return state->sample(strq, lod);
}

float4 tex2D(sampler2D s, float2 st)
{
    return s.sample(float4(st,0,0), 0.0f);
}

float4 tex2D(sampler2D s, float2 st, float2 dx, float2 dy)
{
    return s.sample(float4(st,0,0), 0.0f);
}

float4 tex2D(sampler2D s, float3 str)
{
    return s.sample(float4(str,0), 0.0f);
}

float4 tex2D(sampler2D s, float3 str, float3 dx, float3 dy)
{
    return s.sample(float4(str,0), 0.0f);
}

float4 tex2Dproj(sampler2D s, float3 stq)
{
    float3 str = float3(stq.st / stq.p, 0);

    return s.sample(float4(str,0), 0.0f);
}

float4 tex2Dproj(sampler2D s, float4 strq)
{
    float3 str = strq.stp / strq.q;

    return s.sample(float4(str,0), 0.0f);
}

float4 tex2D(sampler2D s, float4 strq, float4 dx, float4 dy)
{
    return s.sample(strq, 0.0f);
}

} // namespace Cg
