
/* 
 * Copyright 2006 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "sampler_state.hpp"

namespace Cg {

class __CGsampler2DARRAY_state : public __CGsampler_state {
    __CGimage image[maxLevels];

    virtual void initImages();

    virtual ~__CGsampler2DARRAY_state() { }

    virtual float4 linearFilter(int level, float4 strq);
    virtual float4 nearestFilter(int level, float4 strq);

    virtual float4 nearest(const __CGimage &image, float4 strq);
    virtual float4 linear(const __CGimage &image, float4 strq);

public:
    __CGsampler2DARRAY_state() {
        initDerivedSampler(GL_TEXTURE_2D_ARRAY_EXT);
    }
    __CGsampler2DARRAY_state(int unit) {
        initDerivedSampler(GL_TEXTURE_2D_ARRAY_EXT, unit);
    }
    float4 sample(float4 strq, float lod);
    int3 size(int1 lod);
};

void __CGsampler2DARRAY_state::initImages()
{
    // Width and height dimensions support borders.
    const int3 borderSupport = int3(1, 0, 0);

    for (int level=0; level<maxLevels; level++) {
        image[level].initImage(texTarget, level, borderSupport);
    }
}

float4 __CGsampler2DARRAY_state::nearest(const __CGimage &image, float4 strq)
{
    int1 u = wrapNearest(wrapS, image.borderlessSize.x, strq.s);
    int1 v = wrapNearest(wrapT, image.borderlessSize.y, strq.t);
    int1 slice = wrapNearest(GL_CLAMP_TO_EDGE, image.borderlessSize.z, strq.r);
    float4 texel = image.fetch(u, v, slice, clampedBorderValues);

    return prefilter(texel, strq.q);
}

float4 __CGsampler2DARRAY_state::linear(const __CGimage &image, float4 strq)
{
    int2 u, v;
    int1 p;
    float4 tex[2][2];
    float s, t;

    u = wrapLinear(wrapS, image.borderlessSize.x, strq.s, s);
    v = wrapLinear(wrapT, image.borderlessSize.y, strq.t, t);
    p = wrapNearest(GL_CLAMP_TO_EDGE, image.borderlessSize.y, strq.p + 0.5);
    // Fetch 2x2 cluster of texels
    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            float4 texel = image.fetch(u[j], v[i], p, clampedBorderValues);

            tex[i][j] = prefilter(texel, strq.q);
        }
    }

    return linear2D(tex, frac(float2(s,t) - 0.5f));
}

float4 __CGsampler2DARRAY_state::nearestFilter(int level, float4 strq)
{
    strq.st *= image[level].borderlessSizeF.xy;
    return nearest(image[level], strq);
}

float4 __CGsampler2DARRAY_state::linearFilter(int level, float4 strq)
{
    strq.st *= image[level].borderlessSizeF.xy;
    return linear(image[level], strq);
}

float4 __CGsampler2DARRAY_state::sample(float4 strq, float lod)
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

int3 __CGsampler2DARRAY_state::size(int1 lod)
{
    lod += trueBaseLevel;

    if (lod <= effectiveMaxLevel) {
        return int3(image[lod].borderlessSize.xy, 0);
    } else {
        return int3(0);
    }
}

sampler2DARRAY::sampler2DARRAY() {
    state = new __CGsampler2DARRAY_state;
}

sampler2DARRAY::sampler2DARRAY(int texUnit) {
    state = new __CGsampler2DARRAY_state(texUnit);
}

sampler2DARRAY::sampler2DARRAY(const sampler2DARRAY &src) {
    state = src.state;
    state->ref();
}

sampler2DARRAY::~sampler2DARRAY() {
    state->unref();
}

sampler2DARRAY & sampler2DARRAY::operator = (const sampler2DARRAY &rhs) {
    if (this != &rhs) {
        state->unref();
        state = rhs.state;
        state->ref();
    }
    return *this;
}

float4 sampler2DARRAY::sample(float4 strq, float lod) {
    return state->sample(strq, lod);
}

} // namespace Cg
