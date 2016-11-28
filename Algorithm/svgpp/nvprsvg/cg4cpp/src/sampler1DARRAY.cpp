
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

class __CGsampler1DARRAY_state : public __CGsampler_state {
    __CGimage image[maxLevels];

    virtual void initImages();

    virtual ~__CGsampler1DARRAY_state() { }

    virtual float4 linearFilter(int level, float4 strq);
    virtual float4 nearestFilter(int level, float4 strq);

    virtual float4 nearest(const __CGimage &image, float4 strq);
    virtual float4 linear(const __CGimage &image, float4 strq);

public:
    __CGsampler1DARRAY_state() {
        initDerivedSampler(GL_TEXTURE_1D_ARRAY_EXT);
    }
    __CGsampler1DARRAY_state(int unit) {
        initDerivedSampler(GL_TEXTURE_1D_ARRAY_EXT, unit);
    }
    float4 sample(float4 strq, float lod);
    int3 size(int1 lod);
};

void __CGsampler1DARRAY_state::initImages()
{
    // Just width dimension supports border.
    const int3 borderSupport = int3(1, 0, 0);

    for (int level=0; level<maxLevels; level++) {
        image[level].initImage(texTarget, level, borderSupport);
    }
}

float4 __CGsampler1DARRAY_state::nearest(const __CGimage &image, float4 strq)
{
    int1 u = wrapNearest(wrapS, image.borderlessSize.x, strq.s);
    int1 slice = wrapNearest(GL_CLAMP_TO_EDGE, image.borderlessSize.y, strq.t);
    float4 texel = image.fetch(u, slice, 0, clampedBorderValues);

    return prefilter(texel, strq.p);
}

float4 __CGsampler1DARRAY_state::linear(const __CGimage &image, float4 strq)
{
    int2 u;
    int1 v;
    float4 tex2[2];
    float s;

    u = wrapLinear(wrapS, image.borderlessSize.x, strq.s, s);
    v = wrapNearest(GL_CLAMP_TO_EDGE, image.borderlessSize.y, strq.t + 0.5);
    // Fetch 2 texels
    for (int i=0; i<2; i++) {
        float4 texel = image.fetch(u[i], v, 0, clampedBorderValues);

        tex2[i] = prefilter(texel, strq.p);
    }

    return linear1D(tex2, frac(float1(s) - 0.5f));
}

float4 __CGsampler1DARRAY_state::nearestFilter(int level, float4 strq)
{
    strq.s *= image[level].borderlessSizeF.x;
    return nearest(image[level], strq);
}

float4 __CGsampler1DARRAY_state::linearFilter(int level, float4 strq)
{
    strq.s *= image[level].borderlessSizeF.x;
    return linear(image[level], strq);
}

float4 __CGsampler1DARRAY_state::sample(float4 strq, float lod)
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

int3 __CGsampler1DARRAY_state::size(int1 lod)
{
    lod += trueBaseLevel;

    if (lod <= effectiveMaxLevel) {
        return int3(image[lod].borderlessSize.xy, 0);
    } else {
        return int3(0);
    }
}

sampler1DARRAY::sampler1DARRAY()
{
    state = new __CGsampler1DARRAY_state;
}

sampler1DARRAY::sampler1DARRAY(const sampler1DARRAY &src) {
    state = src.state;
    state->ref();
}

sampler1DARRAY::~sampler1DARRAY() {
    state->unref();
}

sampler1DARRAY & sampler1DARRAY::operator = (const sampler1DARRAY &rhs) {
    if (this != &rhs) {
        state->unref();
        state = rhs.state;
        state->ref();
    }
    return *this;
}

float4 sampler1DARRAY::sample(float4 strq, float lod) {
    return state->sample(strq, lod);
}

} // namespace Cg
