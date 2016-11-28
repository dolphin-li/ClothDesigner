
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

class __CGsampler1D_state : public __CGsampler_state {
    __CGimage image[maxLevels];

    virtual void initImages();

    virtual ~__CGsampler1D_state() { }

    virtual float4 linearFilter(int level, float4 strq);
    virtual float4 nearestFilter(int level, float4 strq);

    virtual float4 nearest(const __CGimage &image, float4 strq);
    virtual float4 linear(const __CGimage &image, float4 strq);

public:
    __CGsampler1D_state() {
        initDerivedSampler(GL_TEXTURE_1D);
    }
    __CGsampler1D_state(int unit) {
        initDerivedSampler(GL_TEXTURE_1D, unit);
    }
    float4 sample(float4 strq, float lod);
    int3 size(int1 lod);
};

void __CGsampler1D_state::initImages()
{
    // Just width dimension supports border.
    const int3 borderSupport = int3(1, 0, 0);

    for (int level=0; level<maxLevels; level++) {
        image[level].initImage(texTarget, level, borderSupport);
    }
}

float4 __CGsampler1D_state::nearest(const __CGimage &image, float4 strq)
{
    int1 u = wrapNearest(wrapS, image.borderlessSize.x, strq.s);
    float4 texel = image.fetch(u, 0, 0, clampedBorderValues);

    return prefilter(texel, strq.p);
}

float4 __CGsampler1D_state::linear(const __CGimage &image, float4 strq)
{
    int2 u;
    float4 tex2[2];
    float s;

    u = wrapLinear(wrapS, image.borderlessSize.x, strq.s, s);
    // Fetch 2 texels
    for (int i=0; i<2; i++) {
        float4 texel = image.fetch(u[i], 0, 0, clampedBorderValues);

        tex2[i] = prefilter(texel, strq.p);
    }

    return linear1D(tex2, frac(float1(s) - 0.5f));
}

float4 __CGsampler1D_state::nearestFilter(int level, float4 strq)
{
    strq.s *= image[level].borderlessSizeF.x;
    return nearest(image[level], strq);
}

float4 __CGsampler1D_state::linearFilter(int level, float4 strq)
{
    strq.s *= image[level].borderlessSizeF.x;
    return linear(image[level], strq);
}

float4 __CGsampler1D_state::sample(float4 strq, float lod)
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

int3 __CGsampler1D_state::size(int1 lod)
{
    lod += trueBaseLevel;

    if (lod <= effectiveMaxLevel) {
        return int3(image[lod].borderlessSize.x, 0, 0);
    } else {
        return int3(0);
    }
}

sampler1D::sampler1D()
{
    state = new __CGsampler1D_state;
}

sampler1D::sampler1D(const sampler1D &src) {
    state = src.state;
    state->ref();
}

sampler1D::~sampler1D() {
    state->unref();
}

sampler1D & sampler1D::operator = (const sampler1D &rhs) {
    if (this != &rhs) {
        state->unref();
        state = rhs.state;
        state->ref();
    }
    return *this;
}

float4 sampler1D::sample(float4 strq, float lod) {
    return state->sample(strq, lod);
}

} // namespace Cg
