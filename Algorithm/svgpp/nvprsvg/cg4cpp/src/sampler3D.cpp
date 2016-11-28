
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

class __CGsampler3D_state : public __CGsampler_state {

    __CGimage image[maxLevels];

    virtual void initImages();

    virtual ~__CGsampler3D_state() { }

    virtual float4 linearFilter(int level, float4 strq);
    virtual float4 nearestFilter(int level, float4 strq);

    virtual float4 nearest(const __CGimage &image, float4 strq);
    virtual float4 linear(const __CGimage &image, float4 strq);

public:
    __CGsampler3D_state() {
        initDerivedSampler(GL_TEXTURE_3D);
    }
    __CGsampler3D_state(int unit) {
        initDerivedSampler(GL_TEXTURE_3D, unit);
    }
    float4 sample(float4 strq, float lod);
    int3 size(int1 lod);
};

void __CGsampler3D_state::initImages()
{
    // Width, height, and depth dimensions all support borders.
    const int3 borderSupport = int3(1, 1, 1);

    for (int level=0; level<maxLevels; level++) {
        image[level].initImage(texTarget, level, borderSupport);
    }
}

float4 __CGsampler3D_state::nearest(const __CGimage &image, float4 strq)
{
    int1 u = wrapNearest(wrapS, image.borderlessSize.x, strq.s);
    int1 v = wrapNearest(wrapT, image.borderlessSize.y, strq.t);
    int1 p = wrapNearest(wrapR, image.borderlessSize.z, strq.p);
    float4 texel = image.fetch(u, v, p, clampedBorderValues);

    return prefilter(texel, 
        /* no shadow mapping for 3D textures */0.0);
}

float4 __CGsampler3D_state::linear(const __CGimage &image, float4 strq)
{
    int2 u, v, p;
    float4 tex[2][2][2];
    float s, t, r;

    u = wrapLinear(wrapS, image.borderlessSize.x, strq.s, s);
    v = wrapLinear(wrapT, image.borderlessSize.y, strq.t, t);
    p = wrapLinear(wrapR, image.borderlessSize.z, strq.p, r);
    // Fetch 2x2x2 cluster of texels
    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            for (int k=0; k<2; k++) {
                float4 texel = image.fetch(u[k], v[j], p[i], clampedBorderValues);

                tex[i][j][k] = prefilter(texel,
                    /* no shadow mapping for 3D textures */0.0);
            }
        }
    }

    return linear3D(tex, frac(float3(s,t,r) - 0.5f));
}

float4 __CGsampler3D_state::nearestFilter(int level, float4 strq)
{
    strq.stp *= image[level].borderlessSizeF.xyz;
    return nearest(image[level], strq);
}

float4 __CGsampler3D_state::linearFilter(int level, float4 strq)
{
    strq.stp *= image[level].borderlessSizeF.xyz;
    return linear(image[level], strq);
}

float4 __CGsampler3D_state::sample(float4 strq, float lod)
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

int3 __CGsampler3D_state::size(int1 lod)
{
    lod += trueBaseLevel;

    if (lod <= effectiveMaxLevel) {
        return image[lod].borderlessSize.xyz;
    } else {
        return int3(0);
    }
}

sampler3D::sampler3D() {
    state = new __CGsampler3D_state;
}

sampler3D::sampler3D(int texUnit) {
    state = new __CGsampler3D_state(texUnit);
}

sampler3D::sampler3D(const sampler3D &src) {
    state = src.state;
    state->ref();
}

sampler3D::~sampler3D() {
    state->unref();
}

sampler3D & sampler3D::operator = (const sampler3D &rhs) {
    if (this != &rhs) {
        state->unref();
        state = rhs.state;
        state->ref();
    }
    return *this;
}

float4 sampler3D::sample(float4 strq, float lod) {
    return state->sample(strq, lod);
}

} // namespace Cg
