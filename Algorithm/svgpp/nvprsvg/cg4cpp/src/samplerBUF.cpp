
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

class __CGsamplerBUF_state : public __CGsampler_state {

    __CGimage image[1];

    virtual void initImages();

    virtual ~__CGsamplerBUF_state() { }

    // Never actually used but must be implemented
    virtual float4 linearFilter(int level, float4 strq) { return 0; }
    virtual float4 nearestFilter(int level, float4 strq) { return 0; }
    virtual float4 nearest(const __CGimage &image, float4 strq) { return 0; }
    virtual float4 linear(const __CGimage &image, float4 strq) { return 0; }

    virtual void initSampler();

public:
    __CGsamplerBUF_state() {
        initDerivedSampler(GL_TEXTURE_BUFFER_EXT);
    }
    __CGsamplerBUF_state(int unit) {
        initDerivedSampler(GL_TEXTURE_BUFFER_EXT, unit);
    }
    float4 sample(float4 strq, float lod);
    int3 size(int1 lod);
};

void __CGsamplerBUF_state::initImages()
{
    // No border support.
    const int3 borderSupport = int3(0, 0, 0);

    // Texture rectangles have just one level.
    image[0].initImage(GL_TEXTURE_BUFFER_EXT, 0, borderSupport);
}

float4 __CGsamplerBUF_state::sample(float4 strq, float lod)
{
    int u = int(floor(strq.s));

    // Technically EXT_texture_buffer_object does not support formats that require prefilter.
    return prefilter(image[0].fetch(u, 0, 0, clampedBorderValues), strq.p);
}


void __CGsamplerBUF_state::initSampler()
{
    //refcnt = 1;

    wrapS = GL_NONE;
    wrapT = GL_NONE;
    wrapR = GL_NONE;

    baseLevel = 0;
    maxLevel = 0;

    minLod = 0;
    maxLod = 0;

    lodBias = 0;

    borderValues[0] = 0;
    borderValues[1] = 0;
    borderValues[2] = 0;
    borderValues[3] = 0;

    minFilter = GL_NONE;
    magFilter = GL_NONE;

    compareMode = GL_NONE;
    compareFunc = GL_NONE;
    depthTextureMode = GL_NONE;

    magnifyTransition = 0.0f;

    clampedLodBias = 0;

    trueBaseLevel = 0;  // needs clamping
    effectiveMaxLevel = 0;  // needs clamping

    clampedBorderValues = float4(0);  // needs clamping based on format

    prefilterMode = GL_NONE;
}

samplerBUF::samplerBUF()
{
    state = new __CGsamplerBUF_state;
}

samplerBUF::samplerBUF(int texUnit) {
    state = new __CGsamplerBUF_state(texUnit);
}

samplerBUF::samplerBUF(const samplerBUF &src) {
    state = src.state;
    state->ref();
}

samplerBUF::~samplerBUF() {
    state->unref();
}

samplerBUF & samplerBUF::operator = (const samplerBUF &rhs) {
    if (this != &rhs) {
        state->unref();
        state = rhs.state;
        state->ref();
    }
    return *this;
}

float4 samplerBUF::sample(float4 strq, float lod) {
    return state->sample(strq, lod);
}

float4 texBUF(samplerBUF buf, float1 s)
{
    return buf.sample(float4(s,0,0,0), 0.0f);
}

} // namespace Cg
