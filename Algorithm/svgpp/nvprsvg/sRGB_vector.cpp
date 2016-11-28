
// sRGB.cpp - sRGB color space conversion utilities

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <Cg/vector/rgba.hpp>
#include <Cg/vector.hpp>

#include "sRGB_math.h"
#include "sRGB_vector.hpp"

using namespace Cg;

float3 srgb2linear(float3 srgb)
{
    float3 linear;

    linear.r = convertSRGBColorComponentToLinearf(srgb.r);
    linear.g = convertSRGBColorComponentToLinearf(srgb.g);
    linear.b = convertSRGBColorComponentToLinearf(srgb.b);
    return linear;
}

float3 linear2srgb(float3 linear)
{
    float3 srgb;

    srgb.r = convertLinearColorComponentToSRGBf(linear.r);
    srgb.g = convertLinearColorComponentToSRGBf(linear.g);
    srgb.b = convertLinearColorComponentToSRGBf(linear.b);
    return srgb;
}

float4 srgb2linear(float4 srgb)
{
    float4 linear;

    
    linear.r = convertSRGBColorComponentToLinearf(srgb.r);
    linear.g = convertSRGBColorComponentToLinearf(srgb.g);
    linear.b = convertSRGBColorComponentToLinearf(srgb.b);
#if 0 // VS 2008 compiler bug??
    linear.a = srgb.a;
#else
    linear[3] = srgb[3];
#endif
    return linear;
}

float4 linear2srgb(float4 linear)
{
    float4 srgb;

    srgb.r = convertLinearColorComponentToSRGBf(linear.r);
    srgb.g = convertLinearColorComponentToSRGBf(linear.g);
    srgb.b = convertLinearColorComponentToSRGBf(linear.b); 
#if 0 // VS 2008 compiler bug??
    srgb.a = linear.a;
#else
    srgb[3] = linear[3];
#endif
    return srgb;
}

