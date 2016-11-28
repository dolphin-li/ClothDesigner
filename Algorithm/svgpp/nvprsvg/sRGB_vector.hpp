
// sRGB.cpp - sRGB color space conversion utilities

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <Cg/vector/rgba.hpp>
#include <Cg/vector.hpp>

using namespace Cg;

extern float3 srgb2linear(float3 srgb);
extern float3 linear2srgb(float3 linear);
extern float4 srgb2linear(float4 srgb);
extern float4 linear2srgb(float4 linear);

