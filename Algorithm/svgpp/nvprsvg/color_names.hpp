
/* color_names.hpp - map SVG color names to RGB values */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#pragma once
#ifndef __color_names_hpp__
#define __color_names_hpp__

#include <Cg/vector.hpp>

// returns true if matched a color name, and then updated RGB components of color
bool parse_color_name(const char *name, Cg::float4 &color);

#endif // __color_names_hpp__
