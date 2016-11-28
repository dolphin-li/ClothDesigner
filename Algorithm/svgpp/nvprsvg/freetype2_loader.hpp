
// freetype2_loader.cpp - use FreeType2 library to load outlines of glyphs from fonts

// Copyright (c) NVIDIA Corporation. All rights reserved.

#pragma once
#ifndef __freetype2_loader_hpp__
#define __freetype2_loader_hpp__

#include "path.hpp"

void init_freetype2();
int lookup_font(const char *name);
PathPtr load_freetype2_glyph(unsigned int font, unsigned char c);
int num_fonts();
const char *font_name(int font_index);

#endif // __freetype2_loader_hpp__
