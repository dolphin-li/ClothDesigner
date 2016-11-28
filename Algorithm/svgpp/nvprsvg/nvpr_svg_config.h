
/* nvpr_svg_config.h - configure renderers for path rendering test bed */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __nvpr_svg_config_h__
#define __nvpr_svg_config_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

// FreeType2
// http://www.freetype.org/
#define USE_FREETYPE2 1

// Khronos OpenVG 1.1 royalty-free, cross-platform API is a low-level hardware acceleration interface for vector graphics
// http://www.khronos.org/openvg/
// http://www.khronos.org/registry/vg/specs/openvg-1.1.pdf
//#define USE_OPENVG 1

#ifdef _WIN32
// Microsoft's Direct2D API introduced with Windows Vista (not supported on XP)
// http://msdn.microsoft.com/en-us/library/dd370990%28VS.85%29.aspx
// http://msdn.microsoft.com/en-us/library/dd372337%28v=VS.85%29.aspx
// http://msdn.microsoft.com/en-us/library/dd372349%28v=VS.85%29.aspx
//# define USE_D2D   1
#else
# define USE_D2D   0
#endif

// Cairo Graphics, an open source 2D graphics library for multiple output devices
// http://cairographics.org/
// http://cairographics.org/manual/
//#define USE_CAIRO  1

// Nokia's Qt platform includes the Arthur Paint System
// http://doc.qt.nokia.com/4.6/qt4-arthur.html
// http://doc.qt.nokia.com/4.6/qpainter.html
//#define USE_QT     1

// NVIDIA's NV_path_rendering GPU-accelerated path rendering system
// https://p4viewer.nvidia.com/getfile///sw/docs/gpu/drivers/OpenGL/specs/proposed/GL_NV_path_rendering.txt
//#define USE_NVPR   1

// Google's Skia 2D Graphics Library used by Google's Chrome web browser and Android operating system
// http://code.google.com/p/skia/
//#define USE_SKIA   1

#if defined(__APPLE__)
// It would probably be easy enough to support Skia and OpenVG (since their source is in the tree)
// Supporting Cairo would be easy too if Cairo's source is in the tree
// Qt might be more work (maybe not)
# undef USE_D2D
# define USE_D2D    0
# undef USE_CAIRO
# define USE_CAIRO  0
# undef USE_QT
# define USE_QT     0
# undef USE_OPENVG
# define USE_OPENVG 0
# undef USE_NVPR
# define USE_NVPR   0
# undef USE_SKIA
# define USE_SKIA   0
#endif

#if defined(linux)
// It would probably be easy enough to support Skia and OpenVG (since their source is in the tree)
// Supporting Cairo would be easy too if Cairo's source is in the tree
// Qt might be more work (maybe not)
# undef USE_D2D
# define USE_D2D    0
//# undef USE_CAIRO
//# define USE_CAIRO  0
#undef USE_QT
#define USE_QT     0
# undef USE_OPENVG
# define USE_OPENVG 0
//# undef USE_NVPR
//# define USE_NVPR   0
//# undef USE_SKIA
//# define USE_SKIA   0
#endif

#if defined(_WIN64)
# undef USE_QT
# define USE_QT 0
# undef USE_FREETYPE2
# define USE_FREETYPE2 0
#endif

#if defined(sun)
# undef USE_QT
# define USE_QT 0
# undef USE_OPENVG
# define USE_OPENVG 0
#endif

#endif // __nvpr_svg_config_h__
