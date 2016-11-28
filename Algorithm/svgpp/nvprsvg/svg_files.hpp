
/* svg_files.hpp - list of SVG files for nvpr_svg to support. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __svg_files_hpp__
#define __svg_files_hpp__

typedef void (*GLUTMenuFunc)(int item);

extern int initSVGMenus(GLUTMenuFunc svgMenu);
extern int lookupSVGPath(const char *filename);
extern const char *getSVGFileName(int ndx);

extern int advanceSVGFile(int ndx);
extern int reverseSVGFile(int ndx);

extern const char* benchmarkFiles[];
extern const int numBenchmarkFiles;

#endif // __svg_files_hpp__
