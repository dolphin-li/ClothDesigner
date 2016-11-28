
/* welsh_dragon.c - draw welsh dragon */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <string.h>

#include <GL/glew.h>
#include "nvpr_init.h"

static const int dragon_paths = 158;
static const char *welsh_dragon[158] = {
#include "welsh_dragon_paths.h"
};
#define float3(r,g,b) {r,g,b}
static const float welsh_dragon_colors[158][3] = {
#include "welsh_dragon_colors.h"
};
GLuint dragon_path_base;

void initDragon()
{
  int i;

  dragon_path_base = glGenPathsNV(dragon_paths);

  for (i=0; i<dragon_paths; i++) {
    const char *svg_str = welsh_dragon[i];
    size_t svg_len = strlen(welsh_dragon[i]);

    glPathStringNV(dragon_path_base+i, GL_PATH_FORMAT_SVG_NV,
      (GLsizei)svg_len, svg_str);
  }
}

void drawDragon()
{
  int i;

  for (i=0; i<dragon_paths; i++) {
    const float *color = &welsh_dragon_colors[i][0];
    glStencilFillPathNV(dragon_path_base+i, GL_COUNT_UP_NV, 0x1F);
    glStencilFunc(GL_NOTEQUAL, 0, 0x1F);
    glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
    glColor3f(color[0], color[1], color[2]);
    glCoverFillPathNV(dragon_path_base+i, GL_BOUNDING_BOX_NV);
  }
}
