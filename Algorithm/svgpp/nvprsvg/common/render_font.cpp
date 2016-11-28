
/* render_font.cpp - class for rendering messia via font with NV_path_rendering */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <string.h>

#include "render_font.hpp"

using namespace Cg;

FontFace::FontFace(GLenum target, const char *name, int num_chars_, GLuint path_param_template)
  : font_name(name)
  , num_chars(num_chars_)
  , horizontal_advance(num_chars)
{
  /* Create a range of path objects corresponding to Latin-1 character codes. */
  glyph_base = glGenPathsNV(num_chars);
  glPathGlyphRangeNV(glyph_base,
    target, name, GL_BOLD_BIT_NV,
    0, num_chars,
    GL_USE_MISSING_GLYPH_NV, path_param_template,
    em_scale);
  glPathGlyphRangeNV(glyph_base,
    GL_STANDARD_FONT_NAME_NV, "Sans", GL_BOLD_BIT_NV,
    0, num_chars,
    GL_USE_MISSING_GLYPH_NV, path_param_template,
    em_scale);

  /* Query font and glyph metrics. */
  GLfloat font_data[4];
  glGetPathMetricRangeNV(GL_FONT_Y_MIN_BOUNDS_BIT_NV|GL_FONT_Y_MAX_BOUNDS_BIT_NV|
    GL_FONT_UNDERLINE_POSITION_BIT_NV|GL_FONT_UNDERLINE_THICKNESS_BIT_NV,
    glyph_base+' ', /*count*/1,
    4*sizeof(GLfloat),
    font_data);

  y_min = font_data[0];
  y_max = font_data[1];
  underline_position = font_data[2];
  underline_thickness = font_data[3];
  glGetPathMetricRangeNV(GL_GLYPH_HORIZONTAL_BEARING_ADVANCE_BIT_NV,
    glyph_base, num_chars,
    0, /* stride of zero means sizeof(GLfloat) since 1 bit in mask */
    &horizontal_advance[0]);
}

FontFace::~FontFace()
{
  glDeletePathsNV(glyph_base, num_chars);
}

Message::Message(const FontFace *font_, const char *message_, Cg::float2 to_quad[4])
  : message(message_)
  , font(font_)
  , message_length(strlen(message_))
  , xtranslate(message_length)
  , stroking(true)
  , filling(true)
  , underline(0)
  , fill_gradient(0)
{
  glGetPathSpacingNV(GL_ACCUM_ADJACENT_PAIRS_NV,
    (GLsizei)message_length, GL_UNSIGNED_BYTE, message.c_str(),
    font->glyph_base,
    1.0, 1.0,
    GL_TRANSLATE_X_NV,
    &xtranslate[1]);

  /* Total advance is accumulated spacing plus horizontal advance of
  the last glyph */
  total_advance = xtranslate[message_length-1] +
    font->horizontal_advance[GLubyte(message[message_length-1])];

  bound_before_transform = float4(0, font->y_min, total_advance, font->y_max);
  matrix = float3x3(box2quad(bound_before_transform, to_quad));
}

Message::~Message()
{}

void Message::render()
{
  if (underline) {
    float position = font->underline_position,
      half_thickness = font->underline_thickness/2;
    glDisable(GL_STENCIL_TEST);
    if (underline == 2) {
      glColor3f(1,1,1);
    } else {
      glColor3f(0.5, 0.5, 0.5);
    }
    glBegin(GL_QUAD_STRIP); {
      glVertex2f(0, position+half_thickness);
      glVertex2f(0, position-half_thickness);
      glVertex2f(total_advance, position+half_thickness);
      glVertex2f(total_advance, position-half_thickness);
    } glEnd();
    glEnable(GL_STENCIL_TEST);
  }

  if (stroking) {
    glStencilStrokePathInstancedNV((GLsizei)message_length,
      GL_UNSIGNED_BYTE, message.c_str(), font->glyph_base,
      1, ~0,  /* Use all stencil bits */
      GL_TRANSLATE_X_NV, &xtranslate[0]);
    glColor3f(0.5,0.5,0.5);  // gray
    glCoverStrokePathInstancedNV((GLsizei)message_length,
      GL_UNSIGNED_BYTE, message.c_str(), font->glyph_base,
      GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
      GL_TRANSLATE_X_NV, &xtranslate[0]);
  }

  if (filling) {
    /* STEP 1: stencil message into stencil buffer.  Results in samples
    within the message's glyphs to have a non-zero stencil value. */
    glStencilFillPathInstancedNV((GLsizei)message_length,
      GL_UNSIGNED_BYTE, message.c_str(), font->glyph_base,
      GL_PATH_FILL_MODE_NV, ~0,  /* Use all stencil bits */
      GL_TRANSLATE_X_NV, &xtranslate[0]);

    /* STEP 2: cover region of the message; color covered samples (those
    with a non-zero stencil value) and set their stencil back to zero. */
    switch (fill_gradient) {
    case 0:
      {
        GLfloat rgb_gen[3][3] = { {0,  0, 1},
        {0,  1, 0},
        {0, -1, 1} };
        glPathColorGenNV(GL_PRIMARY_COLOR,
          GL_PATH_OBJECT_BOUNDING_BOX_NV, GL_RGB, &rgb_gen[0][0]);
      }
      break;
    case 1:
      glColor3ub(192, 192, 192);  // gray
      break;
    case 2:
      glColor3ub(255, 255, 255);  // white
      break;
    case 3:
      glColor3ub(0, 0, 0);  // black
      break;
    }

    glCoverFillPathInstancedNV((GLsizei)message_length,
      GL_UNSIGNED_BYTE, message.c_str(), font->glyph_base,
      GL_BOUNDING_BOX_OF_BOUNDING_BOXES_NV,
      GL_TRANSLATE_X_NV, &xtranslate[0]);
    if (fill_gradient == 0) {
      /* Disable gradient. */
      glPathColorGenNV(GL_PRIMARY_COLOR, GL_NONE, 0, NULL);
    }
  }
}
Cg::float3x3 Message::getMatrix()
{
  return matrix;
}
void Message::setMatrix(Cg::float3x3 M)
{
	matrix = M;
}
void Message::multMatrix()
{
  MatrixMultToGL(matrix);
}

void Message::loadMatrix()
{
  MatrixLoadToGL(matrix);
}
