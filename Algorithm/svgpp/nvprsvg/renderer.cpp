
/* renderer.cpp - generic renderer back-end implementation. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include "renderer.hpp"

#include "showfps.h"

void GLBlitRenderer::swapBuffers()
{
    glutSwapBuffers();
}

void GLBlitRenderer::reportFPS()
{
    double thisFPS = handleFPS();
    thisFPS = thisFPS; // force used
}
