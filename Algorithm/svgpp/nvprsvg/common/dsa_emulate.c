
/* dsa_emulate.c - support EXT_direct_state_access for GLEW if unsupported */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <assert.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>
#if __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/glext.h>
#else
#include <GL/freeglut.h>
#endif

#if !defined(GLAPIENTRY)
# if defined(APIENTRY)
#  define GLAPIENTRY APIENTRY
# else
#  define GLAPIENTRY
# endif
#endif

#include "dsa_emulate.h"

void GLAPIENTRY dsa_glDisableClientStateIndexedEXT(GLenum array, GLuint index)
{
    assert(array == GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE0+index);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

void GLAPIENTRY dsa_glEnableClientStateIndexedEXT(GLenum array, GLuint index)
{
    assert(array == GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE0+index);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
}

void GLAPIENTRY dsa_glMultiTexCoordPointerEXT(GLenum texunit, GLint size, GLenum type, GLsizei stride, const void* pointer)
{
    glClientActiveTexture(texunit);
    glTexCoordPointer(size, type, stride, pointer);
}

void GLAPIENTRY dsa_glMatrixLoadIdentityEXT(GLenum matrixMode)
{
    glMatrixMode(matrixMode);
    glLoadIdentity();
}

void GLAPIENTRY dsa_glMatrixMultTransposefEXT(GLenum matrixMode, const GLfloat* m)
{
    glMatrixMode(matrixMode);
    glMultTransposeMatrixf(m);
}

void GLAPIENTRY dsa_glMatrixLoadTransposefEXT(GLenum matrixMode, const GLfloat* m)
{
    glMatrixMode(matrixMode);
    glLoadTransposeMatrixf(m);
}

void GLAPIENTRY dsa_glMatrixPopEXT(GLenum matrixMode)
{
    glMatrixMode(matrixMode);
    glPopMatrix();
}

void GLAPIENTRY dsa_glMatrixPushEXT(GLenum matrixMode)
{
    glMatrixMode(matrixMode);
    glPushMatrix();
}

void GLAPIENTRY dsa_glTextureBufferEXT(GLuint texture, GLenum target, GLenum internalformat, GLuint buffer)
{
    glBindTexture(target, texture);
    glTexBufferARB(target, internalformat, buffer);
}

void GLAPIENTRY dsa_glBindMultiTextureEXT(GLenum texunit, GLenum target, GLuint texture)
{
	glActiveTexture(texunit);
	glBindTexture(target, texture);
}


void emulate_dsa_if_needed(int forceDSAemulation)
{
    if (!forceDSAemulation && glutExtensionSupported("GL_EXT_direct_state_access")) {
        printf("supports EXT_direct_state_access\n");
    } else {
        printf("emulating lack of EXT_direct_state_access support...\n");
        __glewDisableClientStateIndexedEXT = dsa_glDisableClientStateIndexedEXT;
        __glewEnableClientStateIndexedEXT = dsa_glEnableClientStateIndexedEXT;
        __glewMultiTexCoordPointerEXT = dsa_glMultiTexCoordPointerEXT;
        __glewMatrixLoadIdentityEXT = dsa_glMatrixLoadIdentityEXT;
        __glewMatrixMultTransposefEXT = dsa_glMatrixMultTransposefEXT;
        __glewMatrixLoadTransposefEXT = dsa_glMatrixLoadTransposefEXT;
        __glewMatrixPopEXT = dsa_glMatrixPopEXT;
        __glewMatrixPushEXT = dsa_glMatrixPushEXT;
        __glewTextureBufferEXT = dsa_glTextureBufferEXT;
		__glewBindMultiTextureEXT = dsa_glBindMultiTextureEXT;
    }
}
