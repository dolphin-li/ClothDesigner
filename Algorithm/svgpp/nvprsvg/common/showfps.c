
/* showfps.c - OpenGL code for rendering frames per second */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifdef _WIN32
#include <windows.h>  /* for QueryPerformanceCounter */
#endif

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <stdio.h>
#include <string.h>
#include "showfps.h"

static int reportFPS = 1;
static GLfloat textColor[3];
static int validFPS = 0;

static void drawFPS(double fpsRate)
{
  GLubyte dummy;
  char buffer[200], *c;

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
      glLoadIdentity();
      glOrtho(0, 1, 1, 0, -1, 1);
      //glDisable(GL_DEPTH_TEST);
      glColor3fv(textColor);
      glRasterPos2f(1,1);
      glBitmap(0, 0, 0, 0, -10*9, 15, &dummy);
      if (fpsRate > 0 || !validFPS) {
        sprintf(buffer, "fps %0.1f", fpsRate);
      } else {
        strcpy(buffer, "fps --");
      }
      for (c = buffer; *c != '\0'; c++)
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
      //glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

#ifdef _WIN32
static __int64 freq = 0;
#else
#include <sys/time.h> /* for gettimeofday and struct timeval */
#endif

double getElapsedTime()
{
  static int firstTime = 1;
  double secs;
#ifdef _WIN32
  /* Use Win32 performance counter for high-accuracy timing. */
  static __int64 startTime = 0;  /* Timer count for last fps update */
  __int64 newCount;

  if (!freq) {
    QueryPerformanceFrequency((LARGE_INTEGER*) &freq);
  }

  /* Update the frames per second count if we have gone past at least
     a second since the last update. */

  QueryPerformanceCounter((LARGE_INTEGER*) &newCount);
  if (firstTime) {
    startTime = newCount;
    firstTime = 0;
  }
  secs = (double) (newCount - startTime) / (double)freq;
#else
  /* Use BSD 4.2 gettimeofday system call for high-accuracy timing. */
  static struct timeval start_tp;
  struct timeval new_tp;
  
  gettimeofday(&new_tp, NULL);
  if (firstTime) {
    start_tp.tv_sec = new_tp.tv_sec;
    start_tp.tv_usec = new_tp.tv_usec;
    firstTime = 0;
  }
  secs = (new_tp.tv_sec - start_tp.tv_sec) + (new_tp.tv_usec - start_tp.tv_usec)/1000000.0;
#endif
  return secs;
}

static double lastFpsRate = 0;
static int frameCount = 0;     /* Number of frames for timing */
#ifdef _WIN32
/* Use Win32 performance counter for high-accuracy timing. */
static __int64 lastCount = 0;  /* Timer count for last fps update */
#else
static struct timeval last_tp = { 0, 0 };
#endif

void invalidateFPS()
{
    validFPS = 0;
}

void restartFPS()
{
    frameCount = 0;
    validFPS = 1;
    lastFpsRate = -1;
#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER*) &lastCount);
#else
    gettimeofday(&last_tp, NULL);
#endif
}

double just_handleFPS(void)
{
#ifdef _WIN32
  /* Use Win32 performance counter for high-accuracy timing. */
  __int64 newCount;

  if (!freq) {
    QueryPerformanceFrequency((LARGE_INTEGER*) &freq);
  }

  /* Update the frames per second count if we have gone past at least
     a second since the last update. */

  QueryPerformanceCounter((LARGE_INTEGER*) &newCount);
  frameCount++;
  if ((newCount - lastCount) > freq) {
    double fpsRate;

    fpsRate = (double) (freq * (__int64) frameCount)  / (double) (newCount - lastCount);
    lastCount = newCount;
    frameCount = 0;
    lastFpsRate = fpsRate;
  }
#else
  /* Use BSD 4.2 gettimeofday system call for high-accuracy timing. */
  struct timeval new_tp;
  double secs;
  
  gettimeofday(&new_tp, NULL);
  secs = (new_tp.tv_sec - last_tp.tv_sec) + (new_tp.tv_usec - last_tp.tv_usec)/1000000.0;
  if (secs >= 1.0) {
    lastFpsRate = frameCount / secs;
    last_tp = new_tp;
    frameCount = 0;
  }
  frameCount++;
#endif
  if (!validFPS) {
    restartFPS();
  }
  return lastFpsRate;
}

double handleFPS(void)
{
  double lastFpsRate = just_handleFPS();
  if (reportFPS) {
    drawFPS(lastFpsRate);
  }
  return lastFpsRate;
}

void colorFPS(float r, float g, float b)
{
  textColor[0] = r;
  textColor[1] = g;
  textColor[2] = b;
}

void toggleFPS(void)
{
  reportFPS = !reportFPS;
}

void enableFPS(void)
{
  reportFPS = 1;
}

void disableFPS(void)
{
  reportFPS = 0;
}
