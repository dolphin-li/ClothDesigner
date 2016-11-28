
/* request_vsync.h - request buffer swap synchroization with vertical sync */

/* NVIDIA Corporation, Copyright 2007-2010 */

#if defined(__APPLE__)
# include <OpenGL/OpenGL.h>
#elif defined(_WIN32)
# include <windows.h>
# ifndef WGL_EXT_swap_control
typedef int (APIENTRY * PFNWGLSWAPINTERVALEXTPROC)(int);
# endif
#else
# include <GL/glx.h>
# ifndef GLX_SGI_swap_control
typedef int ( * PFNGLXSWAPINTERVALSGIPROC) (int interval);
# endif
#endif

/* enableSync true means synchronize buffer swaps to monitor refresh;
   false means do NOT synchornize. */
void requestSynchornizedSwapBuffers(int enableSync)
{
#if defined(__APPLE__)
#ifdef GL_VERSION_1_2
  const GLint sync = enableSync;
#else
  const long sync = enableSync;
#endif
  CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &sync);
#elif defined(_WIN32)
  PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT =
    (PFNWGLSWAPINTERVALEXTPROC) wglGetProcAddress("wglSwapIntervalEXT");

  if (wglSwapIntervalEXT) {
    wglSwapIntervalEXT(enableSync);
  }
#else
  PFNGLXSWAPINTERVALSGIPROC glXSwapIntervalSGI =
    (PFNGLXSWAPINTERVALSGIPROC) glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalSGI");
  if (glXSwapIntervalSGI) {
    glXSwapIntervalSGI(enableSync);
  }
#endif
}

