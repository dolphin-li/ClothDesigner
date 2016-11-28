#ifndef SHOWFPS_H
#define SHOWFPS_H

/* showfps.h - OpenGL code for rendering frames per second */

/* Call handleFPS in your GLUT display callback every frame. */

#ifdef __cplusplus
extern "C" {
#endif

extern double just_handleFPS(void);
extern double handleFPS(void);
extern void toggleFPS();
extern void enableFPS();
extern void disableFPS();
extern void colorFPS(float r, float g, float b);
extern double getElapsedTime();
extern void invalidateFPS();

#ifdef __cplusplus
}
#endif

#endif /* SHOWFPS_H */
