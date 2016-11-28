
/* request_vsync.h - request buffer swap synchroization with vertical sync */

/* NVIDIA Corporation, Copyright 2007-2010 */

/* enableSync true means synchronize buffer swaps to monitor refresh;
   false means do NOT synchornize. */

#ifdef __cplusplus
extern "C" {
#endif

extern void requestSynchornizedSwapBuffers(int enableSync);

#ifdef __cplusplus
}
#endif

