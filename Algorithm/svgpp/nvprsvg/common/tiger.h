
/* tiger.h - draw classic PostScript tiger */

#ifdef __cplusplus
extern "C" {
#endif

extern void initTiger();
extern void drawTiger(int filling, int stroking);
extern void drawTigerRange(int filling, int stroking, unsigned int start, unsigned int end);
extern void renderTiger(int filling, int stroking);
extern void tigerDlistUsage(int b);
extern unsigned int getTigerPathCount();
extern GLuint getTigerBasePath();

#ifdef __cplusplus
}
#endif
