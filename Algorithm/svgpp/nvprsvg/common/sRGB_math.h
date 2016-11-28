
// sRGB.c - sRGB color space conversion utilities

#ifdef __cplusplus
extern "C" {
#endif

extern float convertLinearColorComponentToSRGBf(const float cl);
extern unsigned char convertLinearColorComponentToSRGBub(const float cl);
extern float convertSRGBColorComponentToLinearf(const float cs);

#ifdef __cplusplus
}
#endif
