
// sRGB.c - sRGB color space conversion utilities

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <math.h>

#include "sRGB_math.h"

#define FLOAT_TO_UB(_f) \
        (unsigned char)(floorf((_f) * 255.0f + 0.5f))

float convertLinearColorComponentToSRGBf(const float cl)
{
    float csf;

    if (cl > 1.0f) {
        csf = 1.0f;
    } else if (cl > 0.0f) {
        if (cl < 0.0031308f) {
            csf = 12.92f * cl;
        } else {
            csf = 1.055f * (float)pow(cl, 0.41666f) - 0.055f;
        }
    } else {
        /* IEEE NaN should get here since comparisons with NaN always
           fail. */
        csf = 0.0f;
    }
    return csf;
}

unsigned char convertLinearColorComponentToSRGBub(const float cl)
{
    unsigned char cs;

    if (cl > 1.0f) {
        cs = 255;
    } else if (cl > 0.0f) {
        float csf;

        if (cl < 0.0031308f) {
            csf = 12.92f * cl;
        } else {
            csf = 1.055f * (float)pow(cl, 0.41666f) - 0.055f;
        }
        cs = FLOAT_TO_UB(csf);
    } else {
        /* IEEE NaN should get here since comparisons with NaN always
           fail. */
        cs = 0;
    }
    return cs;
}

float convertSRGBColorComponentToLinearf(const float cs)
{
    float cl;

    if (cs <= 0.04045f) {
        cl = cs / 12.92f;
    } else {
        cl = (float)pow((cs + 0.055f)/1.055f, 2.4f);
    }
    return cl;
}
