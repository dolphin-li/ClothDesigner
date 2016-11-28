
/* color_names.cpp - mapping from SVG color names to RGB values */

// Copyright (c) NVIDIA Corporation. All rights reserved.

// http://www.w3.org/TR/SVG/types.html#ColorKeywords

#include <string.h>

#include <Cg/vector/rgba.hpp>
#include <Cg/vector.hpp>

#include "countof.h"

#include "color_names.hpp"

using Cg::float3;
using Cg::float4;

struct ColorName {
  const char *name;
  float3 rgb;
};

static const ColorName color_names[] = {
#define COLOR(r,g,b) float3(r/255.0, g/255.0, b/255.0)
    { "aliceblue ", COLOR(240, 248, 255) },
    { "antiquewhite", COLOR(250, 235, 215) },
    { "aqua", COLOR( 0, 255, 255) },
    { "aquamarine", COLOR(127, 255, 212) },
    { "azure", COLOR(240, 255, 255) },
    { "beige", COLOR(245, 245, 220) },
    { "bisque", COLOR(255, 228, 196) },
    { "black", COLOR( 0, 0, 0) },
    { "blanchedalmond", COLOR(255, 235, 205) },
    { "blue", COLOR( 0, 0, 255) },
    { "blueviolet", COLOR(138, 43, 226) },
    { "brown", COLOR(165, 42, 42) },
    { "burlywood", COLOR(222, 184, 135) },
    { "cadetblue", COLOR( 95, 158, 160) },
    { "chartreuse", COLOR(127, 255, 0) },
    { "chocolate", COLOR(210, 105, 30) },
    { "coral", COLOR(255, 127, 80) },
    { "cornflowerblue", COLOR(100, 149, 237) },
    { "cornsilk", COLOR(255, 248, 220) },
    { "crimson", COLOR(220, 20, 60) },
    { "cyan", COLOR( 0, 255, 255) },
    { "darkblue", COLOR( 0, 0, 139) },
    { "darkcyan", COLOR( 0, 139, 139) },
    { "darkgoldenrod", COLOR(184, 134, 11) },
    { "darkgray", COLOR(169, 169, 169) },
    { "darkgreen", COLOR( 0, 100, 0) },
    { "darkgrey", COLOR(169, 169, 169) },
    { "darkkhaki", COLOR(189, 183, 107) },
    { "darkmagenta", COLOR(139, 0, 139) },
    { "darkolivegreen", COLOR( 85, 107, 47) },
    { "darkorange", COLOR(255, 140, 0) },
    { "darkorchid", COLOR(153, 50, 204) },
    { "darkred", COLOR(139, 0, 0) },
    { "darksalmon", COLOR(233, 150, 122) },
    { "darkseagreen", COLOR(143, 188, 143) },
    { "darkslateblue", COLOR( 72, 61, 139) },
    { "darkslategray", COLOR( 47, 79, 79) },
    { "darkslategrey", COLOR( 47, 79, 79) },
    { "darkturquoise", COLOR( 0, 206, 209) },
    { "darkviolet", COLOR(148, 0, 211) },
    { "deeppink", COLOR(255, 20, 147) },
    { "deepskyblue", COLOR( 0, 191, 255) },
    { "dimgray", COLOR(105, 105, 105) },
    { "dimgrey", COLOR(105, 105, 105) },
    { "dodgerblue", COLOR( 30, 144, 255) },
    { "firebrick", COLOR(178, 34, 34) },
    { "floralwhite", COLOR(255, 250, 240) },
    { "forestgreen", COLOR( 34, 139, 34) },
    { "fuchsia", COLOR(255, 0, 255) },
    { "gainsboro", COLOR(220, 220, 220) },
    { "ghostwhite", COLOR(248, 248, 255) },
    { "gold", COLOR(255, 215, 0) },
    { "goldenrod", COLOR(218, 165, 32) },
    { "gray", COLOR(128, 128, 128) },
    { "green", COLOR( 0, 128, 0) },
    { "greenyellow", COLOR(173, 255, 47) },
    { "grey", COLOR(128, 128, 128) },
    { "honeydew", COLOR(240, 255, 240) },
    { "hotpink", COLOR(255, 105, 180) },
    { "indianred", COLOR(205, 92, 92) },
    { "indigo", COLOR( 75, 0, 130) },
    { "ivory", COLOR(255, 255, 240) },
    { "khaki", COLOR(240, 230, 140) },
    { "lavender", COLOR(230, 230, 250) },
    { "lavenderblush", COLOR(255, 240, 245) },
    { "lawngreen", COLOR(124, 252, 0) },
    { "lemonchiffon", COLOR(255, 250, 205) },
    { "lightblue", COLOR(173, 216, 230) },
    { "lightcoral", COLOR(240, 128, 128) },
    { "lightcyan", COLOR(224, 255, 255) },
    { "lightgoldenrodyellow", COLOR(250, 250, 210) },
    { "lightgray", COLOR(211, 211, 211) },
    { "lightgreen", COLOR(144, 238, 144) },
    { "lightgrey", COLOR(211, 211, 211) },
    { "lightpink", COLOR(255, 182, 193) },
    { "lightsalmon", COLOR(255, 160, 122) },
    { "lightseagreen", COLOR( 32, 178, 170) },
    { "lightskyblue", COLOR(135, 206, 250) },
    { "lightslategray", COLOR(119, 136, 153) },
    { "lightslategrey", COLOR(119, 136, 153) },
    { "lightsteelblue", COLOR(176, 196, 222) },
    { "lightyellow", COLOR(255, 255, 224) },
    { "lime", COLOR( 0, 255, 0) },
    { "limegreen", COLOR( 50, 205, 50) },
    { "linen", COLOR(250, 240, 230) },
    { "magenta", COLOR(255, 0, 255) },
    { "maroon", COLOR(128, 0, 0) },
    { "mediumaquamarine", COLOR(102, 205, 170) },
    { "mediumblue", COLOR( 0, 0, 205) },
    { "mediumorchid", COLOR(186, 85, 211) },
    { "mediumpurple", COLOR(147, 112, 219) },
    { "mediumseagreen", COLOR( 60, 179, 113) },
    { "mediumslateblue", COLOR(123, 104, 238) },
    { "mediumspringgreen", COLOR( 0, 250, 154) },
    { "mediumturquoise", COLOR( 72, 209, 204) },
    { "mediumvioletred", COLOR(199, 21, 133) },
    { "midnightblue", COLOR( 25, 25, 112) },
    { "mintcream", COLOR(245, 255, 250) },
    { "mistyrose", COLOR(255, 228, 225) },
    { "moccasin", COLOR(255, 228, 181) },
    { "navajowhite", COLOR(255, 222, 173) },
    { "navy", COLOR( 0, 0, 128) },
    { "oldlace", COLOR(253, 245, 230) },
    { "olive", COLOR(128, 128, 0) },
    { "olivedrab", COLOR(107, 142, 35) },
    { "orange", COLOR(255, 165, 0) },
    { "orangered", COLOR(255, 69, 0) },
    { "orchid", COLOR(218, 112, 214) },
    { "palegoldenrod", COLOR(238, 232, 170) },
    { "palegreen", COLOR(152, 251, 152) },
    { "paleturquoise", COLOR(175, 238, 238) },
    { "palevioletred", COLOR(219, 112, 147) },
    { "papayawhip", COLOR(255, 239, 213) },
    { "peachpuff", COLOR(255, 218, 185) },
    { "peru", COLOR(205, 133, 63) },
    { "pink", COLOR(255, 192, 203) },
    { "plum", COLOR(221, 160, 221) },
    { "powderblue", COLOR(176, 224, 230) },
    { "purple", COLOR(128, 0, 128) },
    { "red", COLOR(255, 0, 0) },
    { "rosybrown", COLOR(188, 143, 143) },
    { "royalblue", COLOR( 65, 105, 225) },
    { "saddlebrown", COLOR(139, 69, 19) },
    { "salmon", COLOR(250, 128, 114) },
    { "sandybrown", COLOR(244, 164, 96) },
    { "seagreen", COLOR( 46, 139, 87) },
    { "seashell", COLOR(255, 245, 238) },
    { "sienna", COLOR(160, 82, 45) },
    { "silver", COLOR(192, 192, 192) },
    { "skyblue", COLOR(135, 206, 235) },
    { "slateblue", COLOR(106, 90, 205) },
    { "slategray", COLOR(112, 128, 144) },
    { "slategrey", COLOR(112, 128, 144) },
    { "snow", COLOR(255, 250, 250) },
    { "springgreen", COLOR( 0, 255, 127) },
    { "steelblue", COLOR( 70, 130, 180) },
    { "tan", COLOR(210, 180, 140) },
    { "teal", COLOR( 0, 128, 128) },
    { "thistle", COLOR(216, 191, 216) },
    { "tomato", COLOR(255, 99, 71) },
    { "turquoise", COLOR( 64, 224, 208) },
    { "violet", COLOR(238, 130, 238) },
    { "wheat", COLOR(245, 222, 179) },
    { "white", COLOR(255, 255, 255) },
    { "whitesmoke", COLOR(245, 245, 245) },
    { "yellow", COLOR(255, 255, 0) },
    { "yellowgreen", COLOR(154, 205, 50) },
};

bool parse_color_name(const char *name, float4 &color)
{
    int first = 0,
        last = countof(color_names);

    while (first <= last) {
        int mid = (first + last) / 2;  // compute mid point.
        if (strcmp(name, color_names[mid].name) > 0) {
            first = mid + 1;  // repeat search in top half.
        } else if (strcmp(name, color_names[mid].name) < 0) {
            last = mid - 1; // repeat search in bottom half.
        } else {
            color.rgb = color_names[mid].rgb;
            return true;
        }
    }
    return false;
}
