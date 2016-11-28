
/* PathStyle.hpp - class for managing SVG path styles */

// Copyright (c) NVIDIA Corporation. All rights reserved.

// http://www.w3.org/TR/SVG/painting.html#FillProperties
// http://www.w3.org/TR/SVG/painting.html#StrokeProperties

#pragma once
#ifndef __PathStyle_hpp__
#define __PathStyle_hpp__

#include <vector>

using std::vector;

// Properties of a path necessary to determine stencil coverage.
// (PathStyle does not contain shading/covering parameters.)
struct PathStyle {
    // fill properties
    bool do_fill;
    enum FillRule {
        EVEN_ODD = 0,
        NON_ZERO
    };
    FillRule fill_rule;

    // stroke properties
    bool do_stroke;
    float stroke_width;
    enum LineCap {
        BUTT_CAP,
        ROUND_CAP,
        SQUARE_CAP,
        TRIANGLE_CAP
    };
    LineCap line_cap;
    enum LineJoin {
        NONE_JOIN, // "naked" joins, don't add special join geometry
        MITER_REVERT_JOIN,  // SVG-style miter joins, snap to bevel when miter limit exceeded
        ROUND_JOIN,
        BEVEL_JOIN,
        // Qt limits the length of the miter edges to
        // stroke_width*miter_limit instead of snapping to bevel.
        MITER_TRUNCATE_JOIN // consistent with Qt's miter join style
    };
    LineJoin line_join;
    float miter_limit;
    vector<float> dash_array;
    float dash_offset;
    enum DashPhase {
        MOVETO_RESETS = 0,
        MOVETO_CONTINUES
    };
    DashPhase dash_phase;

    // Initializations match the initial Fill Properties and Stroke Properites of SVG
    PathStyle()
        : do_fill(true)
        , fill_rule(NON_ZERO)
        , do_stroke(false)
        , stroke_width(1)
        , line_cap(BUTT_CAP)
        , line_join(MITER_REVERT_JOIN)
        , miter_limit(4)
        , dash_array()
        , dash_offset(0)
        , dash_phase(MOVETO_CONTINUES)
    {}
};

#endif // __PathStyle_hpp__
