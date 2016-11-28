
/* path_process.cpp - path methods for processing commands of the path. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.0 or higher). */

#include "nvpr_svg_config.h"  // configure path renderers to use

#include "path.hpp"

#include <Cg/iostream.hpp>

#include <Cg/abs.hpp>
#include <Cg/acos.hpp>
#include <Cg/all.hpp>
#include <Cg/any.hpp>
#include <Cg/clamp.hpp>
#include <Cg/distance.hpp>
#include <Cg/dot.hpp>
#include <Cg/isinf.hpp>
#include <Cg/isnan.hpp>
#include <Cg/length.hpp>
//#include <Cg/lerp.hpp>  // This has problems with Visual Studio 2008
#define lerp(a,b,t) ((a) + (t)*((b)-(a)))
#include <Cg/max.hpp>
#include <Cg/mul.hpp>
#include <Cg/normalize.hpp>
#include <Cg/radians.hpp>
#include <Cg/sqrt.hpp>
#include <Cg/transpose.hpp>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

using namespace Cg;

// Equation F.6.5.4
static double arc_angle_helper(double2 u, double2 v)
{
    double cos_value = dot(u,v)/(length(u)*length(v)),
           clamped_cos_value = clamp(cos_value, -1.0, 1.0);
    double theta_sign = double(u.x*v.y - u.y*v.x < 0 ? -1 : 1);
    double angle = theta_sign * acos(clamped_cos_value);
    return angle;
}

template <typename T>
static bool illbehaved(T a)
{
    return any(isnan(a)) || any(isinf(a));
}

CenterPointArc::CenterPointArc(const EndPointArc &src)
{
    // Handle these two easy/common degenerate cases first.
    if (all(src.p[0] == src.p[1])) {
        // "If the endpoints (x1, y1) and (x2, y2) are identical, then this
        // is equivalent to omitting the elliptical arc segment entirely."
        CenterPointArc::form = DEGENERATE_POINT;
        // other members left undefined
        return;
    }
    if (src.radii.x == 0 || src.radii.y == 0) {
        CenterPointArc::form = DEGENERATE_LINE;
        CenterPointArc::p[0] = src.p[0];
        CenterPointArc::p[1] = src.p[1];
        // other members left undefined
        return;
    }

    // Conversion from endpoint arc to center arc parameterization
    // See http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
    const double2 v1 = double2(src.p[0]),  // v1.x = x1, v1.y = y1
                  v2 = double2(src.p[1]);  // v2.x = x2, v2.y = x2

    // Ensure radii are positive
    // Take the absolute value of rX and rY:
    double2 r = double2(abs(src.radii));  // r not const since could be adjusted if too small

    // Any nonzero value for either of the flags fA or fS is taken to mean the value 1
    const bool fA = src.large_arc_flag,
               fS = src.sweep_flag;

    // psi is taken mod 360 degrees.
    // [fine since sin and cos functions do this (better)]
    const double1 psi = radians(double1(src.x_axis_rotation));

    // Step 1: Compute (x1', y1') according to the formula
    // cw = clockwise
    const double2x2 cw_rotate_by_psi = double2x2(cos(psi), sin(psi),
                                                 -sin(psi), cos(psi));

    const double2 half_v_diff = (v1-v2)/2,
                  v1p = mul(cw_rotate_by_psi, half_v_diff);

    { // BEGIN F.6.6 Correction of out-of-range radii

        // Step 1: Ensure radii are non-zero
        // If rX = 0 or rY = 0, then treat this as a straight line
        // from (x1, y1) to (x2, y2) and stop.
        // [already earlier]

        // Otherwise,
        // Step 2: Ensure radii are positive
        // Take the absolute value of rX and rY:
        // [already done]

        // Step 3: Ensure radii are large enough
        // Using the primed coordinate values of equation (F.6.5.1), compute
        const double Lambda = dot(v1p/r,v1p/r);

        // If the result of the above equation is less than or equal to 1,
        // then no further change need be made to rX and rY.  If the result
        // of the above equation is greater than 1, then make the replacements
        if (Lambda > 1) {
            r = sqrt(Lambda)*r;
        }
        // Step 4: Proceed with computations

        // Proceed with the remaining elliptical arc computations, such as
        // those in section F.6.5.  Note: As a consequence of the radii
        // corrections in this section, equation (F.6.5.2) for the center
        // of the ellipse always has at least one solution (i.e. the
        // radicand is never negative).  In the case that the radii are
        // scaled up using equation (F.6.6.3), the radicand of (F.6.5.2)
        // is zero and there is exactly one solution for the center of the
        // ellipse.
    } // END F.6.6 Correction of out-of-range radii

    // Step 2: Compute (cX', cY') according to the [re-arranged] formula
    // Rearrange sqrt((rx^2*ry^2 - rx^2*y1p^2 - ry^2*x1p^2)/
    //                (rx^2*y1p^2 + ry^2*x1p^2))
    // to be sqrt(rx^2*ry^2)/(rx^2*y1p^2 + ry^2*x1p^2))-1
    // because sqrt((- rx^2*y1p^2 - ry^2*x1p^2)/
    //              (rx^2*y1p^2 + ry^2*x1p^2))
    // equals -1
    double numer = r.x*r.x * r.y*r.y,
           denom = dot(r*r, (v1p*v1p).yx);
    double radicand = numer /denom - 1;
    // Add max to make sure numer/denom-1 can't be negative
    double k = sqrt(max(0, radicand));
    // where the + sign is chosen if fA notequal fS
    // and the - sign is chosen if fA equal fS
    if (fA == fS) {
        k = -k;
    }
    double2 cp = k*(r*(v1p/r).yx);
    cp.y = -cp.y;

    // Step 3: Compute (cX, cY) from (cX', cY')
    // ccw = counterclockwise
    const double2x2 ccw_rotate_by_psi = transpose(cw_rotate_by_psi);
    const double2 average_v = (v1+v2)/2,
                  c = mul(ccw_rotate_by_psi, cp) + average_v;

    // Step 4: Compute theta1 and delta-theta

    // This angle function can be used to express theta1 and delta-theta as follows
    double1 theta1 = arc_angle_helper(double2(1,0), (v1p-cp)/r),
            delta_theta = arc_angle_helper((v1p-cp)/r, (-v1p-cp)/r);
    // where theta1 is fixed in the range -360 degrees < delta-theta < 360
    // degrees such that:
    // if fS equal 0, then delta-theta < 0,
    // else if fS equal 1, then delta-theta > 0.
    //
    // In other words, if fS = 0 and the right side of (F.6.5.6) is > 0,
    // then subtract 360 degrees, whereas if fS = 1 and the right side of
    // (F.6.5.6) is < 0, then add 360 degrees.  In all other cases leave it as is.
    const double pi = M_PI;
    if (!fS && delta_theta > 0) {
        delta_theta = delta_theta - 2*pi;
    } else if (fS && delta_theta < 0) {
        delta_theta = delta_theta + 2*pi;
    }

    // Weird stuff could happen due to division by zero or
    // overflow; treat any such situation as a degenerate line.
    if (illbehaved(c) || illbehaved(r) ||
        illbehaved(theta1) || illbehaved(delta_theta) || illbehaved(psi)) {
        CenterPointArc::form = DEGENERATE_LINE;
        CenterPointArc::p[0] = src.p[0];
        CenterPointArc::p[1] = src.p[1];
        // other members left undefined
        return;
    }

    // Assign back to the member variables 
    // (scoping avoids ambiguity because several locals are declared
    // double precision with identical names to the member variables)
    CenterPointArc::form = BEHAVED;
    CenterPointArc::center = float2(c);
    CenterPointArc::radii = float2(r);
    CenterPointArc::p[0] = src.p[0];
    CenterPointArc::p[1] = src.p[1];
    CenterPointArc::theta1 = float(theta1);
    CenterPointArc::delta_theta = float(delta_theta);
    CenterPointArc::psi = float(psi);

    // Early return possible for degenerate cases!
}

float2 CenterPointArc::eval(double theta) {
    double2 result;
    double dpsi = psi,
           cos_psi = cos(dpsi),
           sin_psi = sin(dpsi),
           cos_theta = cos(theta),
           sin_theta = sin(theta);
    result.x = cos_psi*radii.x*cos_theta - sin_psi*radii.y*sin_theta;
    result.y = sin_psi*radii.x*cos_theta + cos_psi*radii.y*sin_theta;
    result += center;
    return float2(result);
}

static inline bool isCubicCommand(char c)
{
    switch (c) {
    case 'C':
    case 'c':
    case 'S':
    case 's':
        return true;
    default:
        return false;
    }
}

static inline bool isQuadraticCommand(char c)
{
    switch (c) {
    case 'Q':
    case 'q':
    case 'T':
    case 't':
        return true;
    default:
        return false;
    }
}

struct PathSegmentCount {
    int count;

    PathSegmentCount() : count(0) { }
};

struct SegmentCountProcessor : PathSegmentProcessor {
    PathSegmentCount &state;

    SegmentCountProcessor(PathSegmentCount &count) : state(count) {}

    void beginPath(PathPtr p) {}
    void moveTo(const float2 p[2], size_t coord_index, char cmd) { };
    void lineTo(const float2 p[2], size_t coord_index, char cmd) {
        state.count++;
    }
    void quadraticCurveTo(const float2 p[3], size_t coord_index, char cmd) {
        state.count++;
    }
    void cubicCurveTo(const float2 p[4], size_t coord_index, char cmd) {
        state.count++;
    }
    void arcTo(const EndPointArc &arc, size_t coord_index, char cmd) {
        state.count++;
    }
    void close(char cmd) { }
    void endPath(PathPtr p) {}
};

struct ControlPointHitProcessor : PathSegmentProcessor {
    ActiveControlPoint &state;
    PathPtr path;
    bool hit_path;

    ControlPointHitProcessor(ActiveControlPoint &hit) : state(hit), hit_path(false) {}

    void beginPath(PathPtr p) {
        path = p;
    }
    void moveTo(const float2 plist[2], size_t coord_index, char cmd) {
        float2 relative_to = float2(0,0);
        switch (cmd) {
        default:
            assert(!"unexpected cmd");
        case 'M':
            break;
        case 'm':
            relative_to = plist[0];
            break;
        }
        state.testForControlPointHit(plist[1],
                                     coord_index, relative_to,
                                     ActiveControlPoint::X_AND_Y, hit_path, path);
        state.testForControlPointHit(plist[1],
                                   coord_index, relative_to,
                                   ActiveControlPoint::X_AND_Y, hit_path, path);
    };
    void lineTo(const float2 plist[2], size_t coord_index, char cmd) {
        float2 relative_to = float2(0,0);
        ActiveControlPoint::CoordUsage coord_usage = ActiveControlPoint::X_AND_Y;
        switch (cmd) {
        default:
            assert(!"unexpected cmd");
        case 'L':
            break;
        case 'H':
            coord_usage = ActiveControlPoint::X_ONLY;
            break;
        case 'V':
            coord_usage = ActiveControlPoint::Y_ONLY;
            break;
        case 'l':
            relative_to = plist[0];
            break;
        case 'h':
            relative_to = plist[0];
            coord_usage = ActiveControlPoint::X_ONLY;
            break;
        case 'v':
            relative_to = plist[0];
            coord_usage = ActiveControlPoint::Y_ONLY;
            break;
        }
        state.testForControlPointHit(plist[1],
                                   coord_index, relative_to,
                                   coord_usage, hit_path, path);
    }
    void quadraticCurveTo(const float2 plist[3], size_t coord_index, char cmd) {
        float2 relative_to = float2(0,0);
        int num_points = 2;
        switch (cmd) {
        default:
            assert(!"unexpected cmd");
        case 'Q':
            break;
        case 'q':
            relative_to = plist[0];
            break;
        case 'T':
            num_points = 1;
            break;
        case 't':
            num_points = 1;
            relative_to = plist[0];
            break;
        }
        for (int i=0; i<num_points; i++) {
            state.testForControlPointHit(plist[i+1],
                                       coord_index+2*i, relative_to,
                                       ActiveControlPoint::X_AND_Y, hit_path, path);
        }
    }
    void cubicCurveTo(const float2 plist[4], size_t coord_index, char cmd) {
        float2 relative_to = float2(0,0);
        int num_points = 3;
        switch (cmd) {
        default:
            assert(!"unexpected cmd");
        case 'C':
            break;
        case 'c':
            relative_to = plist[0];
            break;
        case 'S':
            num_points = 2;
            break;
        case 's':
            num_points = 2;
            relative_to = plist[0];
            break;
        }
        for (int i=0; i<num_points; i++) {
            state.testForControlPointHit(plist[i+1],
                                       coord_index+2*i, relative_to,
                                       ActiveControlPoint::X_AND_Y, hit_path, path);
        }
    }
    void arcTo(const EndPointArc &arc, size_t coord_index, char cmd) {
        float2 relative_to = float2(0,0);
        switch (cmd) {
        default:
            assert(!"unexpected cmd");
        case 'A':
            break;
        case 'a':
            relative_to = arc.p[0];
            break;
        }
        const int to_point_offset = 5;
        state.testForControlPointHit(arc.p[1],
                                   coord_index+to_point_offset, relative_to,
                                   ActiveControlPoint::X_AND_Y, hit_path, path);
    }
    void close(char c) { }
    void endPath(PathPtr p) {
        path = PathPtr();
    }
};

void Path::findNearerControlPoint(ActiveControlPoint &hit)
{
    ControlPointHitProcessor processor(hit);
    processSegments(processor);
}

int Path::countSegments()
{
    PathSegmentCount count;

    SegmentCountProcessor counter(count);
    processSegments(counter);
    return count.count;
}

void Path::processSegments(PathSegmentProcessor &processor)
{
    float2 current = float2(0,0),
           initial_point = float2(0,0),
           prior_curve_point;
    char last_c = 0, c;
    const size_t n = cmd.size();

    processor.beginPath(shared_from_this());
    for (size_t i=0, j=0; i<n; i++, last_c = c) {
        c = cmd[i];
        switch (c) {
        case 'M':
        case 'm':
            {
                float2 plist[] = {
                    current,
                    float2(coord[j+0], coord[j+1]),
                };
                if (c == 'm') {  // relative
                    plist[1] += current;
                }
                processor.moveTo(plist, j, c);
                current = plist[1];
                initial_point = plist[1];
                j += 2;
            }
            break;
        case 'H':
        case 'h':
            {
                float2 plist[] = {
                    current,
                    float2(coord[j+0], current.y),
                };
                if (c == 'h') {  // relative
                    plist[1].x += current.x;
                }
                processor.lineTo(plist, j, c);
                current = plist[1];
                j += 1;
            }
            break;
        case 'V':
        case 'v':
            {
                float2 plist[] = {
                    current,
                    float2(current.x, coord[j+0]),
                };
                if (c == 'v') {  // relative
                    plist[1].y += current.y;
                }
                processor.lineTo(plist, j, c);
                current = plist[1];
                j += 1;
            }
            break;
        case 'L':
        case 'l':
            {
                float2 plist[] = {
                    current,
                    float2(coord[j+0], coord[j+1]),
                };
                if (c == 'l') {  // relative
                    plist[1] += current;
                }
                processor.lineTo(plist, j, c);
                current = plist[1];
                j += 2;
            }
            break;
        case 'Q':
        case 'q':
            {
                float2 plist[] = {
                    current,
                    float2(coord[j+0], coord[j+1]),
                    float2(coord[j+2], coord[j+3]),
                };
                if (c == 'q') {  // relative
                    plist[1] += current;
                    plist[2] += current;
                }
                processor.quadraticCurveTo(plist, j, c);
                prior_curve_point = plist[1];
                current = plist[2];
                j += 2*2;
            }
            break;
        case 'T':
        case 't':
            {
                float2 plist[] = {
                    current,
                    // "The control point is assumed to be the reflection of the control
                    // point on the previous command relative to the current point. (If 
                    // there is no previous command or if the previous command was not a Q,
                    // q, T or t, assume the control point is coincident with the current point.)
                    isQuadraticCommand(last_c) ? 2*current - prior_curve_point : current,
                    float2(coord[j+0], coord[j+1]),
                };
                if (c == 't') {  // relative
                    plist[2] += current;
                }
                processor.quadraticCurveTo(plist, j, c);
                prior_curve_point = plist[1];
                current = plist[2];
                j += 2*1;
            }
            break;
        case 'C':
        case 'c':
            {
                float2 plist[] = {
                    current,
                    float2(coord[j+0], coord[j+1]),
                    float2(coord[j+2], coord[j+3]),
                    float2(coord[j+4], coord[j+5]),
                };
                if (c == 'c') {  // relative
                    plist[1] += current;
                    plist[2] += current;
                    plist[3] += current;
                }
                processor.cubicCurveTo(plist, j, c);
                prior_curve_point = plist[2];
                current = plist[3];
                j += 2*3;
            }
            break;
        case 'S':
        case 's':
            {
                float2 plist[] = {
                    current,
                    // "The first control point is assumed to be the reflection
                    // of the second control point on the previous command relative
                    // to the current point. (If there is no previous command or if
                    // the previous command was not an C, c, S or s, assume the first
                    // control point is coincident with the current point.)"
                    isCubicCommand(last_c) ? 2*current - prior_curve_point : current,
                    float2(coord[j+0], coord[j+1]),
                    float2(coord[j+2], coord[j+3]),
                };
                if (c == 's') {  // relative
                    plist[2] += current;
                    plist[3] += current;
                }
                processor.cubicCurveTo(plist, j, c);
                prior_curve_point = plist[2];
                current = plist[3];
                j += 2*2;
            }
            break;
        case 'A':
        case 'a':
            {
                EndPointArc arc;

                // 0: rx -- ellipse X radius
                // 1: ry -- ellipse Y radius
                // 2: x-axis-rotation
                // 3: large-arc-flag
                // 4: sweep-flag
                // 5: x -- X of to-point
                // 6: y -- Y of to-point
                arc.p[0] = current;
                arc.radii = float2(coord[j+0], coord[j+1]);
                arc.x_axis_rotation = coord[j+2];
                arc.large_arc_flag = coord[j+3] != 0;
                arc.sweep_flag = coord[j+4] != 0;
                arc.p[1] = float2(coord[j+5], coord[j+6]);
                if (c == 'a') {
                    arc.p[1] += current;
                }
                current = arc.p[1];
                processor.arcTo(arc, j, c);
                j += 7;
            }
            break;
        case 'Z':
        case 'z':
            {
                // Optimization: closing a filled path doesn't need to generate
                // a line when this contour's initial point and current are identical.
                // NOTE:  This optimization isn't legal for a stroked path because
                // closing the path eliminates end-caps.
                if (any(initial_point != current)) {
                    float2 plist[] = {
                        current,
                        initial_point,
                    };
                    // "The "closepath" (Z or z) ends the current subpath and causes
                    // an automatic straight line to be drawn from the current point
                    // to the initial point of the current subpath."
                    processor.lineTo(plist, j, 'L');
                    // "If a "closepath" is followed immediately by any other command,
                    // then the next subpath starts at the same initial point as the
                    // current subpath."
                    current = initial_point;
                }
                processor.close(c);
            }
            break;
        default:
            assert(!"bad command");
            break;
        }
    }
    processor.endPath(shared_from_this());
}

struct DrawControlPointProcessor : PathSegmentProcessor {
    void beginPath(PathPtr p) { }
    void moveTo(const float2 p[2], size_t coord_index, char cmd) {
        glColor3f(1,1,0.66f);  // ???
        glVertex2f(p[1].x, p[1].y);
    };
    void lineTo(const float2 p[2], size_t coord_index, char cmd) {
        glColor3f(0,0,1);  // blue
        glVertex2f(p[1].x, p[1].y);
    }
    void quadraticCurveTo(const float2 p[3], size_t coord_index, char cmd) {
        glColor3f(0,1,1);  // cyan
        glVertex2f(p[1].x, p[1].y);
        glColor3f(1,0,1); // magenta
        glVertex2f(p[2].x, p[2].y);
    }
    void cubicCurveTo(const float2 p[4], size_t coord_index, char cmd) {
        glColor3f(1,0.5,0.5);  // red
        glVertex2f(p[1].x, p[1].y);
        glColor3f(1,0.5,0.5); // red
        glVertex2f(p[2].x, p[2].y);
        glColor3f(0.5,1,0.5); // green
        glVertex2f(p[3].x, p[3].y);
    }
    void arcTo(const EndPointArc &arc, size_t coord_index, char cmd) {
        // Convert to a center point arc to be able to render the arc center.
        CenterPointArc center_point_arc(arc);
        if (center_point_arc.form == CenterPointArc::BEHAVED) {
            glColor3f(0.33f,0,0);
            glVertex2f(center_point_arc.p[1].x, center_point_arc.p[1].y);
            glColor3f(0.66f,0,0);
        } else {
            glColor3f(0.66f,0,1);  // hint the control point is for a degenerate arc
        }
        glVertex2f(center_point_arc.center.x, center_point_arc.center.y);
    }
    void close(char cmd) { }
    void endPath(PathPtr p) {}
};

void Path::drawControlPoints()
{
    glPointSize(5.0);
    glBegin(GL_POINTS); {
        DrawControlPointProcessor processor;
        processSegments(processor);
    } glEnd();
}

// This version also draws tangent offset curve points
static void draw_cubic_reference_points(const float2 b[4], const int steps)
{
    const float2 P0 = b[0],
                 P1 = b[1],
                 P2 = b[2],
                 P3 = b[3];

    for (int i=0; i<=steps; i++) {
        const float t = float(i) / steps;

        const float2 p = (1-t)*(1-t)*(1-t)*P0 + 3*(1-t)*(1-t)*t*P1 + 3*(1-t)*t*t*P2+ t*t*t*P3;
        const float2 dpdt = -3*(1-t)*(1-t)*P0-6*(1-t)*t*P1+3*(1-t)*(1-t)*P1-3*t*t*P2+6*(1-t)*t*P2+3*t*t*P3;
        const float2 tangent = normalize(dpdt);
        const float2 normal = 30*float2(-tangent.y, tangent.x);

        glColor3f(1,1,1);
        glVertex2f(p.x, p.y);

#if 0  // add offset curve points (for stroking)
        glColor3f(1,0,0);
        glVertex2f(p.x + normal.x, p.y + normal.y);
        glColor3f(0,1,0);
        glVertex2f(p.x - normal.x, p.y - normal.y);
        glColor3f(0,0,1);
        glVertex2f(p.x + 6*tangent.x, p.y + 6*tangent.y);
#endif
    }
}

static void draw_quadratic_reference_points(const float2 b[3], const int steps)
{
    glColor3f(1,1,1);
    for (int i=0; i<=steps; i++) {
        float t = float(i) / steps;

        float2 pa = lerp(b[0], b[1], t),
               pb = lerp(b[1], b[2], t),
               p  = lerp(pa, pb, t);

        glVertex2f(p.x, p.y);
    }
}

static void draw_line_reference_points(const float2 b[2], const int steps)
{
    glColor3f(1,1,1);
    for (int i=0; i<=steps; i++) {
        float t = float(i) / steps;

        float2 p = lerp(b[0], b[1], t);

        glVertex2f(p.x, p.y);
    }
}

static void draw_arc_reference_points(CenterPointArc center_point_arc, const int steps)
{
    // "If the endpoints (x1, y1) and (x2, y2) are identical, then this
    // is equivalent to omitting the elliptical arc segment entirely."
    if (center_point_arc.form == CenterPointArc::DEGENERATE_POINT) {
        return;
    }

    // "If rX = 0 or rY  = 0 then this arc is treated as a straight
    // line segment (a "lineto") joining the endpoints."
    if (center_point_arc.form == CenterPointArc::DEGENERATE_LINE) {
        glColor3f(1,0,0); // red
        for (int i=1; i<steps; i++) {
            float t = float(i) / steps;

            float2 p = lerp(center_point_arc.p[0],
                            center_point_arc.p[1], t);

            glVertex2f(p.x, p.y);
        }
        return;
    }

    const float &theta1 = center_point_arc.theta1,
                &delta_theta = center_point_arc.delta_theta,
                theta2 = theta1 + delta_theta,
                theta_step = delta_theta / steps,
                &psi = center_point_arc.psi;
    const float2 &radii = center_point_arc.radii,
                 &center = center_point_arc.center;

    glColor3ub(255, 140, 0); // dark orange
    const float2x2 rotate = float2x2(cos(psi), -sin(psi),
                                     sin(psi), cos(psi));
    for (int i=0; i<steps; i++) {
        const float theta = theta1 + i*theta_step;
        const float2 scale = radii * float2(cos(theta), sin(theta));

        const float2 p = mul(rotate, scale) + center;
        glVertex2f(p.x, p.y);
    }
    const float2 scale = radii * float2(cos(theta2), sin(theta2));
    const float2 p = mul(rotate, scale) + center;
    glVertex2f(p.x, p.y);
}

struct DrawReferencePointProcessor : PathSegmentProcessor {
    void beginPath(PathPtr p) {}
    void moveTo(const float2 p[2], size_t coord_index, char cmd) { };
    void lineTo(const float2 p[2], size_t coord_index, char cmd) {
        draw_line_reference_points(p, 5);
    }
    void quadraticCurveTo(const float2 p[3], size_t coord_index, char cmd) {
        draw_quadratic_reference_points(p, 8);
    }
    void cubicCurveTo(const float2 p[4], size_t coord_index, char cmd) {
        draw_cubic_reference_points(p, 20);
    }
    void arcTo(const EndPointArc &arc, size_t coord_index, char cmd) {
        CenterPointArc center_point_arc(arc);
        draw_arc_reference_points(center_point_arc, 12);
    }
    void close(char cmd) { }
    void endPath(PathPtr p) {}
};

void Path::drawReferencePoints()
{
    glPointSize(3.0);
    glBegin(GL_POINTS); {
        DrawReferencePointProcessor processor;
        processSegments(processor);
    } glEnd();
}
