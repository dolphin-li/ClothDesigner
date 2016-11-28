
/* path.cpp - path rendering demo. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.0 or higher). */

#include "nvpr_svg_config.h"  // configure path renderers to use

#include <limits.h>
#include <math.h>
#include <float.h>

#include "path.hpp"

#include <Cg/iostream.hpp>
#include <iterator>

#include <Cg/all.hpp>
#include <Cg/cross.hpp>
#include <Cg/dot.hpp>
#include <Cg/isfinite.hpp>
#include <Cg/inverse.hpp>
#include <Cg/length.hpp>
//#include <Cg/lerp.hpp>  // This has problems with Visual Studio 2008
#define lerp(a,b,t) ((a) + (t)*((b)-(a)))
#include <Cg/max.hpp>
#include <Cg/min.hpp>
#include <Cg/mul.hpp>
#include <Cg/normalize.hpp>
#include <Cg/sqrt.hpp>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include "countof.h"
#include "path_parse_svg.h"

// Grumble, Microsoft (and probably others) define these as macros
#undef min
#undef max

using namespace Cg;
using namespace boost;
using std::string;
using std::vector;
using std::cout;
using std::endl;

// Release builds shouldn't have verbose conditions.
#ifdef NDEBUG
const static int verbose = 0;
#else
const static int verbose = 1;
#endif

// For debugging to print STL vectors
template <typename T>
std::ostream& operator<<(std::ostream& s, std::vector<T> const& vec)
{
    s << "[" << vec.size() << "]";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(s, " "));
    return s;
}

static string float_to_string(float v)
{
    string s = lexical_cast<string>(v);
    return s;
}

string Path::convert_to_svg_path(const float4x4 &transform)
{
    string result;
    size_t ndx = 0;
    const size_t n = cmd.size();
    for (size_t i=0; i<n; i++) {
        char c = cmd[i];
        switch (c) {
        case 'M':
        case 'm':
            assert(ndx+2 <= coord.size());
            result += string(1,c) + " " + float_to_string(coord[ndx+0]) + "," + float_to_string(coord[ndx+1]);
            ndx += 2;
            break;
        case 'H':
        case 'h':
        case 'V':
        case 'v':
            assert(ndx+1 <= coord.size());
            result += string(1,c) + " " + float_to_string(coord[ndx+0]);
            ndx += 1;
            break;
        case 'C':
        case 'c':
            assert(ndx+6 <= coord.size());
            result += string(1,c) 
                   + " " + float_to_string(coord[ndx+0]) + "," + float_to_string(coord[ndx+1])
                   + " " + float_to_string(coord[ndx+2]) + "," + float_to_string(coord[ndx+3]) 
                   + " " + float_to_string(coord[ndx+4]) + "," + float_to_string(coord[ndx+5]);
            ndx += 6;
            break;
        case 'Q':
        case 'q':
        case 'S':
        case 's':
            assert(ndx+4 <= coord.size());
            result += string(1,c) 
                   + " " + float_to_string(coord[ndx+0]) + "," + float_to_string(coord[ndx+1]) 
                   + " " + float_to_string(coord[ndx+2]) + "," + float_to_string(coord[ndx+3]);
            ndx += 4;
            break;
        case 'L':
        case 'l':
        case 'T':
        case 't':
            assert(ndx+2 <= coord.size());
            result += string(1,c) 
                   + " " + float_to_string(coord[ndx+0]) + "," + float_to_string(coord[ndx+1]);
            ndx += 2;
            break;
        case 'A':
        case 'a':
            assert(ndx+7 <= coord.size());
            result += string(1,c) 
                   + " " + float_to_string(coord[ndx+0]) + "," + float_to_string(coord[ndx+1])
                   + " " + float_to_string(coord[ndx+2]) + " " + float_to_string(coord[ndx+3]) + " " + float_to_string(coord[ndx+4])
                   + " " + float_to_string(coord[ndx+5]) + "," + float_to_string(coord[ndx+6]);
            ndx += 7;
            break;
        case 'z':
        case 'Z':
            result += string(1,c);
            break;
        }
    }
    assert(ndx == coord.size());
    return result;
}

#if 0 // XXX remove me?
static bool sameVertex(double3 a, double3 b)
{
    bool3 sameComponents = (a == b);
    bool allSame = all(sameComponents);
    return allSame;
}

static bool sameVertex(float2 a, float2 b)
{
    bool2 sameComponents = (a == b);
    bool allSame = all(sameComponents);
    return allSame;
}
#endif

Path::Path(const vector<char> &cmds, const vector<float> &coords)
    : HasRendererState<Path>(this)
    , has_logical_bbox(false)
    , cmd(cmds)
	, coord(coords)
	, ldp_poly_id(-1)
	, ldp_poly_cylinder_dir(false)
{
    owner = this;
	memset(ldp_poly_3dCenter, 0, sizeof(ldp_poly_3dCenter));
	memset(ldp_poly_3dRot, 0, sizeof(ldp_poly_3dRot));
}

Path::Path(const PathStyle &s, const vector<char> &cmds, const vector<float> &coords)
    : HasRendererState<Path>(this)
    , has_logical_bbox(false)
    , cmd(cmds)
    , coord(coords)
	, style(s)
	, ldp_poly_id(-1)
	, ldp_poly_cylinder_dir(false)
{
	memset(ldp_poly_3dCenter, 0, sizeof(ldp_poly_3dCenter));
	memset(ldp_poly_3dRot, 0, sizeof(ldp_poly_3dRot));
}

Path::Path(const char *string)
    : HasRendererState<Path>(this)
	, has_logical_bbox(false)
	, ldp_poly_id(-1)
	, ldp_poly_cylinder_dir(false)
{
    int ok = parse_svg_path(string, cmd, coord);
    if (!ok) {
        printf("Mis-formed path: <%s>\n", string);
        cmd.clear();
        coord.clear();
	}		
	memset(ldp_poly_3dCenter, 0, sizeof(ldp_poly_3dCenter));
	memset(ldp_poly_3dRot, 0, sizeof(ldp_poly_3dRot));
}

Path::Path(const PathStyle &s, const char *string)
    : HasRendererState<Path>(this)
    , has_logical_bbox(false)
    , style(s)
	, ldp_poly_id(-1)
	, ldp_poly_cylinder_dir(false)
{
    int ok = parse_svg_path(string, cmd, coord);
    if (!ok) {
        printf("Mis-formed path: <%s>\n", string);
        cmd.clear();
        coord.clear();
    }
	memset(ldp_poly_3dCenter, 0, sizeof(ldp_poly_3dCenter));
	memset(ldp_poly_3dRot, 0, sizeof(ldp_poly_3dRot));
}

void Path::invalidate()
{
    invalidateRenderStates();
}

void Path::validate()
{
    assert(!"rethink");
}

void Path::gatherStats(PathStats &stats)
{
    stats.num_cmds = cmd.size();
    stats.num_coords = coord.size();
}

// Solve the quadratic equation in form x^2 + b*x + c = 0
template <typename REAL>
int quadratic(REAL b, REAL c, REAL rts[2])
{
   int solutions = 0;
   rts[0] = 0;
   rts[1] = 0;

   REAL discriminant = b*b - 4*c;
   if (b == 0) {
      if (c == 0) {
         solutions = 2;
      } else {
         if (c < 0) {
            solutions = 2;
            rts[0] = sqrt(-c);
            rts[1] = -rts[0];
         } else {
            solutions = 0;
         }         
      }
   } else if (c == 0) {
      solutions = 2;
      rts[0] = -b;
   } else if (discriminant >= 0) {
      solutions = 2 ;
      REAL discriminant_sqrt = sqrt(discriminant);
      if (b > 0) {
         rts[0] = (-b - discriminant_sqrt)*(1/REAL(2));
      } else {
         rts[0] = (-b + discriminant_sqrt)*(1/REAL(2));
      }
      if (rts[0] == 0) {
         rts[1] = -b;
      } else {
         rts[1] = c/rts[0];
      }
   }
   return solutions;
}

struct GetBoundsPathSegmentProcessor : PathSegmentProcessor {
    float4 bbox;

    GetBoundsPathSegmentProcessor()
        : bbox(FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX)
    {}

    void addPointToBounds(const float2 &p) {
        bbox.xy = min(bbox.xy, p);
        bbox.zw = max(bbox.zw, p);
    }
    void addXToBounds(const float x) {
        bbox.x = min(bbox.x, x);
        bbox.z = max(bbox.z, x);
    }
    void addYToBounds(const float y) {
        bbox.y = min(bbox.y, y);
        bbox.w = max(bbox.w, y);
    }
    void addComponentToBounds(int c, const float v) {
        bbox[c+0] = min(bbox[c+0], v);
        bbox[c+2] = max(bbox[c+2], v);
    }

    void beginPath(PathPtr p) { 
    }
    void moveTo(const float2 plist[2], size_t coord_index, char cmd) {
    }
    void lineTo(const float2 plist[2], size_t coord_index, char cmd) {
        addPointToBounds(plist[0]);
        addPointToBounds(plist[1]);
    }
    void quadraticCurveTo(const float2 plist[3], size_t coord_index, char cmd) {
        const double2 P0 = double2(plist[0]),
                      P1 = double2(plist[1]),
                      P2 = double2(plist[2]);

        // Quadratic Bezier parametric equation:
        // > Qeq:=P0*(1-t)^2+2*P1*(1-t)*t+P2*t^2;
        // One-half derivative of quadratic bezier parametric equation:
        // > Qteq:=diff(Qeq/2,t);
        //   Qteq:=-P0*(1-t)-P1*t+P1*(1-t)+P2*t
        // Collect coefficients of t terms:
        // > collect(Qteq/2,t);
        //   (P0-2*P1+P2)*t-P0+P1
        double2 a = P0 - 2 * P1 + P2,
                b = P1 - P0;

        // For X (j=0) and Y (j=1) components...
        for (int j=0; j<2; j++) {
            if (a[j] == 0) {
                // Constant equation.  End-points will suffice as bounds.
            } else {
                // Linear equation.
                double t = -b[j]/a[j];
                // Is t in parametric [0,1] range of the segment?
                if (t > 0 && t < 1) {
                    double v = (a[j]*t+2*b[j])*t+P0[j];
                    addComponentToBounds(j, float(v));
                }
            }
        }

        addPointToBounds(plist[0]);
        addPointToBounds(plist[2]);
    }
    void cubicCurveTo(const float2 plist[4], size_t coord_index, char cmd)
    {
        // Cubic bezier segment control points:
        const double2 P0 = double2(plist[0]),
                      P1 = double2(plist[1]),
                      P2 = double2(plist[2]),
                      P3 = double2(plist[3]);

        // Cubic Bezier parametric equation:
        // > Ceq:=P0*(1-t)^3+3*P1*(1-t)^2*t+3*P2*t^2*(1-t)+P3*t^3;
        // One-third derivative of cubic bezier parametric equation:
        // > Cteq:=diff(Ceq/3,t);
        //   Cteq := -P0*(1-t)^2-2*P1*(1-t)*t+P1*(1-t)^2+2*P2*t*(1-t)-P2*t^2+P3*t^2
        // Collect coefficients of t terms:
        // > collect(Cteq/3,t);
        //   (P3-3*P2+3*P1-P0)*t^2+(-4*P1+2*P2+2*P0)*t+P1-P0
        const double2 a = P3 - 3*P2 + 3*P1 - P0,
                      b = 2*P2 - 4*P1 + 2*P0,
                      c = P1 - P0;

        // For X (j=0) and Y (j=1) components...
        for (int j=0; j<2; j++) {
            // Solve for "t=0" for Cteq
            if (a[j] == 0) {
                // Not quadratic.
                if (b[j] == 0) {
                    // Constant equation.  End-points will suffice as bounds.
                } else {
                    // Is linear equation.
                    const double t = -c[j]/b[j];
                    // Is t in parametric [0,1] range of the segment?
                    if (t > 0 && t < 1) {
                        // Form original cubic equation in Horner form and evaluate:
                        const double v = P0[j]+(-3*P0[j]+3*P1[j]+(3*P0[j]+3*P2[j]-6*P1[j]+(3*P1[j]-P0[j]-3*P2[j]+P3[j])*t)*t)*t;
                        addComponentToBounds(j, float(v));
                    }
                }
            } else {
                // Need the quadratic equation.
                double t_array[2];
                const int solutions = quadratic(b[j]/a[j], c[j]/a[j], t_array);
                // For each quadratic equation solution...
                for (int i=0; i<solutions; i++) {
                    const double t = t_array[i];
                    // Is t in parametric [0,1] range of the segment?
                    if (t > 0 && t < 1) {
                        // Form original cubic equation in Horner form and evaluate:
                        const double v = P0[j]+(-3*P0[j]+3*P1[j]+(3*P0[j]+3*P2[j]-6*P1[j]+(3*P1[j]-P0[j]-3*P2[j]+P3[j])*t)*t)*t;
                        addComponentToBounds(j, float(v));
                    }
                }
            }
        }

        // Add initial and terminal points of cubic Bezier segment.
        addPointToBounds(plist[0]);
        addPointToBounds(plist[3]);
    }
    inline double wrapAngle(double theta) {
        if (std::isfinite(theta) && fabs(theta) < 4 * 2*M_PI) {
            // XXX fmod is slow, but it may already to this optimization
            // we may want to do a perf test and see if this is worth it ...
            while (theta >= 2*M_PI) {
                theta -= 2*M_PI;
            }
            while (theta < 0) {
                theta += 2*M_PI;
            }
        } else {
            theta = fmod(theta, 2*M_PI);
        }

        return theta;
    }
    void arcTo(const EndPointArc &arc, size_t coord_index, char cmd) {
        // Convert to a center point arc to be able to render the arc center.
        CenterPointArc center_point_arc(arc);

        // Add the arc's two end points to the bounding box.
        addPointToBounds(arc.p[0]);
        addPointToBounds(arc.p[1]);

        if (center_point_arc.form == CenterPointArc::BEHAVED) {
            double tan_psi = tan(center_point_arc.psi),
                   rx = center_point_arc.radii.x,
                   ry = center_point_arc.radii.y,
                   theta1 = wrapAngle(center_point_arc.theta1),
                   theta2 = wrapAngle(theta1 + center_point_arc.delta_theta);

            if (center_point_arc.delta_theta < 0) {
                double tmp = theta1;
                theta1 = theta2;
                theta2 = tmp;
            }

            // Use the partial elliptical arc segment center point parametric form...
            //
            // Solve for where gradient is zero in the interval [theta1,theta2].
            //
            // Ax:=cos(phi)*rx*cos(theta) - sin(phi)*ry*sin(theta) + cx;
            // diff(Ax,theta);
            //   -cos(phi)*rx*sin(theta)-sin(phi)*ry*cos(theta)
            // Ay:=sin(phi)*rx*cos(theta) + cos(phi)*ry*sin(theta) + cy;
            // diff(Ay,theta);
            //   -sin(phi)*rx*sin(theta)+cos(phi)*ry*cos(theta)
            // solve(diff(Ax,theta)=0,theta);
            //   -arctan(tan(phi)*ry/rx)
            // solve(diff(Ay,theta)=0,theta);
            //   arctan(ry/(tan(phi)*rx))
            double theta_x = wrapAngle(-atan2(tan_psi*ry, rx)),
                   theta_y = wrapAngle(atan2(ry, tan_psi*rx));

            // Here we have two scenarios: theta1 < theta2, and theta1 > theta2 
            // (if they're equal the ellipse is degenerate). Conceptually, an angle is
            // within the partial arc if it falls within the set of angles passed over by
            // moving from theta1 to theta2 in the direction of positive angles.

            // This can be accoomplish by saying
            // if (theta1 > theta2)
            //    on_path = theta1 <= angle <= theta2
            // else
            //    on_path = angle >= theta1 && angle <= theta2

            // If we swap theta1 and theta2 when theta2 > theta1, then a simple
            // if ((theta1 < theta_x && theta_x < theta2) == not_swapped)
            // can also do the job because there's no need to worry about </<= boundaries -
            // we already called addPointToBounds() before for the points at theta1 and theta2

            bool not_swapped = true;
            if (theta1 > theta2) {
                not_swapped = false;
                double tmp = theta1;
                theta1 = theta2;
                theta2 = tmp;
            }

            // Is theta_x in [theta1,theta2] range?
            if ((theta1 < theta_x && theta_x < theta2) == not_swapped) {
                float2 v = center_point_arc.eval(theta_x);
                addXToBounds(v.x);
            }
            // Is theta_y in [theta1,theta2] range?
            if ((theta1 < theta_y && theta_y < theta2) == not_swapped) {
                float2 v = center_point_arc.eval(theta_y);
                addYToBounds(v.y);
            }
            theta_x = wrapAngle(theta_x + M_PI);
            theta_y = wrapAngle(theta_y + M_PI);
            // Is theta_x in [theta1,theta2] range?
            if ((theta1 < theta_x && theta_x < theta2) == not_swapped) {
                float2 v = center_point_arc.eval(theta_x);
                addXToBounds(v.x);
            }
            // Is theta_y in [theta1,theta2] range?
            if ((theta1 < theta_y && theta_y < theta2) == not_swapped) {
                float2 v = center_point_arc.eval(theta_y);
                addYToBounds(v.y);
            }
        } else {
            // In degenerate cases, end points are enough.
            switch (center_point_arc.form) {
            case CenterPointArc::DEGENERATE_LINE:
            case CenterPointArc::DEGENERATE_POINT:
                // Do nothing.
                break;
            default:
                assert(!"bogus CenterPointArc form");
                break;
            }
        }
    }
    void close(char cmd) {
    }
    void endPath(PathPtr p) {}
};

float4 Path::getActualFillBounds()
{
    GetBoundsPathSegmentProcessor bounder;

    processSegments(bounder);
    if (verbose) {
        cout << "got: " << bounder.bbox << endl;
    }
    if (style.do_stroke) {
        float half_stroke_width = style.stroke_width/2;
        bounder.bbox += float2(-half_stroke_width,half_stroke_width).xxyy;
    }
    return bounder.bbox;
}

float4 Path::getDilatedFillBounds()
{
    float4 bbox = getActualFillBounds();
    if (style.do_stroke) {
        float half_stroke_width = style.stroke_width/2;
        bbox += float2(-half_stroke_width,half_stroke_width).xxyy;
    }
    return bbox;
}

float4 Path::getBounds()
{
    if (has_logical_bbox) {
        return logical_bbox;
    } else {
#if 0  // This gets the bounds of all the GL renderer's vertices
        extern GLRendererPtr gl_renderer;

        assert(gl_renderer);
        RendererPtr renderer = dynamic_pointer_cast<Renderer>(gl_renderer);
        assert(renderer);
        PathRendererStatePtr renderer_state = getRendererState(renderer);
        GLPathRendererStatePtr path_renderer_state = dynamic_pointer_cast<GLPathRendererState>(renderer_state);
        assert(path_renderer_state);
        if (verbose) {
            cout << "old: " << path_renderer_state->getBounds() << endl;
        }
        return path_renderer_state->getBounds();
#endif
        return getDilatedFillBounds();
    }
}

void Path::setLogicalBounds(const float4 &bbox)
{
    logical_bbox = bbox;
    has_logical_bbox = true;
}

void Path::unsetLogicalBounds()
{
    has_logical_bbox = false;
}

bool Path::isEmpty()
{
#if 1
    // Quick way to know a path is empty.  Path may still have stroke or fill area.
    return cmd.size() <= 0;
#else  // below is a more sophisticated check
    // XXX hack fixme
    {
        extern GLRendererPtr gl_renderer;

        assert(gl_renderer);
        RendererPtr renderer = dynamic_pointer_cast<Renderer>(gl_renderer);
        assert(renderer);
        PathRendererStatePtr renderer_state = getRendererState(renderer);
        GLPathRendererStatePtr path_renderer_state = dynamic_pointer_cast<GLPathRendererState>(renderer_state);
        assert(path_renderer_state);
        path_renderer_state->validate();
        bool is_empty = path_renderer_state->fill_vertex_array_buffer.isEmpty() &&
                        path_renderer_state->stroke_vertex_array_buffer.isEmpty();
        return is_empty;
    }
    assert(!"implement me");
#if 0
    validate();
    bool is_empty = fill_vertex_array_buffer.isEmpty() &&
                    stroke_vertex_array_buffer.isEmpty();
    return is_empty;
#else
    return true;
#endif
#endif
}

static inline bool isZero(double v)
{
#if 0
    const double eps = 6e-008;

    if (fabs(v) < eps) {
        return true;
    } else {
        return false;
    }
#else
    return v == 0.0;
#endif
}
