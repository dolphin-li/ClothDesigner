
/* scene.c - scene management for nvpr_svg. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
   2.0 or higher). */

#include "scene.hpp"
#include "glmatrix.hpp"

// Grumble, Microsoft (and probably others) define these as macros
#undef min
#undef max

using namespace Cg;
using namespace boost;
using std::string;
using std::vector;

#ifdef NDEBUG
const static int verbose = 1;
#else
const static int verbose = 1;
#endif

// XXX stuff that still comes from nvpr_svg.cpp
extern int do_unsafe_stencil;

static const char svg_head[] = 
"<?xml version=\"1.0\" standalone=\"no\"?>\n"
"<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \n"
"  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
"<svg width=\"%dpx\" height=\"%dpx\" viewBox=\"0 0 %d %d\"\n"
"     xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n"
"  <title>A path from nvpr_svg</title>\n"
"  <desc>A path from nvpr_svg</desc>\n"
"  <rect x=\"1\" y=\"1\" width=\"%d\" height=\"%d\"\n"
"        fill=\"none\" stroke=\"blue\" />\n";

static const char svg_tail[] =
"</svg>\n";


///////////////////////////////////////////////////////////////////////////////
// ImagePaint
#include "stb/stb_image.h"
ImagePaint::ImagePaint(RasterImagePtr image_) : image(image_)
{
    GLubyte alpha_reduce = 0xff;
    for (int i = 0; i < image->width*image->height; i++) {
        alpha_reduce &= image->pixels[i].a;
    }
    opaque = alpha_reduce == 0xff;
}

///////////////////////////////////////////////////////////////////////////////
// Node
void Node::dumpSVG(FILE *file, const float4x4 &clip_to_svg_coords)
{
    // Figure out the window size from the clip_to_svg_coords transform.
    // Transform both the clip space upper-left corner (-1,1) and lower-left corner (1,-1) to window space.
    float4 upper_left = mul(clip_to_svg_coords, float4(-1,1,0,1)),
           lower_right = mul(clip_to_svg_coords, float4(1,-1,0,1));
    // Difference these two (projected) locations to get the width and height.
    float2 wh = round(lower_right.xy/lower_right.w -
                      upper_left.xy/upper_left.w);
    int w = int(wh.x),
        h = int(wh.y);
    fprintf(file, svg_head, w, h, w+2, h+2, w+1, h+1);
    dumpSVGHelper(file, clip_to_svg_coords);
    fprintf(file, "%s", svg_tail);
}

void Node::traverse(VisitorPtr visitor)
{
    GenericTraversal traversal;
    traverse(visitor, traversal);
}

///////////////////////////////////////////////////////////////////////////////
// text
void Text::traverse(VisitorPtr visitor, Traversal &traversal)
{
	traversal.traverse(shared_from_this(), visitor);
}


///////////////////////////////////////////////////////////////////////////////
// Shape
int Shape::countSegments()
{
    return path->countSegments();
}

void Shape::drawControlPoints()
{
    path->drawControlPoints();
}

void Shape::drawReferencePoints()
{
    path->drawReferencePoints();
}

static string floatToString(float v)
{
    string s = lexical_cast<string>(v);
    return s;
}

typedef unsigned char uchar;

string glColorToSVGColor(const string fillOrStroke, const float4 &color)
{
    static const char *hexdigits = "0123456789ABCDEF";
    string result(" " + fillOrStroke + "=\"#");
    float4 saturated_color = saturate(color);

    for (int i=0; i<3; i++) {
        uchar c(uchar(round(255 * saturated_color[i])));
        result.push_back(hexdigits[c>>4]);
        result.push_back(hexdigits[c&0xF]);
    }
    if (color.a != 1.0) {
        result += "\" " + fillOrStroke + "-opacity=\"" + floatToString(float1(color.a)) + "\"";
    } else {
        result += "\"";
    }
    return result;
}

string strokeProperties(const PathStyle &style)
{
    string result;

    result += " stroke-linecap=\"";
    switch (style.line_cap) {
    case PathStyle::BUTT_CAP:
        result += "butt";
        break;
    case PathStyle::ROUND_CAP:
        result += "round";
        break;
    case PathStyle::SQUARE_CAP:
        result += "square";
        break;
    case PathStyle::TRIANGLE_CAP:
        result += "triangle";
        break;
    }
    result += "\" stroke-linejoin=\"";
    switch (style.line_join) {
    case PathStyle::NONE_JOIN:
        result += "none";
        break;
    case PathStyle::MITER_REVERT_JOIN:
        result += "miter";
        break;
    case PathStyle::ROUND_JOIN:
        result += "round";
        break;
    case PathStyle::BEVEL_JOIN:
        result += "bevel";
        break;
    case PathStyle::MITER_TRUNCATE_JOIN:
        result += "miter-truncate";
        break;
    }
    result += "\" stroke-dashoffset=\"";
    result += lexical_cast<string,double>(style.dash_offset);
    result += "\" stroke-width=\"";
    result += lexical_cast<string,double>(style.stroke_width);
    result += "\" stroke-dasharray=\"";
    if (style.dash_array.size() > 0) {
        result += lexical_cast<string,double>(style.dash_array[0]);
        for (size_t i=1; i<style.dash_array.size(); i++) {
            result += "," + lexical_cast<string,double>(style.dash_array[i]);
        }
    } else {
        result += "none";
    }
    result += "\"";
    if (style.dash_phase == PathStyle::MOVETO_CONTINUES) {
        result += " stroke-dashphase=\"movetoContinues\"";
    }
    return result;
}

void Shape::dumpSVGHelper(FILE *file, const float4x4 &transform)
{
    string svg_path = path->convert_to_svg_path(transform);
    string svg_fill_color = " fill=\"none\"";
    if (path->style.do_fill && fill_paint) {
        svg_fill_color = fill_paint->toSVG("fill");
    }
    string svg_stroke_color = " stroke=\"none\"";
    if (path->style.do_stroke && stroke_paint) {
        svg_stroke_color = stroke_paint->toSVG("stroke");
    }
    fprintf(file, "  <path style=\"fill-rule:%s\"%s%s%s d=\"%s\" transform=\"matrix(%f,%f,%f,%f,%f,%f)\" />\n",
        path->style.fill_rule==PathStyle::NON_ZERO ? "nonzero" : "evenodd",
        svg_fill_color.c_str(),
        svg_stroke_color.c_str(),
        strokeProperties(path->style).c_str(),
        svg_path.c_str(),
        transform[0][0], transform[1][0],
        transform[0][1], transform[1][1],
        transform[0][3], transform[1][3]);
}

void Shape::processSegments(PathSegmentProcessor &processor)
{
    path->processSegments(processor);
}

void Shape::traverse(VisitorPtr visitor, Traversal &traversal)
{
    traversal.traverse(shared_from_this(), visitor);
}


///////////////////////////////////////////////////////////////////////////////
// Transform
const GLfloat *glptr(float4 &v)
{
    return reinterpret_cast<const GLfloat*>(&v);
}

// Identity transform
Transform::Transform(NodePtr node_)
    : node(node_)
    , matrix(float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)) {

    // inverse of identity is itself
    inverse_matrix = matrix;
}

// Projective 3D transform (4x4 matrix)
Transform::Transform(NodePtr node_, const float4x4 &matrix_)
    : node(node_)
    , matrix(matrix_) {

    inverse_matrix = inverse(matrix);
}

// Projective 2D transform (3x3 matrix)
Transform::Transform(NodePtr node_, const double3x3 &matrix_)
    : node(node_)
    , matrix(float4x4(float4(matrix_[0].xy,0,matrix_[0].z),
                      float4(matrix_[1].xy,0,matrix_[1].z),
                      float4(0,0,1,0),
                      float4(matrix_[2].xy,0,matrix_[2].z))) {

    inverse_matrix = inverse(matrix);
}

void Transform::dumpSVGHelper(FILE *file, const float4x4 &transform)
{
    float4x4 updated_transform = mul(transform, matrix);
    node->dumpSVGHelper(file, updated_transform);
}

void Transform::setMatrix(const float4x4 &transform)
{
    matrix = transform;
    inverse_matrix = inverse(transform);
}

float4 Transform::getBounds()
{
    float4 bounds = node->getBounds();

    if (bounds.x <= bounds.z && bounds.y <= bounds.w)
    {
        // Transform the 4 extreme points of the points box by the matrix
        float4 v[4] = { mul(matrix, float4(bounds.xy, 0, 1)),
                        mul(matrix, float4(bounds.zw, 0, 1)),
                        mul(matrix, float4(bounds.xw, 0, 1)),
                        mul(matrix, float4(bounds.zy, 0, 1)) };

        // Now find the new bounding box of the (perspective-divided) transformed points.
        v[0].xy /= v[0].w;
        bounds.xy = v[0].xy;
        bounds.zw = v[0].xy;
        for (int i=1; i<4; i++) {
            v[i].xy /= v[i].w;
            bounds.xy = Cg::min(v[i].xy, bounds.xy);
            bounds.zw = Cg::max(v[i].xy, bounds.zw);
        }
    }

    return bounds;
}

void Transform::traverse(VisitorPtr visitor, Traversal &traversal)
{
    traversal.traverse(shared_from_this(), visitor);
}

///////////////////////////////////////////////////////////////////////////////
// WarpTransform

// The WarpTransform computes its transform matrix from a set of "from" points
// that represent the corners of a bounding box for the node heirarchy under
// the transform node.
//
// Call updateMatrix method if from points or scale changes.
void WarpTransform::updateMatrix()
{
    double3x3 xmap = quad2quad(from, to);
    double4x4 ortho = make_float4x4(xmap);
    setMatrix(mul(scale4x4(scale),ortho));
}

// Identity transform
WarpTransform::WarpTransform(NodePtr node_, const float2 to_[4], const float2 from_[4], const float2 &scale_)
    : Transform(node_)
{
    scale = scale_;
    for (int i=0; i<4; i++) {
        to[i] = to_[i];
        from[i] = from_[i];
    }
    updateMatrix();
}

void WarpTransform::findNearerControlPoint(ActiveWarpPoint &hit)
{
    bool hit_transform = false;
    for (int i=0; i<4; i++) {
        hit.testForControlPointHit(to[i], i, hit_transform,
            static_pointer_cast<WarpTransform>(shared_from_this()));
    }
}

void WarpTransform::drawWarpPoints() {
    glEnable(GL_LINE_STIPPLE);
    glColor3f(1,1,0);
    glBegin(GL_LINE_LOOP); {
        for (int i=0; i<4; i++) {
            glVertex2f(from[i].x, from[i].y);
        }
    } glEnd();
    glDisable(GL_LINE_STIPPLE);
    glColor3f(1,0,1);
    glPointSize(7);
    glBegin(GL_POINTS); {
        for (int i=0; i<4; i++) {
            glVertex2f(from[i].x, from[i].y);
        }
    } glEnd();
}

void WarpTransform::setWarpPoint(int ndx, const float2 &xy) {
    assert(ndx >= 0);
    assert(ndx < 4);
    //float2 old = from[ndx];
    to[ndx] = xy;
    updateMatrix();
    //from[ndx] = old;
}

///////////////////////////////////////////////////////////////////////////////
// Group
void Group::dumpSVGHelper(FILE *file, const float4x4 &transform)
{
    for (size_t i=0; i<list.size(); i++) {
        list[i]->dumpSVGHelper(file, transform);
    }
}

float4 Group::getBounds()
{
    float4 group_bounds = Node::getBounds();

    for (size_t i=0; i<list.size(); i++) {
        float4 node_bounds = list[i]->getBounds();
        if (node_bounds.x <= node_bounds.z &&
            node_bounds.y <= node_bounds.w) {
            if (group_bounds.x <= group_bounds.z &&
                group_bounds.y <= group_bounds.w) {
                group_bounds.xy = Cg::min(group_bounds.xy, node_bounds.xy);
                group_bounds.zw = Cg::max(group_bounds.zw, node_bounds.zw);
            } else {
                group_bounds = node_bounds;
            }
        }
    }
    return group_bounds;
}

void Group::traverse(VisitorPtr visitor, Traversal &traversal)
{
    traversal.traverse(shared_from_this(), visitor);
}


///////////////////////////////////////////////////////////////////////////////
// Clip
RectBounds Clip::getClipBounds()
{
    ClipPtr clip_clipper = dynamic_pointer_cast<Clip>(path);
    if (clip_clipper) {
        return clip_clipper->getClipBounds() & clip_clipper->node->getBounds();
    } else {
        return path->getBounds();
    }
}

void Clip::traverse(VisitorPtr visitor, Traversal &traversal)
{
    traversal.traverse(shared_from_this(), visitor);
}

///////////////////////////////////////////////////////////////////////////////
// ViewBox
ViewBox::ViewBox(NodePtr node_, RectBounds &viewbox)
    : Clip(NodePtr(), node_, SUM_WINDING_NUMBERS_MOD_2) // SUM_WINDING_NUMBERS_MOD_2 is the fastest and will work for a rect
{
    vector<char> cmds;
    vector<float> coords;

    cmds.push_back('M');
    coords.push_back(viewbox.x);
    coords.push_back(viewbox.y);
    cmds.push_back('h');
    coords.push_back(viewbox.width());
    cmds.push_back('v');
    coords.push_back(viewbox.height());
    cmds.push_back('h');
    coords.push_back(-viewbox.width());
    cmds.push_back('z');

    PathPtr p(new Path(PathStyle(), cmds, coords));
    path = ShapePtr(new Shape(p));
}

///////////////////////////////////////////////////////////////////////////////
// RectBounds
RectBounds::RectBounds() 
    : valid(false) { }
RectBounds::RectBounds(const float4 &bounds) 
    : float4(bounds), valid(true) { }
RectBounds::RectBounds(const RectBounds &bounds) 
    : float4(bounds), valid(bounds.valid) { }
RectBounds::RectBounds(float2 top_left, float2 bottom_right) 
    : float4(top_left, bottom_right), valid(true) { }
RectBounds::RectBounds(float left, float top, float right, float bottom) 
    : float4(left, top, right, bottom), valid(true) { }

RectBounds& RectBounds::operator =(const float4 &bounds)
{
    xyzw = bounds.xyzw;
    valid = true;
    return *this;
}

RectBounds& RectBounds::operator =(const RectBounds &bounds)
{
    xyzw = bounds.xyzw;
    valid = bounds.valid;
    return *this;
}

RectBounds RectBounds::operator |(const float4 &bounds)
{
    if (valid) {
        return RectBounds(min(xy, bounds.xy), max(zw, bounds.zw));
    } else {
        return RectBounds(bounds);
    }
}

RectBounds RectBounds::operator |(const RectBounds &bounds)
{
    if (bounds.valid) {
        return *this | bounds.xyzw;
    } else {
        return *this;
    }
}

RectBounds& RectBounds::operator |=(const float4 &bounds)
{
    *this = *this | bounds;
    return *this;
}

RectBounds& RectBounds::operator |=(const RectBounds &bounds)
{
    *this = *this | bounds;
    return *this;
}

RectBounds RectBounds::operator &(const float4 &bounds)
{
    if (valid) {
        return RectBounds(max(xy, bounds.xy), min(zw, bounds.zw));
    } else {
        return RectBounds(bounds);
    }
}

RectBounds RectBounds::operator &(const RectBounds &bounds)
{
    if (bounds.valid) {
        return *this & bounds.xyzw;
    } else {
        return *this;
    }
}

RectBounds& RectBounds::operator &=(const float4 &bounds)
{
    *this = *this & bounds;
    return *this;
}

RectBounds& RectBounds::operator &=(const RectBounds &bounds)
{
    *this = *this & bounds;
    return *this;
}

RectBounds RectBounds::include(float2 point)
{
    if (!valid) {
        return RectBounds(point, point);
    } else {
        return RectBounds(min(xy, point), max(zw, point));
    }
}

RectBounds RectBounds::transform(const float4x4 &matrix)
{
    float4 coords[] = {
        mul(matrix, float4(x, y, 0, 1)),
        mul(matrix, float4(x, w, 0, 1)),
        mul(matrix, float4(z, w, 0, 1)),
        mul(matrix, float4(z, y, 0, 1)),
    };

    // Find the bbox of all 4 transformed corners, accounting for perspective
    RectBounds rb;
    for (int i = 0; i < 4; i++) {
        const float z_clip = 1e-5;
        if (coords[i].w > z_clip) {
            rb = rb.include(coords[i].xy / coords[i].w);
        } else {
            // we could clip by the 4 planes of each edge of the view here, but this 
            // is easier and quicker, and if something is crossing this z plane, the resulting 
            // bbox will likely take up either none of the screen or a large portion of it
            for (int j = 1; j < 4; j += 2) { // let j be 1 and 3 (equivalent to 1 and -1 mod 4)
                float4 neighbor = coords[(i+j)%4];
                if (neighbor.w > z_clip) {
                    double delta = (z_clip - coords[i].w) / (neighbor.w - coords[i].w);
                    float4 cross(coords[i] + (neighbor - coords[i]) * delta);
                    rb = rb.include(cross.xy / cross.w);
                }
            }
        }
    }

    return rb;
}

RectBounds RectBounds::transform(const float3x3 &matrix)
{
    return transform(float4x4(matrix[0].x, matrix[0].y, 0, matrix[0].z, 
                              matrix[1].x, matrix[1].y, 0, matrix[1].z,
                              0, 0, 1, 0,
                              matrix[2].x, matrix[2].y, 0, matrix[2].z));
}

RectBounds RectBounds::dilate(float x_, float y_)
{
    return RectBounds(x - x_, y - y_, z + x_, w + y_);
}


///////////////////////////////////////////////////////////////////////////////
// Traverseral implementations
void GenericTraversal::traverse(ShapePtr shape, VisitorPtr visitor)
{
    visitor->visit(shape);
}

void GenericTraversal::traverse(TextPtr shape, VisitorPtr visitor)
{
	visitor->visit(shape);
}

void GenericTraversal::traverse(TransformPtr transform, VisitorPtr visitor)
{
    visitor->apply(transform);
    transform->node->traverse(visitor, *this);
    visitor->unapply(transform);
}

void GenericTraversal::traverse(ClipPtr clip, VisitorPtr visitor)
{
    visitor->apply(clip);
    clip->node->traverse(visitor, *this);
    visitor->unapply(clip);
}

void GenericTraversal::traverse(GroupPtr group, VisitorPtr visitor)
{
    visitor->apply(group);
    for (size_t i = 0; i < group->list.size(); i++)
       group->list[i]->traverse(visitor, *this);
    visitor->unapply(group);
}

void ReverseTraversal::traverse(GroupPtr group, VisitorPtr visitor)
{
    visitor->apply(group);
    for (size_t i = group->list.size(); i; i--)
        group->list[i-1]->traverse(visitor, *this);
    visitor->unapply(group);
}

void ForEachShapeTraversal::traverse(ClipPtr clip, VisitorPtr visitor)
{
    visitor->apply(clip);
    clip->path->traverse(visitor, *this);
    clip->node->traverse(visitor, *this);
    visitor->unapply(clip);
}

///////////////////////////////////////////////////////////////////////////////
// Visitor implementations
MatrixSaveVisitor::MatrixSaveVisitor()
{
    matrix_stack.push(float4x4(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1));
}

MatrixSaveVisitor::~MatrixSaveVisitor()
{
    matrix_stack.pop();
    if (!matrix_stack.empty()) {
        printf("warning : bad scene graph; the matrix stack should be empty at this point\n");
    }
}

void MatrixSaveVisitor::apply(TransformPtr transform)
{
    matrix_stack.push(mul(matrix_stack.top(), transform->getMatrix()));
}

void MatrixSaveVisitor::unapply(TransformPtr transform)
{
    matrix_stack.pop();
}

ClipSaveVisitor::~ClipSaveVisitor()
{
    if (!clip_stack.empty()) {
        printf("warning : bad scene graph; the clip stack should be empty at this point\n");
    }
}

void ClipSaveVisitor::apply(ClipPtr clip)
{
    clip_stack.push_back(ClipAndMatrix(clip, matrix_stack.top()));
}

void ClipSaveVisitor::unapply(ClipPtr clip)
{
    assert(clip_stack.back().clip == clip);
    clip_stack.pop_back();
}

void Invalidate::visit(ShapePtr shape)
{
    shape->invalidate();
}

void FlushRendererStates::visit(ShapePtr shape)
{
    shape->getPath()->flushRenderStates();
    PaintPtr p;
    p = shape->getFillPaint();
    if (p) {
        p->flushRenderStates();
    }
    p = shape->getStrokePaint();
    if (p) {
        p->flushRenderStates();
    }
    shape->flushRenderStates();
}

void FlushRendererState::visit(ShapePtr shape)
{
    shape->getPath()->flushRenderState(renderer);
    PaintPtr p;
    p = shape->getFillPaint();
    if (p) {
        p->flushRenderState(renderer);
    }
    p = shape->getStrokePaint();
    if (p) {
        p->flushRenderState(renderer);
    }
    shape->flushRenderState(renderer);
}

void InvalidateRendererStates::visit(ShapePtr shape)
{
    shape->getPath()->invalidateRenderStates();
    shape->invalidateRenderStates();
}

void InvalidateRendererState::visit(ShapePtr shape)
{
    shape->getPath()->invalidateRenderState(renderer);
    if (shape->getFillPaint()) {
        shape->getFillPaint()->invalidateRenderState(renderer);
    }
    if (shape->getStrokePaint()) {
        shape->getStrokePaint()->invalidateRenderState(renderer);
    }
    shape->invalidateRenderState(renderer);
}

void CountSegments::visit(ShapePtr shape)
{
    count += shape->getPath()->countSegments();
}

void CountNonOpaqueObjects::visit(ShapePtr shape)
{
    count += int(shape->isNonOpaque());
}

// Utility matrix routines

float2 clipToSurfaceScales(float w, float h, float scene_ratio)
{
    float window_ratio = h/w;
    if (scene_ratio < 1.0) {
        if (scene_ratio < window_ratio) {
            if (verbose) {
                printf("0: s=%f, w=%f\n", scene_ratio, window_ratio);
            }
            return float2(1,1.0/window_ratio);
        } else {
            if (verbose) {
                printf("1: s=%f, w=%f\n", scene_ratio, window_ratio);
            }
            return float2(window_ratio/scene_ratio,1.0/scene_ratio);
        }
    } else {
        if (scene_ratio < window_ratio) {
            if (verbose) {
                printf("2: s=%f, w=%f\n", scene_ratio, window_ratio);
            }
            return float2(scene_ratio,scene_ratio/window_ratio);
        } else {
            if (verbose) {
                printf("3: s=%f, w=%f\n", scene_ratio, window_ratio);
            }
            return float2(window_ratio,1);
        }
    }
}

// Generates a 4x4 matrix to map from surface space having the scene
// centered within [-1,+1]^2 to clip space [-1,+1]^3.
float4x4 surfaceToClip(float w, float h, float scene_ratio)
{
    float2 scale = clipToSurfaceScales(w, h, scene_ratio);
    float4x4 surface_to_clip = float4x4(scale.x,0,0,0,
                                        0,scale.y,0,0,
                                        0,0,1,0,
                                        0,0,0,1);
    return surface_to_clip;
}
