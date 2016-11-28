
/* scene.hpp - path rendering scene graph. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __scene_hpp__
#define __scene_hpp__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <vector>
#include <stack>
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/lexical_cast.hpp>

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sin and cos */
#include <GL/glew.h>
#if __APPLE__
#include <OpenGL/glext.h>
#endif

#include <Cg/vector/xyzw.hpp>
#include <Cg/vector/rgba.hpp>
#include <Cg/double.hpp>
#include <Cg/vector.hpp>
#include <Cg/matrix.hpp>
#include <Cg/dot.hpp>
#include <Cg/any.hpp>
#include <Cg/all.hpp>
#include <Cg/mul.hpp>
#include <Cg/round.hpp>
#include <Cg/cross.hpp>
#include <Cg/min.hpp>
#include <Cg/max.hpp>
#include <Cg/distance.hpp>
#include <Cg/saturate.hpp>
#include <Cg/inverse.hpp>
#include <Cg/iostream.hpp>

#include "renderer.hpp"

#include "ActiveControlPoint.hpp"
#include "path.hpp"
#include "path_process.hpp"
#include "glmatrix.hpp"

// Grumble, Microsoft (and probably others) define these as macros
#undef min
#undef max

using namespace boost;
using std::string;
using std::vector;
using std::stack;
using namespace Cg;


class Traversal;
typedef shared_ptr<class Visitor> VisitorPtr;

class RectBounds : public float4
{
public:
    RectBounds();
    RectBounds(const float4 &bounds);
    RectBounds(const RectBounds &bounds);
    RectBounds(float2 top_left, float2 bottom_right);
    RectBounds(float left, float top, float right, float bottom);

    inline float width() { return z-x; }
    inline float height() { return w-y; }
    inline bool isValid() { return valid; }
    inline void invalidate() { valid = false; }
    inline RectBounds dilate(float amount) { return dilate(amount, amount); }

    RectBounds& operator =(const float4 &bounds);
    RectBounds& operator =(const RectBounds &bounds);
    RectBounds operator |(const float4 &bounds);
    RectBounds operator |(const RectBounds &bounds);
    RectBounds& operator |=(const float4 &bounds);
    RectBounds& operator |=(const RectBounds &bounds);
    RectBounds operator & (const float4 &bounds);
    RectBounds operator & (const RectBounds &bounds);
    RectBounds& operator &=(const float4 &bounds);
    RectBounds& operator &=(const RectBounds &bounds);
    RectBounds include(float2 point);
    RectBounds transform(const float4x4 &matrix);
    RectBounds transform(const float3x3 &matrix);
    RectBounds dilate(float x_, float y_);

protected:
    bool valid;
};

struct Node {
    virtual ~Node() {};

    virtual void dumpSVGHelper(FILE *file, const float4x4 &transform) { };
    void dumpSVG(FILE *file, const float4x4 &transform);

    virtual float4 getBounds() { return float4(0,0,-1,-1); } // bogus bounds (x>z and y>w);

    // Force every subclass to implement a traverse() method
    virtual void traverse(VisitorPtr visitor, Traversal &traversal) = 0;
    
    // Make it easier to traverse by automatically using a generic traversal
    void traverse(VisitorPtr visitor);
};

struct Shape;
typedef shared_ptr<struct RendererState<Shape> > ShapeRendererStatePtr;

typedef shared_ptr<struct Text> TextPtr;
struct Text : Node, HasRendererState<Text>, enable_shared_from_this<Text> {
protected:
public:
	std::string font_family;
	std::string text;
	double font_size;
	double3x3 transform;

    Text(const char* font_, const char* text_, double font_size_,
		double3x3 matrix_)
        : HasRendererState<Text>(this),
		font_family(font_),
		text(text_),
		font_size(font_size_),
		transform(matrix_)
    {}
    virtual ~Text() {}

    void invalidate() {
        invalidateRenderStates();
    }

	void traverse(VisitorPtr visitor, Traversal &traversal);
};

enum OpacityTreatment {
    FULLY_OPAQUE,
    VARIABLE_PREMULTIPLIED_ALPHA,
    VARIABLE_INDEPENDENT_ALPHA,
};

// Gradient-related enumerations
enum SpreadMethod {
    PAD,      // clamp to edge
    REFLECT,  // mirror
    REPEAT,   // repeat
    NONE      // clamp to border with (0,0,0,0) border
};
enum GradientUnits {
    USER_SPACE_ON_USE,
    OBJECT_BOUNDING_BOX
};
// http://www.w3.org/TR/SVG/painting.html#ColorInterpolationProperty
enum ColorSpace {
    CORRECTED_SRGB,     // Indicates that sRGB color values should be converted into linearized RGB color space 
    UNCORRECTED         // Indicates that color interpolation should be uncorrected
};

struct Paint : HasRendererState<Paint> {
    virtual bool isNonOpaque() = 0;
    virtual bool isOpaque() = 0;
    virtual bool isTransparent() = 0;
    virtual string toSVG(string fillOrStroke) = 0;
    virtual ~Paint() {}

    Paint()
        : HasRendererState<Paint>(this)
    { }
};
typedef shared_ptr<Paint> PaintPtr;
typedef shared_ptr<RendererState<Paint> > PaintRendererStatePtr;

string glColorToSVGColor(const string fillOrStroke, const float4 &color);

struct SolidColorPaint : Paint {
protected:
    float4 color;

public:
    SolidColorPaint(float4 c) 
        : color(c)
    {}

    bool isNonOpaque() {
        return color.a != 1;
    }
    bool isOpaque() {\
        return color.a == 1;
    }
    bool isTransparent() {
        return color.a != 1;
    }
    string toSVG(string fillOrStroke) {
        return glColorToSVGColor(fillOrStroke, color);
    }

    inline float4 getColor() const { return color; }
};
typedef shared_ptr<SolidColorPaint> SolidColorPaintPtr;

struct GradientStop {
    float offset;
    float4 color;  // RGB is color, alpha is opacity

    GradientStop()
        : color(0,0,0,1)
    { }  // opaque black is default

    GradientStop operator = (const GradientStop &src) {
        if (&src != this) {
            offset = src.offset;
            color = src.color;
        }
        return *this;
    }
};

struct GradientStops {
    vector<GradientStop> stop_array;
};
typedef shared_ptr<GradientStops> GradientStopsPtr;

struct GradientPaint : Paint {
protected:
    // Generic gradient attributes
    GradientUnits gradient_units;
    float3x3 gradient_transform,
             inverse_gradient_transform;  // could be float4x4
    SpreadMethod spread_method;
    GradientStopsPtr gradient_stops;

public:
    GradientPaint()
        : gradient_units(OBJECT_BOUNDING_BOX)
        , gradient_transform(1,0,0, 0,1,0, 0,0,1)  // 3x3 identity
        , inverse_gradient_transform(1,0,0, 0,1,0, 0,0,1)  // 3x3 identity
        , spread_method(PAD)
    {}

    inline const vector<GradientStop> &getStopArray() const { return gradient_stops->stop_array; }
    inline SpreadMethod getSpreadMethod() const { return spread_method; }
    inline const float3x3 &getGradientTransform() const { return gradient_transform; }
    inline const float3x3 &getInverseGradientTransform() const { return inverse_gradient_transform; }
    inline GradientUnits getGradientUnits() const { return gradient_units; }

    inline void setGradientStops(GradientStopsPtr stops) {
        gradient_stops = stops;
    }
    inline void setGradientTransform(const float3x3 &matrix) {
        gradient_transform = matrix;
        // OpenGL wants to map path-space positions to texture
        // coordinates so needs the reverse transform so take the inverse.
        inverse_gradient_transform = inverse(matrix);
    }
    inline void setSpreadMethod(SpreadMethod v) {
        spread_method = v;
    }
    inline void setGradientUnits(GradientUnits v) {
      gradient_units = v;
    }
};
typedef shared_ptr<GradientPaint> GradientPaintPtr;

struct LinearGradientPaint : GradientPaint {
protected:
    float2 v1, v2;

public:
    LinearGradientPaint(float2 v1_, float2 v2_) 
        : v1(v1_)
        , v2(v2_)
    {}

    OpacityTreatment startShading();
    void stopShading();
    bool isNonOpaque() {
        return false;
    }
    bool isOpaque() {
        return false;
    }
    bool isTransparent() {
        return false;
    }
    string toSVG(string fillOrStroke) {
        printf("LinearGradientPaint lacks proper toSVG, outputting red\n");
        return glColorToSVGColor(fillOrStroke, float4(1,0,0,1));
    }

    inline float2 getV1() const { return v1; }
    inline float2 getV2() const { return v2; }
};
typedef shared_ptr<LinearGradientPaint> LinearGradientPaintPtr;

struct RadialGradientPaint : GradientPaint {
protected:
    float2 center, focal_point;
    float radius;

public:
    RadialGradientPaint(float2 c, float2 f, float r) 
        : center(c)
        , focal_point(f)
        , radius(r)
    {}

    OpacityTreatment startShading();
    void stopShading();
    bool isNonOpaque() {
        return false;
    }
    bool isOpaque() {
        return false;
    }
    bool isTransparent() {
        return false;
    }
    string toSVG(string fillOrStroke) {
        printf("RadialGradientPaint lacks proper toSVG, outputting cyan\n");
        return glColorToSVGColor(fillOrStroke, float4(0,1,1,1));
    }

    inline float2 getCenter() const { return center; }
    inline float2 getFocalPoint() const { return focal_point; }
    inline float getRadius() const { return radius; }
};
typedef shared_ptr<RadialGradientPaint> RadialGradientPaintPtr;

struct RasterImage {
#pragma pack(1)
    struct Pixel { 
        GLubyte r, g, b, a; 
    };
#pragma pack()

    RasterImage(Pixel *pixels_, int width_, int height_) 
        : pixels(pixels_), width(width_), height(height_) { }
    RasterImage()
        : pixels(NULL)
        , width(0)
        , height(0)
    { }
    ~RasterImage() { if (pixels) free(pixels); }

    Pixel *pixels;
    int width, height;
};
typedef shared_ptr<RasterImage> RasterImagePtr;

struct ImagePaint : Paint {
    RasterImagePtr image;
    bool opaque;

    ImagePaint(RasterImagePtr image_);

    bool isNonOpaque() {
        return !opaque;
    }
    bool isOpaque() {
        return opaque;
    }
    bool isTransparent() {
        return !opaque;
    }
    string toSVG(string fillOrStroke) {
        printf("LinearGradientPaint lacks proper toSVG, outputting green\n");
        return glColorToSVGColor(fillOrStroke, float4(0,1,0,1));
    }
};
typedef shared_ptr<ImagePaint> ImagePaintPtr;

typedef shared_ptr<struct Shape> ShapePtr;
struct Shape : Node, HasRendererState<Shape>, enable_shared_from_this<Shape> {
protected:
    PathPtr path;
    PaintPtr fill_paint, stroke_paint;
public:
	LdpPolyGroupPtr ldpPoly;
public:
    float net_fill_opacity, net_stroke_opacity;

    Shape()
        : HasRendererState<Shape>(this)
    {}
    Shape(PathPtr p)
        : HasRendererState<Shape>(this)
        , path(p)
        , net_fill_opacity(1)
        , net_stroke_opacity(1)
    { }
    Shape(PathPtr p, PaintPtr f, PaintPtr s)
        : HasRendererState<Shape>(this)
        , path(p)
        , fill_paint(f)
        , stroke_paint(s)
        , net_fill_opacity(1)
        , net_stroke_opacity(1)
    { }

    virtual ~Shape() {}

    int countSegments();

    inline PathPtr getPath() const { return path; }
    inline PaintPtr getFillPaint() const { return fill_paint; }
    inline PaintPtr getStrokePaint() const { return stroke_paint; }

    void dumpSVGHelper(FILE *file, const float4x4 &transform);

    void drawControlPoints();
    void drawReferencePoints();

    float4 getBounds() {
        return path ? path->getBounds() : Node::getBounds();
    }

    using Node::traverse; // Lame ... Node::traverse(VisitorPtr visitor) gets "shadowed"
    void traverse(VisitorPtr visitor, Traversal &traversal);

    bool isEmpty() { 
        return path->isEmpty();
    }
    bool isFillable() { 
        return path->isFillable();
    }
    bool isStrokable() { 
        return path->isStrokable();
    }
    void invalidate() {
        invalidateRenderStates();
        path->invalidate();
    }

    bool isOpaqueFill() {
        if (fill_paint) {
            return fill_paint->isOpaque();
        } else {
            return true;
        }
    }
    bool isOpaqueStroke() {
        if (stroke_paint) {
            return stroke_paint->isOpaque();
        } else {
            return true;
        }
    }
    bool isOpaque() {
        return isOpaqueFill() && isOpaqueStroke();
    }
    bool isNonOpaque() {
        return !isOpaque();
    }

    void processSegments(PathSegmentProcessor &processor);
};

struct Transform : Node, enable_shared_from_this<Transform> {
public:
    NodePtr node;
protected:
    float4x4 matrix;
    float4x4 inverse_matrix;

public:
    // Identity transform
    Transform(NodePtr node_);

    // Projective 3D transform (4x4 matrix)
    Transform(NodePtr node_, const float4x4 &matrix_);

    // Projective 2D transform (3x3 matrix)
    Transform(NodePtr node_, const double3x3 &matrix_);

    void dumpSVGHelper(FILE *file, const float4x4 &transform);
    void setMatrix(const float4x4 &transform);
    inline float4x4 getMatrix() const { return matrix; }
    inline float4x4 getInverseMatrix() const { return inverse_matrix; }

    float4 getBounds();

    using Node::traverse; // Lame ... Node::traverse(VisitorPtr visitor) gets "shadowed"
    void traverse(VisitorPtr visitor, Traversal &traversal);
};
typedef shared_ptr<Transform> TransformPtr;

struct WarpTransform : Transform {
protected:
    float2 to[4];
    float2 from[4];  // Points for a quadrilateral that should map to a [-1..+1,-1..+1] square.
    float2 scale;    // Extra scaling term (typically 90%).

    void updateMatrix();

public:
    // Identity transform
    WarpTransform(NodePtr node_, const float2 to_[4], const float2 from_[4], const float2 &scale);

    void drawWarpPoints();

    void setWarpPoint(int ndx, const float2 &xy);

    void findNearerControlPoint(ActiveWarpPoint &hit);

    inline float4x4 scaledTransform(const float4x4 &m) const {
        return mul(m, scale4x4(scale));
    }
};
typedef shared_ptr<WarpTransform> WarpTransformPtr;

enum ClipMerge {
    SUM_WINDING_NUMBERS,
    SUM_WINDING_NUMBERS_MOD_2,
    CLIP_COVERAGE_UNION
};
enum ClipRule {
    NON_ZERO,
    EVEN_ODD
};

typedef shared_ptr<struct Clip> ClipPtr;
struct Clip : Node, enable_shared_from_this<Clip> {
    NodePtr path;
    NodePtr node;
    ClipMerge clip_merge;

    Clip(NodePtr path_, NodePtr node_, ClipMerge clip_merge_)
        : path(path_)
        , node(node_)
        , clip_merge(clip_merge_) {
    }

    float4 getBounds() { return node->getBounds(); }
    RectBounds getClipBounds();
    using Node::traverse; // Lame ... Node::traverse(VisitorPtr visitor) gets "shadowed"
    void traverse(VisitorPtr visitor, Traversal &traversal);
};

struct ViewBox : Clip {
    ViewBox(NodePtr node_, RectBounds &viewbox);
};
typedef shared_ptr<ViewBox> ViewBoxPtr;

struct Group : public Node, enable_shared_from_this<Group> {
    vector<NodePtr> list;

    void push_back(NodePtr node) { list.push_back(node); }

    void dumpSVGHelper(FILE *file, const float4x4 &transform);

    float4 getBounds();

    using Node::traverse; // Lame ... Node::traverse(VisitorPtr visitor) gets "shadowed"
    void traverse(VisitorPtr visitor, Traversal &traversal);

	std::string ldp_layer_name;
};
typedef shared_ptr<Group> GroupPtr;

struct SvgScene : public Group {
    int width, height;
    RectBounds view_box;
    string preserve_aspect_ratio;
	double ldp_pixel2meter;
};
typedef shared_ptr<SvgScene> SvgScenePtr;


///////////////////////////////////////////////////////////////////////////////
// Visitor/Traversal Interfaces

// This class lets us 
//   1. Customize how the different types of nodes are treated during traversal
//   2. Confidently maintain state throughout the process
class Visitor 
{
public:
    virtual ~Visitor() { } // Make the destructor virtual

    virtual void visit(ShapePtr shape) = 0;

	virtual void visit(TextPtr text) {}

    virtual void apply(TransformPtr transform) { }
    virtual void unapply(TransformPtr transform) { }

    virtual void apply(ClipPtr clip) { }
    virtual void unapply(ClipPtr clip) { }

    virtual void apply(GroupPtr group) { }
    virtual void unapply(GroupPtr group) { }
};

// This class lets us customize
//   1. The order a node gets traversed in
//   2. What nodes get traversed
class Traversal 
{
public:
    virtual ~Traversal() { } // Make the destructor virtual

    virtual void traverse(ShapePtr shape, VisitorPtr visitor) = 0;
    virtual void traverse(TransformPtr transform, VisitorPtr visitor) = 0;
    virtual void traverse(ClipPtr clip, VisitorPtr visitor) = 0;
	virtual void traverse(GroupPtr group, VisitorPtr visitor) = 0;
	virtual void traverse(TextPtr group, VisitorPtr visitor) = 0;
};

///////////////////////////////////////////////////////////////////////////////
// Traversal implementations
class GenericTraversal : public Traversal
{
public:
    virtual void traverse(ShapePtr shape, VisitorPtr visitor);
    virtual void traverse(TransformPtr transform, VisitorPtr visitor);
    virtual void traverse(ClipPtr clip, VisitorPtr visitor);
	virtual void traverse(GroupPtr group, VisitorPtr visitor);
	virtual void traverse(TextPtr group, VisitorPtr visitor);
};

class ReverseTraversal : public GenericTraversal 
{
public:
    void traverse(GroupPtr group, VisitorPtr visitor);
};

class ForEachShapeTraversal : public GenericTraversal
{
public:
    void traverse(ClipPtr clip, VisitorPtr visitor);
};

class ForEachTransformTraversal : public GenericTraversal
{
public:
    void traverse(ShapePtr shape, VisitorPtr visitor) { }  // ignore shapes
};

///////////////////////////////////////////////////////////////////////////////
// Visitor implementations
class MatrixSaveVisitor : public Visitor 
{
protected:
    stack<float4x4> matrix_stack;

public:
    MatrixSaveVisitor();
    virtual ~MatrixSaveVisitor();
    void apply(TransformPtr transform);
    void unapply(TransformPtr transform);
};

// OK to just inherit from MatrixSaveVisitor because the ClipPathSaveVisitor
// requires a matrix anyway
class ClipSaveVisitor : public MatrixSaveVisitor 
{
protected:
    struct ClipAndMatrix {
        ClipAndMatrix(ClipPtr c, float4x4 m)
            : clip(c)
            , matrix(m) { }
        ClipPtr clip;
        float4x4 matrix;
    };
    vector<ClipAndMatrix> clip_stack;
    	
public:
    virtual ~ClipSaveVisitor();
    void apply(ClipPtr clip);
    void unapply(ClipPtr clip);
};


///////////////////////////////////////////////////////////////////////////////
// Other visitors 
class Invalidate : public Visitor 
{
public:
    void visit(ShapePtr shape);
};

class FlushRendererStates : public Visitor 
{
public:
    void visit(ShapePtr shape);
};

class FlushRendererState : public Visitor
{
public:
    FlushRendererState(RendererPtr renderer_) : renderer(renderer_) { }
    void visit(ShapePtr shape);

protected:
    RendererPtr renderer;
};

class InvalidateRendererStates : public Visitor 
{
public:
    void visit(ShapePtr shape);
};

class InvalidateRendererState : public Visitor 
{
public:
    InvalidateRendererState(RendererPtr renderer_) : renderer(renderer_) { }
void visit(ShapePtr shape);

protected:
    RendererPtr renderer;
};

class CountVisitor : public Visitor {
public:
    CountVisitor()
        : count(0)
    { }
    inline int getCount() { return count; }

protected:
    int count;
};

class CountSegments : public CountVisitor  {
public:
    void visit(ShapePtr shape);
};
typedef shared_ptr<CountSegments> CountSegmentsPtr;

class CountNonOpaqueObjects : public CountVisitor  {
public:
    void visit(ShapePtr shape);
};
typedef shared_ptr<CountNonOpaqueObjects> CountNonOpaqueObjectsPtr;

extern float2 clipToSurfaceScales(float w, float h, float scene_ratio);
extern float4x4 surfaceToClip(float w, float h, float scene_ratio);

#endif // __scene_hpp__
