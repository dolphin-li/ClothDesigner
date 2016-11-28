
/* svg_loader.hpp - class declaration for SVG loader */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __svg_loader_hpp__
#define __svg_loader_hpp__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include "scene.hpp"
#include <string>
#include <map>

using std::map;

extern float pixels_per_millimeter;

SvgScenePtr svg_loader(const char *xmlFile);

struct Gradient;
typedef shared_ptr<Gradient> GradientPtr;
struct ClipInfo;
typedef shared_ptr<ClipInfo> ClipInfoPtr;
struct Use;
typedef shared_ptr<Use> UsePtr;
typedef map<string,GradientPtr> GradientMap;
typedef map<string,class TiXmlElement*> UseMap;
typedef map<string,ClipInfoPtr> ClipPathMap;

// http://www.w3.org/TR/2002/WD-SVG11-20020215/painting.html#SpecifyingPaint
struct PaintServer {
    virtual void resolve(const GradientMap &gradient_map) {}
    virtual PaintPtr makePaintPtr() = 0;
    virtual ~PaintServer() {}
};
typedef shared_ptr<PaintServer> PaintServerPtr;
typedef map<string,PaintServerPtr> PaintServerMap;

// http://www.w3.org/TR/2002/WD-SVG11-20020215/color.html#SolidColorElement
// The solidColor paint server seems to have gone away in SVG 1.1 2nd edition
struct SolidColor : PaintServer {
    float4 color;

    SolidColor(const float4 &c)
        : color(c)
    {
        color.a = 1;
    }
    SolidColor()
        : color(float4(0,0,0,1))
    {}
    PaintPtr makePaintPtr() {
        return SolidColorPaintPtr(new SolidColorPaint(color));
    }
};
typedef shared_ptr<SolidColor> SolidColorPtr;

// http://www.w3.org/TR/2002/WD-SVG11-20020215/pservers.html#Gradients
struct Gradient : PaintServer {
    // Generic gradient attributes
    enum GradientType {
        LINEAR,
        RADIAL
    } gradient_type;
    GradientUnits gradient_units;
    float3x3 gradient_transform;  // could be float4x4
    SpreadMethod spread_method;
    GradientStopsPtr gradient_stops;
    string href;

    // Linear gradient attributes
    float2 v1, v2;

    // Radial gradient attributes
    float2 c;  // center
    float2 f;  // focal point
    float r;   // radius

    enum UndefinedAttribs {
        // Gradient generic
        GRADIENT_UNITS     = 0x01,
        GRADIENT_TRANSFORM = 0x02,
        SPREAD_METHOD      = 0x04,
        STOP_ARRAY         = 0x08,

        // LinearGradient specific
        X1 = 0x0100,
        Y1 = 0x0200,
        X2 = 0x0400,
        Y2 = 0x0800,

        // RadialGradient specific
        CX = 0x010000,
        CY = 0x020000,
        FX = 0x040000,
        FY = 0x080000,
        R  = 0x100000,

        ALL = ~0
    };
    UndefinedAttribs unspecified_attribs;
    bool resolved;

    Gradient(GradientType type);

    virtual void resolve(const GradientMap &gradient_map);
    void resolveUnspecifiedAttributes(const GradientPtr);

    inline void markAttribSpecified(UndefinedAttribs attr) {
        unspecified_attribs = UndefinedAttribs(int(unspecified_attribs) & ~attr);
    }
    void setX1(float v) {
        v1.x = v;
        markAttribSpecified(X1);
    }
    void setY1(float v) {
        v1.y = v;
        markAttribSpecified(Y1);
    }
    void setX2(float v) {
        v2.x = v;
        markAttribSpecified(X2);
    }
    void setY2(float v) {
        v2.y = v;
        markAttribSpecified(Y2);
    }

    void setCX(float v) {
        c.x = v;
        markAttribSpecified(CX);
    }
    void setCY(float v) {
        c.y = v;
        markAttribSpecified(CY);
    }
    void setFX(float v) {
        f.x = v;
        markAttribSpecified(FX);
    }
    void setFY(float v) {
        f.y = v;
        markAttribSpecified(FY);
    }
    void setR(float v) {
        r = v;
        markAttribSpecified(R);
    }
    void setUnits(GradientUnits v) {
        gradient_units = v;
        markAttribSpecified(GRADIENT_UNITS);
    }
    void setTransform(const double3x3 &m) {
        gradient_transform = m;
        markAttribSpecified(GRADIENT_TRANSFORM);
    }
    void setSpreadMethod(SpreadMethod v) {
        spread_method = v;
        markAttribSpecified(SPREAD_METHOD);
    }
    void setHref(string v) {
        href = v;
    }

    inline const float3x3 &getGradientTransform() { return gradient_transform; }

    void initGenericGradientParameters(GradientPaintPtr gradient);
};

struct LinearGradient;
typedef shared_ptr<LinearGradient> LinearGradientPtr;
struct LinearGradient : Gradient {
    LinearGradient()
        : Gradient(LINEAR)
    {}

    PaintPtr makePaintPtr();
};

struct RadialGradient;
typedef shared_ptr<RadialGradient> RadialGradientPtr;
struct RadialGradient : Gradient {
    RadialGradient()
        : Gradient(RADIAL)
    {}

    PaintPtr makePaintPtr();
};

struct ClipInfo {
    GroupPtr group;
    ClipInfoPtr clip_info;
    double3x3 transform;
    ClipMerge clip_merge;

    enum {
        USER_SPACE_ON_USE,
        OBJECT_BOUNDING_BOX
    } units;

    ClipInfo();

    NodePtr createClip(NodePtr node);
};

#endif // __svg_loader_hpp__
