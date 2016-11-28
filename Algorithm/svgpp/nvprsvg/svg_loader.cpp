
// svg_loader.svg - loader for Scalable Vector Graphics web format

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include "tinyxml\tinyxml.h"

#include <iostream>
#include <sstream>

#include "scene.hpp"

#include <Cg/clamp.hpp>
#include <Cg/max.hpp>
#include <Cg/radians.hpp>
#include <Cg/inverse.hpp>

#include "svg_loader.hpp"
#include "color_names.hpp"

#include <string>
#include <map>
#include <sstream>

using std::map;

#ifdef NDEBUG
const static int verbose = 0;
#else
const static int verbose = 1;
#endif

const unsigned int NUM_INDENTS_PER_SPACE=2;
static const double3x3 identity = double3x3(1,0,0, 0,1,0, 0,0,1);
bool sameMatrix(double3x3 a, double3x3 b)
{
    return all(a[0] == b[0]) && all(a[1] == b[1]) && all(a[2] == b[2]);
}

Gradient::Gradient(GradientType type)
        : gradient_type(type)
        , gradient_units(OBJECT_BOUNDING_BOX)
        , gradient_transform(float3x3(1,0,0, 0,1,0, 0,0,1))
        , spread_method(PAD)
        // Linear gradient attributes
        , v1(float2(0,0))
        , v2(float2(1,0))
        // Radial gradient attributes
        , c(float2(0.5,0.5))  // 50%
        , f(float2(0.5,0.5))
        , r(0.5)
        , unspecified_attribs(ALL)
        , resolved(false)
{
}

void Gradient::initGenericGradientParameters(GradientPaintPtr gradient)
{
    gradient->setGradientStops(gradient_stops);
    gradient->setSpreadMethod(spread_method);
    gradient->setGradientUnits(gradient_units);
    // http://dev.w3.org/SVG/profiles/1.1F2/publish/pservers.html#LinearGradientElementGradientTransformAttribute
    /* Contains the definitions of an optional additional
       transformation from the gradient coordinate system 
       onto the target coordinate system (i.e., 
       userSpaceOnUse or objectBoundingBox). */
    gradient->setGradientTransform(gradient_transform);
}

void Gradient::resolveUnspecifiedAttributes(const GradientPtr reference)
{
    if (reference) {
        assert(reference->resolved);
        // Generic attributes
        if (unspecified_attribs & GRADIENT_UNITS) {
            gradient_units = reference->gradient_units;
        }
        if (unspecified_attribs & GRADIENT_TRANSFORM) {
            gradient_transform = reference->gradient_transform;
        }
        if (unspecified_attribs & SPREAD_METHOD) {
            spread_method = reference->spread_method;
        }
        if (unspecified_attribs & STOP_ARRAY) {
            gradient_stops = reference->gradient_stops;
        }
        // Linear attributes
        if (unspecified_attribs & X1) {
            v1.x = reference->v1.x;
        }
        if (unspecified_attribs & Y1) {
            v1.y = reference->v1.y;
        }
        if (unspecified_attribs & X2) {
            v2.x = reference->v2.x;
        }
        if (unspecified_attribs & Y2) {
            v2.y = reference->v2.y;
        }
        // Radial attributes
        if (unspecified_attribs & CX) {
            c.x = reference->c.x;
        }
        if (unspecified_attribs & CY) {
            c.y = reference->c.y;
        }
        if (unspecified_attribs & R) {
            r = reference->r;
        }
    } else {
        // Constructor provides unspecified initialization. 
    }
    // If attribute fx is not specified, fx will coincide with cx.
    if (unspecified_attribs & FX) {
#if 0 // compiler bug??
        f.x = c.x;
#else
        f[0] = c[1];
#endif
    }
    // If attribute fy is not specified, fy will coincide with cy.
    if (unspecified_attribs & FY) {
#if 0 // compiler bug??
        f.y = c.y;
#else
        f[1] = c[1];
#endif
    }
}

void Gradient::resolve(const GradientMap &gradient_map)
{
    if (resolved) {
        return;
    }
    resolved = true;  // mark resolved before actual resolve to avoid infinite looping

    GradientPtr p;
    GradientMap::const_iterator iter = gradient_map.find(href);
    if (iter != gradient_map.end()) {
        p = iter->second;
        p->resolve(gradient_map);   
    }
    resolveUnspecifiedAttributes(p);
}

PaintPtr LinearGradient::makePaintPtr()
{
    // http://dev.w3.org/SVG/profiles/1.1F2/publish/pservers.html#GradientStops
    // "It is necessary that at least two stops defined to have a gradient effect.
    // If no stops are defined, then painting shall occur as if 'none' were
    // specified as the paint style. If one stop is defined, then paint with the
    // solid color fill using the color defined for that gradient stop."
    if (gradient_stops) {
        size_t stop_array_size = gradient_stops->stop_array.size();
        if (stop_array_size >= 2) {
            LinearGradientPaintPtr paint(new LinearGradientPaint(v1, v2));
            initGenericGradientParameters(paint);
            return paint;
        }
        if (stop_array_size == 1) {
            return SolidColorPaintPtr(new SolidColorPaint(gradient_stops->stop_array[0].color));
        }
    }
    return PaintPtr();
}

PaintPtr RadialGradient::makePaintPtr()
{
    // http://dev.w3.org/SVG/profiles/1.1F2/publish/pservers.html#GradientStops
    // "It is necessary that at least two stops defined to have a gradient effect.
    // If no stops are defined, then painting shall occur as if 'none' were
    // specified as the paint style. If one stop is defined, then paint with the
    // solid color fill using the color defined for that gradient stop."
    if (gradient_stops) {
        size_t stop_array_size = gradient_stops->stop_array.size();
        if (stop_array_size >= 2) {
            RadialGradientPaintPtr paint(new RadialGradientPaint(c, f, r));
            initGenericGradientParameters(paint);
            return paint;
        }
        if (stop_array_size == 1) {
            return SolidColorPaintPtr(new SolidColorPaint(gradient_stops->stop_array[0].color));
        }
    }
    return PaintPtr();
}

ClipInfo::ClipInfo()
        : group(GroupPtr(new Group))
        , transform(identity)
        , clip_merge(CLIP_COVERAGE_UNION)
        , units(USER_SPACE_ON_USE)
{
}

NodePtr ClipInfo::createClip(NodePtr node)
{
    ClipPtr clip;
    

    if (clip_info) {
        clip = ClipPtr(new Clip(clip_info->createClip(group), node, clip_merge));
    } else {
        clip = ClipPtr(new Clip(group, node, clip_merge));
    }

    double3x3 matrix = transform;
    if (units == OBJECT_BOUNDING_BOX) {
        // From the spec on objectBoundingBox
        // First, the (minx,miny) and (maxx,maxy) coordinates are determined for the 
        // applicable element and all of its descendants.
        float4 bounds = clip->getBounds();
        float3x3 normalize(
            float3((bounds.z - bounds.x), 0, bounds.x),
            float3(0, (bounds.w - bounds.y), bounds.y),
            float3(0,0,1));

        // Normalization should come before the clip path's transform
        matrix = mul(matrix, normalize);
    }

    if (!sameMatrix(matrix, identity)) {
        clip->path = TransformPtr(new Transform(clip->path, matrix));
    }

    return clip;
}


static SolidColorPtr blackSolidColor(new SolidColor(float4(0,0,0,1)));

enum TextAnchor {
    START,
    MIDDLE,
    END
};

struct StyleInfo {
    PathStyle path;
    float4 color;  // http://www.w3.org/TR/SVGTiny12/painting.html#ColorProperty
    double3x3 matrix;
    float opacity;  // group opacity
    PaintServerPtr fill;
    PaintServerPtr stroke;
    float fill_opacity, stroke_opacity;
    float4 stop_color;
    ClipInfoPtr clip_path;
    TextAnchor text_anchor;
    string font_family;

    StyleInfo()
        : color(float4(0,0,0,1))
        , matrix(identity)
        , opacity(1)
        , fill(blackSolidColor)
        , stroke(PaintServerPtr())
        , fill_opacity(1)
        , stroke_opacity(1)
        , stop_color(float4(0,0,0,1))
        , clip_path(ClipInfoPtr())
        , text_anchor(START)
        , font_family("Arial")
    {
    }

    StyleInfo & operator = (const StyleInfo &src) {
        if (this != &src) {
            path = src.path;
            matrix = src.matrix;
            color = src.color;
            opacity = src.opacity;
            fill_opacity = src.fill_opacity;
            stroke_opacity = src.stroke_opacity;
            fill = src.fill;
            stroke = src.stroke;
            stop_color = src.stop_color;
            clip_path = src.clip_path;
        }
        return *this;
    }

    StyleInfo inheritFrom(const StyleInfo &parent) {
        StyleInfo child = parent;
        
        // Certain values aren't inherited
        child.matrix = matrix;
        child.stop_color = stop_color;

        return child;
    }
};

struct StyleInfoStack : vector<StyleInfo> { // Can't use stack because we need access to the second-to-top
    StyleInfoStack() {
        // Push default values for a pseudo-parent
        push_back(StyleInfo());
    }

    inline void pushChild() {
        push_back(StyleInfo().inheritFrom(top()));
    }

    template <typename T>
    inline T popAndReturn(T ret) {
        pop();
        return ret;
    }

    inline StyleInfo &top() {
        return (*this)[size()-1];
    }

    inline StyleInfo &secondToTop() {
        return (*this)[size()-2];
    }

    inline void pop() {
        if (size() > 1) {
            erase(end() - 1);
        } else {
            assert(!"You can't pop the initial default values!");
        }
    }
};

struct SVGParser {
    typedef map<string,string> StyleMap;

    StyleInfoStack style_stack;
    StyleMap style_map;
    UseMap use_map;
    GradientMap gradient_map;
    ClipPathMap clipPath_map;
    PaintServerMap paint_server_map;
    TiXmlDocument doc;
    SvgScenePtr scene;
    string root_dir;

    SVGParser(const char *xmlFile);

    inline StyleInfo &style() { return style_stack.top(); }
    inline StyleInfo &parentStyle() { return style_stack.secondToTop(); }

    // Helper routines
    static const char *parseColor(const char *s, float4 &color, const float4 &current_color, const float4 &inherit_color);
    static const char *parseColor(const char *s, float4 &color, const float4 &current_color) { return parseColor(s, color, current_color, color); }
    static const char *parseColor(const char *s, float4 &color) { return parseColor(s, color, color); }
    static const char *parseStrokeLinejoin(const char *s, PathStyle::LineJoin &line_join);
    static const char *parseTextAnchor(const char *s, TextAnchor &text_anchor);
    static const char *parseFontFamily(const char *s, string &font_family);
    static const char *parseStrokeDashArraySpace(const char *s, vector<float> &dash_array);
    static const char *parseStrokeDashArray(const char *s, vector<float> &dash_array);
    static int skipProblem(const char *bad_tag, const char *s);
    static void parseTransform(const char *ss, double3x3 &matrix);
    static const char *parseStrokeLineCap(const char *s, PathStyle::LineCap &line_cap);
    static const char *parseStrokeDashPhase(const char *s, PathStyle::DashPhase &dash_phase);
    static const char *parseFillRule(const char *s, PathStyle::FillRule &fill_rule);
    static const char *parseUnit(const char *s, float &value, bool percentBounds = false);
    static const char *parseCoordinate(const char *s, float &value, bool percentBounds = false);
    static const char *parseLength(const char *s, float &value);
    static const char *parseViewBox(const char *s, float4 &viewbox);
    static const char *parseOpacity(const char *s, float &opacity, float current_opacity, float inherit_opacity);
    static const char *parseOpacity(const char *s, float &opacity, float current_opacity) { return parseOpacity(s, opacity, current_opacity, opacity); }
    static const char *parseOpacity(const char *s, float &opacity) { return parseOpacity(s, opacity, opacity); }
    static const char *parseOffset(const char *s, float &offset);
    
    const char *parsePaintServer(const char *s, const StyleInfo &style, PaintServerPtr &paint_server);
    ClipInfoPtr parseClipPathReference(const char *name);
    const char *parseStyle(const char *ss, StyleInfo &style);

    bool parseGenericShapeProperty(TiXmlAttribute* a, TiXmlElement* elem);

    GradientStop parseGradientStop(TiXmlElement *elem);
    NodePtr parseCircleOrEllipse(TiXmlElement* elem, bool circle_only);
    NodePtr parseCircle(TiXmlElement* elem);
    NodePtr parseEllipse(TiXmlElement* elem);
    NodePtr parsePolygonOrPolyline(TiXmlElement* elem, bool is_polygon);
    NodePtr parsePolygon(TiXmlElement* elem);
    NodePtr parsePolyline(TiXmlElement* elem);
    NodePtr parseLine(TiXmlElement* elem);
    NodePtr parseRect(TiXmlElement* elem);
	NodePtr parsePath(TiXmlElement* elem);
	NodePtr parseLdpPolyGroup(TiXmlElement* elem);
    NodePtr parseGroup(TiXmlElement *elem);
    NodePtr parseText(TiXmlElement* elem);
    NodePtr parseSwitch(TiXmlElement *elem);
    NodePtr parseView(TiXmlElement *elem);
    NodePtr parseImage(TiXmlElement *elem);

    NodePtr createShape(PathPtr path, const StyleInfo &style);

    NodePtr decorateNode(NodePtr node);

    SvgScenePtr parseScene(TiXmlElement *svg);
    
    // Definitions
    void parseSolidColor(TiXmlElement *elem);
    void parseGradient(GradientPtr gradient, TiXmlElement *elem);
    void parseLinearGradient(TiXmlElement *elem);
    void parseRadialGradient(TiXmlElement *elem);
    void parseClipPath(TiXmlElement *elem);
    void parsePattern(TiXmlElement *elem);
    void parseDefs(TiXmlElement *elem);
    NodePtr parseUse(TiXmlElement *elem);
    NodePtr parseNode(TiXmlElement *elem);
    NodePtr parseUsedNode(TiXmlElement *elem, const UsePtr &use);
    void parseSymbol(TiXmlElement *elem);

    void loadStyles(TiXmlNode* pParent);

    TiXmlElement* findSVG(TiXmlNode* pParent, unsigned int indent = 0);
};

struct Use : Node {
protected:
public:
    Use()
        : x(0)
        , y(0)
        , width(1)
        , height(1)
    {}
    virtual ~Use() {}

    string href;
    float x, y, width, height;
    StyleInfoStack style_stack;

    void traverse(VisitorPtr visitor, Traversal &traversal) { /* nop for now */ }
};

bool isShapeTextOrPath(NodePtr node)
{
    if (!node) {
        return false;
    }

    UsePtr use = dynamic_pointer_cast<Use>(node);
    if (use) {
        return true; // The use may not be parsed yet; defer for now
    }

    TransformPtr transform = dynamic_pointer_cast<Transform>(node);
    if (transform) {
        assert("!Transforms are not supported inside of <clipPath>'s");
        return isShapeTextOrPath(transform->node);
    }

    GroupPtr group = dynamic_pointer_cast<Group>(node);
    if (group) {
        for (size_t i = 0; i < group->list.size(); i++) {
            if (!isShapeTextOrPath(group->list[i])) {
                return false;
            }
        }
        return true;
    }

    if (dynamic_pointer_cast<Shape>(node)) {
        return true;
    }

    if (dynamic_pointer_cast<Text>(node)) {
        return true;
    }

    if (dynamic_pointer_cast<Path>(node)) {
        return true;
    }

    return false;
}

// It would be difficult to implement this as a visitor since it modifies the
// actual structure instead of just reading it (it requires a NodePtr& instead
// of a NodePtr) we'll just do it like this
template<typename T>
void foreachUseTraversal(NodePtr &node, T functor = T())
{
    UsePtr use = dynamic_pointer_cast<Use>(node);
    if (use) {
        functor(node, use);
        return;
    }

    TransformPtr transform = dynamic_pointer_cast<Transform>(node);
    if (transform) {
        foreachUseTraversal<T>(transform->node, functor);
        // FIXME node could get null'ed
        assert(transform->node);
        return;
    }

    GroupPtr group = dynamic_pointer_cast<Group>(node);
    if (group) {
        vector<NodePtr> &list = group->list;
        for (size_t i=0; i<list.size(); i++) {
            foreachUseTraversal<T>(list[i], functor);
        }
        std::vector<NodePtr>::iterator where;
        where = std::remove(list.begin(), list.end(), NodePtr());
        group->list.erase(where, list.end());

        return;
    }

    ClipPtr clip = dynamic_pointer_cast<Clip>(node);
    if (clip) {
        foreachUseTraversal<T>(clip->path, functor);
        assert(clip->path);
        if (!isShapeTextOrPath(clip->path) && !dynamic_pointer_cast<Clip>(node)) {
            assert(!"bogus clip path");
        }

        foreachUseTraversal<T>(clip->node, functor);
        assert(clip->node);
    }
}

struct ResolveUses {  // used with foreachUseTraversal
    SVGParser *parser;

    void operator()(NodePtr &node, const UsePtr &use) {
        UseMap::const_iterator iter = parser->use_map.find(use->href);
        if (iter != parser->use_map.end()) {
            node = parser->parseUsedNode(iter->second, use);
            foreachUseTraversal(node, ResolveUses(parser));
        } else {
            printf("could not resolve <use> href \"%s\"\n", use->href.c_str());
            // Assign a null Node to replace the dangling Use.
            node = NodePtr();
        }
    }

    ResolveUses(SVGParser *p) 
        : parser(p)
    {
    }
};

const char * getIndent(unsigned int numIndents)
{
    static const char * pINDENT="                                      + ";
    static const size_t LENGTH=strlen( pINDENT );
    size_t n = numIndents*NUM_INDENTS_PER_SPACE;
    if ( n > LENGTH ) {
        n = LENGTH;
    }

    return &pINDENT[ LENGTH-n ];
}

float pixels_per_viewport_width = 500;
static float pixels_per_millimeter = 5;

// same as getIndent but no "+" at the end
const char * getIndentAlt(unsigned int numIndents)
{
    static const char * pINDENT = "                                        ";
    static const size_t LENGTH = strlen( pINDENT );
    size_t n = numIndents*NUM_INDENTS_PER_SPACE;
    if ( n > LENGTH ) {
        n = LENGTH;
    }

    return &pINDENT[ LENGTH-n ];
}

int dump_attribs_to_stdout(TiXmlElement* element, unsigned int indent)
{
    int i=0;
    if (element) {
        TiXmlAttribute* attrib = element->FirstAttribute();
        int ival;
        double dval;
        const char* indentStr = getIndent(indent);
        printf("\n");
        while (attrib) {
            printf( "%s%s: value=[%s]", indentStr, attrib->Name(), attrib->Value());

            if (attrib->QueryIntValue(&ival)==TIXML_SUCCESS)    printf( " int=%d", ival);
            if (attrib->QueryDoubleValue(&dval)==TIXML_SUCCESS) printf( " d=%1.1f", dval);
            printf( "\n" );
            i++;
            attrib=attrib->Next();
        }
    }
    return i;   
}

static string RemoveCommas(string input)
{
        //SVG allows for arbitrary whitespace and commas everywhere
        //sscanf ignores whitespace, so the only problem is commas
        for(unsigned int i=0; i < input.length(); i++)
                if(input[i] == ',')
                        input[i] = ' ';
        return input;
}

// From SVG 1.1's transform grammar:
// comma-wsp:
//     (wsp+ comma? wsp*) | (comma wsp*)
int skip_comma_wsp_plus(const char *s, int pos)
{
    if (s == NULL) {
        return 0;
    }
    while (isspace(s[pos]) || s[pos]==',') {
        pos++;
    }
    return pos;
}

const char *skip_semicolons(const char *s)
{
    if (s == NULL) {
        return NULL;
    }
    while (isspace(*s) || *s==';') {
        s++;
    }
    return s;
}

const char *skip_past_semicolon(const char *s)
{
    if (s == NULL) {
        return NULL;
    }
    while (*s!=';' && *s!='\0') {
        s++;
    }
    return skip_semicolons(s);
}

static int hexvalue(int a)
{
    if (a >= '0' && a <= '9') {
        return a-'0';
    } else if (a >= 'a' && a <= 'f') {
        return a-'a' + 10;
    } else if (a >= 'A' && a <= 'F') {
        return a-'A' + 10;
    } else {
        assert(!"bad hexvalue");
        return 0;
    }
}

static int hexvalue2(int a, int b)
{
    int v = hexvalue(a) * 16 + hexvalue(b);
    return v;
}

const char *SVGParser::parseColor(const char *s, float4 &color, const float4 &current_color, const float4 &inherit_color)
{
    // http://www.w3.org/TR/SVG11/types.html#DataTypeColor
    char c[7];
    int count = 0;
    // "The format of an RGB value in hexadecimal notation is a '#' immediately 
    // followed by either three or six hexadecimal characters."
    int rc = sscanf(s, " #%6[0-9a-fA-F]%n", c, &count);
    if (rc == 1 && count > 0) {
        size_t l = strlen(c);
        if (l == 3) {
            // "The three-digit RGB notation (#rgb) is converted into six-digit form (#rrggbb) 
            // by replicating digits, not by adding zeros. For example, #fb0 expands to 
            // #ffbb00. This ensures that white (#ffffff) can be specified with the short
            // notation (#fff) and removes any dependencies on the color depth of the display."
            color.r = hexvalue2(c[0], c[0]) / 255.0f;
            color.g = hexvalue2(c[1], c[1]) / 255.0f;
            color.b = hexvalue2(c[2], c[2]) / 255.0f;
        } else if (l == 6) {
            color.r = hexvalue2(c[0], c[1]) / 255.0f;
            color.g = hexvalue2(c[2], c[3]) / 255.0f;
            color.b = hexvalue2(c[4], c[5]) / 255.0f;
        }
        s += count;
    } else {
        // RGB values as 8-bit 0 to 255 values
        int red, green, blue;
        int rc = sscanf(s, " rgb ( %d, %d, %d )%n", &red, &green, &blue, &count);
        if (rc != 3) {
            // CSS2 says it is case-insensitive so allow this too (though SVG itself probably wouldn't)
            rc = sscanf(s, " RGB ( %d, %d, %d )%n", &red, &green, &blue, &count);
        }
        if (rc == 3 && count > 0) {
            color.r = red / 255.0f;
            color.g = green / 255.0f;
            color.b = blue / 255.0f;
            s += count;
        } else {
            // RGB values as percentages
            float red, green, blue;
            int rc = sscanf(s, " rgb ( %f%%, %f%%, %f%% )%n", &red, &green, &blue, &count);
            if (rc != 3) {
                // CSS2 says it is case-insensitive so allow this too (though SVG itself probably wouldn't)
                rc = sscanf(s, " RGB ( %f%%, %f%%, %f%% )%n", &red, &green, &blue, &count);
            }
            if (rc == 3 && count > 0) {
                color.r = red / 100.0f;
                color.g = green / 100.0f;
                color.b = blue / 100.0f;
                s += count;
            } else {
                char color_name[41];

                int rc = sscanf(s, " %40[A-Za-z]%n", color_name, &count);
                if (rc == 1 && count > 0) {
                    if (!strcmp("inherit", color_name)) {
                        color.rgb = inherit_color.rgb;
                    } else if (!strcmp("currentColor", color_name)) {
                        color.rgb = current_color.rgb;
                    } else {
                        bool success = parse_color_name(color_name, color);
                        if (success) {
                        } else {
                            if (verbose) {
                                printf("failed to match color <%s>\n", color_name);
                            }
                            color = float4(0,0,0,1);
                        }
                    }
                    s += count;
                }
            }
        }
    }
    return s;
}

const char *SVGParser::parsePaintServer(const char *s, const StyleInfo &style, PaintServerPtr &paint_server)
{
    char uri[200];
    int count = 0;
    char expecting_char[2] = { 0, 0 };
    int rc = sscanf(s, " non%[e]%n", expecting_char, &count);
    if (rc == 1 && count > 0) {
        paint_server = PaintServerPtr();
        return s + count;
    }

    rc = sscanf(s, " url ( # %[^) \t\n])%n", uri, &count);
    if (rc == 0) {
        // Allow the url to have quotes around it, example:
        // style="fill: url(&quot;#fieldGradient&quot;)"
        rc = sscanf(s, " url ( \" # %[^) \t\n\"] \" )%n", uri, &count);
    }
    if (rc == 1 && count>0) {
        PaintServerMap::iterator iter = paint_server_map.find(string(uri));
        if (iter != paint_server_map.end()) {
            paint_server = iter->second;
        } else {
            printf("uri %s used in class attribute doesn't exist!\n", uri);
        }
        return s + count;
    }  else if (strcmp(s, "inherit")) {
        float4 color;
        const char *ss = parseColor(s, color, style.color);
        paint_server = SolidColorPtr(new SolidColor(color));
        return ss;
    } else {
        // Paints are already inherited
        return s + sizeof("inherit")-1;
    }
}

ClipInfoPtr SVGParser::parseClipPathReference(const char *name)
{
    char uri[200];
    int count = 0;
    int rc = sscanf(name, " url ( # %[^) \t\n]s)%n", uri, &count);
    if (rc == 1) {
        return ClipInfoPtr(clipPath_map[uri]);
    }

    return ClipInfoPtr();
}

const char *SVGParser::parseOpacity(const char *s, float &opacity, float current_opacity, float inherit_opacity)
{
    int count = 0;
    float alpha = 1.0;

    if (!strcmp(s, "inherit")) {
        opacity = inherit_opacity;
    } else if (!strcmp(s, "currentColor")) {
        opacity = current_opacity;
    } else {
        int rc = sscanf(s, " %f%n", &alpha, &count);
        if (rc == 1 && count > 0) {
            alpha = clamp(alpha, 0, 1);
            opacity = alpha;
        }
    }
    s += count;
    return s;
}

const char *SVGParser::parseOffset(const char *s, float &offset)
{
    int count = 0;
    offset = 0;
    int rc = sscanf(s, " %f%%%n", &offset, &count);
    if (rc == 1 && count > 0) {
        // Percentage
        offset /= 100;
        s += count;
    } else {
        int rc = sscanf(s, " %f%n", &offset, &count);
        if (rc == 1 && count > 0) {
            s += count;
        }
    }
    if (offset > 1) {
        offset = 1;
    } else if (offset < 0) {
        offset = 0;
    }
    return s;
}

#define literal_match(string,literal) (!strncmp(string, literal, sizeof(literal)-1))

// http://www.w3.org/TR/SVGTiny12/painting.html#FillRuleProperty
const char *SVGParser::parseFillRule(const char *s, PathStyle::FillRule &fill_rule)
{
    int count = 0;
    char value[9];
    int rc = sscanf(s, " %8[A-Za-z]%n", value, &count);
    if (rc == 1 && count > 0) {
        if (literal_match(value, "evenodd")) {
            fill_rule = PathStyle::EVEN_ODD;
            s += count;
        } else if (literal_match(value, "nonzero")) {
            fill_rule = PathStyle::NON_ZERO;
            s += count;
        }
    }
    return s;
}

// NOT standard SVG
const char *SVGParser::parseStrokeDashPhase(const char *s, PathStyle::DashPhase &dash_phase)
{
    int count = 0;
    char value[20];
    int rc = sscanf(s, " %20[A-Za-z]%n", value, &count);
    if (rc == 1 && count > 0) {
        if (literal_match(value, "movetoResets")) {
            dash_phase = PathStyle::MOVETO_RESETS;
            s += count;
        } else if (literal_match(value, "movetoContinues")) {
            dash_phase = PathStyle::MOVETO_CONTINUES;
            s += count;
        }
    }
    return s;
}

// http://www.w3.org/TR/SVGTiny12/painting.html#StrokeLineCapProperty
const char *SVGParser::parseStrokeLineCap(const char *s, PathStyle::LineCap &line_cap)
{
    int count = 0;
    char value[9];
    int rc = sscanf(s, " %8[A-Za-z]%n", value, &count);
    if (rc == 1 && count > 0) {
        if (literal_match(value, "butt")) {
            line_cap = PathStyle::BUTT_CAP;
            s += count;
        } else if (literal_match(value, "round")) {
            line_cap = PathStyle::ROUND_CAP;
            s += count;
        } else if (literal_match(value, "square")) {
            line_cap = PathStyle::SQUARE_CAP;
            s += count;
        } else if (literal_match(value, "triangle")) {
            line_cap = PathStyle::TRIANGLE_CAP;
            s += count;
        }
    }
    return s;
}

// http://www.w3.org/TR/SVGTiny12/painting.html#StrokeLineJoinProperty
const char *SVGParser::parseStrokeLinejoin(const char *s, PathStyle::LineJoin &line_join)
{
    int count = 0;
    char value[20];
    int rc = sscanf(s, " %20[A-Za-z-]%n", value, &count);
    if (rc == 1 && count > 0) {
        if (literal_match(value, "miter-truncate")) {
            line_join = PathStyle::MITER_TRUNCATE_JOIN;
            s += count;
        } else if (literal_match(value, "none")) {
            line_join = PathStyle::NONE_JOIN;
            s += count;
        } else if (literal_match(value, "miter")) {
            line_join = PathStyle::MITER_REVERT_JOIN;
            s += count;
        } else if (literal_match(value, "round")) {
            line_join = PathStyle::ROUND_JOIN;
            s += count;
        } else if (literal_match(value, "bevel")) {
            line_join = PathStyle::BEVEL_JOIN;
            s += count;
        }
    }
    return s;
}

// http://www.w3.org/TR/SVG/text.html#TextAnchorProperty
const char *SVGParser::parseTextAnchor(const char *s, TextAnchor &text_anchor)
{
    int count = 0;
    char value[20];
    int rc = sscanf(s, " %20[A-Za-z-]%n", value, &count);
    if (rc == 1 && count > 0) {
        if (literal_match(value, "start")) {
            text_anchor = START;
            s += count;
        } else if (literal_match(value, "end")) {
            text_anchor = END;
            s += count;
        } else if (literal_match(value, "middle")) {
            text_anchor = MIDDLE;
            s += count;
        }
    }
    return s;
}

// http://www.w3.org/TR/SVG/text.html#TextAnchorProperty
const char *SVGParser::parseFontFamily(const char *s, string &font_family)
{
    const char *ss = s;
    while (*ss != '\0' && *ss != ';') {
        ss++;
    }
    font_family = string(s, ss-s);
    return ss;
}

// http://www.w3.org/TR/SVGTiny12/coords.html#Units
const char *SVGParser::parseUnit(const char *s, float &value, bool percentBounds)
{
    // http://www.w3.org/TR/SVG11/coords.html#UnitIdentifiers
    if (!strncmp("mm", s, 2)) {  // millimeter
        value = value * pixels_per_millimeter;
        s += strlen("mm");
    } else
    if (!strncmp("cm", s, 2)) {  // centimeter (10 per millimeter)
        value = value * 10 * pixels_per_millimeter;
        s += strlen("cm");
    } else
    if (!strncmp("in", s, 2)) {  // inch (2.54 per millimeter)
        value = value * 25.4f * pixels_per_millimeter;
        s += strlen("in");
    } else
    if (!strncmp("px", s, 2)) {  // pixels
        value = value * 1;  // XXX perhaps account for svg element's width attribute?
        s += strlen("px");
    } else
    if (!strncmp("pt", s, 2)) {  // point (72 per inch)
        value = value / 72 * 25.4f * pixels_per_millimeter;
        s += strlen("pt");
    } else
    if (!strncmp("pc", s, 2)) {  // pica (6 per inch)
        value = value / 6 * 25.4f * pixels_per_millimeter;
        s += strlen("pc");
    } else
    if (!strncmp("em", s, 2)) {  // the width of a lower case "m" in your text'
        assert(!"support em");
        value = value;  // XXX fix me
        s += strlen("em");
    } else
    if (!strncmp("ex", s, 2)) {  // 1 ex is the x-height of a font (x-height is usually about half the font-size)
        assert(!"support ex");
        value = value;  // XXX fix me
        s += strlen("ex");
    } else
    if (!strncmp("%", s, 1)) {  // A percentage represents a distance as a percentage of the current viewport.
        value /= 100;
        if (percentBounds) {
            // Value is a straight percent.
        } else {
            // Value is a percent of the viewport width.
            value *= pixels_per_viewport_width;
        }
        s += strlen("%");
    } else {
        // nothing
    }
    return s;
}

// http://www.w3.org/TR/SVGTiny12/types.html#DataTypeCoordinate
const char *SVGParser::parseCoordinate(const char *s, float &value, bool percentBounds)
{
    int count = 0;
    int rc = sscanf(s, " %f%n", &value, &count);
    if (rc == 1 && count > 0) {
        s += count;
        s = parseUnit(s, value, percentBounds);
    }
    return s;
}

// http://www.w3.org/TR/SVGTiny12/types.html#DataTypeLength
const char *SVGParser::parseLength(const char *s, float &value)
{
    int count = 0;
    int rc = sscanf(s, " %f%n", &value, &count);
    if (rc == 1 && count > 0) {
        s += count;
        s = parseUnit(s, value);
    }
    return s;
}

const char *SVGParser::parseViewBox(const char *s, float4 &viewbox)
{
    int count = 0;
    float x, y, w, h;
    int rc = sscanf(s, " %f %f %f %f%n", &x, &y, &w, &h, &count);
    if (rc == 4 && count > 0) {
        viewbox = float4(x,y,x+w,y+h);
        s += count;
    }
    return s;
}

const char *SVGParser::parseStrokeDashArraySpace(const char *s, vector<float> &dash_array)
{
    float value = -666;
    const char *ss = parseLength(s, value);
    if (ss && s!=ss) {
        dash_array.push_back(value);
        int pos = skip_comma_wsp_plus(ss, 0);
        ss += pos;
    }
    return ss;
}

// http://www.w3.org/TR/SVGTiny12/painting.html#StrokeDasharrayProperty
const char *SVGParser::parseStrokeDashArray(const char *s, vector<float> &dash_array)
{
    int count = 0;
    char value[20];
    int rc = sscanf(s, " %20[A-Za-z-]%n", value, &count);
    if (rc == 1 && count > 0) {
        if (literal_match(value, "none")) {
            dash_array.clear();
            s += count;
            return s;
        } else if (literal_match(value, "inherit")) {
            // no change to dash array
            s += count;
            return s;
        }
    }
    while (s && *s != '\0' && *s != ';') {
        s = parseStrokeDashArraySpace(s, dash_array);
    }
    // Assume if dash pattern is zero length, that means no dashing.
    // XXX Though the painting-stroke-06-t.svg SVG test requires this,
    // where does the spec say this?
    double total = 0;
    for (size_t i=0; i<dash_array.size(); i++) {
        total += dash_array[i];
    }
    if (total == 0) {
        dash_array.clear();
    }
    return s;
}

const char *SVGParser::parseStyle(const char *ss, StyleInfo &style)
{
    int count=0;

    while (ss && *ss) {
        char expecting_colon[2] = { 0, 0 };

        ss = skip_semicolons(ss);
        const char *start = ss;

        // Hack for Microsoft content where attributs can be capitalized in styles
        // piechart.svg is an example of this
        char lowered_ss[200];
        int i;
        for (i=0; ss[i]!='\0' && ss[i]!=':'; i++) {
            lowered_ss[i] = tolower(ss[i]);
        }
        if (ss[i]!='\0') {
            lowered_ss[i] = ss[i];
            i++;
        }
        lowered_ss[i] = '\0';

        if (ss && 1==sscanf(lowered_ss, " opacity %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseOpacity(ss, style.opacity);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " fill-opacity %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseOpacity(ss, style.fill_opacity);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " stroke-opacity %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseOpacity(ss, style.stroke_opacity);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " fill-rule %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseFillRule(ss, style.path.fill_rule);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " fill %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parsePaintServer(ss, style, style.fill);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " color %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseColor(ss, style.color);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " stop-color %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseColor(ss, style.stop_color, style.color, parentStyle().stop_color);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " stop-opacity %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            float opacity;

            ss = parseOpacity(ss, opacity, style.color.a, parentStyle().stop_color.a);
            style.stop_color.a = opacity;
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " stroke %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parsePaintServer(ss, style, style.stroke);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " stroke-width %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            float stroke_width;
            const char *sss = parseLength(ss, stroke_width);
            if (sss != ss) {
                if (stroke_width < 0) {
                    printf("SVG error: stroke-width %f is negative\n", stroke_width);
                } else {
                    style.path.stroke_width = stroke_width;
                }
                ss = sss;
            }
        }
        if (ss && 1==sscanf(lowered_ss, " stroke-linecap %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseStrokeLineCap(ss, style.path.line_cap);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " stroke-linejoin %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseStrokeLinejoin(ss, style.path.line_join);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " text-anchor %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseTextAnchor(ss, style.text_anchor);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " font-family %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseFontFamily(ss, style.font_family);
            ss = skip_semicolons(ss);
        }
        if (ss && 1==sscanf(lowered_ss, " stroke-miterlimit %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            float miter_limit;
            int rc = sscanf(ss, "%f%n", &miter_limit, &count);
            if (rc == 1 && count > 0) {
                if (miter_limit >= 1.0) {
                    style.path.miter_limit = miter_limit;
                    ss += count;
                } else {
                    // The value of <miterlimit> must be a number greater than or equal
                    // to 1. Any other value shall be treated as unsupported and processed
                    // as if the property had not been specified. 
                }
            }
        }
        // http://www.w3.org/TR/SVGTiny12/painting.html#StrokeDashOffsetProperty
        if (ss && 1==sscanf(lowered_ss, " stroke-dashoffset %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            float dash_offset;
            const char *sss = parseLength(ss, dash_offset);
            if (sss != ss) {
                style.path.dash_offset = dash_offset;
                ss = sss;
            }
        }
        if (ss && 1==sscanf(lowered_ss, " stroke-dasharray %1[:]%n", expecting_colon, &count)) {
            assert(expecting_colon[0] == ':');
            ss += count;
            ss = parseStrokeDashArray(ss, style.path.dash_array);
            ss = skip_semicolons(ss);
        }
        // If ss still equals start, no progress was made so...
        if (ss == start) {
            ss = skip_past_semicolon(ss);
        }
    }
    return ss;
}

int SVGParser::skipProblem(const char *bad_tag, const char *s)
{
    fprintf(stderr, "SVG parse problem: skipping %s tag: \"%s\"\n", bad_tag, s);

    char c = *s;
    int pos = 0;
    while (c != ')' && c != '\0') {
        pos++;
        c = s[pos];
    }
    if (c == ')') {
        pos++;
    }
    return pos;
}

void SVGParser::parseTransform(const char *ss, double3x3 &matrix)
{
    int pos=0;
    int count=0;
    while(ss[pos]) {
        if(strncmp(&ss[pos], "translate", 9)==0) {
            double tx, ty;
            int rc = sscanf(&ss[pos], "translate ( %lf %lf )%n", &tx, &ty, &count);
            if (rc == 2 && count > 0) {
                double3x3 m = double3x3(1,0,tx,
                                        0,1,ty,
                                        0,0,1);
                matrix = mul(matrix, m);
                pos = skip_comma_wsp_plus(ss, pos+count);
                continue;
            } else {
                int rc = sscanf(&ss[pos], "translate ( %lf )%n", &tx, &count);
                if (rc == 1 && count > 0) {
                    double3x3 m = double3x3(1,0,tx,
                                            0,1,0,
                                            0,0,1);
                    matrix = mul(matrix, m);
                    pos = skip_comma_wsp_plus(ss, pos+count);
                    continue;
                } else {
                    pos += skipProblem("translate", &ss[pos]);
                }
            }
            break;;
        }
        if(strncmp(&ss[pos], "matrix", 6)==0) {
            double a, b, c, d, e, f;
            int rc = sscanf(&ss[pos],"matrix ( %lf %lf %lf %lf %lf %lf )%n", &a, &b, &c, &d, &e, &f, &count);
            if (rc == 6 && count > 0) {
                double3x3 m = double3x3(a, c, e,
                    b, d, f,
                    0, 0, 1);
                matrix = mul(matrix, m);
                pos = skip_comma_wsp_plus(ss, pos+count);
                continue;
            } else {
                pos += skipProblem("matrix", &ss[pos]);
            }
            break;
        }
        if(strncmp(&ss[pos], "scale", 5)==0) {
            double sx, sy;
            int rc = sscanf(&ss[pos], "scale ( %lf %lf )%n", &sx,&sy,&count);
            if (rc == 2 && count > 0) {
                double3x3 m = double3x3(sx,0,0,
                    0,sy,0,
                    0,0,1);
                matrix = mul(matrix, m);
                pos = skip_comma_wsp_plus(ss, pos+count);
                continue;
            } else {
                int rc = sscanf(&ss[pos], "scale ( %lf )%n", &sx,&count);
                if (rc == 1 && count > 0) {
                    double3x3 m = double3x3(sx,0,0,
                        0,sx,0,
                        0,0,1);
                    matrix = mul(matrix, m);
                    pos = skip_comma_wsp_plus(ss, pos+count);
                    continue;
                } else {
                    pos += skipProblem("scale", &ss[pos]);
                }
            }
            break;
        }
        if(strncmp(&ss[pos],"rotate",6)==0) {
            char expecting_close_paren[2] = { 0, 0 };
            double a;
            int rc;
            bool got_angle = false;
            bool got_translate = false;
            double cx = 0, cy = 0;
            
            rc = sscanf(&ss[pos],"rotate ( %lf %1[)]%n", &a, expecting_close_paren, &count);
            if (rc == 2 && count > 0) {
                assert(expecting_close_paren[0] == ')');
                a = radians(a);
                got_angle = true;
            }
            if (!got_angle) {
                rc = sscanf(&ss[pos],"rotate ( %lf %lf %lf  %1[)]%n", &a, &cx, &cy, expecting_close_paren, &count);
                if (rc == 4 && count > 0) {
                    assert(expecting_close_paren[0] == ')');
                    a = radians(a);
                    got_angle = true;
                    got_translate = true;
                }
            }
            if (got_angle) {
                double3x3 m = double3x3(cos(a),-sin(a),0,
                                        sin(a),cos(a),0,
                                        0,0,1);
                if (got_translate) {
                    double3x3 t =double3x3(1,0,cx,
                                           0,1,cy,
                                           0,0,1);
                    double3x3 ti =double3x3(1,0,-cx,
                                            0,1,-cy,
                                            0,0,1);
                    m = mul(t,mul(m,ti));
                }
                matrix = mul(matrix, m);
                pos = skip_comma_wsp_plus(ss, pos+count);
                continue;
            }
            break;
        }
        if(strncmp(&ss[pos],"skewX",5)==0 ||
           strncmp(&ss[pos],"skewY",5)==0) {
            char expecting_close_paren[2] = { 0, 0 };
            char expecting_X_or_Y[2] = { 0, 0 };
            double a;
            int rc;
            bool got_angle = false;

            rc = sscanf(&ss[pos],"skew%1[XY] ( %lf %1[)]%n",
                expecting_X_or_Y, &a, expecting_close_paren, &count);
            if (rc != 3) {
                rc = sscanf(&ss[pos],"skew%1[XY] ( %lf deg %1[)]%n",
                    expecting_X_or_Y, &a, expecting_close_paren, &count);
            }
            if (rc == 3 && count > 0) {
                assert(expecting_close_paren[0] == ')');
                a = radians(a);
                got_angle = true;
            }
            if (!got_angle) {
                rc = sscanf(&ss[pos],"skew%1[XY] ( %lf grad %1[)]%n",
                    expecting_X_or_Y, &a, expecting_close_paren, &count);
                if (rc == 3 && count > 0) {
                    assert(expecting_close_paren[0] == ')');
                    a = a * M_PI/200;
                    got_angle = true;
                }
            }
            if (!got_angle) {
                rc = sscanf(&ss[pos],"skew%1[XY] ( %lf rad %1[)]%n",
                    expecting_X_or_Y, &a, expecting_close_paren, &count);
                if (rc == 3 && count > 0) {
                    assert(expecting_close_paren[0] == ')');
                    // a is already in radians.
                    got_angle = true;
                } else {
                    pos += skipProblem("skew", &ss[pos]);
                }
            }
            if (got_angle) {
                double3x3 m;
                
                switch(expecting_X_or_Y[0]) {
                default:
                    assert(!"must be X or Y");
                case 'X':
                    m = double3x3(1,tan(a),0,
                                  0,1,0,
                                  0,0,1);
                    break;
                case 'Y':
                    m = double3x3(1,0,0,
                                  tan(a),1,0,
                                  0,0,1);
                    break;
                }
                matrix = mul(matrix, m);
                pos = skip_comma_wsp_plus(ss, pos+count);
                continue;
            }
            break;
        }
        break;
    }
}

bool SVGParser::parseGenericShapeProperty(TiXmlAttribute* a, TiXmlElement* elem)
{
    string name(a->Name());

    if(name == "transform") {
        string s = a->Value();
        s = RemoveCommas(s);
        const char* ss = s.c_str();
        parseTransform(ss, style().matrix);
        return true;
    }
    if(name == "color") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parseColor(ss, style().color);
        return true;
    }
    if(name == "fill") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parsePaintServer(ss, style(), style().fill);
        return true;
    }
    if(name == "stroke") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parsePaintServer(ss, style(), style().stroke);
        return true;
    }
    if (name == "stroke-width") {
        // http://www.w3.org/TR/SVGTiny12/painting.html#StrokeWidthProperty
        string s = a->Value();
        const char* ss = s.c_str();
        float stroke_width;
        const char *sss = parseLength(ss, stroke_width);
        if (sss != ss) {
            style().path.stroke_width = stroke_width;
            ss = sss;
        }
        return true;
    }
    if (name == "stroke-dashoffset") {
        // http://www.w3.org/TR/SVGTiny12/painting.html#StrokeDashOffsetProperty
        string s = a->Value();
        const char* ss = s.c_str();
        float dash_offset;
        const char *sss = parseLength(ss, dash_offset);
        if (sss != ss) {
            style().path.dash_offset = dash_offset;
            ss = sss;
        }
        return true;
    }
    if (name == "stroke-dasharray") {
        // http://www.w3.org/TR/SVGTiny12/painting.html#StrokeDashOffsetProperty
        string s = a->Value();
        const char* ss = s.c_str();
        ss = parseStrokeDashArray(ss, style().path.dash_array);
        return true;
    }
    if (name == "stroke-dashphase") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parseStrokeDashPhase(ss, style().path.dash_phase);
        return true;
    }
    if(name == "style") {
        const string s = a->Value();
        const char* ss = s.c_str();

        ss = parseStyle(ss, style());
        return true;
    }
    if(name == "class") {
        string s = a->Value();
        StyleMap::iterator iter = style_map.find(s);
        if (iter != style_map.end()) {
            parseStyle(iter->second.c_str(), style());
        } else {
            printf("style %s used in class attribute doesn't exist!\n", s.c_str());
        }
        return true;
    }
    if(name == "opacity") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parseOpacity(ss, style().opacity);
        return true;
    }
    if(name == "fill-opacity") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parseOpacity(ss, style().fill_opacity);
        return true;
    }    
    if (name == "stop-color") {
        string s = a->Value();
        const char* ss = s.c_str();

        parseColor(ss, style().stop_color, style().color, parentStyle().stop_color);
        return true;
    }
    if (name == "stop-opacity") {
        string s = a->Value();
        const char* ss = s.c_str();

        float opacity;
        parseOpacity(ss, opacity, style().color.a, parentStyle().stop_color.a);
        style().stop_color.a = opacity;
        return true;
    }
    if(name == "fill-rule") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parseFillRule(ss, style().path.fill_rule);
        return true;
    }
    if(name == "stroke-opacity") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parseOpacity(ss, style().stroke_opacity);
        return true;
    }
    if (name == "stroke-linecap") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parseStrokeLineCap(ss, style().path.line_cap);
        return true;
    }
    if (name == "stroke-linejoin") {
        string s = a->Value();
        const char* ss = s.c_str();

        ss = parseStrokeLinejoin(ss, style().path.line_join);
        return true;
    }
    if (name == "stroke-miterlimit") {
        string s = a->Value();
        const char* ss = s.c_str();
        float miter_limit;

        int count = 0;
        int rc = sscanf(ss, " %f%n", &miter_limit, &count);
        if (rc == 1 && count > 0) {
            style().path.miter_limit = miter_limit;
            ss += count;
        }
        return true;
    }
    if (name == "clip-path") {
        style().clip_path = parseClipPathReference(a->Value());
        return true;
    }
    if(name == "id") {
        string s = a->Value();
        use_map[s] = elem;
        return true;
    }
	if (name == "font-family") {
		parseFontFamily(a->Value(), style().font_family);
		return true;
	}
    return false;
}

NodePtr SVGParser::createShape(PathPtr path, const StyleInfo &style)
{
    PaintPtr fill_paint, stroke_paint;
    if (style.fill) {
        style.fill->resolve(this->gradient_map);
        fill_paint = style.fill->makePaintPtr();
        path->style.do_fill = true;
    } else {
        fill_paint = PaintPtr();
        path->style.do_fill = false;
    }
    if (style.stroke) {
        style.stroke->resolve(this->gradient_map);
        stroke_paint = style.stroke->makePaintPtr();
        path->style.do_stroke = true;
    } else {
        stroke_paint = PaintPtr();
        path->style.do_stroke = false;
    }

    ShapePtr shape = ShapePtr(new Shape(path,
                                        fill_paint,
                                        stroke_paint));

    shape->net_fill_opacity = style.fill_opacity * style.opacity;
    shape->net_stroke_opacity = style.stroke_opacity * style.opacity;

    return shape;
}

NodePtr SVGParser::decorateNode(NodePtr node)
{
    if (style().clip_path != parentStyle().clip_path) {
        if (!style().clip_path) {
            printf("warning: bogus clip-path id\n");
        } else {
            node = style().clip_path->createClip(node);
        }
    }

    if (!sameMatrix(style().matrix, identity)) {
        node = TransformPtr(new Transform(node, style().matrix));
    } 
    
    return node;
}

NodePtr SVGParser::parseCircleOrEllipse(TiXmlElement* elem, bool circle_only)
{
    style_stack.pushChild();

    PathPtr path = PathPtr();
    float cx = 0, cy = 0, rx = 0, ry = 0;

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        // http://www.w3.org/TR/SVGTiny12/shapes.html#CircleElementCXAttribute
        // http://www.w3.org/TR/SVGTiny12/shapes.html#EllipseElementCXAttribute
        if(name == "cx") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, cx);
            continue;
        }
        // http://www.w3.org/TR/SVGTiny12/shapes.html#CircleElementCYAttribute
        // http://www.w3.org/TR/SVGTiny12/shapes.html#EllipseElementCYAttribute
        if(name == "cy") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, cy);
            continue;
        }
        if (circle_only) {
            // http://www.w3.org/TR/SVGTiny12/shapes.html#CircleElementRAttribute
            if(name == "r") {
                float r;

                string s = a->Value();
                const char* ss = s.c_str();
                ss = parseLength(ss, r);
                rx = r;
                ry = r;
                continue;
            }
        } else {
            // http://www.w3.org/TR/SVGTiny12/shapes.html#EllipseElementRXAttribute
            if(name == "rx") {
                string s = a->Value();
                const char* ss = s.c_str();
                ss = parseLength(ss, rx);
                continue;
            }
            // http://www.w3.org/TR/SVGTiny12/shapes.html#EllipseElementRYAttribute
            if(name == "ry") {
                string s = a->Value();
                const char* ss = s.c_str();
                ss = parseLength(ss, ry);
                continue;
            }
        }
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }

    if (rx <= 0 || ry <= 0) {
        // A negative value is unsupported. A value of zero disables
        // rendering of the element.
        return style_stack.popAndReturn(ShapePtr());
    }

    if ((style().path.do_fill && rx>0 && ry>0) || style().path.do_stroke) {
        vector<char> cmds;
        cmds.reserve(3);
        vector<float> coords;
        coords.reserve(2+7+7);

        // "The arc of a 'circle'  element begins at the "3 o'clock" point on
        // the radius and progresses towards the "9 o'clock" point. The starting
        // point and direction of the arc are affected by the user space transform
        // in the same manner as the geometry of the element."
        cmds.push_back('M');
        coords.push_back(cx+rx);
        coords.push_back(cy);
        cmds.push_back('A');
        coords.push_back(rx);
        coords.push_back(ry);
        coords.push_back(0);  // x-axis-rotation
        coords.push_back(1);  // large-arc-flag 
        coords.push_back(1);  // sweep-flag 
        coords.push_back(cx-rx);
        coords.push_back(cy);
        cmds.push_back('A');
        coords.push_back(rx);
        coords.push_back(ry);
        coords.push_back(0);  // x-axis-rotation
        coords.push_back(1);  // large-arc-flag 
        coords.push_back(1);  // sweep-flag 
        coords.push_back(cx+rx);
        coords.push_back(cy);
        path = PathPtr(new Path(style().path, cmds, coords));

        NodePtr shape = createShape(path, style());
        
        return style_stack.popAndReturn(decorateNode(shape));
    }

    return style_stack.popAndReturn(NodePtr());
}

// http://www.w3.org/TR/SVGTiny12/shapes.html#CircleElement
NodePtr SVGParser::parseCircle(TiXmlElement* elem)
{
    return parseCircleOrEllipse(elem, true);
}

// http://www.w3.org/TR/SVGTiny12/shapes.html#EllipseElement
NodePtr SVGParser::parseEllipse(TiXmlElement* elem)
{
    return parseCircleOrEllipse(elem, false);
}

// "The points that make up the polygon. All coordinate values
// are in the user coordinate system."
// http://www.w3.org/TR/SVGTiny12/shapes.html#PointsBNF
const char *parse_points_data(const char *ss, vector<float2> &points)
{
    points.clear();
    while (ss && *ss) {
        float x, y;
        int count = 0;

        if (ss && 2==sscanf(ss, " %f%*[, \n\r\t]%f%n", &x, &y, &count)) {
            if (count > 0 && ss[count-1] == '\0') {
                ss = NULL;
            } else {
                ss += count;
                int pos = skip_comma_wsp_plus(ss, 0);
                ss += pos;
            }
            points.push_back(float2(x,y));
        } else {
            return NULL;
        }
    }
    return ss;
}

// "The 'polygon' element defines a closed shape consisting of a set 
// of connected straight line segments."
NodePtr SVGParser::parsePolygonOrPolyline(TiXmlElement* elem, bool is_polygon)
{
    style_stack.pushChild();

    PathPtr path = PathPtr();
    vector<float2> points;

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        if(name == "points") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parse_points_data(ss, points);
            continue;
        }
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }

    size_t vertex_count = points.size();
    if ((style().path.do_fill && vertex_count>=3) || (style().path.do_stroke && vertex_count>=1)) {
        // "Mathematically, a 'polygon' element can be mapped to an
        // equivalent 'path' element as follows:"
        vector<char> cmds;
        cmds.reserve(vertex_count);
        vector<float> coords;
        coords.reserve(vertex_count*2);

        // "perform an absolute moveto operation to the first
        // coordinate pair in the list of points"
        cmds.push_back('M');
        coords.push_back(points[0].x);
        coords.push_back(points[0].y);
        // "for each subsequent coordinate pair, perform an
        // absolute lineto  operation to that coordinate pair"
        for (size_t i=1; i<vertex_count; i++) {
            cmds.push_back('L');
            coords.push_back(points[i].x);
            coords.push_back(points[i].y);
        }
        if (is_polygon) {
            // "perform a closepath  command"
            cmds.push_back('z');
        } else {
            // Polyline isn't closed.
        }
        path = PathPtr(new Path(style().path, cmds, coords));

        NodePtr shape = createShape(path, style());
        
        return style_stack.popAndReturn(decorateNode(shape));
    }

    return style_stack.popAndReturn(NodePtr());
}

// http://www.w3.org/TR/SVGTiny12/shapes.html#PolygonElement
NodePtr SVGParser::parsePolygon(TiXmlElement* elem)
{
    return parsePolygonOrPolyline(elem, true);
}

// http://www.w3.org/TR/SVGTiny12/shapes.html#PolylineElement
NodePtr SVGParser::parsePolyline(TiXmlElement* elem)
{
    return parsePolygonOrPolyline(elem, false);
}

// "The 'line' element defines a line segment that starts at one point and ends at another."
// http://www.w3.org/TR/SVGTiny12/shapes.html#LineElement
NodePtr SVGParser::parseLine(TiXmlElement* elem)
{
    style_stack.pushChild();

    PathPtr path = PathPtr();
    float2 p1 = float2(0),
           p2 = float2(0);

    style().path.do_fill = false;
    style().path.do_stroke = true;

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        if(name == "x1") {
            string s = a->Value();
            const char* ss = s.c_str();
            float v;
            const char *sss = parseCoordinate(ss, v);
            if (sss != ss) {
                p1.x = v;
                ss = sss;
            }
            continue;
        }
        if(name == "y1") {
            string s = a->Value();
            const char* ss = s.c_str();
            float v;
            const char *sss = parseCoordinate(ss, v);
            if (sss != ss) {
                p1.y = v;
                ss = sss;
            }
            continue;
        }
        if(name == "x2") {
            string s = a->Value();
            const char* ss = s.c_str();
            float v;
            const char *sss = parseCoordinate(ss, v);
            if (sss != ss) {
                p2.x = v;
                ss = sss;
            }
            continue;
        }
        if(name == "y2") {
            string s = a->Value();
            const char* ss = s.c_str();
            float v;
            const char *sss = parseCoordinate(ss, v);
            if (sss != ss) {
                p2.y = v;
                ss = sss;
            }
            continue;
        }
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }

    if (style().path.do_stroke) {
        // "Mathematically, a 'polygon' element can be mapped to an
        // equivalent 'path' element as follows:"
        vector<char> cmds;
        cmds.reserve(2);
        vector<float> coords;
        coords.reserve(4);

        // "perform an absolute moveto operation to the first
        // coordinate pair in the list of points"
        cmds.push_back('M');
        coords.push_back(p1.x);
        coords.push_back(p1.y);
        cmds.push_back('L');
        coords.push_back(p2.x);
        coords.push_back(p2.y);
        path = PathPtr(new Path(style().path, cmds, coords));

        NodePtr shape = createShape(path, style());
        
        return style_stack.popAndReturn(decorateNode(shape));
    }

    return style_stack.popAndReturn(NodePtr());
}

NodePtr SVGParser::parseRect(TiXmlElement* elem)
{
    style_stack.pushChild();

    PathPtr path = PathPtr();
    float x = 0, y = 0, width = 0, height = 0, rx = 0, ry = 0;
    bool rx_set = false, ry_set = false;

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        if(name == "x") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, x);
            continue;
        }
        if(name == "y") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, y);
            continue;
        }
        if(name == "rx") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseLength(ss, rx);
            rx_set = true;
            continue;
        }
        if(name == "ry") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseLength(ss, ry);
            ry_set = true;
            continue;
        }
        if(name == "width") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseLength(ss, width);
            continue;
        }
        if(name == "height") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseLength(ss, height);
            continue;
        }
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }

    // Enforce several "rules" stated in the SVG specification...

    // "If a properly specified value is provided for rx but not for ry,
    // then the user agent processes the 'rect' element with the effective
    // value for ry as equal to rx."
    if (rx_set && !ry_set) {
        ry = rx;
    }
    // "If a properly specified value is provided for ry but not for rx,
    // then the user agent processes the 'rect' element with the effective
    // value for rx as equal to ry."
    else if (ry_set && !rx_set) {
        rx = ry;
    }
    if (!rx_set && !ry_set) {
        // "If neither rx nor ry has a properly specified value,
        // then the user agent processes the 'rect' element as if
        // no rounding had been specified, resulting in square corners."
        assert(rx == 0);
        assert(ry == 0);
    }
    // "If rx is greater than half of the width of the rectangle,
    // then the user agent processes the 'rect' element with the effective
    // value for rx as half of the width of the rectangle."
    if (rx > width/2) {
        rx = width/2;
    }
    // "If ry is greater than half of the height of the rectangle,
    // then the user agent processes the 'rect' element with the effective
    // value for ry as half of the height of the rectangle."
    if (ry > height/2) {
        ry = height/2;
    }

    // Specs says for width and height "A value of zero disables rendering of the element."
    if (width <= 0) {
        return style_stack.popAndReturn(NodePtr());
    }
    if (height <= 0) {
        return style_stack.popAndReturn(NodePtr());
    }
    if (rx < 0) {
        printf("negative value %f for rx is an error\n", rx);
        return style_stack.popAndReturn(NodePtr());
    }
    if (ry < 0) {
        printf("negative value %f for ry is an error\n", ry);
        return style_stack.popAndReturn(NodePtr());
    }

    if (style().path.do_fill || style().path.do_stroke) {
        // "Mathematically, a 'rect' element can be mapped to an equivalent
        // 'path' element as follows: (Note: all coordinate and length values
        // are first converted into user space coordinates according to Units.)"
        vector<char> cmds;
        cmds.reserve(9);
        vector<float> coords;
        coords.reserve(2+4*(1+7));


        // "perform an absolute moveto operation to location (x+rx,y), where x
        // is the value of the 'rect' element's x attribute converted to user
        // space, rx is the effective value of the rx attribute converted to
        // user space and y is the value of the y attribute converted to user space"
        cmds.push_back('M');
        coords.push_back(x+rx);
        coords.push_back(y);
        // "perform an absolute horizontal lineto operation to location (x+width-rx,y),
        // where width is the 'rect' element's width attribute converted to user space"
        cmds.push_back('H');
        coords.push_back(x+width-rx);
        if (rx != 0 && ry != 0) {
            // "perform an absolute elliptical arc operation to coordinate (x+width,y+ry),
            // where the effective values for the rx and ry attributes on the 'rect'
            // element converted to user space are used as the rx and ry  attributes
            // on the elliptical arc command, respectively, the x-axis-rotation  is set
            // to zero, the large-arc-flag is set to zero, and the sweep-flag is set to one"
            cmds.push_back('A');
            coords.push_back(rx);
            coords.push_back(ry);
            coords.push_back(0);  // x-axis-rotation
            coords.push_back(0);  // large-arc-flag 
            coords.push_back(1);  // sweep-flag 
            coords.push_back(x+width);
            coords.push_back(y+ry);
        }
        // "perform a absolute vertical lineto to location (x+width,y+height-ry),
        // where height is the 'rect' element's height attribute converted to
        // user space"
        cmds.push_back('V');
        coords.push_back(y+height-ry);
        if (rx != 0 && ry != 0) {
            // "perform an absolute elliptical arc operation to coordinate
            // (x+width-rx,y+height)"
            cmds.push_back('A');
            coords.push_back(rx);
            coords.push_back(ry);
            coords.push_back(0);  // x-axis-rotation
            coords.push_back(0);  // large-arc-flag 
            coords.push_back(1);  // sweep-flag 
            coords.push_back(x+width-rx);
            coords.push_back(y+height);
        }
        // "perform an absolute horizontal lineto to location (x+rx,y+height)"
        cmds.push_back('H');
        coords.push_back(x+rx);
        if (rx != 0 && ry != 0) {
            // "perform an absolute elliptical arc operation to
            // coordinate (x,y+height-ry)"
            cmds.push_back('A');
            coords.push_back(rx);
            coords.push_back(ry);
            coords.push_back(0);  // x-axis-rotation
            coords.push_back(0);  // large-arc-flag 
            coords.push_back(1);  // sweep-flag 
            coords.push_back(x);
            coords.push_back(y+height-ry);
        }
        // "perform an absolute absolute vertical lineto to 
        // location (x,y+ry)"
        cmds.push_back('V');
        coords.push_back(y+ry);
        if (rx != 0 && ry != 0) {
            // "perform an absolute elliptical arc operation to coordinate (x+rx,y)"
            cmds.push_back('A');
            coords.push_back(rx);
            coords.push_back(ry);
            coords.push_back(0);  // x-axis-rotation
            coords.push_back(0);  // large-arc-flag 
            coords.push_back(1);  // sweep-flag 
            coords.push_back(x+rx);
            coords.push_back(y);
        }

        // SVG 1.1 doesn't specify a "closepath" command
        //   http://www.w3.org/TR/SVG/shapes.html#RectElement
        // but presumably that was the intent and the SVG Tiny 1.2 does
        //   http://www.w3.org/TR/SVGTiny12/shapes.html#RectElement
        cmds.push_back('z');

        path = PathPtr(new Path(style().path, cmds, coords));

        NodePtr shape = createShape(path, style());
        
        return style_stack.popAndReturn(decorateNode(shape));
    }

    return style_stack.popAndReturn(NodePtr());
}

NodePtr SVGParser::parsePath(TiXmlElement* elem)
{
    style_stack.pushChild();

    string path_string;

	int ldp_poly = -1;
	float ldp_c[3] = { 0 }, ldp_r[4] = { 0 };
	bool ldp_poly_cylinder = false;
	std::vector<int> cidx;
    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        if(name == "d") {
            path_string = a->Value();
            continue;
        }
		if (name == "ldp_poly"){
			ldp_poly = a->IntValue();
			continue;
		}
		if (name == "ldp_3dx"){
			ldp_c[0] = a->DoubleValue();
			continue;
		}
		if (name == "ldp_3dy"){
			ldp_c[1] = a->DoubleValue();
			continue;
		}
		if (name == "ldp_3dz"){
			ldp_c[2] = a->DoubleValue();
			continue;
		}
		if (name == "ldp_3drx"){
			ldp_r[0] = a->DoubleValue();
			continue;
		}
		if (name == "ldp_3dry"){
			ldp_r[1] = a->DoubleValue();
			continue;
		}
		if (name == "ldp_3drz"){
			ldp_r[2] = a->DoubleValue();
			continue;
		}
		if (name == "ldp_3drw"){
			ldp_r[3] = a->DoubleValue();
			continue;
		}
		if (name == "ldp_cylinder_dir")
		{
			ldp_poly_cylinder = !!a->IntValue();
			continue;
		}
		if (name == "ldp_corner"){
			std::stringstream stm(a->Value());
			while (!stm.eof()){
				int id = 0;
				stm >> id;
				cidx.push_back(id);
			}
		}
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }
    PathPtr path = PathPtr(new Path(style().path, path_string.c_str()));
	path->ldp_poly_id = ldp_poly;
	path->ldp_corner_ids = cidx;
	path->ldp_poly_cylinder_dir = ldp_poly_cylinder;
	for (int k = 0; k < 3; k++)
		path->ldp_poly_3dCenter[k] = ldp_c[k];
	for (int k = 0; k < 4; k++)
		path->ldp_poly_3dRot[k] = ldp_r[k];
    if (!path->isEmpty()) {
        NodePtr shape = createShape(path, style());
        
        return style_stack.popAndReturn(decorateNode(shape));
    }

    return style_stack.popAndReturn(NodePtr());
}

NodePtr SVGParser::parseLdpPolyGroup(TiXmlElement* elem)
{
	style_stack.pushChild();

	string color_str, cmd_str;

	for (TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
		string name(a->Name());
		if (name == "color") {
			color_str = a->Value();
			continue;
		}
		if (name == "cmd"){
			cmd_str = a->Value();
			continue;
		}
	}
	LdpPolyGroupPtr path(new LdpPolyGroup());
	float3 c;
	if (3 == sscanf_s(color_str.c_str(), "%f %f %f", &c[0], &c[1], &c[2]))
		path->color = c;
	std::stringstream stm(cmd_str);
	while (!stm.eof()){
		int p = 0;
		stm >> p;
		path->cmds.push_back(p);
	}
	if (path->cmds.size()) {
		ShapePtr shape = ShapePtr(new Shape());
		shape->ldpPoly = path;
		return style_stack.popAndReturn(decorateNode(shape));
	}

	return style_stack.popAndReturn(NodePtr());
}

NodePtr SVGParser::parseText(TiXmlElement* elem)
{
    style_stack.pushChild();

    float x = 0, y = 0, font_size=0;

    TiXmlNode* child = elem->FirstChild();
	std::string textValue = "";
    if (child) {
        if (child->Type() == TiXmlNode::TEXT) {
			textValue = child->Value();
        }
    }

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        if(name == "x") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, x);
            continue;
        }
        if(name == "y") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, y);
            continue;
        }
        if(name == "text-anchor") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, x);
            continue;
        }
		if (name == "font-size"){
			string s = a->Value();
			const char* ss = s.c_str();
			font_size = atof(ss);
			continue;
		}
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }


	TextPtr text = TextPtr(new Text(style().font_family.c_str(),  
		textValue.c_str(), font_size, style().matrix));

    return style_stack.popAndReturn(text);
}

NodePtr SVGParser::parseGroup(TiXmlElement *elem)
{
    style_stack.pushChild();

	std::string ldp_layer_name = "";
    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
		if (a->Name() == std::string("ldp_layer_name"))
			ldp_layer_name = a->Value();
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }

    GroupPtr group = GroupPtr(new Group);
	group->ldp_layer_name = ldp_layer_name;
    for(TiXmlElement *subelem=elem->FirstChildElement(); subelem; subelem = subelem->NextSiblingElement()) {
        NodePtr node = parseNode(subelem);
        if (node) {
            group->push_back(node);
        }
    }
    if (style().opacity != 1) {
        // The group opacity property is complex to implement; it would need a temporary buffer.
        // http://www.w3.org/TR/SVG11/masking.html#OpacityProperty
        printf("WARNING: non-1.0 group opacity is NOT properly supported\n");
    }
    
    return style_stack.popAndReturn(decorateNode(group));
}

NodePtr SVGParser::parseView(TiXmlElement *elem)
{
    style_stack.pushChild();

    float x = 0, y = 0, width = 1, height = 1;
    float4 viewbox = float4(0, 0, 0, 0);

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());

        if (name == "width") {
            /// http://www.w3.org/TR/SVG/struct.html#SVGElementWidthAttribute
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseLength(ss, width);
            continue;
        }
        if (name == "height") {
            /// http://www.w3.org/TR/SVG/struct.html#SVGElementHeightAttribute
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseLength(ss, height);
            continue;
        }
        if (name == "x") {
            /// http://www.w3.org/TR/SVG/struct.html#SVGElementXAttribute
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, x);
            continue;
        }
        if (name == "y") {
            /// http://www.w3.org/TR/SVG/struct.html#SVGElementYAttribute
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, y);
            continue;
        }
        if (name == "version") {
            // http://www.w3.org/TR/SVG/struct.html#SVGElementVersionAttribute
        }
        if (name == "baseProfile") {
            // http://www.w3.org/TR/SVG/struct.html#SVGElementBaseProfileAttribute
        }
        if (name == "viewBox") {
            string s = a->Value();
            s = RemoveCommas(s);
            const char* ss = s.c_str();
            parseViewBox(ss, viewbox);
        }
    }

    const float x2 = x, y2 = y, w2 = width, h2 = height,
                x1 = viewbox.x, y1 = viewbox.y, w1 = viewbox.z, h1 = viewbox.w;

    // Map viewbox to [x..x+width,y..y+width]
    if (w1 != 0 && h1 != 0) {
        style().matrix = double3x3(w2/w1, 0, -w2*x1/w1 + x2,
                                 0, h2/h1, -h2*y1/h1 + y2,
                                 0, 0, 1);
    } else {
        // We were either given bogus data or the viewBox was not specified
        style().matrix = double3x3(1, 0, x2,
                                 0, 1, y2,
                                 0, 0, 1);
    }

    if (verbose) {
        std::cout << "matrix = " << style().matrix << std::endl;
    }

    NodePtr group = parseGroup(elem);
    
    return style_stack.popAndReturn(decorateNode(group));
}

int base64Decode(unsigned char *s)
{
    static unsigned char base64_to_ascii[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static unsigned char ascii_to_base64[256];
    if (!ascii_to_base64[int('A')]) {
        memset(ascii_to_base64, -1, sizeof(ascii_to_base64)); // Mark characters not in the table as such
        for (int i = 0; i < 64; i++) {
            ascii_to_base64[base64_to_ascii[i]] = i;
        }
    }

    // Base 64 is 6 bits; base 256 is 8
    int idx = 0;
    for (int i = 0; s[i] && s[i] != '='; i++) {
        if (ascii_to_base64[s[i]] != 0xff) {
            int bit_idx = 6 * idx++;
            unsigned char val = ascii_to_base64[s[i]] << 2;
            s[bit_idx/8] = (s[bit_idx/8] & ~(0xfc >> (bit_idx%8))) | (val >> (bit_idx%8));
            s[1 + bit_idx/8] = (s[1 + bit_idx/8] & ~(0xfc << (8 - (bit_idx%8)))) | (val << (8 - (bit_idx%8)));
        }
    }

    // Always round down because base64 overshoots
    return idx * 6/8;
}

typedef shared_ptr<class ImageProvider> ImageProviderPtr;
class ImageProvider
{
protected:
    ImageProvider() { }
    virtual RectBounds getViewBox() = 0;
    virtual bool deferAspectRatio() = 0;
    virtual NodePtr getBaseImage() = 0;

public:
    static ImageProviderPtr FromFile(const char* filename);
    virtual ~ImageProvider() { }
    
    // http://www.w3.org/TR/SVG/struct.html#ImageElement
    // The "viewport" is the section, in path space, to fill
    // The "view box" is the section within the image to fill the viewport with
    void setViewport(RectBounds viewport_, const char *preserve_aspect_ratio)
    {
        viewport = viewport_;
        view_box = getViewBox();
        view_box_to_viewport = double3x3(1,0,0, 0,1,0, 0,0,1);

        if (view_box.isValid()) {
            parsePreserveAspectRatio(preserve_aspect_ratio);
        } else {
            // Where no value is readily available (e.g. an SVG file with no 'viewBox' 
            // attribute on the outermost 'svg' element) the preserveAspectRatio attribute 
            // is ignored, and only the translate due to the 'x' & 'y' attributes of the 
            // viewport is used to display the content.
            view_box_to_viewport = double3x3(1, 0, viewport.x,
                                             0, 1, viewport.y,
                                             0, 0, 1);
        }
    }

    void parsePreserveAspectRatio(const char *s)
    {
        if (!strncmp(s, "defer", sizeof("defer")-1)) {
            s += sizeof("defer")-1;
            if (deferAspectRatio()) {
                return;
            }
        }

        char align[9];
        char meetOrSlice[9];
#if defined(_MSC_VER) && _MSC_VER >= 1400
        int count = sscanf_s(s, " %s %s ", align, sizeof(align), meetOrSlice, sizeof(meetOrSlice));
#else
        int count = sscanf(s, " %s %s ", align, meetOrSlice);
#endif

        if (count < 1 || !strcmp(align, "none")) {
            // just stretch the view_box to fill the whole viewport
            float2 scale(viewport.width()/view_box.width(),
                         viewport.height()/view_box.height());
            view_box_to_viewport = double3x3(scale.x, 0, viewport.x - scale.x*view_box.x,
                                             0, scale.y, viewport.y - scale.y*view_box.y,
                                             0, 0, 1);
            return;
        }

        // These need to be called in this order
        parseScale(meetOrSlice);
        parseAlign(align);
    }

    // Must be called first; can't preserve align
    void parseScale(const char *s)
    {
        float2 scale(viewport.width()/view_box.width(),
                     viewport.height()/view_box.height());
        
        if (!strcmp(s, "slice")) {
            scale = float2(max(scale.x, scale.y));
        } else { // if (!strcmp(s, "meet")) {
            scale = float2(min(scale.x, scale.y));
        }

        view_box_to_viewport[0].x *= scale.x;
        view_box_to_viewport[1].y *= scale.y;
    }

    void parseAlign(const char *s)
    {
        RectBounds current_view_box = view_box.transform(view_box_to_viewport);
        assert(current_view_box.isValid());

        // The (x,y) align is any permutation of [min,mid,max] x [min,mid,max] (i.e xMidYMax)
        char align[2][4];
#if defined(_MSC_VER) && _MSC_VER >= 1400
        int count = sscanf_s(s, "x%[^Y]Y%s", align[0], sizeof(align[0]), align[1], sizeof(align[1]));
#else
        int count = sscanf(s, "x%[^Y]Y%s", align[0], align[1]);
#endif

        if (count != 2) {
            printf("parseAlign: malformed align\n");
            return;
        }

        // Determine how far to move both x and y
        float2 current_position, destination_position;
        for (int i = 0; i < 2; i++) {
            if (!strcmp(align[i], "Min")) {
                current_position[i] = current_view_box[i];
                destination_position[i] = viewport[i];
            }
            else if (!strcmp(align[i], "Mid")) {
                current_position[i] = (current_view_box[i] + current_view_box[i+2]) / 2;
                destination_position[i] = (viewport[i] + viewport[i+2]) / 2;
            }
            else { // if (!strcmp(align[i], "Max")) {
                current_position[i] = current_view_box[i+2];
                destination_position[i] = viewport[i+2];
            }
        }

        view_box_to_viewport[0].z += destination_position.x - current_position.x;
        view_box_to_viewport[1].z += destination_position.y - current_position.y;
    }

    NodePtr getImage()
    {
        NodePtr image = TransformPtr(new Transform(getBaseImage(), view_box_to_viewport));
        RectBounds box = viewport;
        if (view_box.isValid()) {
            box &= view_box.transform(view_box_to_viewport);
        }
        return ViewBoxPtr(new ViewBox(image, box));
    }

protected:
    double3x3 view_box_to_viewport;
    RectBounds viewport;
    RectBounds view_box;

};

#include "stb/stb_image.h"
typedef shared_ptr<class RasterImageProvider> RasterImageProviderPtr;
class RasterImageProvider : public ImageProvider
{
public:
    static RasterImageProviderPtr FromFile(const char *filename)
    {
        int original_bpp;
        RasterImageProviderPtr provider(new RasterImageProvider);
        // Force it to resample to 4 bytes/pixel
        provider->image->pixels = (RasterImage::Pixel *)
            stbi_load(filename, &provider->image->width, &provider->image->height, &original_bpp, 4);

        if (provider->image->pixels) {
            assert(provider->image->width >= 1);
            assert(provider->image->height >= 1);
            // Pre-multiply the image by its alpha channel.
            int num_pixels = provider->image->width*provider->image->height;
            RasterImage::Pixel *pixels = provider->image->pixels;
            for (int i=0; i<num_pixels; i++) {
                float r = pixels[i].r / 255.0f,
                    g = pixels[i].g / 255.0f,
                    b = pixels[i].b / 255.0f,
                    a = pixels[i].a / 255.0f;
                r *= a;
                g *= a;
                b *= a;
                pixels[i].r = GLubyte(r * 255 + 0.5);
                pixels[i].g = GLubyte(g * 255 + 0.5);
                pixels[i].b = GLubyte(b * 255 + 0.5);
            }
            return provider;
        } else {
            return RasterImageProviderPtr();
        }
    }

    static RasterImageProviderPtr FromString(const char *s)
    {
        size_t size = 1 + strlen(s);
        unsigned char *data = new unsigned char[size];
        char img_type[32];
#if defined(_MSC_VER) && _MSC_VER >= 1400
        int matches = sscanf_s(s, " data : image/%[^\\;] ; base64, %[a-zA-Z0-9+/= \\n\\t]", img_type, sizeof(img_type), data, size);
#else
        int matches = sscanf(s, " data : image/%[^\\;] ; base64, %[a-zA-Z0-9+/= \\n\\t]", img_type, data);
#endif
        if (matches == 2) {
            int buf_size = base64Decode(data);
            RasterImageProviderPtr provider(new RasterImageProvider);
            int bpp;
            provider->image->pixels = (RasterImage::Pixel *)
                stbi_load_from_memory((stbi_uc *)data, buf_size, &provider->image->width, &provider->image->height, &bpp, 4);
            delete[] data;

            if (provider->image->pixels) {
                assert(bpp == 4);
                return provider;
            } else {
                printf("STBI: stbi_load_from_memory failure reason: %s\n", stbi_failure_reason());
            }
        }
        
        return RasterImageProviderPtr();
    }


protected:
    virtual RectBounds getViewBox()
    {
        return RectBounds(0,0, image->width,image->height);
    }

    virtual bool deferAspectRatio()
    {
        // Images don't say how to clip themselves to the view box - only SVG files
        return false;
    }

    virtual NodePtr getBaseImage()
    {
        // Transform the corners of the image to (0,0), (1,1)
        double3x3 normalize(image->width, 0, 0,
                            0, image->height, 0,
                            0, 0, 1);

        // Build a unit rect
        vector<char> cmds;
        vector<float> coords;

        cmds.push_back('M');
        coords.push_back(0);
        coords.push_back(0);
        cmds.push_back('h');
        coords.push_back(1);
        cmds.push_back('v');
        coords.push_back(1);
        cmds.push_back('h');
        coords.push_back(-1);
        cmds.push_back('Z');

        // The resource referenced by the 'image' element represents a separate document which 
        // generates its own parse tree and document object model (if the resource is XML). 
        // Thus, there is no inheritance of properties into the image.
        PathPtr path(new Path(PathStyle(), cmds, coords));
        ShapePtr shape(new Shape(path, PaintPtr(new ImagePaint(image)), PaintPtr()));
        return TransformPtr(new Transform(shape, normalize));
    }

private:
    RasterImageProvider() : image(new RasterImage()) { }
    RasterImagePtr image;
};

typedef shared_ptr<class SvgImageProvider> SvgImageProviderPtr;
class SvgImageProvider : public ImageProvider
{
public:
    static SvgImageProviderPtr FromFile(const char *filename)
    {
        SVGParser parser(filename);
        if (parser.scene) {
            return SvgImageProviderPtr(new SvgImageProvider(parser.scene));
        }

        return SvgImageProviderPtr();
    }

protected:
    virtual RectBounds getViewBox()
    {
        return scene->view_box;
    }

    virtual bool deferAspectRatio()
    {
        if (scene->preserve_aspect_ratio == "") {
            return false;
        }
        if (deferring_aspect_ratio) {
            // The svg file marked its preserve aspect ratio as defer; might as well give up
            return false;
        }

        deferring_aspect_ratio = true;
        parsePreserveAspectRatio(scene->preserve_aspect_ratio.c_str());
        deferring_aspect_ratio = false;

        return true;
    }

    virtual NodePtr getBaseImage()
    {
        return scene;
    }

private:
    SvgImageProvider(SvgScenePtr scene_) : scene(scene_), deferring_aspect_ratio(false) { }
    SvgScenePtr scene;
    bool deferring_aspect_ratio;
};

ImageProviderPtr ImageProvider::FromFile(const char* filename)
{
    RasterImageProviderPtr rip = RasterImageProvider::FromFile(filename);
    if (rip) {
        return rip;
    }

    SvgImageProviderPtr sip = SvgImageProvider::FromFile(filename);
    if (sip) {
        return sip;
    }

    return ImageProviderPtr();
}

static int hexdigit(char c)
{
    if (c >= '0' && c <= '9') {
        return c-'0';
    }
    if (c >= 'A' && c <= 'F') {
        return c-'A'+10;
    }
    if (c >= 'a' && c <= 'f') {
        return c-'a'+10;
    }
    return -1000; // error
}

// http://en.wikipedia.org/wiki/Percent-encoding
// http://tools.ietf.org/html/rfc3986
static string percent_decode(const string input)
{
    string output;

    for (size_t i=0; i<input.size(); i++) {
        if (input[i] == '%') {
            if (i < input.size()+2) {
                int c = hexdigit(input[i+1])*16 + hexdigit(input[i+2]);
                if (c >= 0) {
                    output.push_back(c);
                    i+=2;
                    continue;
                }
            }
        }
        output.push_back(input[i]);
    }
    return output;
}

NodePtr SVGParser::parseImage(TiXmlElement *elem)
{
    float x = 0, y = 0, width = 0, height = 0;
    string preserve_aspect_ratio = "none";
    ImageProviderPtr image_provider;

    style_stack.pushChild();

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        string value(a->Value());
        const char* ss = value.c_str();

        if (name == "x") {
            ss = parseLength(ss, x);
        } else if (name =="y") {
            ss = parseLength(ss, y);
        } else if (name =="width") {
            ss = parseLength(ss, width);
        } else if (name =="height") {
            ss = parseLength(ss, height);
        } else if (name == "preserveAspectRatio") {
            preserve_aspect_ratio = value;
        } else if (name == "xlink:href") {
            string uri = percent_decode(value);
            const char* ss = uri.c_str();
            if (!strncmp(ss, "file:///", sizeof("file:///")-1)) {
                ss += sizeof("file:///")-1;
            }

            #define MAX_PATH 260 // from WinDef.h
            if (strlen(root_dir.c_str()) + strlen(ss) < MAX_PATH) {
                char relative[1+MAX_PATH];
#if defined(_MSC_VER) && _MSC_VER >= 1400
                strcpy_s(relative, MAX_PATH, root_dir.c_str());
                strcat_s(relative, MAX_PATH, ss);
#else
                strcpy(relative, root_dir.c_str());
                strcat(relative, ss);
#endif
                if ((image_provider = ImageProvider::FromFile(relative)) ||
                    (image_provider = ImageProvider::FromFile(ss))) {
                    continue;
                }
            }

            // http://www.w3.org/TR/SVGMobile12/linking.html
            // Yes, but the content within the data: IRI reference must be a raster image.
            if (!(image_provider = RasterImageProvider::FromString(ss))) {
                printf("STBI: stbi_load failure reason for %s: %s\n", ss, stbi_failure_reason());
                printf("warning : could not load image: %s\n", ss);
               // assert(0);
            }
        } else {
            parseGenericShapeProperty(a, elem);
        }
    }

    if (!image_provider) {
        printf("warning : invalid <image>; drawing nothing\n");
        return style_stack.popAndReturn(NodePtr());
    }

    image_provider->setViewport(RectBounds(x,y, x+width,y+height), preserve_aspect_ratio.c_str());
    return style_stack.popAndReturn(decorateNode(image_provider->getImage()));
}

NodePtr SVGParser::parseSwitch(TiXmlElement *elem)
{
    style_stack.pushChild();

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }

    GroupPtr group = GroupPtr(new Group);
    // XXX hack to groak SVG files exported from Adobe Illustrator 10
    // http://www.w3.org/TR/SVG/struct.html#SwitchElement
    for(TiXmlElement *subelem=elem->FirstChildElement(); subelem; subelem = subelem->NextSiblingElement()) {
        NodePtr node = parseNode(subelem);
        if (node) {
            group->push_back(node);
        }
    }
    
    return style_stack.popAndReturn(decorateNode(group));
}

// Very simplistic Cascading Style Sheet parsing
void SVGParser::loadStyles(TiXmlNode* pParent)
{
	TiXmlNode* pChild;
	TiXmlText* pText;

	for ( pChild = pParent->FirstChild(); pChild != 0; pChild = pChild->NextSibling()) {
        int t = pChild->Type();

        switch (t) {
        case TiXmlNode::TEXT:
            pText = pChild->ToText();
            const char *t = pText->Value();
            while (t && *t!='\0') {
                // Are we beginning a C-style comment?
                if (t[0] == '/' && t[1] == '*') {
                    t+=2;
                    while (*t!='\0') {
                        // Are we at the end of a C-style comment?
                        if (t[0]=='*' && t[1]=='/') {
                            // Comment done; resume normal processing.
                            t+=2;
                            break;
                        }
                        t++;
                    }
                } else {
                    // Do we see the start of a simple style like ".st0 {"
                    char style_name[41];
                    int count = 0;
                    char expecting_curly[2] = { 0, 0 };
                    int rc = sscanf(t, " .%40[A-Za-z0-9] %[{] %n",
                        style_name, expecting_curly, &count);
                    if (rc == 2) {
                        // Yes, look for the style value like " Fill:red }"
                        t += count;
                        char style_value[801];
                        int rc = sscanf(t, " %800[^}]%[}] %n",
                            style_value, expecting_curly, &count);
                        if (rc == 2) {
                            style_map[style_name] = style_value;
                            t += count;
                        }
                    } else {
                        // Skip characters we don't recognize
                        t++;
                    }
                }
            }
        }
	}
}

GradientStop SVGParser::parseGradientStop(TiXmlElement *elem)
{
    GradientStop rv;
    style_stack.pushChild();

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        string value(a->Value());
        const char* ss = value.c_str();

        if (name == "offset") {
            parseOffset(ss, rv.offset);
            continue;
        }
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }
    rv.color = style().stop_color;
    return style_stack.popAndReturn(rv);
}

void SVGParser::parseGradient(GradientPtr gradient, TiXmlElement *elem)
{
    style_stack.pushChild();
    string id;

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        string value(a->Value());
        const char* ss = value.c_str();
        float v;
        switch (gradient->gradient_type) {
        case Gradient::LINEAR:
            if(name == "x1") {
                ss = parseCoordinate(ss, v, true);
                gradient->setX1(v);
                continue;
            }
            if(name == "y1") {
                ss = parseCoordinate(ss, v, true);
                gradient->setY1(v);
                continue;
            }
            if(name == "x2") {
                ss = parseCoordinate(ss, v, true);
                gradient->setX2(v);
                continue;
            }
            if(name == "y2") {
                ss = parseCoordinate(ss, v, true);
                gradient->setY2(v);
                continue;
            }
            break;
        case Gradient::RADIAL:
            if(name == "cx") {
                ss = parseCoordinate(ss, v, true);
                gradient->setCX(v);
                continue;
            }
            if(name == "cy") {
                ss = parseCoordinate(ss, v, true);
                gradient->setCY(v);
                continue;
            }
            if(name == "fx") {
                ss = parseCoordinate(ss, v, true);
                gradient->setFX(v);
                continue;
            }
            if(name == "fy") {
                ss = parseCoordinate(ss, v, true);
                gradient->setFY(v);
                continue;
            }
            if(name == "r") {
                ss = parseCoordinate(ss, v, true);
                gradient->setR(v);
                continue;
            }
            break;
        default:
            assert(!"bogus gradient type");
            break;
        }
        if(name == "id") {
            id = value;
            continue;
        }
        if(name == "gradientUnits") {
            if (value == "userSpaceOnUse") {
                gradient->setUnits(USER_SPACE_ON_USE);
            } else
            if (value == "objectBoundingBox") {
                gradient->setUnits(OBJECT_BOUNDING_BOX);
            }
            continue;
        }
        if(name == "gradientTransform") {
            double3x3 matrix = identity;
            value = RemoveCommas(value);
            parseTransform(value.c_str(), matrix);
            gradient->setTransform(matrix);
        }
        if(name == "spreadMethod") {
            if (value == "pad") {
                gradient->setSpreadMethod(PAD);
            } else
            if (value == "reflect") {
                gradient->setSpreadMethod(REFLECT);
            } else
            if (value == "repeat") {
                gradient->setSpreadMethod(REPEAT);
            }
            continue;
        }
        if(name == "xlink:href") {
            value = percent_decode(value);
            if (value.size()>0 && value[0] == '#') {
                value.erase(0, 1);
                gradient->setHref(value);
            } else {
                printf("malformed xlink:href: %s\n", value.c_str());
            }
            continue;
        }
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }

    for(TiXmlElement *subelem=elem->FirstChildElement(); subelem; subelem = subelem->NextSiblingElement()) {
        string name(subelem->Value());

        if (name == "stop") {
            if (!gradient->gradient_stops) {
                gradient->gradient_stops = GradientStopsPtr(new GradientStops());
            }
            GradientStop stop = parseGradientStop(subelem);
            gradient->gradient_stops->stop_array.push_back(stop);
        }
    }

    if (id.empty()) {
        // Skip null ids?
    } else {
        gradient_map[id] = gradient;
        paint_server_map[id] = gradient;
    }

    style_stack.pop();
}

void SVGParser::parseSolidColor(TiXmlElement *elem)
{
    SolidColorPtr solid_color(new SolidColor());
    string id;

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        string value(a->Value());
        const char* ss = value.c_str();

        if(name == "id") {
            id = value;
            continue;
        }
        if(name == "solid-color") {
            parseColor(ss, solid_color->color);
            continue;
        }
        if(name == "solid-opacity") {
            float opacity = solid_color->color.a;
            parseOpacity(ss, opacity);
            solid_color->color.a = opacity;
            continue;
        }
    }

    paint_server_map[id] = solid_color;
}

void SVGParser::parseLinearGradient(TiXmlElement *elem)
{
    LinearGradientPtr gradient(new LinearGradient());

    parseGradient(gradient, elem);
}

void SVGParser::parseRadialGradient(TiXmlElement *elem)
{
    RadialGradientPtr gradient(new RadialGradient());

    parseGradient(gradient, elem);
}

void SVGParser::parseClipPath(TiXmlElement *elem)
{
    string id;
    string clipPathId;

    ClipInfoPtr clip_path = ClipInfoPtr(new ClipInfo);

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        string value(a->Value());

        if (name == "id") {
            id = value;
            continue;
        }
        if (name == "clipPathUnits") {
            if (value == "userSpaceOnUse") {
                clip_path->units = ClipInfo::USER_SPACE_ON_USE;
            } else {
                if (value != "objectBoundingBox") {
                    printf("warning : bogus clipPathUnits %s\n", value.c_str());
                }
                clip_path->units = ClipInfo::OBJECT_BOUNDING_BOX;
            }
            continue;
        }
        if (name == "clip-path") {
            clip_path->clip_info = parseClipPathReference(value.c_str());
            continue;
        }
        if (name == "transform") {
            parseTransform(value.c_str(), clip_path->transform);
            continue;
        }
        if (name == "clip-merge") {
            if (value == "sumWindingNumbers") {
                clip_path->clip_merge = SUM_WINDING_NUMBERS;
            } else if (value == "sumWindingNumbersMod2") {
                clip_path->clip_merge = SUM_WINDING_NUMBERS_MOD_2;
            } else {
                if (value != "clipCoverageUnion") {
                    printf("warning : %s : bogus value for clip_merge", value.c_str());
                }
                clip_path->clip_merge = CLIP_COVERAGE_UNION;
            }
            continue;
        }

        printf("bogus clip path attribute %s\n", name.c_str());
    }

    for(TiXmlElement *subelem=elem->FirstChildElement(); subelem; subelem = subelem->NextSiblingElement()) {
        string name(subelem->Value());
        NodePtr node = parseNode(subelem);
        do {
            if (isShapeTextOrPath(node)) {
                clip_path->group->push_back(NodePtr(node));
                break;
            } 

            ClipPtr clip = dynamic_pointer_cast<Clip>(node);
            if (clip) {
                printf("warning: clip paths on elements of clip paths are not supported - just clip paths on other whole clip paths\n");
                node = clip->node;
            } else {
                printf("warning: bogus clip path element\n");
                break;
            }
        } while (node);
    }

    if (!id.empty()) {
        clipPath_map[id] = clip_path;
    }
}

void SVGParser::parsePattern(TiXmlElement *elem)
{
}


void SVGParser::parseDefs(TiXmlElement *elem)
{
    for(TiXmlElement *subelem=elem->FirstChildElement(); subelem; subelem = subelem->NextSiblingElement()) {
        string name(subelem->Value());

        if (name == "linearGradient") {
            parseLinearGradient(subelem);
            continue;
        }
        if (name == "radialGradient") {
            parseRadialGradient(subelem);
            continue;
        }
        if (name == "pattern") {
            parsePattern(subelem);
            continue;
        }
        if (name == "solidColor") {
            parseSolidColor(subelem);
            continue;
        }
        (void) parseNode(subelem);
    }
}

NodePtr SVGParser::parseUse(TiXmlElement *elem)
{
    style_stack.pushChild();

    UsePtr use(new Use());

    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        string value(a->Value());

        if(name == "x") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, use->x);
            continue;
        }
        if(name == "y") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseCoordinate(ss, use->y);
            continue;
        }
        if(name == "width") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseLength(ss, use->width);
            continue;
        }
        if(name == "height") {
            string s = a->Value();
            const char* ss = s.c_str();
            ss = parseLength(ss, use->height);
            continue;
        }
        if(name == "xlink:href") {
            value = percent_decode(value);
            if (value.size()>0 && value[0] == '#') {
                value.erase(0, 1);
                use->href = value;
            } else {
                printf("malformed xlink:href: %s\n", value.c_str());
            }
            continue;
        }
        bool got_one = parseGenericShapeProperty(a, elem);
        if (got_one) {
            continue;
        }
    }

    double3x3 trans = double3x3(1,0,use->x,
                                0,1,use->y,
                                0,0,1);
    style().matrix = mul(trans, style().matrix);
    use->style_stack = style_stack;

    return style_stack.popAndReturn(decorateNode(use));
}

void SVGParser::parseSymbol(TiXmlElement *elem)
{
    for(TiXmlAttribute* a = elem->FirstAttribute(); a; a = a->Next()) {
        string name(a->Name());
        string value(a->Value());

        if(name == "id") {
            string s = a->Value();
            use_map[s] = elem;
        }
    }
}

NodePtr SVGParser::parseNode(TiXmlElement *elem)
{
    string name(elem->Value());

    if (name == "g") {
        NodePtr group = parseGroup(elem);
        return group;
    }
    if (name == "svg") {
        NodePtr group = parseView(elem);
        return group;
    }
    if (name == "symbol") {
        parseSymbol(elem);
        return NodePtr();
    }
    if (name == "defs") {
        parseDefs(elem);
        return NodePtr();
    }
    if (name == "line") {
        NodePtr shape = parseLine(elem);
        return shape;
    }
    if (name == "rect") {
        NodePtr shape = parseRect(elem);
        return shape;
    }
    if (name == "circle") {
        NodePtr shape = parseCircle(elem);
        return shape;
    }
    if (name == "ellipse") {
        NodePtr shape = parseEllipse(elem);
        return shape;
    }
    if (name == "polygon") {
        NodePtr shape = parsePolygon(elem);
        return shape;
    }
    if (name == "polyline") {
        NodePtr shape = parsePolyline(elem);
        return shape;
    }
    if (name == "path") {
        NodePtr shape = parsePath(elem);
        return shape;
	}
	if (name == "ldp_poly_group") {
		NodePtr shape = parseLdpPolyGroup(elem);
		return shape;
	}
    if (name == "image") {
        NodePtr image = parseImage(elem);
        return image;
    }
    if (name == "text") {
        NodePtr text = parseText(elem);
        return text;
    }
    if (name == "foreignObject") {
        return NodePtr();
    }
    if (name == "switch") {
        NodePtr switch_group = parseSwitch(elem);
        return switch_group;
    }
    if (name == "style") {
        loadStyles(elem);
        return NodePtr();
    }

    // Normally these would be in a <defs> but they don't have to be.
    if (name == "linearGradient") {
        parseLinearGradient(elem);
        return NodePtr();
    }
    if (name == "radialGradient") {
        parseRadialGradient(elem);
        return NodePtr();
    }
    if (name == "pattern") {
        parsePattern(elem);
        return NodePtr();
    }
    if (name == "solidColor") {
        parseSolidColor(elem);
        return NodePtr();
    }
    if (name == "clipPath") {
        parseClipPath(elem);
        return NodePtr();
    }

    if (name == "use") {
        NodePtr use = parseUse(elem);
        return use;
    }

    return NodePtr();
}

// When resolving <use>, treat symbol like <group>.
NodePtr SVGParser::parseUsedNode(TiXmlElement *elem, const UsePtr &use)
{
    StyleInfoStack old_stack = style_stack;
    style_stack = use->style_stack;
    NodePtr ret;

    string name(elem->Value());

    if (name == "symbol") {
        NodePtr group = parseGroup(elem);
        ret = group;
    } else {
        ret = parseNode(elem);
    }

    style_stack = old_stack;

    return ret;
}

// Do a depth first search to find the first <svg> tag in the document.
TiXmlElement* SVGParser::findSVG(TiXmlNode* pParent, unsigned int indent)
{
	if ( !pParent ) return NULL;

	TiXmlNode* pChild;
	TiXmlText* pText;
	int t = pParent->Type();
    if (verbose) {
	    printf( "%s", getIndent(indent));
    }
	int num;
    string name;

	switch ( t ) {
    case TiXmlNode::DOCUMENT:
        if (verbose) {
            printf( "Document" );
        }
        break;

    case TiXmlNode::ELEMENT:
        name = pParent->Value();
        // Allow styles to be specified prior to the SVG node.
        if (name == "style") {
            loadStyles(pParent);
        }
        if (name == "svg") {
            // Found an SVG element; done!
            return (TiXmlElement*)(pParent);
        }
        if (verbose) {
            printf( "Element [%s]", pParent->Value() );
            num=dump_attribs_to_stdout(pParent->ToElement(), indent+1);
            switch(num)
            {
            case 0:  printf( " (No attributes)"); break;
            case 1:  printf( "%s1 attribute", getIndentAlt(indent)); break;
            default: printf( "%s%d attributes", getIndentAlt(indent), num); break;
            }
        }
        break;

    case TiXmlNode::COMMENT:
        if (verbose) {
            printf( "Comment: [%s]", pParent->Value());
        }
        break;

    case TiXmlNode::UNKNOWN:
        if (verbose) {
            printf( "Unknown" );
        }
        break;

    case TiXmlNode::TEXT:
        if (verbose) {
            pText = pParent->ToText();
            printf( "Text: [%s]", pText->Value() );
        }
        break;

    case TiXmlNode::DECLARATION:
        if (verbose) {
            printf( "Declaration" );
        }
        break;
	default:
		break;
	}
    if (verbose) {
	    printf( "\n" );
    }
	for ( pChild = pParent->FirstChild(); pChild != 0; pChild = pChild->NextSibling()) 
	{
		TiXmlElement* result = findSVG( pChild, indent+1 );
        if (result) {
            return result;
        }
	}
    return NULL;
}

SvgScenePtr SVGParser::parseScene(TiXmlElement *svg)
{
    style_stack.pushChild();

    SvgScenePtr scene = SvgScenePtr(new SvgScene);
    if (scene) {
        for(TiXmlAttribute* a = svg->FirstAttribute(); a; a = a->Next()) {
            string name(a->Name());
            if (name == "width") {
                scene->width = atoi(a->Value());
                continue;
            }
            if (name == "height") {
                scene->height = atoi(a->Value());
                continue;
            }
            if (name == "viewBox") {
                float4 view_box;
                parseViewBox(a->Value(), view_box);
                scene->view_box = view_box;
                continue;
            }
            if (name == "preserveAspectRatio") {
                scene->preserve_aspect_ratio = a->Value();
                continue;
            }
			if (name == "ldp_pixel2meter"){
				scene->ldp_pixel2meter = a->DoubleValue();
				continue;
			}
            bool got_one = parseGenericShapeProperty(a, svg);
            if (got_one) {
                continue;
            }
        }
        for(TiXmlElement *elem=svg->FirstChildElement(); elem; elem = elem->NextSiblingElement()) {
            string name(elem->Value());
            NodePtr node = parseNode(elem);
            if (node) {
                scene->push_back(node);
            }
        }

        NodePtr node(scene);
        foreachUseTraversal(node, ResolveUses(this));
    }

    return style_stack.popAndReturn(scene);
}

SVGParser::SVGParser(const char *xmlFile)
    : doc(xmlFile)
{
    // Parse the XML file with TinyXML.
    bool loadOkay = doc.LoadFile();
    if (loadOkay) {

        // Compute the path to the svg's containing directory for opening images
        root_dir = xmlFile;
        size_t i = root_dir.length()-1;
        for (; i >= 0 && root_dir[i] != '/' && root_dir[i] != '\\'; i--) {
            root_dir[i] = 0;
        }

        TiXmlElement* svg = findSVG(&doc);
        if (svg) {
            scene = parseScene(svg);
        } else {
            printf("\n** XML PARSE ERROR **  Can't find the <svg> tag.\n"
                   "    (Does the document contain square brackets? TinyXml can't handle things like \"<!DOCTYPE ... []>\") ..."); 
        }
    }
}

SvgScenePtr svg_loader(const char *xmlFile)
{
    SVGParser parser(xmlFile);

     if (parser.doc.Error()) {
        printf("\n** XML PARSE ERROR **  %s line: %d col: %d ...", parser.doc.ErrorDesc(), parser.doc.ErrorRow(), parser.doc.ErrorCol());
    }

    return parser.scene;
}
