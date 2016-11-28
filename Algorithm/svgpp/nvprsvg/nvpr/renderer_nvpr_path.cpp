
/* renderer_nvpr_path.cpp - paths for use with NV_path_rendering path rendering */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include "nvpr_svg_config.h"

#if USE_NVPR

#include <GL/glew.h>
#include "nvpr_init.h"

#include "renderer_nvpr.hpp"

struct NVprPathCacheProcessor : PathSegmentProcessor {
    Path *p;

    GLuint &path;
    GLenum &fill_rule;

    vector<GLubyte> cmds;
    vector<GLfloat> coords;

    NVprPathCacheProcessor(Path *p_, GLuint &path_, GLenum &fill_rule_)
        : p(p_)
        , path(path_)
        , fill_rule(fill_rule_)
        , cmds(p->cmd.size())
        , coords(p->coord.size())
    {
        cmds.clear();
        coords.clear();
    }

    void beginPath(PathPtr p) {
        switch (p->style.fill_rule) {
        case PathStyle::EVEN_ODD:
            fill_rule = GL_INVERT;
            break;
        default:
            assert(!"bogus style.fill_rule");
            break;
        case PathStyle::NON_ZERO:
            fill_rule = GL_COUNT_UP_NV;
            break;
        }
    }
    void moveTo(const float2 plist[2], size_t coord_index, char cmd) {
        cmds.push_back(GL_MOVE_TO_NV);
        coords.push_back(plist[1].x);
        coords.push_back(plist[1].y);
    }
    void lineTo(const float2 plist[2], size_t coord_index, char cmd) {
        cmds.push_back(GL_LINE_TO_NV);
        coords.push_back(plist[1].x);
        coords.push_back(plist[1].y);
    }
    void quadraticCurveTo(const float2 plist[3], size_t coord_index, char cmd) {
        cmds.push_back(GL_QUADRATIC_CURVE_TO_NV);
        coords.push_back(plist[1].x);
        coords.push_back(plist[1].y);
        coords.push_back(plist[2].x);
        coords.push_back(plist[2].y);
    }
    void cubicCurveTo(const float2 plist[4], size_t coord_index, char cmd) {
        cmds.push_back(GL_CUBIC_CURVE_TO_NV);
        coords.push_back(plist[1].x);
        coords.push_back(plist[1].y);
        coords.push_back(plist[2].x);
        coords.push_back(plist[2].y);
        coords.push_back(plist[3].x);
        coords.push_back(plist[3].y);
    }
    void arcTo(const EndPointArc &arc, size_t coord_index, char cmd) {
        if (arc.large_arc_flag) {
            if (arc.sweep_flag) {
                cmds.push_back(GL_LARGE_CCW_ARC_TO_NV);
            } else {
                cmds.push_back(GL_LARGE_CW_ARC_TO_NV);
            }
        } else {
            if (arc.sweep_flag) {
                cmds.push_back(GL_SMALL_CCW_ARC_TO_NV);
            } else {
                cmds.push_back(GL_SMALL_CW_ARC_TO_NV);
            }
        }
        coords.push_back(arc.radii.x);
        coords.push_back(arc.radii.y);
        coords.push_back(arc.x_axis_rotation);
        coords.push_back(arc.p[1].x);
        coords.push_back(arc.p[1].y);
    }
    void close(char cmd) {
        cmds.push_back(GL_CLOSE_PATH_NV);
    }
    void endPath(PathPtr p) {
        if (!path) {
            path = glGenPathsNV(1);
        }
        glPathCommandsNV(path,
                         GLsizei(cmds.size()), &cmds[0],
                         GLsizei(coords.size()), GL_FLOAT, &coords[0]);
    }
};

static GLenum lineCapConverter(const Path *path)
{
    switch (path->style.line_cap) {
    default:
        assert(!"bad line_cap");
    case PathStyle::BUTT_CAP:
        return GL_FLAT;
    case PathStyle::ROUND_CAP:
        return GL_ROUND_NV;
    case PathStyle::SQUARE_CAP:
        return GL_SQUARE_NV;
    case PathStyle::TRIANGLE_CAP:
        return GL_TRIANGULAR_NV;
    }
}

static GLenum lineJoinConverter(const Path *path)
{
    switch (path->style.line_join) {
    default:
        assert(!"bad line_join");
    case PathStyle::MITER_TRUNCATE_JOIN:
        return GL_MITER_TRUNCATE_NV;
    case PathStyle::MITER_REVERT_JOIN:
        return GL_MITER_REVERT_NV;
    case PathStyle::ROUND_JOIN:
        return GL_ROUND_NV;
    case PathStyle::BEVEL_JOIN:
        return GL_BEVEL_NV;
    case PathStyle::NONE_JOIN:
        return GL_NONE;
    }
}

void NVprPathRendererState::validate()
{
    if (valid) {
        return;
    }

    NVprPathCacheProcessor processor(owner, path, fill_rule);
    owner->processSegments(processor);
    if (owner->style.do_stroke) {
        glPathParameteriNV(path, GL_PATH_JOIN_STYLE_NV, lineJoinConverter(owner));
        glPathParameteriNV(path, GL_PATH_END_CAPS_NV, lineCapConverter(owner));
        glPathParameterfNV(path, GL_PATH_STROKE_WIDTH_NV, owner->style.stroke_width);
        glPathParameterfNV(path, GL_PATH_MITER_LIMIT_NV, owner->style.miter_limit);
        if (owner->style.dash_array.size()) {
            glPathDashArrayNV(path, GLsizei(owner->style.dash_array.size()), &owner->style.dash_array[0]);
            glPathParameteriNV(path, GL_PATH_DASH_CAPS_NV, lineCapConverter(owner));
            glPathParameterfNV(path, GL_PATH_DASH_OFFSET_NV, owner->style.dash_offset);
            glPathParameteriNV(path, GL_PATH_DASH_OFFSET_RESET_NV, owner->style.dash_phase);
        } else {
            glPathDashArrayNV(path, 0, NULL);
        }
    }
    valid = true;
}

void NVprPathRendererState::invalidate()
{
    if (path) {
        glDeletePathsNV(path, 1);
        path = 0;
    }
    valid = false;
}

NVprPathRendererState::NVprPathRendererState(RendererPtr renderer, Path *owner)
    : RendererState<Path>(renderer, owner)
    , path(0)
    , valid(false)
    , fill_cover_mode(GL_CONVEX_HULL_NV)
    , stroke_cover_mode(GL_CONVEX_HULL_NV)
{ 
}

NVprPathRendererState::~NVprPathRendererState()
{
    invalidate();
}

void NVprPathRendererState::stencilModeFill(StencilMode mode, GLuint stencil_write_mask,
                                            GLenum stencil_func, GLint stencil_ref,
                                            GLuint stencil_read_mask)
{
    GLenum stencil_fill_rule;
    switch (mode)
    {
    case COUNT_UP:
        stencil_fill_rule = GL_COUNT_UP_NV;
        break;
    default:
        assert(!"bad stencil mode");
        // Fallthrough
    case COUNT_DOWN:
        stencil_fill_rule = GL_COUNT_DOWN_NV;
        break;
    case INVERT:
        stencil_fill_rule = GL_INVERT;
        break;
    }

    stencilModeFill(stencil_fill_rule, stencil_write_mask, stencil_func, stencil_ref, stencil_read_mask);
}

void NVprPathRendererState::stencilModeFill(GLenum stencil_fill_rule, GLuint stencil_write_mask,
                                            GLenum stencil_func, GLint stencil_ref,
                                            GLuint stencil_read_mask)
{
    glPathStencilFuncNV(stencil_func, stencil_ref, stencil_read_mask);
    glStencilFillPathNV(path, stencil_fill_rule, stencil_write_mask);
}

void NVprPathRendererState::coverFill()
{
    glCoverFillPathNV(path, fill_cover_mode);
}

void NVprPathRendererState::stencilStroke(GLint replace_value, GLuint stencil_write_mask,
                                          GLenum stencil_func,
                                          GLint stencil_ref,
                                          GLuint read_mask)
{
    glPathStencilFuncNV(stencil_func, stencil_ref, read_mask);
    glStencilStrokePathNV(path, replace_value, stencil_write_mask);
}

void NVprPathRendererState::coverStroke()
{
    glCoverStrokePathNV(path, stroke_cover_mode);
}

void NVprPathRendererState::coverFillDilated(const float4x4 &mvp, float4 &dilations)
{
    coverDilated(GL_PATH_FILL_BOUNDING_BOX_NV, mvp, dilations);
}

void NVprPathRendererState::coverStrokeDilated(const float4x4 &mvp, float4 &dilations)
{
    coverDilated(GL_PATH_STROKE_BOUNDING_BOX_NV, mvp, dilations);
}

void NVprPathRendererState::coverDilated(GLenum bbox_id, const float4x4 &mvp, float4 &dilations)
{
    float bbox[4];
    glGetPathParameterfvNV(path, bbox_id, bbox);

    RectBounds bounds(bbox[0], bbox[1], bbox[2], bbox[3]);

    // XXX TODO: dilate the box

    // XXX TODO: get rid of this once the driver returns the correct bbox
    bounds = RectBounds(-1.1, -1.1, 1.1, 1.1).transform(inverse(mvp));

    glBegin(GL_QUADS); {
        glVertex2f(bounds.x, bounds.y);
        glVertex2f(bounds.x, bounds.w);
        glVertex2f(bounds.z, bounds.w);
        glVertex2f(bounds.z, bounds.y);
    } glEnd();
}

PathPtr NVprPathRendererState::getPath()
{
    return owner->shared_from_this();
}

#endif // USE_NVPR
