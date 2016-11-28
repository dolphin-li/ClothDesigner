
/* renderer_stc.cpp - rendering algorithms for "stencil, then cover" path rendering. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <vector>
using std::vector;

#include <Cg/vector/xyzw.hpp>
#include <Cg/double.hpp>
#include <Cg/vector.hpp>
using namespace Cg;

// OpenGL API
#include <GL/glew.h>
#if __APPLE__
#include <OpenGL/glext.h>
#endif

#include "scene.hpp"
#include "renderer_stc.hpp"
#include "showfps.h"

void StCRenderer::update()
{
    float w = view_width_height.x;
    float h = view_width_height.y;
    float4x4 clip_to_window = float4x4(w/2, 0, 0, w/2,
                                       0, h/2, 0, h/2,
                                       0, 0, 1, 0,
                                       0, 0, 0, 1);
    const float4x4 surface_to_clip = surfaceToClip(w, h, scene_ratio);
    surface_to_window = mul(clip_to_window, surface_to_clip);
    glMatrixLoadTransposefEXT(GL_PROJECTION, &surface_to_clip[0][0]);
}

void StCRenderer::setViewport(int x, int y, int width, int height)
{
    view_width_height = int2(width, height);
    glViewport(x, y, width, height);
    update();
}

void StCRenderer::setSceneRatio(float scene_ratio_)
{
    scene_ratio = scene_ratio_;
    update();
}

bool StCRenderer::setColorSpace(ColorSpace cs)
{
    bool old_state = render_sRGB;
    switch (cs) {
    case UNCORRECTED:
        render_sRGB = false;
        if (has_EXT_framebuffer_sRGB) {
            if (sRGB_capable) {
                glDisable(GL_FRAMEBUFFER_SRGB_EXT);
            }
        } else {
            // Only color space supported is linear.
        }
        break;
    case CORRECTED_SRGB:
        render_sRGB = true;
        if (has_EXT_framebuffer_sRGB) {
            if (sRGB_capable) {
                glEnable(GL_FRAMEBUFFER_SRGB_EXT);
            } else {
                printf("Framebuffer lacks sRGB support\n");
            }
        } else {
            printf("GL implementation lacks EXT_framebuffer_sRGB\n");
        }
    }
    // Any color ramps or textures need to be re-initialized now to get an sRGB or linear format.
    return old_state ^ render_sRGB;
}

void StCRenderer::swapBuffers()
{
    glutSwapBuffers();
}

void StCRenderer::reportFPS()
{
    double thisFPS = handleFPS();
    thisFPS = thisFPS; // force used
}

static void getStencilInfo(PathPtr path, GLuint stencil_read_mask, GLuint stencil_func, 
                    GLuint &cover_stencil_read_mask, StencilMode &mode)
{
    if (path->style.fill_rule == PathStyle::EVEN_ODD) {
        cover_stencil_read_mask = 0x1;
        mode = INVERT;
    } else {
        GLuint write_mask = ~0;
        if (stencil_func != GL_ALWAYS) {
            GLuint bit;
            for (int i=0; i<32; i++) {
                bit = 1<<i;
                if (stencil_read_mask & bit) {
                    break;
                }
            }
            write_mask = bit-1;
        }
        cover_stencil_read_mask = write_mask;
        mode = COUNT_UP;
    }
}

void StCPathRenderer::stencilFill(GLenum stencil_func, GLenum stencil_ref, GLuint stencil_read_mask)
{
    GLuint cover_stencil_read_mask;
    StencilMode mode;
    getStencilInfo(getPath(), stencil_read_mask, stencil_func, cover_stencil_read_mask, mode);
    stencilModeFill(mode, cover_stencil_read_mask, stencil_func, stencil_ref, stencil_read_mask);
}


// From shape's renderer info, get its path's renderer info
StCPathRendererPtr StCShapeRenderer::getPathRenderer()
{
    PathRendererStatePtr path_renderer = owner->getPath()->getRendererState(getRenderer());
    StCPathRendererPtr stc_path_renderer = dynamic_pointer_cast<StCPathRenderer>(path_renderer);
    stc_path_renderer->validate();
    return stc_path_renderer;
}

void StCShapeRenderer::invalidate()
{
    getPathRenderer()->invalidate();
}

void StCShapeRenderer::fill(GLenum stencil_func,
                                GLenum stencil_ref,
                                GLuint stencil_read_mask,
                                AlphaTreatment alpha_treatment)
{
    if (!owner->isFillable() || !owner->getFillPaint()) {
        return;
    }

    StCRendererPtr sc = getRenderer();
    PaintRendererStatePtr renderer_state = owner->getFillPaint()->getRendererState(sc);
    StCPaintRendererPtr stc_paint = dynamic_pointer_cast<StCPaintRenderer>(renderer_state);
    if (!stc_paint) {
        return;
    }
    stc_paint->validate();

    glColorMask(0,0,0,0);
    glDepthMask(0);

    GLuint cover_stencil_read_mask;
    StencilMode mode;
    getStencilInfo(owner->getPath(), stencil_read_mask, stencil_func, cover_stencil_read_mask, mode);

    StCPathRendererPtr path_renderer = getPathRenderer();
    path_renderer->stencilModeFill(mode, cover_stencil_read_mask, stencil_func, stencil_ref, stencil_read_mask);

    glDepthMask(1);
    glColorMask(1,1,1,1);

    OpacityTreatment opacity_treatment = stc_paint->startShading(owner, owner->net_fill_opacity, StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);

    switch (opacity_treatment) {
    case FULLY_OPAQUE:
        if (alpha_treatment == ZERO_ALPHA) {
            glEnable(GL_BLEND);
            glBlendFuncSeparate(GL_ONE, GL_ZERO, GL_ZERO, GL_ZERO);
        } else {
            glDisable(GL_BLEND);
        }
        break;
    case VARIABLE_INDEPENDENT_ALPHA:
        glEnable(GL_BLEND);
        // Configure Porter & Duff "over" blending assuming independent (non-pre-multiplied) alpha.
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        break;
    default:
        assert(!"bogus opacity treatment");
    case VARIABLE_PREMULTIPLIED_ALPHA:
        glEnable(GL_BLEND);
        // Configure Porter & Duff "over" blending assuming pre-multiplied alpha.
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        break;
    }

    glEnable(GL_STENCIL_TEST);

    glStencilMask(cover_stencil_read_mask);
    glStencilFunc(GL_NOTEQUAL, 0, cover_stencil_read_mask);
    glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);

    path_renderer->coverFill();
    stc_paint->stopShading(StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
}

void StCShapeRenderer::stroke(GLenum stencil_func,
                                  GLenum stencil_ref,
                                  GLuint stencil_read_mask,
                                  AlphaTreatment alpha_treatment)
{
    if (!owner->isStrokable() || !owner->getStrokePaint()) {
        return;
    }

    StCRendererPtr sc = getRenderer();
    PaintRendererStatePtr renderer_state = owner->getStrokePaint()->getRendererState(sc);
    StCPaintRendererPtr stc_paint = dynamic_pointer_cast<StCPaintRenderer>(renderer_state);
    if (!stc_paint) {
        return;
    }
    stc_paint->validate();

    glColorMask(0,0,0,0);
    glDepthMask(0);
    
    GLint replace_value = 0x1;
    GLuint stencil_write_mask = 0x1;

    StCPathRendererPtr path_renderer = getPathRenderer();
    
    path_renderer->stencilStroke(replace_value, stencil_write_mask,
                              stencil_func, stencil_ref, stencil_read_mask);

    glColorMask(1,1,1,1);

    GLuint cover_stencil_read_mask = stencil_write_mask;
    OpacityTreatment opacity_treatment = stc_paint->startShading(owner, owner->net_stroke_opacity, StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
    switch (opacity_treatment) {
    case FULLY_OPAQUE:
        if (alpha_treatment == ZERO_ALPHA) {
            glEnable(GL_BLEND);
            glBlendFuncSeparate(GL_ONE, GL_ZERO, GL_ZERO, GL_ZERO);
        } else {
            glDisable(GL_BLEND);
        }
        break;
    case VARIABLE_INDEPENDENT_ALPHA:
        glEnable(GL_BLEND);
        // Configure Porter & Duff "over" blending assuming independent (non-pre-multiplied) alpha.
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        break;
    default:
        assert(!"bogus opacity treatment");
    case VARIABLE_PREMULTIPLIED_ALPHA:
        glEnable(GL_BLEND);
        // Configure Porter & Duff "over" blending assuming pre-multiplied alpha.
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        break;
    }
    glEnable(GL_STENCIL_TEST);

    glStencilMask(cover_stencil_read_mask);
    glStencilFunc(GL_EQUAL, 0x1, cover_stencil_read_mask);
    glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
    
    path_renderer->coverStroke();
    stc_paint->stopShading(StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
}

static int floatToFixed(float maxAlpha, float f)
{
    int v = int(floor(f * maxAlpha + 0.5));
    return v;
}

void StCShapeRenderer::fillDilated(int xsteps, int ysteps, bool stipple,
                                       float2 spread,
                                       const float4x4 &mvp,
                                       GLenum stencil_func,
                                       GLenum stencil_ref,
                                       GLuint stencil_read_mask)
{
    assert(xsteps*ysteps > 1);

    if (!owner->isFillable() || !owner->getFillPaint()) {
        return;
    }

    StCRendererPtr sc = getRenderer();
    assert(sc->alpha_bits > 0);  // framebuffer alpha required!
    PaintRendererStatePtr renderer_state = owner->getFillPaint()->getRendererState(sc);
    StCPaintRendererPtr stc_paint = dynamic_pointer_cast<StCPaintRenderer>(renderer_state);
    if (!stc_paint) {
        return;
    }
    stc_paint->validate();

    glColorMask(0,0,0,0);
    glDepthMask(0);

    GLuint cover_stencil_read_mask;
    StencilMode mode;
    getStencilInfo(owner->getPath(), stencil_read_mask, stencil_func, cover_stencil_read_mask, mode);

    /* Care is taken to avoid accumulation bias in the the blend
       color term. */
    // XXX there's still lsb stratification error in this scheme
    const unsigned int gridSize = xsteps*ysteps;
    const unsigned int passes = stipple ? gridSize/2 : gridSize;
    float maxAlpha = (1<<sc->alpha_bits)-1;
    const GLfloat alphaPerPass = 1.0f/passes;

    const int fixedAlpha = floatToFixed(maxAlpha, alphaPerPass);
    const float alphaQuanta = fixedAlpha / maxAlpha;
    const float alphaQuantaStepDown = (fixedAlpha - 1) / maxAlpha;
    const float alphaQuantaStepUp = (fixedAlpha + 1) / maxAlpha;
    bool showGrid = false;

#if 0
    fill(stencil_func, stencil_ref, stencil_read_mask);
    return;
#endif

    glEnable(GL_BLEND);
    glBlendFunc(GL_CONSTANT_COLOR, GL_ONE);  // accumulate weighted alpha
    glEnable(GL_STENCIL_TEST);

    StCPathRendererPtr path_renderer = getPathRenderer();

    float startX = -0.5+1.0/(2*xsteps),
          startY = -0.5+1.0/(2*ysteps),
          oneStepX = 1.0/xsteps,
          oneStepY = 1.0/ysteps;

    float4 dilations = float4(0);

    double sumAlphaQuantas = 0;
    float lastAlpha = -1;  // out-of-range initial value
    int step = 0;
    for (int i=0; i<ysteps; i++) {
        for (int j=0; j<xsteps; j++) {

            if (stipple) {
                if (gridSize & 1) {
                    if (0==((i*xsteps+j) & 1)) {
                        if (showGrid) printf("x");
                        continue;
                    }
                } else {
                    if ((i & 1) ^ (j & 1)) {
                        if (showGrid) printf("x");
                        continue;
                    }
                }
            }
            float2 jxy = float2(startX + oneStepX*j,
                                startY + oneStepY*i);

            step++;
            const float idealAlphaSoFar = alphaPerPass * step;
            float tenativeSumOfAlphaQuantas = sumAlphaQuantas + alphaQuanta;
            float alphaThisPass;
            if (floatToFixed(maxAlpha, tenativeSumOfAlphaQuantas) == floatToFixed(maxAlpha, idealAlphaSoFar)) {
                // right on track
                alphaThisPass = alphaQuanta;
                if (showGrid) printf(".");
            } else if (tenativeSumOfAlphaQuantas > idealAlphaSoFar) {
                if (alphaQuantaStepDown <= 0) {
                    if (showGrid) printf("0");
                    continue;
                }
                if (showGrid) printf("-");
                alphaThisPass = alphaQuantaStepDown;
            } else {
                if (showGrid) printf("+");
                assert(tenativeSumOfAlphaQuantas < idealAlphaSoFar);
                alphaThisPass = alphaQuantaStepUp;
            }
            if (alphaThisPass != lastAlpha) {
                glBlendColor(0,0,0, alphaThisPass);
                lastAlpha = alphaThisPass;
            }
            sumAlphaQuantas += alphaThisPass;

            glMatrixPushEXT(GL_PROJECTION); {
                float2 nudge = 1.0f/sc->view_width_height * jxy;

#if 1  // blur it!
                nudge *= spread;  // value in (XXX half?) pixels
#endif

                dilations.xy = min(dilations.xy, nudge);
                dilations.zw = max(dilations.zw, nudge);

                glMatrixTranslatefEXT(GL_PROJECTION, nudge.x, nudge.y, 0);

                // Stencil step
                glColorMask(0,0,0,0);

                path_renderer->stencilModeFill(mode, cover_stencil_read_mask,
                    stencil_func, stencil_ref, stencil_read_mask);

                // Cover step
                glStencilMask(cover_stencil_read_mask);
                glStencilFunc(GL_NOTEQUAL, 0, cover_stencil_read_mask);
                glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
                
                
                glColorMask(0,0,0,1); // just update alpha
                stc_paint->startShading(owner, owner->net_fill_opacity, StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
                path_renderer->coverFill();
                stc_paint->stopShading(StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);

            } glMatrixPopEXT(GL_PROJECTION);
        }
        if (showGrid) printf("\n");
    }

    glDepthMask(1);
    glColorMask(1,1,1,1);

    // Now final shaded cover step
    OpacityTreatment opacity_treatment = stc_paint->startShading(owner, owner->net_fill_opacity, StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
    glColorMask(1,1,1,1);
    glDisable(GL_STENCIL_TEST);
    switch (opacity_treatment) {
    case FULLY_OPAQUE:
    case VARIABLE_INDEPENDENT_ALPHA:
        glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA,  // RGB = blend with coverage in dst alpha
                            GL_ZERO, GL_ZERO);                     // alpha = clear coverage to zero
        path_renderer->coverFillDilated(mvp, dilations);
        break;
    case VARIABLE_PREMULTIPLIED_ALPHA:
        assert(!"premultiplied alpha not allowed");
    default:
        assert(!"bogus opacity treatment");
        break;
    }

    stc_paint->stopShading(StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
}

void StCShapeRenderer::strokeDilated(int xsteps, int ysteps, bool stipple,
                                         float2 spread,
                                         const float4x4 &mvp,
                                         GLenum stencil_func,
                                         GLenum stencil_ref,
                                         GLuint stencil_read_mask)
{
    assert(xsteps*ysteps > 1);

    if (!owner->isStrokable() || !owner->getStrokePaint()) {
        return;
    }

    StCRendererPtr sc = getRenderer();
    assert(sc->alpha_bits > 0);  // framebuffer alpha required!
    PaintRendererStatePtr renderer_state = owner->getStrokePaint()->getRendererState(sc);
    StCPaintRendererPtr stc_paint = dynamic_pointer_cast<StCPaintRenderer>(renderer_state);
    if (!stc_paint) {
        return;
    }
    stc_paint->validate();
    
    glDepthMask(0);

    GLint replace_value = 0x1;
    GLuint stencil_write_mask = 0x1;
    GLuint cover_stencil_read_mask = stencil_write_mask;

    /* Care is taken to avoid accumulation bias in the the blend
       color term. */
    // XXX there's still lsb stratification error in this scheme
    const unsigned int gridSize = xsteps*ysteps;
    const unsigned int passes = stipple ? gridSize/2 : gridSize;
    float maxAlpha = (1<<sc->alpha_bits)-1;
    const GLfloat alphaPerPass = 1.0f/passes;

    const int fixedAlpha = floatToFixed(maxAlpha, alphaPerPass);
    const float alphaQuanta = fixedAlpha / maxAlpha;
    const float alphaQuantaStepDown = (fixedAlpha - 1) / maxAlpha;
    const float alphaQuantaStepUp = (fixedAlpha + 1) / maxAlpha;
    bool showGrid = false;

    glEnable(GL_BLEND);
    glBlendFunc(GL_CONSTANT_COLOR, GL_ONE);  // accumulate weighted alpha
    glEnable(GL_STENCIL_TEST);

    StCPathRendererPtr path_renderer = getPathRenderer();

    float startX = -0.5+1.0/(2*xsteps),
          startY = -0.5+1.0/(2*ysteps),
          oneStepX = 1.0/xsteps,
          oneStepY = 1.0/ysteps;

    float4 dilations = float4(0);

    double sumAlphaQuantas = 0;
    float lastAlpha = -1;  // out-of-range initial value
    int step = 0;
    for (int i=0; i<ysteps; i++) {
        for (int j=0; j<xsteps; j++) {

            if (stipple) {
                if (gridSize & 1) {
                    if (0==((i*xsteps+j) & 1)) {
                        if (showGrid) printf("x");
                        continue;
                    }
                } else {
                    if ((i & 1) ^ (j & 1)) {
                        if (showGrid) printf("x");
                        continue;
                    }
                }
            }
            float2 jxy = float2(startX + oneStepX*j,
                                startY + oneStepY*i);

            step++;
            const float idealAlphaSoFar = alphaPerPass * step;
            float tenativeSumOfAlphaQuantas = sumAlphaQuantas + alphaQuanta;
            float alphaThisPass;
            if (floatToFixed(maxAlpha, tenativeSumOfAlphaQuantas) == floatToFixed(maxAlpha, idealAlphaSoFar)) {
                // right on track
                alphaThisPass = alphaQuanta;
                if (showGrid) printf(".");
            } else if (tenativeSumOfAlphaQuantas > idealAlphaSoFar) {
                if (alphaQuantaStepDown <= 0) {
                    if (showGrid) printf("0");
                    continue;
                }
                if (showGrid) printf("-");
                alphaThisPass = alphaQuantaStepDown;
            } else {
                if (showGrid) printf("+");
                assert(tenativeSumOfAlphaQuantas < idealAlphaSoFar);
                alphaThisPass = alphaQuantaStepUp;
            }
            if (alphaThisPass != lastAlpha) {
                glBlendColor(0,0,0, alphaThisPass);
                lastAlpha = alphaThisPass;
            }
            sumAlphaQuantas += alphaThisPass;

            glMatrixPushEXT(GL_PROJECTION); {
                float2 nudge = 1.0f/sc->view_width_height * jxy;

#if 1  // blur it!
                nudge *= spread;  // value in (XXX half?) pixels
#endif

                dilations.xy = min(dilations.xy, nudge);
                dilations.zw = max(dilations.zw, nudge);

                glMatrixTranslatefEXT(GL_PROJECTION, nudge.x, nudge.y, 0);

                // Stencil step
                glColorMask(0,0,0,0);

                path_renderer->stencilStroke(replace_value, stencil_write_mask,
                    stencil_func, stencil_ref, stencil_read_mask);

                // Cover step
                glStencilMask(cover_stencil_read_mask);
                glStencilFunc(GL_EQUAL, 0x1, cover_stencil_read_mask);
                glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
                
                glColorMask(0,0,0,1); // just update alpha
                OpacityTreatment opacity_treatment = stc_paint->startShading(owner, owner->net_stroke_opacity, StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
                opacity_treatment = opacity_treatment;  // XXX force used
                path_renderer->coverStroke();
                stc_paint->stopShading(StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);

            } glMatrixPopEXT(GL_PROJECTION);
        }
        if (showGrid) printf("\n");
    }

    glDepthMask(1);

    // Now final shaded cover step
    OpacityTreatment opacity_treatment = stc_paint->startShading(owner, owner->net_stroke_opacity, StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
    glColorMask(1,1,1,1);
    glDisable(GL_STENCIL_TEST);
    switch (opacity_treatment) {
    case FULLY_OPAQUE:
    case VARIABLE_INDEPENDENT_ALPHA:
        glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA,  // RGB = blend with coverage in dst alpha
                            GL_ZERO, GL_ZERO);                     // alpha = clear coverage to zero
        path_renderer->coverStrokeDilated(mvp, dilations);
        break;
    case VARIABLE_PREMULTIPLIED_ALPHA:
        assert(!"premultiplied alpha not allowed");
    default:
        assert(!"bogus opacity treatment");
        break;
    }

    stc_paint->stopShading(StCPaintRenderer::BOTTOM_FIRST_TOP_LAST);
}

void StCShapeRenderer::fillTopToBottom(GLuint filled_bit,
                                           GLenum stencil_func,
                                           GLenum stencil_ref,
                                           GLuint stencil_read_mask)
{
    if (!owner->isFillable() || !owner->getFillPaint()) {
        return;
    }

    assert(!(filled_bit & (filled_bit-1)));

    StCRendererPtr sc = getRenderer();
    assert(sc->alpha_bits > 0);  // framebuffer alpha required!
    PaintRendererStatePtr renderer_state = owner->getFillPaint()->getRendererState(sc);
    StCPaintRendererPtr stc_paint = dynamic_pointer_cast<StCPaintRenderer>(renderer_state);
    if (!stc_paint) {
        return;
    }
    stc_paint->validate();

    glColorMask(0,0,0,0);
    glDepthMask(0);

    GLuint cover_stencil_read_mask;
    StencilMode mode;
    getStencilInfo(owner->getPath(), stencil_read_mask, stencil_func, cover_stencil_read_mask, mode);

    StCPathRendererPtr path_renderer = getPathRenderer();
    path_renderer->stencilModeFill(mode, cover_stencil_read_mask, stencil_func, stencil_ref, stencil_read_mask);

    glDepthMask(1);
    glColorMask(1,1,1,1);

    OpacityTreatment opacity_treatment = stc_paint->startShading(owner, owner->net_fill_opacity, StCPaintRenderer::TOP_FIRST_BOTTOM_LAST);
    switch (opacity_treatment) {
    case FULLY_OPAQUE:
        // Assume blending is initially disabled but will
        // be enabled as soon as any layer uses variable pre-multiplied alpha.
        break;
    case VARIABLE_PREMULTIPLIED_ALPHA:
        // Configure Porter & Duff "under" blending; requires
        // pre-multiplied alpha and assumes framebuffer cleared to (0,0,0,0)
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_ONE_MINUS_DST_ALPHA, GL_ONE, GL_ONE, GL_ONE);
        break;
    case VARIABLE_INDEPENDENT_ALPHA:
        assert(!"variable independent alpha is incompatible with top-to-bottom; use pre-multiplied alpha");
        break;
    default:
        assert(!"bogus opacity treatment");
        break;
    }
    glEnable(GL_STENCIL_TEST);
    GLuint stencil_mask = opacity_treatment==FULLY_OPAQUE ? 
        filled_bit | cover_stencil_read_mask : 
        cover_stencil_read_mask;
    glStencilMask(stencil_mask);
    glStencilFunc(GL_NOTEQUAL, filled_bit, cover_stencil_read_mask);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    path_renderer->coverFill();
    stc_paint->stopShading(StCPaintRenderer::TOP_FIRST_BOTTOM_LAST);
}

void StCShapeRenderer::strokeTopToBottom(GLuint filled_bit,
                                             GLenum stencil_func,
                                             GLenum stencil_ref,
                                             GLuint stencil_read_mask)
{
    if (!owner->isStrokable() || !owner->getStrokePaint()) {
        return;
    }

    assert(!(filled_bit & (filled_bit-1)));

    StCRendererPtr sc = getRenderer();
    assert(sc->alpha_bits > 0);  // framebuffer alpha required!
    PaintRendererStatePtr renderer_state = owner->getStrokePaint()->getRendererState(sc);
    StCPaintRendererPtr stc_paint = dynamic_pointer_cast<StCPaintRenderer>(renderer_state);
    if (!stc_paint) {
        return;
    }
    stc_paint->validate();

    glColorMask(0,0,0,0);
    glDepthMask(0);
    
    GLint replace_value = 0x1;
    GLuint stencil_write_mask = 0x1;

    StCPathRendererPtr path_renderer = getPathRenderer();

    path_renderer->stencilStroke(replace_value, stencil_write_mask,
                              stencil_func, stencil_ref, stencil_read_mask);

    glDepthMask(1);
    glColorMask(1,1,1,1);

    OpacityTreatment opacity_treatment = stc_paint->startShading(owner, owner->net_fill_opacity, StCPaintRenderer::TOP_FIRST_BOTTOM_LAST);
    switch (opacity_treatment) {
    case FULLY_OPAQUE:
        // Assume blending is initially disabled but will
        // be enabled as soon as any layer uses variable pre-multiplied alpha.
        break;
    case VARIABLE_PREMULTIPLIED_ALPHA:
        // Configure Porter & Duff "under" blending; requires
        // pre-multiplied alpha and assumes framebuffer cleared to (0,0,0,0)
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_ONE_MINUS_DST_ALPHA, GL_ONE, GL_ONE, GL_ONE);
        break;
    case VARIABLE_INDEPENDENT_ALPHA:
        assert(!"variable independent alpha is incompatible with top-to-bottom; use pre-multiplied alpha");
        break;
    default:
        assert(!"bogus opacity treatment");
        break;
    }
    glEnable(GL_STENCIL_TEST);
    GLuint stencil_mask = opacity_treatment==FULLY_OPAQUE ? filled_bit | 0x1 : 0x1;
    glStencilMask(stencil_mask);
    // filled_bit | 0x1 because we only want to draw the pixel of filled_bit == 0
    // (this pixel hasn't been drawn yet) and we're on the stroke
    glStencilFunc(GL_EQUAL, 0x01, filled_bit | 0x1);
    // Inverting filled_bit | 0x1 clears the value to zero and marks the pixel as covered
    glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
    path_renderer->coverStroke();
    stc_paint->stopShading(StCPaintRenderer::TOP_FIRST_BOTTOM_LAST);
}

void StCShapeRenderer::fillToStencilBit(GLuint bit)
{
    // Make sure there is only one bit set in this variable
    assert(!(bit & (bit-1)));

    StCPathRendererPtr path_renderer = getPathRenderer();

    path_renderer->stencilFill(GL_EQUAL, bit-1, bit);
    path_renderer->validate();

    glEnable(GL_STENCIL_TEST);
    glStencilMask(bit | (bit - 1));

    // Anybody who's nonzero needs to get marked as drawable
    glStencilFunc(GL_NOTEQUAL, bit, bit - 1);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    path_renderer->coverFill();
}
