
/* renderer_nvpr.hpp - NV_path_rendering renderer class. */

#ifndef __renderer_nvpr_hpp__
#define __renderer_nvpr_hpp__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include "nvpr_svg_config.h"  // configure path renderers to use

#if USE_NVPR

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

#include "path.hpp"
#include "scene.hpp"

//#include "nvpr_init.h"
#include "showfps.h"

#include "stc/renderer_stc.hpp"

typedef shared_ptr<struct NVprRenderer> NVprRendererPtr;
typedef shared_ptr<struct NVprPathRendererState> NVprPathRendererStatePtr;
typedef shared_ptr<struct NVprPaintRendererState> NVprPaintRendererStatePtr;

struct NVprRenderer : StCRenderer {
    // Path rendering controls
    bool use_DSA;
    bool making_dlist;

    // Initialization routines (called by GLRenderer's constructor)
    void initExtensions(bool noDSA);
    void interrogateFramebuffer();
    void reportGL();
    void configureGL();

    CGcontext myCgContext;
    CGprofile myCgFragmentProfile, myCgVertexProfile;
    CGprogram radial_center_gradient_program,
              radial_focal_gradient_program;

    // Cg usage
    typedef enum {
        // Gradient shader
        RADIAL_CENTER_GRADIENT,
        RADIAL_FOCAL_GRADIENT,
    } ShadingMode;

    void checkForCgError(const char *situation);
    void load_gradient_shaders();

    void init_cg(const char *vertex_profile_name,
                 const char *fragment_profile_name);
    void report_cg_profiles();
    void load_shaders();
    void shutdown_shaders();
    void enableFragmentShading();
    void disableFragmentShading();
    bool configureShading(ShadingMode mode);

    const char *getWindowTitle();
    const char *getName();

    NVprRenderer(bool useDSA,
                 const char *vertex_profile_name,
                 const char *fragment_profile_name);

    shared_ptr<RendererState<Shape> > alloc(Shape *owner);
    shared_ptr<RendererState<Path> > alloc(Path *owner);
    shared_ptr<RendererState<Paint> > alloc(Paint *owner);
};

template <typename T>
struct NVprRendererState : SpecificRendererState<T,NVprRenderer> {
    typedef T *OwnerPtr;  // intentionally not a SharedPtr
    NVprRendererState(RendererPtr renderer_, OwnerPtr owner_) 
        : SpecificRendererState<T,NVprRenderer>(renderer_, owner_)
    {}
};

struct NVprPathRendererState : RendererState<Path>, StCPathRenderer {
    GLenum fill_rule;
    GLuint path;
    bool valid;

    GLenum fill_cover_mode,
           stroke_cover_mode;

    NVprPathRendererState(RendererPtr renderer, Path *owner);
    ~NVprPathRendererState();

    void validate();
    void invalidate();

    void stencilModeFill(StencilMode mode, GLuint stencil_write_mask,
                         GLenum stencil_func, GLint stencil_ref,
                         GLuint stencil_read_mask);
    void stencilModeFill(GLenum stencil_fill_rule, GLuint stencil_write_mask,
                         GLenum stencil_func, GLint stencil_ref,
                         GLuint stencil_read_mask);
    void coverFill();

    void stencilStroke(GLint replace_value, GLuint stencil_write_mask,
                       GLenum stencil_func,
                       GLint stencil_ref,
                       GLuint read_mask);
    void coverStroke();

    void coverStrokeDilated(const float4x4 &mvp, float4 &dilations);
    void coverFillDilated(const float4x4 &mvp, float4 &dilations);
    void coverDilated(GLenum bbox_id, const float4x4 &mvp, float4 &dilations);

    PathPtr getPath();
    float4 getBounds();
};

struct NVprPaintRendererState : NVprRendererState<Paint>, StCPaintRenderer {
protected:
    bool valid;

public:
    NVprPaintRendererState(RendererPtr renderer, Paint *paint)
        : NVprRendererState<Paint>(renderer, paint)
        , valid(false)
    {}

    virtual ~NVprPaintRendererState() {
    }

    virtual void validate() = 0;
    void invalidate();

    virtual OpacityTreatment startShading(Shape *shape, float opacity, RenderOrder render_order) = 0;
    virtual void stopShading(RenderOrder render_order) = 0;
};

typedef shared_ptr<struct NVprSolidColorPaintRendererState> NVprSolidColorPaintRendererStatePtr;
struct NVprSolidColorPaintRendererState : NVprPaintRendererState {
    NVprSolidColorPaintRendererState(RendererPtr renderer, SolidColorPaint *paint)
        : NVprPaintRendererState(renderer, paint)
    {}

    void validate();
    OpacityTreatment startShading(Shape *shape, float opacity, RenderOrder render_order);
    void stopShading(RenderOrder render_order);
};

typedef shared_ptr<struct NVprGradientPaintRendererState> NVprGradientPaintRendererStatePtr;
struct NVprGradientPaintRendererState : NVprPaintRendererState {
    GLuint texobj;
    bool fullyOpaqueGradient;

    NVprGradientPaintRendererState(RendererPtr renderer, GradientPaint *paint)
        : NVprPaintRendererState(renderer, paint)
        , texobj(0)
        , fullyOpaqueGradient(true)
    {}

    ~NVprGradientPaintRendererState() {
        if (texobj) {
            glDeleteTextures(1, &texobj);
            texobj = 0;
        }
    }

    void validateGenericGradientState(GradientPaint *paint);
};

typedef shared_ptr<struct NVprLinearGradientPaintRendererState> NVprLinearGradientPaintRendererStatePtr;
struct NVprLinearGradientPaintRendererState : NVprGradientPaintRendererState {
    NVprLinearGradientPaintRendererState(RendererPtr renderer, LinearGradientPaint *paint)
        : NVprGradientPaintRendererState(renderer, paint)
    {}

    void validate();
    OpacityTreatment startShading(Shape *shape, float opacity, RenderOrder render_order);
    void stopShading(RenderOrder render_order);
};

typedef shared_ptr<struct NVprRadialGradientPaintRendererState> NVprRadialGradientPaintRendererStatePtr;
struct NVprRadialGradientPaintRendererState : NVprGradientPaintRendererState {
    NVprRadialGradientPaintRendererState(RendererPtr renderer, RadialGradientPaint *paint)
        : NVprGradientPaintRendererState(renderer, paint)
    {}

    void validate();
    OpacityTreatment startShading(Shape *shape, float opacity, RenderOrder render_order);
    void stopShading(RenderOrder render_order);
};

typedef shared_ptr<struct NVprImagePaintRendererState> NVprImagePaintRendererStatePtr;
struct NVprImagePaintRendererState : NVprPaintRendererState {
    GLuint texture;
    NVprImagePaintRendererState(RendererPtr renderer, ImagePaint *paint)
        : NVprPaintRendererState(renderer, paint)
        , texture(0)
    {}
    virtual ~NVprImagePaintRendererState();

    void validate();
    OpacityTreatment startShading(Shape *shape, float opacity, RenderOrder render_order);
    void stopShading(RenderOrder render_order);
};

#endif // USE_NVPR

#endif // __renderer_nvpr_hpp__
