#ifndef __renderer_stc_hpp__
#define __renderer_stc_hpp__

/* renderer_stc.hpp - rendering interfaces for "stencil, then cover" path rendering. */

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include "renderer.hpp"

enum StencilMode {
    COUNT_UP,
    COUNT_DOWN,
    INVERT
};

enum AlphaTreatment {
    ZERO_ALPHA,
    SEND_ALPHA,
};

struct StCRenderer : Renderer {

    StCRenderer()
        : scene_ratio(1.0)
        , min_filter(GL_LINEAR_MIPMAP_LINEAR)
        , mag_filter(GL_LINEAR)
        , max_anisotropy_limit(1)
        , max_anisotropy(1)
        , color_ramp_size(512)
        , render_sRGB(false)
    {}

    // Extension support booleans
    bool has_ARB_sample_shading;
    bool has_EXT_direct_state_access;
    bool has_EXT_texture_filter_anisotropic;
    bool has_NV_explicit_multisample;
    bool has_EXT_framebuffer_sRGB;
    bool has_EXT_texture_sRGB;
    // Combinations of extensions
    bool has_texture_non_power_of_two;

    // Framebuffer limits
    int num_samples;
    int alpha_bits;
    int supersample_ratio;
    bool multisampling;
    vector<float2> sample_positions;
    bool sRGB_capable;

    // Framebuffer & scene dimensions
    int2 view_width_height;
    float4x4 surface_to_window;
    float scene_ratio;

    // Path rendering controls
    GLenum min_filter;
    GLenum mag_filter;
    GLint max_anisotropy_limit;
    GLint max_anisotropy;
    int color_ramp_size;
    bool render_sRGB;

    void setViewport(int x, int y, int width, int height);
    void setSceneRatio(float scene_ratio);
    bool setColorSpace(ColorSpace cs);
    void swapBuffers();
    void reportFPS();

private:
    void update(); // call when width, height, or scene ratio changes
};

template <typename T>
struct StCRendererState : SpecificRendererState<T,StCRenderer> {
    typedef T *OwnerPtr;  // intentionally not a SharedPtr
    StCRendererState(RendererPtr renderer_, OwnerPtr owner_) 
        : SpecificRendererState<T,StCRenderer>(renderer_, owner_)
    {}
};
typedef shared_ptr<StCRenderer> StCRendererPtr;

struct StCPathRenderer {
    virtual void validate() = 0;
    virtual void invalidate() = 0;

    virtual void stencilStroke(GLint replace_value, GLuint stencil_write_mask,
                               GLenum stencil_func,
                               GLint stencil_ref,
                               GLuint read_mask) = 0;
    virtual void stencilModeFill(StencilMode mode, GLuint stencil_write_mask,
                                 GLenum stencil_func,
                                 GLint stencil_ref,
                                 GLuint stencil_read_mask) = 0;

    void stencilFill(GLenum stencil_func, GLenum stencil_ref, GLuint stencil_read_mask);

    virtual void coverFill() = 0;
    virtual void coverStroke() = 0;

    virtual void coverFillDilated(const float4x4 &mvp, float4 &dilations) = 0;
    virtual void coverStrokeDilated(const float4x4 &mvp, float4 &dilations) = 0;

    virtual PathPtr getPath() = 0;
    virtual ~StCPathRenderer() {}  // satisfy g++ that there be a virtual dtor when there are virtual functions
};
typedef shared_ptr<struct StCPathRenderer> StCPathRendererPtr;

struct StCPaintRenderer {
    enum RenderOrder {
        BOTTOM_FIRST_TOP_LAST,
        TOP_FIRST_BOTTOM_LAST
    };
    virtual void validate() = 0;
    virtual OpacityTreatment startShading(Shape *shape, float opacity, RenderOrder render_order) = 0;
    virtual void stopShading(RenderOrder render_order) = 0;
    virtual ~StCPaintRenderer() {}  // satisfy g++ that there be a virtual dtor when there are virtual functions
};
typedef shared_ptr<StCPaintRenderer> StCPaintRendererPtr;

struct StCShapeRenderer : StCRendererState<Shape> {
    StCShapeRenderer(RendererPtr renderer, Shape *shape)
        : StCRendererState<Shape>(renderer, shape)
    {}

    StCPathRendererPtr getPathRenderer();

    void fill(GLenum stencil_func, GLenum stencil_ref, GLuint stencil_read_mask, AlphaTreatment);
    void stroke(GLenum stencil_func, GLenum stencil_ref, GLuint stencil_read_mask, AlphaTreatment);

    void coverFill();
    void coverStroke();

    void fillDilated(int xsteps, int ysteps, bool stipple,
                     float2 spread,
                     const float4x4 &mvp,
                     GLenum stencil_func,
                     GLenum stencil_ref,
                     GLuint stencil_read_mask);
    void strokeDilated(int xsteps, int ysteps, bool stipple,
                       float2 spread,
                       const float4x4 &mvp,
                       GLenum stencil_func,
                       GLenum stencil_ref,
                       GLuint stencil_read_mask);

    void fillTopToBottom(GLuint filled_bit,
                         GLenum stencil_func,
                         GLenum stencil_ref,
                         GLuint stencil_read_mask);
    void strokeTopToBottom(GLuint filled_bit,
                           GLenum stencil_func,
                           GLenum stencil_ref,
                           GLuint stencil_read_mask);

    void fillToStencilBit(GLuint bit);

    ShapePtr getShape() { return owner->shared_from_this(); }

    virtual bool isNonOpaque() { return false; }
    virtual void invalidate();
};
typedef shared_ptr<StCShapeRenderer> StCShapeRendererPtr;

#endif // __renderer_stc_hpp__
