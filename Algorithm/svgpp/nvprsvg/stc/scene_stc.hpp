#ifndef __scene_stc_hpp__
#define __scene_stc_hpp__

/* scene_stc.hpp - scene interfaces for "stencil, then cover" path rendering. */

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include "scene.hpp"
#include "renderer_stc.hpp"

namespace StCVisitors {

    ///////////////////////////////////////////////////////////////////////////////
    // StCVisitor
    class StCVisitor : public ClipSaveVisitor, public enable_shared_from_this<StCVisitor>
    {
    protected:
        StCRendererPtr renderer;
        StCVisitor(StCRendererPtr renderer_, const float4x4 &view_to_clip);
        StCVisitor(StCRendererPtr renderer_);

    public:
        virtual void visit(ShapePtr shape);
        virtual void fill(StCShapeRendererPtr shape_renderer_renderer) { }
        virtual void stroke(StCShapeRendererPtr shape_renderer_renderer) { }

        virtual void apply(TransformPtr transform);
        void unapply(TransformPtr transform);

        void coverRect(RectBounds &bounds);
        void loadCurrentMatrix();
    };
    typedef shared_ptr<StCVisitor> StCVisitorPtr;

    ///////////////////////////////////////////////////////////////////////////////
    // Draw
    class Draw : public StCVisitor 
    {
    public:
        Draw(StCRendererPtr renderer_, GLuint clip_msb_,  
             const float4x4 &view_to_clip);
        ~Draw();

        void visit(ShapePtr shape);
        void fill(StCShapeRendererPtr shape_renderer);
        void stroke(StCShapeRendererPtr shape_renderer);

        void apply(ClipPtr clip);
        void unapply(ClipPtr clip);

    protected:
        int4 clipScissorBox();
        void validateClipPath();
        inline GLenum stencilClipFunc() { return clip_stack.size() ? GL_EQUAL : GL_ALWAYS; }
        inline GLuint stencilClipBit() { return clip_stack.size() ? clip_msb : 0; }

    protected:
        bool clip_path_valid;
        // already-transformed bounds of the current clip path
        RectBounds clip_bounds;
        GLuint clip_msb;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // ClipVisitor
    typedef shared_ptr<struct ClipFiller> ClipFillerPtr;
    class ClipVisitor : public StCVisitor
    {
    public:
        ClipVisitor(GLuint bit_, GLuint clip_msb_, ClipMerge clip_merge, 
                    StCRendererPtr renderer_, 
                    const float4x4 &view_to_clip,
                    RectBounds &current_clip_bounds_);

        void fill(StCShapeRendererPtr shape_renderer);

        // No stroke virtual method; inherit the no-op stroke method
        // because stroking doesn't affect the clip.

        void apply(ClipPtr clip);
        void unapply(ClipPtr clip);

        void beginFill();
        void endFill();

    protected:
        void intersectTop2StencilBits();

    protected:
        GLuint bit;
        ClipFillerPtr clip_filler;
        RectBounds bounds;
        
        // already-transformed bounds of the existing clip path
        RectBounds &current_clip_bounds;
        GLuint clip_msb;
    };
    typedef shared_ptr<ClipVisitor> ClipVisitorPtr;

    ///////////////////////////////////////////////////////////////////////////////
    // DrawDilated
    class DrawDilated : public Draw {
    public:
        DrawDilated(StCRendererPtr renderer_, GLuint clip_msb_,  
                    const float4x4 &view_to_clip,
                    int xsteps_, int ysteps_, 
                    bool stipple_, float2 spread_);

        void visit(ShapePtr shape);
        void fill(StCShapeRendererPtr shape_renderer);
        void stroke(StCShapeRendererPtr shape_renderer);

        // For now, clip paths don't get any special treatment. Ideally, we would want
        // to allow them to take advantage of the better anti-aliasing also
        // void apply(ClipPtr clip);
        // void unapply(ClipPtr clip);

    protected:
        int alpha_bits;
        int xsteps, ysteps;
        bool stipple;
        float2 spread;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // ReversePainter
    class ReversePainter : public Draw {
    public:
        ReversePainter(StCRendererPtr renderer_, GLuint clip_msb_,  
                       const float4x4 &view_to_clip);

        void visit(ShapePtr shape);
        void fill(StCShapeRendererPtr shape_renderer);
        void stroke(StCShapeRendererPtr shape_renderer);

    protected:
        GLuint filled_bit;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // CoverFill
    class CoverFill : public StCVisitor {
    public:
        CoverFill(StCRendererPtr renderer_, float4x4 view_to_clip) 
            : StCVisitor(renderer_, view_to_clip) { }
        void fill(StCShapeRendererPtr shape_renderer) { 
            shape_renderer->getPathRenderer()->coverFill(); 
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // CoverStroke
    class CoverStroke : public StCVisitor {
    public:
        CoverStroke(StCRendererPtr renderer_, float4x4 view_to_clip) 
            : StCVisitor(renderer_, view_to_clip) { }
        void fill(StCShapeRendererPtr shape_renderer) { 
            shape_renderer->getPathRenderer()->coverStroke(); 
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // VisualizeClipScissor
    class VisualizeClipScissor : public Draw
    {
    public:
        VisualizeClipScissor(StCRendererPtr renderer_, GLuint clip_msb_, const float4x4 &view_to_clip)
            : Draw(renderer_, clip_msb_, view_to_clip) { }

        void visit(ShapePtr shape);
    };

    class DrawControlPoints : public StCVisitor
    {
    public:
        DrawControlPoints(StCRendererPtr renderer_, const float4x4 &view_to_clip)
          : StCVisitor(renderer_, view_to_clip) { }
        void visit(ShapePtr shape);
    };

    class DrawWarpPoints : public StCVisitor
    {
    public:
        DrawWarpPoints(StCRendererPtr renderer_, const float4x4 &view_to_clip)
          : StCVisitor(renderer_, view_to_clip) { }
        void visit(ShapePtr shape) {}

        void apply(TransformPtr transform) {
            MatrixSaveVisitor::apply(transform);
            loadCurrentMatrix();

            WarpTransformPtr warp_transform = dynamic_pointer_cast<WarpTransform>(transform);
            // Is this transform a warp transform?
            if (warp_transform) {
                warp_transform->drawWarpPoints();
            }
        }

    };


    class DrawReferencePoints : public StCVisitor
    {
    public:
        DrawReferencePoints(StCRendererPtr renderer_, const float4x4 &view_to_clip)
          : StCVisitor(renderer_, view_to_clip) { }
        void visit(ShapePtr shape);
    };

    class GatherStats : public StCVisitor {
    public:
        GatherStats(StCRendererPtr renderer, PathStats &t, PathStats &m);
        void visit(ShapePtr shape);

    protected:
        PathStats &total_stats;
        PathStats &max_stats;
    };

};

#endif // __scene_stc_hpp__
