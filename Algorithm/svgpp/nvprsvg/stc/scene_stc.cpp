
/* scene_stc.cpp - scene algorithms for "stencil, then cover" path rendering. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include "nvpr_svg_config.h" 
#include "scene_stc.hpp"

// These need to be moved to a global structure somewhere
static bool doFilling = 1;
static bool doStroking = 1;
static bool makingDlist = 1;

namespace StCVisitors {

///////////////////////////////////////////////////////////////////////////////
// clip-merge stuff
struct ClipFiller {
    virtual void beginFill(GLuint bit) { }
    virtual void fill(StCShapeRendererPtr shape_renderer, GLuint bit) = 0;
    virtual void endFill(GLuint bit, RectBounds &bounds) { }
    virtual ~ClipFiller() {}  // add no-op virtual dtor so g++ -Wall won't complain about virtual functions without virtual dtor
};

struct UnionFiller : ClipFiller {
    void fill(StCShapeRendererPtr shape_renderer, GLuint bit) {
        shape_renderer->fillToStencilBit(bit);
    }
};

struct InvertFiller : ClipFiller {
    void fill(StCShapeRendererPtr shape_renderer, GLuint bit) {
        shape_renderer->getPathRenderer()->stencilModeFill(INVERT, bit, GL_ALWAYS, 0, 0);
    }
};

struct SumFiller : ClipFiller {
    StCVisitor &visitor;
    SumFiller(StCVisitor &visitor_) : visitor(visitor_) { }

    void fill(StCShapeRendererPtr shape_renderer, GLuint bit) {
        shape_renderer->getPathRenderer()->stencilModeFill(COUNT_UP, bit-1, GL_ALWAYS, 0, 0);
    }
    void endFill(GLuint bit, RectBounds &bounds) {
        glEnable(GL_STENCIL_TEST);
        glStencilMask(bit | (bit-1));

        // Anybody who's nonzero needs to get marked as drawable
        glStencilFunc(GL_NOTEQUAL, bit, bit - 1);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        visitor.coverRect(bounds);
    }
};


///////////////////////////////////////////////////////////////////////////////
// Shared StC visitor methods
StCVisitor::StCVisitor(StCRendererPtr renderer_, const float4x4 &view_to_clip)
        : renderer(renderer_)
{ 
    matrix_stack.pop();
    matrix_stack.push(view_to_clip);
}

StCVisitor::StCVisitor(StCRendererPtr renderer_)
        : renderer(renderer_)
{ 
}

void StCVisitor::visit(ShapePtr shape)
{
    StCShapeRendererPtr shape_renderer = 
        dynamic_pointer_cast<StCShapeRenderer>(shape->getRendererState(renderer));

    assert(shape_renderer);

    if (doFilling && !shape->isEmpty()) {
        fill(shape_renderer);
    }

    // No need for a stroke_width != 0 clause - the parser should rip these out
    if (doStroking && shape->isStrokable()) {
        stroke(shape_renderer);
    }
}

void StCVisitor::apply(TransformPtr transform)
{
    MatrixSaveVisitor::apply(transform);
    loadCurrentMatrix();
}

void StCVisitor::unapply(TransformPtr transform)
{
    MatrixSaveVisitor::unapply(transform);
    loadCurrentMatrix();
}

void StCVisitor::loadCurrentMatrix()
{
    GLfloat *m = reinterpret_cast<GLfloat*>(&matrix_stack.top()[0][0]);
    if (makingDlist) {
        // "Pop, then Push" allows us to combine with a base view transform.
        glMatrixPopEXT(GL_MODELVIEW);
        glMatrixPushEXT(GL_MODELVIEW);
        glMatrixMultTransposefEXT(GL_MODELVIEW, m);
    } else {
        glMatrixLoadTransposefEXT(GL_MODELVIEW, m);
    }
}

void StCVisitor::coverRect(RectBounds &bounds)
{
    // XXX this absolute matrix loading usage is incompatible
    // with compiled display list mode (makingDlist).
    glMatrixLoadIdentityEXT(GL_MODELVIEW);

    RectBounds dilated = bounds.dilate(
        2.0f / renderer->view_width_height.x,
        2.0f / renderer->view_width_height.y);

    glBegin(GL_QUADS); {
        glVertex2f(dilated.x, dilated.y);
        glVertex2f(dilated.x, dilated.w);
        glVertex2f(dilated.z, dilated.w);
        glVertex2f(dilated.z, dilated.y);
    } glEnd();

    loadCurrentMatrix();
}


///////////////////////////////////////////////////////////////////////////////
// Draw
Draw::Draw(StCRendererPtr renderer_, GLuint clip_msb_,  
           const float4x4 &view_to_clip)
    : StCVisitor(renderer_, view_to_clip)
    , clip_path_valid(true)
    , clip_msb(clip_msb_)
{
}

Draw::~Draw()
{
    // May need to clear out the clip path in the stencil buffer
    validateClipPath();
}

void Draw::visit(ShapePtr shape)
{
    validateClipPath();
    StCVisitor::visit(shape);
}

void Draw::fill(StCShapeRendererPtr shape_renderer)
{
    if (shape_renderer->owner->isFillable()) {
        shape_renderer->fill(stencilClipFunc(), stencilClipBit(), stencilClipBit(), SEND_ALPHA);
    }
}

void Draw::stroke(StCShapeRendererPtr shape_renderer)
{
    shape_renderer->stroke(stencilClipFunc(), stencilClipBit(), stencilClipBit(), SEND_ALPHA);
}

void Draw::apply(ClipPtr clip)
{
    ClipSaveVisitor::apply(clip);
	clip_path_valid = false;
}

void Draw::unapply(ClipPtr clip)
{
    ClipSaveVisitor::unapply(clip);
	clip_path_valid = false;
}

int4 Draw::clipScissorBox()
{
    RectBounds scissor = clip_bounds.transform(renderer->surface_to_window).dilate(1);
    return int4(scissor + 0.5f);
}

void Draw::validateClipPath()
{
    if (!clip_path_valid) {
        glEnable(GL_STENCIL_TEST);
        glStencilMask(clip_msb | (clip_msb - 1));
        glColorMask(0,0,0,0);
        glDepthMask(0);

        if (clip_bounds.isValid()) {
            // A clip path is already etched into the stencil buffer; erase it
            if (!makingDlist) {
                glDisable(GL_SCISSOR_TEST);
            }
            glStencilFunc(GL_ALWAYS, 0, 0);
            glStencilOp(GL_ZERO, GL_ZERO, GL_ZERO);
            coverRect(clip_bounds);
            clip_bounds.invalidate();
        }

        for (size_t i = 0; i < clip_stack.size(); i++) {
            RectBounds bounds = clip_stack[i].clip->getClipBounds();
            clip_bounds &= bounds.transform(clip_stack[i].matrix);
        }
        if (clip_bounds.isValid()) {
            if (makingDlist) {
                // Don't put the scissor in display lists since scissor
                // is positioned absolutely.
            } else {
                int4 bbox = clipScissorBox();
                glEnable(GL_SCISSOR_TEST);
                glScissor(bbox.x, bbox.y, bbox.z-bbox.x, bbox.w-bbox.y);
            }

            RectBounds current_clip_bounds;
	        for (size_t i = 0; i < clip_stack.size(); i++) {
		        // Stencil the first clip path into the msb and the rest
		        // into the 2nd msb. They will do the intersection in end_fill()
                ClipVisitorPtr clip_visitor(new ClipVisitor( 
                    i == 0 ? clip_msb : clip_msb>>1, clip_msb,
                    clip_stack[i].clip->clip_merge, renderer,
                    clip_stack[i].matrix, current_clip_bounds));
    		    
                if (makingDlist) {
                    glMatrixPopEXT(GL_MODELVIEW);
                    glMatrixPushEXT(GL_MODELVIEW);
                    glMatrixMultTransposefEXT(GL_MODELVIEW, reinterpret_cast<GLfloat*>(&clip_stack[i].matrix));
                } else {
                    glMatrixLoadTransposefEXT(GL_MODELVIEW, 
                        reinterpret_cast<GLfloat*>(&clip_stack[i].matrix));
                }

                clip_visitor->beginFill();
                clip_stack[i].clip->path->traverse(clip_visitor);
		        clip_visitor->endFill();
            }

            loadCurrentMatrix();
        }

        glColorMask(1,1,1,1);
        glDepthMask(1);
    	clip_path_valid = true;
	}
}


///////////////////////////////////////////////////////////////////////////////
// ClipVisitor
ClipVisitor::ClipVisitor(GLuint bit_, GLuint clip_msb_, ClipMerge clip_merge, 
                         StCRendererPtr renderer_, 
                         const float4x4 &view_to_clip,
                         RectBounds &current_clip_bounds_)
    : StCVisitor(renderer_, view_to_clip) 
    , bit(bit_)
    , current_clip_bounds(current_clip_bounds_)
    , clip_msb(clip_msb_)
{
    assert(!(clip_msb & (clip_msb-1)));
    assert(!(bit & (bit-1)));

    switch (clip_merge) {
    case CLIP_COVERAGE_UNION:
        clip_filler = ClipFillerPtr(new UnionFiller);
        break;
    case SUM_WINDING_NUMBERS_MOD_2:
        clip_filler = ClipFillerPtr(new InvertFiller);
        break;
    default: // SUM_WINDING_NUMBERS:
        clip_filler = ClipFillerPtr(new SumFiller(*this));
        break;
    }
}

void ClipVisitor::fill(StCShapeRendererPtr shape_renderer)
{
    // Clipping should NOT check if shape_renderer->owner->isFillable() is false
    // and skip drawing the path.

    // http://www.w3.org/TR/SVG/masking.html#EstablishingANewClippingPath
    // "The raw geometry of each child element exclusive of rendering properties
    // such as ‘fill’, ‘stroke’, ‘stroke-width’ within a ‘clipPath’ conceptually
    // defines a 1-bit mask (with the possible exception of anti-aliasing along
    // the edge of the geometry) which represents the silhouette of the graphics
    // associated with that element. Anything outside the outline of the object
    // is masked out. If a child element is made invisible by ‘display’ or
    // ‘visibility’ it does not contribute to the clipping path."
    clip_filler->fill(shape_renderer, bit);

    RectBounds shape_bounds = shape_renderer->getShape()->getBounds();
    bounds |= shape_bounds.transform(matrix_stack.top());
}

void ClipVisitor::apply(ClipPtr clip)
{
    // This is a clip path for the real clip path
    ClipVisitorPtr clip_visitor(new ClipVisitor(
        bit, clip_msb, clip->clip_merge, renderer, 
        matrix_stack.top(), current_clip_bounds));
    clip_visitor->beginFill();
	clip->path->traverse(clip_visitor);
	clip_visitor->endFill();
	
    // We didn't get the msb anymore - they did; all Clips within a clip
    // path are _guaranteed_ (by the parser) to all occur before shapes
	if (bit == clip_msb) {
		bit >>= 1;
	}
}

void ClipVisitor::unapply(ClipPtr clip)
{
    // ignore
}

void ClipVisitor::beginFill()
{
    clip_filler->beginFill(bit);
}

void ClipVisitor::endFill()
{
    clip_filler->endFill(bit, bounds);
    if (bit != clip_msb) {
		intersectTop2StencilBits();
    }
    current_clip_bounds &= bounds;
}

void ClipVisitor::intersectTop2StencilBits()
{
    // Turn these on to visualize the coverage of these stencil ops
    // glColorMask(1,1,1,1);
    // glColor4f(0,0,0,1);

    // Since we weren't the first to draw, a clip path was already stencilled into the
    // msb, so we stencilled our clip path into the second-msb. Now we need to compute
    // the intersection of the two clip paths currently in the stencil buffer and replace
    // the msb with it.

    // This is what we want to do
    // for each pixel:
    //    if (stencil & clip_msb && stencil & bit) // If the pixel is within both clip paths
    //        stencil = clip_msb; // mark the top bit with a 1
    //    else
    //        stencil = 0; // 0 means we can't draw there - the msb is not a 1

    GLuint bit = clip_msb >> 1;
    RectBounds union_bounds = current_clip_bounds | bounds;

    glStencilMask(clip_msb | (clip_msb - 1));
    glStencilFunc(GL_EQUAL, clip_msb | bit, clip_msb | bit);
    glStencilOp(GL_ZERO, GL_KEEP, GL_REPLACE);
    coverRect(union_bounds);

    glStencilFunc(GL_EQUAL, clip_msb, clip_msb);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    coverRect(union_bounds);
}


///////////////////////////////////////////////////////////////////////////////
// DrawDilated
DrawDilated::DrawDilated(StCRendererPtr renderer_, GLuint clip_msb_,  
                         const float4x4 &view_to_clip,
                         int xsteps_, int ysteps_, 
                         bool stipple_, float2 spread_)
    : Draw(renderer_, clip_msb_, view_to_clip)
    , xsteps(xsteps_)
    , ysteps(ysteps_)
    , stipple(stipple_)
    , spread(spread_)
{
}

void DrawDilated::visit(ShapePtr shape) 
{
#if 0
    // XXX worth revisiting this!
    // Make sure to not form hulls from completely empty paths.
    PathRendererStatePtr renderer_state = shape->getPath()->getRendererState(renderer);
    StCPathRendererPtr path_renderer = dynamic_pointer_cast<StCPathRenderer>(renderer_state);
    assert(path_renderer);
    path_renderer->validate();
    bool is_empty = path_renderer->fill_vertex_array_buffer.isEmpty() &&
                    path_renderer->stroke_vertex_array_buffer.isEmpty();
    if (is_empty) {
        return;
    }
#endif

    Draw::visit(shape);
}

void DrawDilated::fill(StCShapeRendererPtr shape_renderer)
{
    if (shape_renderer->owner->isFillable()) {
        // If the path is also going to be stroked, don't worry about performing a dilated fill
        // as the stroke should cover the aliasing from a non-dilated fill.
        // XXX assumption might not be great if stroke is blended (probably probably still ok).
        if (doStroking && shape_renderer->getShape()->isStrokable()) {
            shape_renderer->fill(stencilClipFunc(), stencilClipBit(), stencilClipBit(), ZERO_ALPHA);
        } else {
            shape_renderer->fillDilated(xsteps, ysteps, stipple, spread, matrix_stack.top(),
                                        stencilClipFunc(), stencilClipBit(), stencilClipBit());
        }
    }
}

void DrawDilated::stroke(StCShapeRendererPtr shape_renderer)
{
    shape_renderer->strokeDilated(xsteps, ysteps, stipple, spread, matrix_stack.top(),
                                  stencilClipFunc(), stencilClipBit(), stencilClipBit());
}


///////////////////////////////////////////////////////////////////////////////
// ReversePainter
ReversePainter::ReversePainter(StCRendererPtr renderer_, GLuint filled_bit_,  
         const float4x4 &view_to_clip)
    : Draw(renderer_, filled_bit_ >> 1, view_to_clip)
    , filled_bit(filled_bit_)
{
}

void ReversePainter::visit(ShapePtr shape)
{
    StCShapeRendererPtr shape_renderer = 
        dynamic_pointer_cast<StCShapeRenderer>(shape->getRendererState(renderer));

    assert(shape_renderer);

    validateClipPath();
    
    // Done in the oppositte order of StCVisitor
    if (doStroking && shape->isStrokable()) {
        stroke(shape_renderer);
    }

    if (doFilling && shape->isFillable() && !shape->isEmpty()) {
        fill(shape_renderer);
    }
}

void ReversePainter::fill(StCShapeRendererPtr shape_renderer)
{
    if (shape_renderer->owner->isFillable()) {
        shape_renderer->fillTopToBottom(filled_bit, GL_EQUAL, stencilClipBit(), filled_bit | stencilClipBit());
    }
}

void ReversePainter::stroke(StCShapeRendererPtr shape_renderer)
{
    shape_renderer->strokeTopToBottom(filled_bit, GL_EQUAL, stencilClipBit(), filled_bit | stencilClipBit());
}


///////////////////////////////////////////////////////////////////////////////
// VisualizeClipScissor
void VisualizeClipScissor::visit(ShapePtr shape)
{
    bool was_valid = clip_path_valid;
    validateClipPath();
    if (!was_valid && clip_path_valid && clip_bounds.isValid()) {
        // Make a dashed line for glDrawPixels
        // We want this to be done in the same space as the scissor box in case there was
        // a bug going from clip coods to window coords
        // (I suppose this could also be done by loading the identity into the projection
        // and modelview, followed by a glOrtho(0,0, w,h, -1, 1)
        int4 bbox = clipScissorBox();
        int max_size = max(bbox.z - bbox.x, bbox.w - bbox.y);
        max_size = 5*((max_size + 4)/5);
        if (max_size < 10*1024) {
            GLuint *dash = new GLuint[max_size];
            for (int i = 0; i < max_size; i += 5) {
                if (i/5 & 1) {
                    dash[i+0] = ~0;
                    dash[i+1] = ~0;
                } else {
                    dash[i+0] = 0;
                    dash[i+1] = 0;
                    ((GLubyte*)&dash[i+0])[3] = 0xff;
                    ((GLubyte*)&dash[i+1])[3] = 0xff;
                }
                dash[i+2] = 0;
                dash[i+3] = 0;
                dash[i+4] = 0;
            }

            if (!makingDlist) {
                glDisable(GL_SCISSOR_TEST);
            }
            glDisable(GL_STENCIL_TEST);
            glEnable(GL_ALPHA_TEST);
            glAlphaFunc(GL_GREATER, 0.5f);
            
            glWindowPos2d(bbox.x-1, bbox.y-1);
            glDrawPixels(1 + bbox.z - bbox.x, 1, GL_RGBA, GL_UNSIGNED_BYTE, dash);

            glWindowPos2d(bbox.z, bbox.y-1);
            glDrawPixels(1, 1 + bbox.w - bbox.y, GL_RGBA, GL_UNSIGNED_BYTE, dash);

            glWindowPos2d(bbox.x, bbox.w);
            glDrawPixels(1 + bbox.z - bbox.x, 1, GL_RGBA, GL_UNSIGNED_BYTE, dash);

            glWindowPos2d(bbox.x-1, bbox.y);
            glDrawPixels(1, 1 + bbox.w - bbox.y, GL_RGBA, GL_UNSIGNED_BYTE, dash);

            glDisable(GL_ALPHA_TEST);

            delete[] dash;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// DrawControlPoints
void DrawControlPoints::visit(ShapePtr shape)
{
    shape->drawControlPoints();
}

void DrawReferencePoints::visit(ShapePtr shape)
{
    shape->drawReferencePoints();
}

GatherStats::GatherStats(StCRendererPtr renderer, PathStats &t, PathStats &m) 
    : StCVisitor(renderer)
    , total_stats(t)
    , max_stats(m)
{
}

void GatherStats::visit(ShapePtr shape)
{
    PathStats stats;

    shape->getPath()->gatherStats(stats);
    total_stats.add(stats);
    max_stats.max(stats);
}

}
