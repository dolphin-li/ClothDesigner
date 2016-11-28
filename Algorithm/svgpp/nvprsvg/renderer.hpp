
/* renderer.hpp - abstract renderer class. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#ifndef __renderer_hpp__
#define __renderer_hpp__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/weak_ptr.hpp>

#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <Cg/vector.hpp>
#include <Cg/matrix.hpp>

using std::vector;
using namespace boost;

// Path scene graph classes to which a Renderer implementation can attach a RenderState
struct Shape;
typedef shared_ptr<struct Shape> ShapePtr;
struct Group;
typedef shared_ptr<Group> GroupPtr;
struct Paint;
typedef shared_ptr<struct Paint> PaintPtr;
struct Path;
typedef shared_ptr<struct Path> PathPtr;
struct Text;
typedef shared_ptr<struct Text> TextPtr;

// Forward declaration
template <typename T> struct RendererState;

// Abstract base class for a path renderer with support for allocating 
// Renderer-specific path and shape renderer states.
struct Renderer : enable_shared_from_this<Renderer> {
    virtual const char *getWindowTitle() = 0;
    virtual const char *getName() = 0;
    virtual void reportFPS() = 0;
    virtual void swapBuffers() = 0;
    virtual shared_ptr<RendererState<Path> > alloc(Path *owner) = 0;
    virtual shared_ptr<RendererState<Shape> > alloc(Shape *owner) = 0;
    virtual shared_ptr<RendererState<Paint> > alloc(Paint *owner) = 0;
    virtual ~Renderer() { };
};
typedef shared_ptr<struct Renderer> RendererPtr;
typedef weak_ptr<struct Renderer> WeakRendererPtr;

typedef shared_ptr<struct BlitRenderer> BlitRendererPtr;
typedef shared_ptr<class Visitor> VisitorPtr;
struct BlitRenderer : Renderer {
    virtual void configureSurface(int width, int height) = 0;
    virtual void beginDraw() { }
    virtual void clear(Cg::float3 clear_color) = 0;
    virtual void setView(Cg::float4x4 view) = 0;
    virtual VisitorPtr makeVisitor() = 0;
    virtual void endDraw() { }
    //virtual void draw(GroupPtr scene, RendererPtr renderer) = 0;
    virtual void copyImageToWindow() = 0;
    virtual void shutdown() = 0;
};

struct GLBlitRenderer : BlitRenderer {
    void reportFPS();
    void swapBuffers();
};

// A RendererState is associated with a Towner object in the path scene graph.
//
// The SpecificRendererState class derives from this abstract base class.
// Implementations of renderer states for a specific Renderer should
// derive from SpecificRendererState (not RendererState itself).
//
// Users of a RendererState (typically the path scene graph) are expected to
// have no knowledge of what renderer actually implements this base class.
template <typename Towner>
struct RendererState {
    typedef Towner *OwnerPtr;  // intentionally not a SharedPtr

    // I know who my owner instance is.
    OwnerPtr owner;
    // I (weakly) know what Renderer implementation instance I am for.
    WeakRendererPtr renderer;

    RendererState(RendererPtr renderer_, OwnerPtr owner_) 
        : owner(owner_)
        , renderer(renderer_)
    {}
    virtual ~RendererState() {}

    // Am I a RendererState that belongs to the specified Renderer?
    bool sameRenderer(const RendererPtr renderer_) {
        RendererPtr locked_renderer = renderer.lock();

        if (locked_renderer) {
            return locked_renderer == renderer_;
        } else {
            return false;
        }
    }
    // The scene graph can invalidate my state at anytime.
    virtual void invalidate() = 0;
};

// The SpecificRendererState base class allows all specific
// renderer state implementations to access their appropriately
// specialized RendererPtr (CairoRendererPtr, etc.)
template <typename Towner, typename Trenderer>
struct SpecificRendererState : RendererState<Towner> {
    typedef Towner *OwnerPtr;  // intentionally not a SharedPtr
    SpecificRendererState(RendererPtr renderer_, OwnerPtr owner_) 
        : RendererState<Towner>(renderer_, owner_)
    {}
    ~SpecificRendererState() {}

    // Returns a (strong) pointer to my specific Renderer.
    shared_ptr<Trenderer> getRenderer() {
        RendererPtr locked_renderer = RendererState<Towner>::renderer.lock();
        assert(locked_renderer);
        shared_ptr<Trenderer> specific_renderer = dynamic_pointer_cast<Trenderer>(locked_renderer);
        assert(specific_renderer);
        return specific_renderer;
    }
};

// Objects that want to have per-renderer state should derive from this class.
//
// This class provides a generic vector of RendererState instances for
// various Renderer implementations.
template <typename T>
struct HasRendererState {
    typedef shared_ptr<T> OwnerType;
    typedef RendererState<T> RenderStateType;
    typedef shared_ptr<RenderStateType> RendererStatePtr;

    // The "this" pointer for my owner.
    T *owner;
    // The vector of RendererState instances, the details of which I am ignorant about.
    vector<RendererStatePtr> renderer_states;

    HasRendererState(T *owner_)
        : owner(owner_)  // I expect owner_ to be my owener's "this" pointer.
    { }

    // Given a shape in the path scene graph, return the specific
    // renderer state for this renderer implementation instance.
    RendererStatePtr getRendererState(RendererPtr renderer) {
        typename vector<RendererStatePtr>::iterator iter;

        for (iter = renderer_states.begin(); iter != renderer_states.end(); iter++) {
            RendererStatePtr renderer_state = *iter;

            if (renderer_state && renderer_state->sameRenderer(renderer) ) {
                return renderer_state;
            }
        }
        RendererStatePtr renderer_state = renderer->alloc(owner);
        renderer_states.push_back(renderer_state);
        return renderer_state;
    }
    // Force all renderers to revalidate their renderer states.
    //
    // Invalidation doesn't necessarily free any memory associated with
    // the renderer state implementations.
    void invalidateRenderStates() {
        typename vector<RendererStatePtr>::iterator iter;

        for (iter = renderer_states.begin(); iter != renderer_states.end(); iter++) {
            RendererStatePtr renderer_state = *iter;

            if (renderer_state) {
                renderer_state->invalidate();
            }
        }
    }

    // Force a particular renderer to revalidate its renderer state.
    void invalidateRenderState(RendererPtr renderer) {
        typename vector<RendererStatePtr>::iterator iter;

        for (iter = renderer_states.begin(); iter != renderer_states.end(); iter++) {
            RendererStatePtr renderer_state = *iter;

            if (renderer_state && renderer_state->sameRenderer(renderer) ) {
                renderer_state->invalidate();
                return;
            }
        }
    }

    // More expensive than simply invalidating the associated renderer
    // states of this object because the renderer states themselves
    // are forgotten.
    void flushRenderStates() {
        // assign empty vector; side-effect will destruct all RendererState in old vector.
        renderer_states = vector<RendererStatePtr>();
    }

    void flushRenderState(RendererPtr renderer) {
        typename vector<RendererStatePtr>::iterator iter;

        for (iter = renderer_states.begin(); iter != renderer_states.end(); iter++) {
            if (*iter && (*iter)->sameRenderer(renderer) ) {
                *iter = RendererStatePtr();
                return;
            }
        }
    }
};

#endif // __renderer_hpp__
