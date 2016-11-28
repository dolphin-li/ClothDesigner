
/* renderer_nvpr.cpp - NV_path_rendering path rendering. */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <iostream>

#include <Cg/vector/xyzw.hpp>
#include <Cg/vector/rgba.hpp>
#include <Cg/double.hpp>
#include <Cg/vector.hpp>
#include <Cg/matrix.hpp>

#include <Cg/distance.hpp>
#include <Cg/max.hpp>
#include <Cg/min.hpp>
#include <Cg/mul.hpp>
//#include <Cg/lerp.hpp>  // This has problems with Visual Studio 2008
#define lerp(a,b,t) ((a) + (t)*((b)-(a)))

#include "nvpr_svg_config.h"  // configure path renderers to use

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "renderer_nvpr.hpp"
#include "scene.hpp"
#include "path.hpp"

#include "sRGB_vector.hpp"

#include "dsa_emulate.h"

#if USE_NVPR

#include "nvpr_init.h"

#ifdef NDEBUG
const static int verbose = 0;
#else
const static int verbose = 1;
#endif

using std::cout;
using std::endl;

void NVprRenderer::initExtensions(bool noDSA)
{
    glewInit();
    bool forceDSAemulation = !use_DSA || noDSA;
    emulate_dsa_if_needed(forceDSAemulation);
}

void NVprRenderer::interrogateFramebuffer()
{
    supersample_ratio = 1;
    num_samples = 1;
    glGetIntegerv(GL_SAMPLES, &num_samples);
    if (num_samples < 1) {
        num_samples = 1;
    } else {
        if (has_ARB_sample_shading) {
            glMinSampleShadingARB(1.0);
        }
    }

    alpha_bits = 0;
    glGetIntegerv(GL_ALPHA_BITS, &alpha_bits);

    // Get sample locations if obtainable via NV_explicit_multisample
    for (int j=0; j<num_samples; j++) {
        GLfloat raw_sample_location[2] = { 0.5, 0.5 };
        if (has_NV_explicit_multisample && num_samples > 1) {
            glGetMultisamplefvNV(GL_SAMPLE_POSITION_NV, j, raw_sample_location);
        }
        float2 sample_location = float2(raw_sample_location[0], raw_sample_location[1]);
        if (verbose) {
            std::cout << "sample_location " << j << " @ " << sample_location << std::endl;
        }
        sample_positions.push_back(sample_location);
    }

    multisampling = num_samples > 1;

    // setWindowSize will set these later
    view_width_height = int2(1);

    if (has_EXT_framebuffer_sRGB) {
        GLint get_sRGB_capable = 0;
        glGetIntegerv(GL_FRAMEBUFFER_SRGB_CAPABLE_EXT, &get_sRGB_capable);
        sRGB_capable = !!get_sRGB_capable;
    } else {
        sRGB_capable = false;
    }
}

void NVprRenderer::reportGL()
{
    printf("NVpr: vendor: %s\n", glGetString(GL_VENDOR));
    printf("NVpr: renderer: %s\n", glGetString(GL_RENDERER));
    printf("NVpr: alpha_bits = %d\n", alpha_bits);
    printf("NVpr: numSamples = %d\n", num_samples);
    if (has_NV_explicit_multisample) {
        printf("NVpr: has NV_explicit_multisample\n");
        if (num_samples == 16) {
            printf("NVpr: info: 16 samples means assuming supersample ratio of %d\n", supersample_ratio);
        }
    }
    if (has_ARB_sample_shading) {
        printf("NVpr: has ARB_sample_shading\n");
    }
    printf("NVpr: sRGB capable = %d\n", sRGB_capable);
}

void NVprRenderer::configureGL()
{
    if (num_samples > 1) {
        if (has_NV_explicit_multisample) {
            glEnable(GL_SAMPLE_MASK_NV);
        }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
}

NVprRenderer::NVprRenderer(bool noDSA,
                           const char *vertex_profile_name,
                           const char *fragment_profile_name)
    : myCgContext(0)
    , myCgFragmentProfile(CG_PROFILE_UNKNOWN)
    , myCgVertexProfile(CG_PROFILE_UNKNOWN)
    , radial_center_gradient_program(0)
    , radial_focal_gradient_program(0)
{
    has_ARB_sample_shading = !!glutExtensionSupported("GL_ARB_sample_shading");
    has_EXT_direct_state_access = !!glutExtensionSupported("GL_EXT_direct_state_access");
    has_EXT_texture_filter_anisotropic = !!glutExtensionSupported("GL_EXT_texture_filter_anisotropic");
    has_NV_explicit_multisample = !!glutExtensionSupported("GL_NV_explicit_multisample");
    has_EXT_framebuffer_sRGB = !!glutExtensionSupported("GL_EXT_framebuffer_sRGB");
    has_EXT_texture_sRGB = !!glutExtensionSupported("GL_EXT_texture_sRGB");

    has_texture_non_power_of_two = !!glutExtensionSupported("GL_ARB_texture_non_power_of_two");

    use_DSA = has_EXT_direct_state_access;

    if (has_EXT_texture_filter_anisotropic) {
        glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_anisotropy_limit);
        max_anisotropy = 2;  // default to 2-to-1 texture filter anisotropy
    }

    initExtensions(noDSA);
   // initializeNVPR("nvpr_svg");

    interrogateFramebuffer();

    reportGL();

    configureGL();

    init_cg(vertex_profile_name, fragment_profile_name);
}

shared_ptr<RendererState<Path> > NVprRenderer::alloc(Path *owner)
{
    return NVprPathRendererStatePtr(new NVprPathRendererState(shared_from_this(), owner));
}

shared_ptr<RendererState<Shape> > NVprRenderer::alloc(Shape *owner)
{
    assert(owner);
    return StCShapeRendererPtr(new StCShapeRenderer(shared_from_this(), owner));
}

void NVprSolidColorPaintRendererState::validate()
{
    valid = true;
}

OpacityTreatment NVprSolidColorPaintRendererState::startShading(Shape *shape, float opacity, RenderOrder render_order)
{
    SolidColorPaint *paint = dynamic_cast<SolidColorPaint*>(owner);
    assert(paint);
    if (paint) {
        float4 color = paint->getColor();

        color.a *= opacity;

        glPathTexGenNV(GL_TEXTURE0, GL_NONE, 0, NULL);

        if (render_order == BOTTOM_FIRST_TOP_LAST) {
            if (getRenderer()->render_sRGB) {
                float4 lin_color = srgb2linear(color);
                glColor4fv(reinterpret_cast<const GLfloat*>(&lin_color));
            } else {
                glColor4fv(reinterpret_cast<const GLfloat*>(&color));
            }
            if (color.a == 1.0) {
                return FULLY_OPAQUE;
            } else {
                return VARIABLE_INDEPENDENT_ALPHA;
            }
        } else {
            assert(render_order == TOP_FIRST_BOTTOM_LAST);

            if (color.a == 1.0) {
                if (getRenderer()->render_sRGB) {
                    float4 lin_color = srgb2linear(color);
                    glColor4fv(reinterpret_cast<const GLfloat*>(&lin_color));
                } else {
                    glColor4fv(reinterpret_cast<const GLfloat*>(&color));
                }
                return FULLY_OPAQUE;
            } else {
                float4 premultipled_alpha_color;
                if (getRenderer()->render_sRGB) {
                    premultipled_alpha_color = srgb2linear(color);
                } else {
                    premultipled_alpha_color = color;
                }
                premultipled_alpha_color.rgb *= premultipled_alpha_color.a;
                glColor4fv(reinterpret_cast<GLfloat*>(&premultipled_alpha_color));
                return VARIABLE_PREMULTIPLIED_ALPHA;
            }
        }
    }
    return FULLY_OPAQUE;
}

void NVprSolidColorPaintRendererState::stopShading(RenderOrder render_order)
{
}

static GLenum convertSVGSpreadMethodToGLWrapMode(SpreadMethod v)
{
    switch (v) {
    case PAD:
        return GL_CLAMP_TO_EDGE;
    case REFLECT:
        return GL_MIRRORED_REPEAT;
    case REPEAT:
        return GL_REPEAT;
    default:
        assert(!"bogus spread method");
    case NONE:
        return GL_CLAMP_TO_BORDER;
    }
}

void NVprGradientPaintRendererState::validateGenericGradientState(GradientPaint *paint)
{
    const bool use_sRGB = getRenderer()->render_sRGB;

    vector<float4> ramp(getRenderer()->color_ramp_size);
    const size_t ramp_size_minus_one = ramp.size()-1;

    const vector<GradientStop> &stop_array = paint->getStopArray();
    assert(stop_array.size() >= 2);
    vector<GradientStop>::const_iterator stop = stop_array.begin();

    fullyOpaqueGradient = true;
    float offset1 = stop->offset;
    float4 color1 = stop->color;
    if (color1.a != 1.0) {
        fullyOpaqueGradient = false;
    }

    size_t i;
    for (i=0; i<ramp_size_minus_one; i++) {
        float texel_offset = float(i)/ramp_size_minus_one;

        if (texel_offset <= offset1) {
            ramp[i] = color1;
        } else {
            break;
        }
    }
    stop++;
    float4 color2;
    if (use_sRGB) {
        float4 lin_color1 = srgb2linear(color1), lin_color2;

        while (stop != stop_array.end()) {
            assert(stop != stop_array.end());
            // "Each gradient offset value is required to be equal to 
            // or greater than the previous gradient stop's offset 
            // value. If a given gradient stop's offset value is not 
            // equal to or greater than all previous offset values,
            // then the offset value is adjusted to be equal to the 
            // largest of all previous offset values."
            float offset2 = max(stop->offset, offset1);
            color2 = stop->color;
            lin_color2 = srgb2linear(color2);
            if (color2.a != 1.0) {
                fullyOpaqueGradient = false;
            }
            
            for (; i<ramp_size_minus_one; i++) {
                float texel_offset = float(i)/ramp_size_minus_one;
                float weight = (texel_offset-offset1)/(offset2-offset1);
                float4 weighted_color;
                if (weight >= 0 && weight <= 1) {
                    // Blend linear colors and converted result to sRGB.
                    weighted_color = linear2srgb(lerp(lin_color1, lin_color2, weight));
                } else {
                    weighted_color = color2;
                }

                if (texel_offset <= offset2) {
                    ramp[i] = weighted_color;
                } else {
                    break;
                }
            }
            color1 = color2;
            lin_color1 = lin_color2;
            offset1 = offset2;
            stop++;
        }
    } else {
        while (stop != stop_array.end()) {
            assert(stop != stop_array.end());
            // "Each gradient offset value is required to be equal to 
            // or greater than the previous gradient stop's offset 
            // value. If a given gradient stop's offset value is not 
            // equal to or greater than all previous offset values,
            // then the offset value is adjusted to be equal to the 
            // largest of all previous offset values."
            float offset2 = max(stop->offset, offset1);
            color2 = stop->color;
            if (color2.a != 1.0) {
                fullyOpaqueGradient = false;
            }
            for (; i<ramp_size_minus_one; i++) {
                float texel_offset = float(i)/ramp_size_minus_one;
                float weight = (texel_offset-offset1)/(offset2-offset1);
                float4 weighted_color;
                if (weight >= 0 && weight <= 1) {
                    weighted_color = lerp(color1, color2, weight);
                } else {
                    weighted_color = color2;
                }

                if (texel_offset <= offset2) {
                    ramp[i] = weighted_color;
                } else {
                    break;
                }
            }
            color1 = color2;
            offset1 = offset2;
            stop++;
        }
    }
    for (; i<ramp_size_minus_one; i++) {
        ramp[i] = color2;
    }
    ramp[ramp_size_minus_one] = stop_array.back().color;

    if (texobj == 0) {
        glGenTextures(1, &texobj);
        assert(texobj != 0);
    }
    glBindTexture(GL_TEXTURE_1D, texobj);
    // Are we rendering in sRGB mode?
    if (use_sRGB) {
        // Yes, so generate the color ramp mipmaps with linear RGB color averaging.
        GLsizei width = GLsizei(ramp.size());
        GLint lod = 0;
    
        assert(getRenderer()->has_EXT_texture_sRGB);
        glTexImage1D(GL_TEXTURE_1D, lod, GL_SRGB8_ALPHA8, width, 0, GL_RGBA, GL_FLOAT, &ramp[0]);
        // Until all the mipmap levels are created...
        while (width > 1) {
            int half_width = width >> 1;  // integer divide rounds down
            assert(half_width > 0);
            // Downsample the 1D texture "in place".
            for (int i=0; i<half_width; i++) {
                // Convert adjacent color pairs to linear RGB.
                float4 a = srgb2linear(ramp[2*i+0]);
                float4 b = srgb2linear(ramp[2*i+1]);
                // Average the linear pairs and convert back to sRGB.
                ramp[i] = linear2srgb(0.5f*(a+b));
            }
            lod += 1;
            width = half_width;
            glTexImage1D(GL_TEXTURE_1D, lod, GL_SRGB8_ALPHA8, width, 0, GL_RGBA, GL_FLOAT, &ramp[0]);
        }
    } else {
        // Let the driver generate the mipmaps.
        glTexParameteri(GL_TEXTURE_1D, GL_GENERATE_MIPMAP, GL_TRUE);
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, GLsizei(ramp.size()), 0, GL_RGBA, GL_FLOAT, &ramp[0]);
    }
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, getRenderer()->min_filter);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, getRenderer()->mag_filter);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, convertSVGSpreadMethodToGLWrapMode(paint->getSpreadMethod()));
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_ANISOTROPY_EXT, getRenderer()->max_anisotropy);
}

void NVprLinearGradientPaintRendererState::validate()
{
    if (valid) {
        return;
    }

    valid = true;

    LinearGradientPaint *paint = dynamic_cast<LinearGradientPaint*>(owner);
    assert(paint);
    if (paint) {
        validateGenericGradientState(paint);
    }
}

// Make 3x3 matrix that converts (x,y) to a t where t=0 at (x1,y1) and t=1 at (x2,y2)
// XXX show derivation of this math
static double3x3 makeMatrix(float x1, float y1, float x2, float y2)
{
    double t1 = x2*x2;
    double t4 = x1*x1;
    double t5 = y2*y2;
    double t8 = y1*y1;
    double t9 = t1-2.0*x2*x1+t4+t5-2.0*y2*y1+t8;
    double t10 = 1/t9;
    double t11 = x2-x1;
    double t12 = t10*t11;
    double t13 = y2-y1;
    double t14 = t10*t13;
    double t18 = sqrt(t9);
    double t19 = 1/t18;
    double t20 = t13*t19;
    double t21 = t11*t19;

    double3x3 Z;
    Z[0][0] = t12;
    Z[0][1] = t14;
    Z[0][2] = -t12*x1-t14*y1;
    Z[1][0] = -t20;
    Z[1][1] = t21;
    Z[1][2] = t20*x1-t21*y1;
    Z[2][0] = 0.0;
    Z[2][1] = 0.0;
    Z[2][2] = 1.0;
    return Z;
}

OpacityTreatment NVprLinearGradientPaintRendererState::startShading(Shape *shape, float opacity, RenderOrder render_order)
{
    if (!valid) {
        validate();
    }

    LinearGradientPaint *paint = dynamic_cast<LinearGradientPaint*>(owner);
    assert(paint);
    if (paint) {
        glBindTexture(GL_TEXTURE_1D, texobj);
        glEnable(GL_TEXTURE_1D);

        const float2 v1 = paint->getV1(),
                     v2 = paint->getV2();
        double3x3 normal_to_gradient = makeMatrix(v1.x, v1.y, v2.x, v2.y);
        double3x3 result;

        if (paint->getGradientUnits() == OBJECT_BOUNDING_BOX) {
            float4 bounds = shape->getPath()->getActualFillBounds();
            float2 p1 = bounds.xy,
                   p2 = bounds.zw,
                   diff = p2-p1,
                   sum = p2+p1;

            double3x3 user_to_normal = double3x3(1/diff.x,0,-0.5*sum.x/diff.x+0.5,
                                                 0,1/diff.y,-0.5*sum.y/diff.y+0.5,
                                                 0,0,1);
            result = mul(normal_to_gradient, mul(user_to_normal, paint->getInverseGradientTransform()));
        } else {
            assert(paint->getGradientUnits() == USER_SPACE_ON_USE);
            result = mul(normal_to_gradient, paint->getInverseGradientTransform());
        }

        GLfloat params[3];
        params[0] = result[0][0];
        params[1] = result[0][1];
        params[2] = result[0][2];
        glPathTexGenNV(GL_TEXTURE0, GL_OBJECT_LINEAR, 1, params);
        glColor4f(1,1,1, opacity);
    }
    if (opacity != 1.0 || !fullyOpaqueGradient) {
        return VARIABLE_INDEPENDENT_ALPHA;
    } else {
        return FULLY_OPAQUE;
    }
}

void NVprLinearGradientPaintRendererState::stopShading(RenderOrder render_order)
{
    glDisable(GL_TEXTURE_1D);
}

void NVprRadialGradientPaintRendererState::validate()
{
    if (valid) {
        return;
    }

    valid = true;

    RadialGradientPaint *paint = dynamic_cast<RadialGradientPaint*>(owner);
    assert(paint);
    if (paint) {
        validateGenericGradientState(paint);
    }
}

/* Detailed explanation by Antti Rasmus:
 *
 * Generation matrix is of the from:
 *
 * [ X 0 0 X ]      X actual value
 * [ 0 X 0 0 ]      0 will always be zero
 * [ X 0 0 X ]      . don't care
 * [ . . . . ]
 *
 * Some assumptions of the input coordinates: 
 * - The radius of the circle is always one.
 * - The center point of the circle is at the origin (0, 0)
 * - The focal point lies on X axis, i.e. focalPoint.y = 0
 * - The focal point is inside the circle, i.e. abs(focalPoint.x) < 1
 *
 * The general radial gradient function is of the form
 *
 * g(vecP) = ( dot(vecQ, vecFC) + sqrt(r^2 * dot(vecQ,vecQ) - cross(vecQ,vecFC)^2) )
 *           / ( r^2 - dot(vecFC,vecFC) ) ,
 * where 
 *      vecP is the vector from the origin to the point being currently drawn vecP = vec2(x,y)
 *      vecF is the vector from the origin to the focal point
 *      vecC is the vector from the origin to the center point
 *      vecQ = vecP - vecF
 *      vecFC is the difference vector between the focal point and the center point (vecF - vecC),
 *      r is the radius of the circle.
 *
 * - By assumption, r = 1
 * - By assumption, vecC = (0,0)  =>  vecFC = vecF
 * - Denominator is a constant. Let's call its inverse A, so that A = 1 / (1 - dot(vecF, vecF))
 *
 * After some rearranging, we get
 *
 * g(vecP) = A * dot(vecQ, vecF) + sqrt(A^2 * dot(vecQ,vecQ) - A^2 * cross(vecQ,vecF)^2),
 *
 * Let's open it up further
 * - By assumption, vecF.y is always zero. Thus,
 *      cross(vecQ, vecF) = -vecF.x * vecQ.y
 * - Dot(vecQ, vecQ) equals vecQ.x^2 + vecQ.y^2.
 * - We can combine constant multipliers of vecQ.y into B = A * sqrt(1 - (-vecF.x)^2)
 *
 * g(vecP) = A * vecF.x*vecQ.x + sqrt( (A*vecQ.x)^2 + (B*vecQ.y)^2 ) 
 *         = A * vecF.x*vecQ.x + length( vecR ) ,
 * where
 *      vecR = vec2( A*vecQ.x, B*vecQ.y) = vec2( A*(x-vecF.x), B*y )
 *            
 * We can add a new scalar variable z = A * dot(vecF*vecQ) = A * vecF.x * vecQ.x and get,
 * g'(vecR, z) = z + length( vecR ),
 *
 * Each row of the m_genMatrix defines one gl_PathCoord[0] component. Hence, 
 * the two first rows will represent the vecR.x and vecR.y, and z will be the third.
 */
static double3x3 radialNormalToCoords(float2 f, float2 c, float radius, float texScale, float texOffset)
{
    const double dist = distance(f,c);
    const double rdist = radius*dist;
    // Don't make epsilon too small because then focal points near or outside the circle
    // can have distances to the circle that are nearly zero, causing unstable
    // texture coordinates.
    const double epsilon = 0.0001;
    double3x3 normal_to_canonical;
    if (rdist < epsilon) {
        double one_over_r = 1.0/radius;
        normal_to_canonical[0] = double3(one_over_r, 0, -c.x*one_over_r);
        normal_to_canonical[1] = double3(0, one_over_r, -c.y*one_over_r);
    } else {
        double one_over_rdist = 1.0/(radius*distance(f,c));
        normal_to_canonical[0] = double3((f-c), dot(c-f,c)) * one_over_rdist;
        normal_to_canonical[1] = double3((c.y-f.y), (f.x-c.x), (f.y-c.y)*c.x - (f.x-c.x)*c.y) * one_over_rdist;
    }
    normal_to_canonical[2] = double3(0,0,1);

    const double fx = min(1.0-epsilon, dist/radius),
                 A = 1.0 / (1 - fx*fx),
                 B = sqrt(A);

    double3x3 canonical_to_coords;
    canonical_to_coords[0] = double3(A*texScale, 0, -A*fx*texScale);
    canonical_to_coords[1] = double3(0, B*texScale, 0);
    canonical_to_coords[2] = double3(A*fx*texScale, 0, -A*fx*fx * texScale + texOffset); 

    double3x3 normal_to_coords = mul(canonical_to_coords, normal_to_canonical);

    return normal_to_coords;
}

OpacityTreatment NVprRadialGradientPaintRendererState::startShading(Shape *shape, float opacity, RenderOrder render_order)
{
    if (!valid) {
        validate();
    }

    RadialGradientPaint *paint = dynamic_cast<RadialGradientPaint*>(owner);
    assert(paint);
    if (paint) {
        const float2 &c = paint->getCenter(),
                     &f = paint->getFocalPoint();
        float r = paint->getRadius();

        double3x3 normal_to_coords = radialNormalToCoords(f, c, r, 1, 0);
        double3x3 positions_to_coords;

        if (paint->getGradientUnits() == OBJECT_BOUNDING_BOX) {
            float4 bounds = shape->getPath()->getActualFillBounds();
            float2 p1 = bounds.xy,
                   p2 = bounds.zw,
                   diff = p2-p1,
                   sum = p2+p1;

            double3x3 user_to_normal = double3x3(1/diff.x,0,-0.5*sum.x/diff.x+0.5,
                                                 0,1/diff.y,-0.5*sum.y/diff.y+0.5,
                                                 0,0,1);
            if (verbose) {
                cout << "user_to_normal = " << user_to_normal << endl;
            }
            positions_to_coords = mul(normal_to_coords, mul(user_to_normal, paint->getInverseGradientTransform()));
        } else {
            assert(paint->getGradientUnits() == USER_SPACE_ON_USE);
            if (verbose) {
                cout << "paint->inverse_gradient_transform = " << paint->getInverseGradientTransform() << endl;
            }
            positions_to_coords = mul(normal_to_coords, paint->getInverseGradientTransform());
        }
        if (verbose) {
            cout << "positions_to_coords = " << positions_to_coords << endl;
        }

        GLfloat params[3][3];
        params[0][0] = positions_to_coords[0][0];
        params[0][1] = positions_to_coords[0][1];
        params[0][2] = positions_to_coords[0][2];
        params[1][0] = positions_to_coords[1][0];
        params[1][1] = positions_to_coords[1][1];
        params[1][2] = positions_to_coords[1][2];
        // Is the radial gradient really just a center radial gradient?
        if (positions_to_coords[2][0] == 0 &&
            positions_to_coords[2][1] == 0 &&
            positions_to_coords[2][2] == 0) {
            // Optimization:  No need to process the R texture coordinate.
            glPathTexGenNV(GL_TEXTURE0, GL_OBJECT_LINEAR, 2, &params[0][0]);
            getRenderer()->configureShading(NVprRenderer::RADIAL_CENTER_GRADIENT);
        } else {
            params[2][0] = positions_to_coords[2][0];
            params[2][1] = positions_to_coords[2][1];
            params[2][2] = positions_to_coords[2][2];
            glPathTexGenNV(GL_TEXTURE0, GL_OBJECT_LINEAR, 3, &params[0][0]);
            getRenderer()->configureShading(NVprRenderer::RADIAL_FOCAL_GRADIENT);
        }
        glBindTexture(GL_TEXTURE_1D, texobj);
        glColor4f(1,1,1, opacity);
        getRenderer()->enableFragmentShading();
    }
    if (opacity != 1.0 || !fullyOpaqueGradient) {
        return VARIABLE_INDEPENDENT_ALPHA;
    } else {
        return FULLY_OPAQUE;
    }
}

void NVprRadialGradientPaintRendererState::stopShading(RenderOrder render_order)
{
    getRenderer()->disableFragmentShading();
}

shared_ptr<RendererState<Paint> > NVprRenderer::alloc(Paint *owner)
{
    SolidColorPaint *solid_color_paint = dynamic_cast<SolidColorPaint*>(owner);
    if (solid_color_paint) {
        return NVprSolidColorPaintRendererStatePtr(new NVprSolidColorPaintRendererState(shared_from_this(), solid_color_paint));
    }

    LinearGradientPaint *linear_gradient_paint = dynamic_cast<LinearGradientPaint*>(owner);
    if (linear_gradient_paint) {
        return NVprLinearGradientPaintRendererStatePtr(new NVprLinearGradientPaintRendererState(shared_from_this(), linear_gradient_paint));
    }

    RadialGradientPaint *radial_gradient_paint = dynamic_cast<RadialGradientPaint*>(owner);
    if (radial_gradient_paint) {
        return NVprRadialGradientPaintRendererStatePtr(new NVprRadialGradientPaintRendererState(shared_from_this(), radial_gradient_paint));
    }

    ImagePaint *image_paint = dynamic_cast<ImagePaint*>(owner);
    if (image_paint) {
        return NVprImagePaintRendererStatePtr(new NVprImagePaintRendererState(shared_from_this(), image_paint));
    }

    assert(!"paint unsupported by NVpr renderer");
    return NVprPaintRendererStatePtr();
}

void NVprPaintRendererState::invalidate()
{
    valid = false;
}

NVprImagePaintRendererState::~NVprImagePaintRendererState()
{
    if (texture) {
        glDeleteTextures(1, &texture);
    }
}

void NVprImagePaintRendererState::validate()
{
    if (!texture) {
        ImagePaint *image_paint = (ImagePaint *)owner;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
        
        // Simple pattern to test image drawing w/o the loader
        /*RasterImage::Pixel* pattern = new RasterImage::Pixel[image_paint->image->width*image_paint->image->height];
        for (int y = 0; y < image_paint->image->height; y++) {
            for (int x = 0; x < image_paint->image->width; x++) {
                RasterImage::Pixel* pix = &pattern[(image_paint->image->width*y + x)];
                pix->r = 255 * x / image_paint->image->width;
                pix->g = 127 * (1 + sinf(x*y));
                pix->b = 255 * y / image_paint->image->height;
                pix->a = 255;
            }
        }
        delete[] pattern;*/
        glTexImage2D(GL_TEXTURE_2D, 0, 4, image_paint->image->width, image_paint->image->height, 
            0, GL_RGBA, GL_UNSIGNED_BYTE, image_paint->image->pixels);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, getRenderer()->min_filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, getRenderer()->mag_filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, getRenderer()->max_anisotropy);
    }
}

OpacityTreatment NVprImagePaintRendererState::startShading(Shape *shape, float opacity, RenderOrder render_order)
{
    //ImagePaint *image_paint = (ImagePaint *)owner;
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    const GLfloat params[2][3] = { {1,0,0},
                                   {0,1,0} };
    glPathTexGenNV(GL_TEXTURE0, GL_OBJECT_LINEAR, 2, &params[0][0]);

    glColor4f(1,1,1,1);

    if (owner->isOpaque()) {
        return FULLY_OPAQUE;
    } else {
        return VARIABLE_PREMULTIPLIED_ALPHA;
    }
}

void NVprImagePaintRendererState::stopShading(RenderOrder render_order)
{
    glDisable(GL_TEXTURE_2D);
}

#endif // USE_NVPR
