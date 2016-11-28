
// freetype2_loader.cpp - use FreeType2 library to load outlines of glyphs from fonts

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include "nvpr_svg_config.h"  // configure path renderers to use

#if USE_FREETYPE2

#include <assert.h>
#include <vector>

#include "freetype2_loader.hpp"

#include <ft2build.h>
#include FT_FREETYPE_H 
#include FT_GLYPH_H
#include FT_BBOX_H
#include FT_OUTLINE_H
#include FT_OUTLINE_H

#include "path.hpp"
#include "countof.h"

using std::vector;

FT_Library library;
FT_Face face;

static bool initialized = false;

struct Font {
    const char *name;
    const char *file;
    bool character_set;
    FT_Face face;
};

// Table mapping font names to font filenames.
Font font_list[] = {
    { "Pacifico", "Pacifico.ttf", true, 0 },
    { "Arial", "arial.ttf", true, 0 },
    { "Georgia", "georgia.ttf", true, 0 },
    { "Courier", "cour.ttf", true, 0 },
    { "Times", "times.ttf", true, 0 },
    { "Times bold", "timesbd.ttf", true, 0 },
    { "Times bold italic", "timesbi.ttf", true, 0 },
    { "Times italic", "timesi.ttf", true, 0 },
    { "Wingding", "wingding.ttf", false, 0 },
    { "Webdings", "webdings.ttf", false, 0 },

    { "LiberationMono-Bold", "LiberationMono-Bold.ttf", true, 0 },
    { "LiberationMono-BoldItalic", "LiberationMono-BoldItalic.ttf", true, 0 },
    { "LiberationMono-Italic", "LiberationMono-Italic.ttf", true, 0 },
    { "LiberationMono-Regular", "LiberationMono-Regular.ttf", true, 0 },
    { "LiberationSans-Bold", "LiberationSans-Bold.ttf", true, 0 },
    { "LiberationSans-BoldItalic", "LiberationSans-BoldItalic.ttf", true, 0 },
    { "LiberationSans-Italic", "LiberationSans-Italic.ttf", true, 0 },
    { "LiberationSans-Regular", "LiberationSans-Regular.ttf", true, 0 },
    { "LiberationSerif-Bold", "LiberationSerif-Bold.ttf", true, 0 },
    { "LiberationSerif-BoldItalic", "LiberationSerif-BoldItalic.ttf", true, 0 },
    { "LiberationSerif-Italic", "LiberationSerif-Italic.ttf", true, 0 },
    { "LiberationSerif-Regular", "LiberationSerif-Regular.ttf", true, 0 },
};
static const int num_font_list = sizeof(font_list)/sizeof(font_list[0]);

#ifndef _WIN32
#include <strings.h>
#define strnicmp strncasecmp
#endif

int lookup_font(const char *name)
{
    static const char prefix_string[] = "fonts/";
    static const size_t prefix_len = sizeof(prefix_string)-1;
    bool prefixed = false;

    if (!strnicmp("fonts/", name, prefix_len)) {
        prefixed = true;
    }
    for (int i=0; i<num_font_list; i++) {
        if (!strcmp(font_list[i].name, name)) {
            return i;
        }
        if (!strcmp(font_list[i].file, name)) {
            return i;
        }
        if (prefixed) {
            if (!strcmp(font_list[i].file, name+prefix_len)) {
                return i;
            }
        }
    }
    printf("font %s not listed as a supported font\n", name);
    exit(1);
    return 0;
}

void init_freetype2()
{
    FT_Error error;

    if (initialized) {
        return;
    }
    
    error = FT_Init_FreeType(&library);
    if (error) {
        fprintf(stderr, "FT_Init_FreeType failed\n");
    }


    initialized = true;
}

struct PathGenInfo {
    vector<char> cmds;
    vector<float> coords;
};

static int move_to(const FT_Vector* to,
                   void* user)
{
    PathGenInfo *info = reinterpret_cast<PathGenInfo*>(user);
    info->cmds.push_back('M');
    info->coords.push_back(float(to->x));
    info->coords.push_back(float(to->y));
    return 0;
}

static int line_to(const FT_Vector* to,
                   void* user)
{
    PathGenInfo *info = reinterpret_cast<PathGenInfo*>(user);
    info->cmds.push_back('L');
    info->coords.push_back(float(to->x));
    info->coords.push_back(float(to->y));
    return 0;
}

static int conic_to(const FT_Vector* control,
                    const FT_Vector* to,
                    void* user)
{
    PathGenInfo *info = reinterpret_cast<PathGenInfo*>(user);
    info->cmds.push_back('Q');
    info->coords.push_back(float(control->x));
    info->coords.push_back(float(control->y));
    info->coords.push_back(float(to->x));
    info->coords.push_back(float(to->y));
    return 0;
}

static int cubic_to(const FT_Vector* control1,
                    const FT_Vector *control2,
                    const FT_Vector* to,
                    void* user)
{
    PathGenInfo *info = reinterpret_cast<PathGenInfo*>(user);
    info->cmds.push_back('C');
    info->coords.push_back(float(control1->x));
    info->coords.push_back(float(control1->y));
    info->coords.push_back(float(control2->x));
    info->coords.push_back(float(control2->y));
    info->coords.push_back(float(to->x));
    info->coords.push_back(float(to->y));
    return 0;
}

FT_Face get_font(int font)
{
    FT_Face face = font_list[font].face;
    if (!face) {
        const char *font_file = font_list[font].file;

        char filename[500];
        const char *dirs[] = { "./fonts", "./fonts/liberation", ".", NULL, "c:/WINDOWS/Fonts", "c:/WINNT/Fonts" };

        for (size_t i=0; i<countof(dirs) && !face; i++) {

            if (dirs[i]) {
                sprintf(filename, "%s/%s", dirs[i], font_file);
            } else {
                const char *windir = getenv("WINDIR");

                if (windir) {
                    sprintf(filename, "%s/Fonts/%s", windir, font_file);
                } else {
                    continue;
                }
            }
            FT_Long face_index = 0;
            printf("loading font %s from %s...", font_list[font].name, filename);
            FT_Error error = FT_New_Face(library, filename, face_index, &face);
            if (!error) {
                printf(" ok\n");
                // Face loaded ok!
                font_list[font].face = face;
                break;
            } else {
                printf(" failed, still looking...\n");
            }
        }
    }
    if (!face) {
        printf("could not locate font, sorry\n");
    }
    return face;
}

int num_fonts()
{
    return countof(font_list);
}

const char *font_name(int i)
{
    return font_list[i].name;
}

PathPtr load_freetype2_glyph(unsigned int font, unsigned char c)
{
    FT_Error error;

    FT_Face face = get_font(font);

    if (!face) {
        fprintf(stderr, "could not open %s\n", font_list[font].name);
        return PathPtr();
    }

    FT_UInt glyph_index;
    if (font_list[font].character_set) {
        glyph_index = FT_Get_Char_Index( face, c); 
    } else {
        // For wingdings & webdings (not actual character sets)
        glyph_index = c + 4 - '!'; 
    }
    error = FT_Load_Glyph(face, glyph_index,
                          FT_LOAD_NO_SCALE |   // Don't scale the outline glyph loaded, but keep it in font units.
                          FT_LOAD_NO_BITMAP);  // Ignore bitmap strikes when loading.
    if (error) {
        return PathPtr();
    }
    FT_GlyphSlot slot = face->glyph;

    FT_Outline outline = slot->outline;

    static const FT_Outline_Funcs funcs = {
        move_to,
        line_to,
        conic_to,
        cubic_to,
        0,  // shift
        0   // delta
    };

    PathGenInfo info;

    FT_Outline_Decompose(&outline, &funcs, &info);

    FT_BBox exact_bounding_box;
    FT_Outline_Get_BBox(&outline, &exact_bounding_box);
    float4 logical_bbox = float4(exact_bounding_box.xMin, exact_bounding_box.yMin,
                                 exact_bounding_box.xMax, exact_bounding_box.yMax);
    float4 face_bbox = float4(face->bbox.xMin, face->bbox.yMin,
                              face->bbox.xMax, face->bbox.yMax);

    PathStyle style;
    if (outline.flags & FT_OUTLINE_EVEN_ODD_FILL) {
        style.fill_rule = PathStyle::EVEN_ODD;
    } else {
        style.fill_rule = PathStyle::NON_ZERO;
    }
    PathPtr path(new Path(style, info.cmds, info.coords));
    path->setLogicalBounds(face_bbox);

    return path;
}

#endif // USE_FREETYPE2
