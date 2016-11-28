
/* path_data.h - declaration for simple path info */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <Cg/vector.hpp>
#include <Cg/matrix.hpp>

#include "PathStyle.hpp"

typedef struct {
    const char *name;
    int num_path_strings;
    const char **path_string;
    const Cg::float3 *fill_color;
    PathStyle::FillRule fill_rule;
} PathInfo;

extern PathInfo path_objects[];
extern int num_path_objects;

