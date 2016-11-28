
/* path_stats.hpp - path statisitics */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#pragma once
#ifndef __path_stats_hpp__
#define __path_stats_hpp__

#include <stdlib.h>

#include "renderer.hpp"

// Grumble, Microsoft's <windef.h> (and probably other headers) define these as macros
#undef min
#undef max

struct PathStats {
    size_t num_paths;
    size_t num_cmds;
    size_t num_coords;

    PathStats()
        : num_paths(0)
        , num_cmds(0)
        , num_coords(0)
    { }

    void add(const PathStats &add_me) {
        num_paths += 1;
        num_cmds += add_me.num_cmds;
        num_coords += add_me.num_coords;
    }
    void max(const PathStats &max_me) {
        if (max_me.num_cmds > num_cmds) {
            num_cmds = max_me.num_cmds;
        }
        if (max_me.num_coords > num_coords) {
            num_coords = max_me.num_coords;
        }
    }
};

#endif // __path_stats_hpp__
