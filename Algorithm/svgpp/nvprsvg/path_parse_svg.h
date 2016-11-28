
/* path_parse_svg.h - interface for parsing SVG path strings */

// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <string>
#include <vector>

int parse_svg_path(const char *input, std::vector<char> &c, std::vector<float> &v);

