// This code is derived from https://github.com/blender/blender/blob/f4e1f62c62e6dae96d865d1fe0a47c1fb8b7a950/source/blender/nodes/shader/nodes/node_shader_mix_rgb.cc under a GPL-2.0-or-later license. 

/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2005 Blender Foundation. All rights reserved. */

#ifndef __MIXRGB__
#define __MIXRGB__

DEVICE_FUNC void node_shader_mix_rgb(
    // params
    int type_,
    int clamp_,
    // input
    float fac,
    float4_nonbuiltin color1,
    float4_nonbuiltin color2,
    // output
    float4_nonbuiltin *color) {
    float results[3]{color1.x, color1.y, color1.z};
    float color2_array[3]{color2.x, color2.y, color2.z};

    ramp_blend(type_, results, CLAMPIS(fac, 0.0f, 1.0f), color2_array);
    if (clamp_) {
        results[0] = clamp_range(results[0], 0.0f, 1.0f);
        results[1] = clamp_range(results[1], 0.0f, 1.0f);
        results[2] = clamp_range(results[2], 0.0f, 1.0f);
    }
    if (color != NULL) {
        *color = float4_nonbuiltin(results[0], results[1], results[2], 0);
    }
}

#endif