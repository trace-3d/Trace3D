#ifndef CUDA_RASTERIZER_ID_TRACE_H_INCLUDED
#define CUDA_RASTERIZER_ID_TRACE_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace ID_TRACE
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void trace_preprocess(int P, int D, int M,
        const float* orig_points,
        const glm::vec2* scales,
        const float scale_modifier,
        const glm::vec4* rotations,
        const float* opacities,
        const float* shs,
        bool* clamped,
        const float* transMat_precomp,
        const float* colors_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const glm::vec3* cam_pos,
        const int W, int H,
        const float tan_fovx, const float tan_fovy,
        const float focal_x, const float focal_y,
        int* radii,
        float2* points_xy_image,
        float* depths,
        float* transMats,
        float* rgb,
        float4* normal_opacity,
        const dim3 grid,
        uint32_t* tiles_touched,
        bool prefiltered);

	// Main trace method.
	void trace(const dim3 grid, dim3 block,
        const uint2* __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        float focal_x, float focal_y,
        const float2* __restrict__ points_xy_image,
        float* __restrict__ weights, 
        const float* __restrict__ transMats,
        const float* __restrict__ depths,
        const float4* __restrict__  normal_opacity,
        float* __restrict__ final_T,
        uint32_t* __restrict__ n_contrib,
        const float* __restrict__ bg_color,
        const int* __restrict__ id_masks,
        const int num_class,
        const bool alpha_w);
}

#endif
