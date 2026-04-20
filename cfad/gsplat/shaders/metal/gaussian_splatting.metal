// 3D Gaussian Splatting - Metal Compute Shader
// Optimized for Apple Silicon (M1/M2/M3/M4)
// 
// This shader performs the core Gaussian splatting operation:
// 1. Projects 3D Gaussians to 2D screen space
// 2. Computes 2D Gaussian parameters (mean, covariance)
// 3. Rasterizes Gaussians as screen-space ellipses

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Data Structures
// ============================================================================

struct GaussianParams {
    float3 mean;           // 3D position (xyz)
    float opacity;         // Opacity scalar
    float4 *sh_coefficients;  // Spherical harmonic coefficients (dynamic size)
};

struct CameraParams {
    float fx;              // Focal length x
    float fy;              // Focal length y
    float cx;              // Principal point x
    float cy;              // Principal point y
    float viewport_scale;// Anti-aliasing viewport scale
};

struct ViewMatrix {
    float4x4 view;       // View transformation matrix
};

// ============================================================================
// Kernel: Project Gaussians to Screen Space
// ============================================================================

kernel void project_gaussians(
    device const float3 *gaussian_means [[buffer(0)]],
    device const float *gaussian_opacities [[buffer(1)]],
    device const float3 *gaussian_scales [[buffer(2)]],
    device const float4 *gaussian_quaternions [[buffer(3)]],
    device float2 *screen_means [[buffer(4)]],
    device float3 *screen_covariances [[buffer(5)]],
    device uint *screen_radii [[buffer(6)]],
    
    constant CameraParams &camera [[buffer(7)]],
    constant ViewMatrix &view_matrix [[buffer(8)]],
    
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 1000000) return; // Max Gaussians
    
    // Load Gaussian parameters
    float3 mean = gaussian_means[gid];
    float opacity = gaussian_opacities[gid];
    float3 scale = gaussian_scales[gid];
    float4 quaternion = gaussian_quaternions[gid];
    
    // Skip invalid Gaussians
    if (opacity <= 0.01) return;
    
    // Convert quaternion to rotation matrix
    float3x3 R = quaternion_to_rotation_matrix(quaternion);
    
    // Compute scale matrix (exponential of log-scale)
    float3 s = exp(scale);
    
    // Compute covariance: R * diag(s^2) * R^T
    float3x3 S = float3x3(
        s.x, 0.0, 0.0,
        0.0, s.y, 0.0,
        0.0, 0.0, s.z
    );
    
    float3x3 cov = R * S * S * transpose(R);
    
    // Transform to camera coordinates
    float4 mean_homo = view_matrix.view * float4(mean, 1.0);
    float3 mean_cam = mean_homo.xyz / mean_homo.w;
    
    // Compute view rotation (inverse of view matrix upper 3x3)
    float3x3 R_view = transpose(float3x3(view_matrix.view));
    
    // Transform covariance to camera space
    float3x3 cov_cam = R_view * cov * transpose(R_view);
    
    // Compute Jacobian of projection matrix
    float fx = camera.fx;
    float fy = camera.fy;
    float z_inv = 1.0 / max(mean_cam.z, 0.001);
    float z_inv_2 = z_inv * z_inv;
    
    // 2D Jacobian (partial derivatives of projection)
    float2x2 J;
    J[0] = float2(fx * z_inv, 0.0);
    J[1] = float2(0.0, fy * z_inv);
    
    // Compute 2D covariance: J * cov_cam[0:2, 0:2] * J^T
    // Simplified for diagonal covariance approximation
    float sigma_x2 = fx * fy * z_inv_2 * cov_cam[0][0];
    float sigma_y2 = fx * fy * z_inv_2 * cov_cam[1][1];
    
    // Compute screen-space mean
    float2 screen_mean;
    screen_mean.x = fx * mean_cam.x / mean_cam.z + camera.cx;
    screen_mean.y = fy * mean_cam.y / mean_cam.z + camera.cy;
    
    // Compute bounding radius (3 sigma for ~99.7% coverage)
    float radius_x = 3.0 * sqrt(max(sigma_x2, 1e-6));
    float radius_y = 3.0 * sqrt(max(sigma_y2, 1e-6));
    float radius = max(radius_x, radius_y);
    
    // Store results (integer radius for tile computation)
    screen_means[gid] = screen_mean;
    screen_covariances[gid] = float3(sigma_x2, sigma_y2, 0.0);
    screen_radii[gid] = uint(radius);
}

// ============================================================================
// Kernel: Splat Gaussians to Image (Tile-based)
// ============================================================================

kernel void splat_gaussians(
    device const float2 *screen_means [[buffer(0)]],
    device const float3 *screen_covs [[buffer(1)]],
    device const uint *screen_radii [[buffer(2)]],
    device const float *gaussian_opacities [[buffer(3)]],
    device const float4 *sh_coefficients_dc [[buffer(4)]],
    device const float4 *sh_coefficients_rest [[buffer(5)]],
    
    device float3 *image_color [[buffer(6)]],
    device float *image_alpha [[buffer(7)]],
    
    constant CameraParams &camera [[buffer(8)]],
    constant int2 &image_size [[buffer(9)]],
    
    uint2 gid [[thread_position_in_grid]]
) {
    int2 pix = gid;
    if (pix.x >= image_size.x || pix.y >= image_size.y) return;
    
    float3 final_color = float3(0.0);
    float final_alpha = 0.0;
    
    // Find the tile this pixel belongs to (16x16 tiles)
    int2 tile_id = pix / 16;
    
    // For each Gaussian that could overlap this pixel
    for (uint g = 0; g < 1024; g++) {
        float2 mean_2d = screen_means[g];
        float3 cov = screen_covs[g];
        uint radius = screen_radii[g];
        
        // Check if Gaussian overlaps this tile (simplified bounding box)
        float dx = abs(float(pix.x) - mean_2d.x);
        float dy = abs(float(pix.y) - mean_2d.y);
        
        if (dx > float(radius) || dy > float(radius)) continue;
        
        // Compute 2D Gaussian at this pixel
        float sigma_x2 = cov.x;
        float sigma_y2 = cov.y;
        
        float dist_x = float(pix.x) - mean_2d.x;
        float dist_y = float(pix.y) - mean_2d.y;
        
        // Compute exponent: -0.5 * (dx^2/sigma_x2 + dy^2/sigma_y2)
        float exponent = -0.5 * (dist_x * dist_x / max(sigma_x2, 1e-6) + 
                                  dist_y * dist_y / max(sigma_y2, 1e-6));
        
        if (exponent > 0.0) continue; // Too far
        
        float gaussian_alpha = exp(exponent);
        
        // Get Gaussian opacity (sigmoid applied)
        float g_opacity = 1.0 / (1.0 + exp(-gaussian_opacities[g]));
        
        // Compute SH color (simplified - just first band)
        float4 sh_dc = sh_coefficients_dc[g];
        float3 sh_color = float3(0.1658, 0.1668, 0.1672); // Neutral color
        sh_color += sh_dc.rgb * 0.2821; // Scale by SH base
        
        // Clamp color to valid range
        sh_color = clamp(sh_color, 0.0, 1.0);
        
        // Alpha composite
        float g_alpha = g_opacity * gaussian_alpha;
        g_alpha = min(g_alpha, 0.999); // Clamp to prevent overflow
        
        final_color += sh_color * g_alpha * (1.0 - final_alpha);
        final_alpha += g_alpha * (1.0 - final_alpha);
        
        if (final_alpha > 0.999) break; // Early exit for opaque regions
    }
    
    image_color[pix.y * image_size.x + pix.x] = final_color;
    image_alpha[pix.y * image_size.x + pix.x] = final_alpha;
}

// ============================================================================
// Helper: Quaternion to Rotation Matrix
// ============================================================================

float3x3 quaternion_to_rotation_matrix(float4 q) {
    float w = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    
    // Normalize
    float norm = sqrt(w*w + x*x + y*y + z*z);
    w /= norm; x /= norm; y /= norm; z /= norm;
    
    float3x3 R = float3x3(
        1.0 - 2.0*(y*y + z*z),  2.0*(x*y + w*z),       2.0*(x*z - w*y),
        2.0*(x*y - w*z),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z + w*x),
        2.0*(x*z + w*y),        2.0*(y*z - w*x),        1.0 - 2.0*(x*x + y*y)
    );
    
    return R;
}

// ============================================================================
// Helper: Spherical Harmonics Evaluation
// ============================================================================

float3 evaluate_sh(float4 *sh_coeffs, float3 view_dir) {
    // SH coefficients are stored as (N, 45) for max degree 3
    // We evaluate the first few bands (simplified)
    
    float3 color = float3(0.5, 0.5, 0.5); // Base neutral color
    
    // First band (DC) - already in sh_coeffs[0].rgb
    float4 dc = sh_coeffs[0];
    color += dc.rgb * 0.2820947913536335;
    
    // Second band (3 coefficients) - view-dependent
    if (length(view_dir) > 0.001) {
        float3 dir = normalize(view_dir);
        
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        
        // L=1 terms
        float3 sh1 = float3(sh_coeffs[1].rgb);
        color += sh1 * (-0.4886025 * y);
        
        float3 sh2 = float3(sh_coeffs[2].rgb);
        color += sh2 * (0.4886025 * z);
        
        float3 sh3 = float3(sh_coeffs[3].rgb);
        color += sh3 * (-0.4886025 * x);
    }
    
    return color;
}
