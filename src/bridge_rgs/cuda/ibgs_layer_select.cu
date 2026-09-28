// Tiled traversal follows the IBGS / Inria diff-plane-rasterization forward
// algorithm. Upstream Inria non-commercial research/evaluation LICENSE.md
// continues to apply to the adapted traversal; this file adds no backward.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
constexpr int BLOCK = 256;
// Status bits: invalid range=1, invalid ID=2, nonfinite referenced inputs=4,
// nonfinite power/alpha/T=8, nonfinite accepted plane depth=16.
__global__ void layer_kernel(
    const float2* means, const float4* conic, const float* planes,
    const int2* ranges, const int* point_list, int n, int instances,
    int width, int height, float fx, float fy, float cx, float cy,
    int* out_ids, float* out_depth, float* out_weight, int* out_order,
    float* out_T, int* out_last, int* out_nonfinite_plane, int* out_status) {
    const int lane = threadIdx.y * 16 + threadIdx.x;
    const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
    const bool inside = x < width && y < height;
    const int pixel = y * width + x;
    const int2 range = ranges[blockIdx.y * gridDim.x + blockIdx.x];
    if (range.x < 0 || range.y < range.x || range.y > instances) {
        if (inside) out_status[pixel] = 1;
        return;  // Uniform block branch: safe before barriers.
    }
    __shared__ int shared_id[BLOCK], shared_bad[BLOCK];
    __shared__ float2 shared_xy[BLOCK];
    __shared__ float4 shared_conic[BLOCK];
    __shared__ float shared_plane[BLOCK][5];
    int selected_id[2][4], selected_order[2][4];
    float selected_z[2][4], selected_w[2][4];
    #pragma unroll
    for (int mode = 0; mode < 2; ++mode) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            selected_id[mode][k] = -1; selected_order[mode][k] = 0;
            selected_z[mode][k] = 0; selected_w[mode][k] = 0;
        }
    }
    float T = 1.f;
    const float rx = (float(x) - cx) / fx, ry = (float(y) - cy) / fy;
    int before_ptr = 0, below_count = 0, last = 0, bad = 0, nonfinite_plane = 0;
    bool done = !inside;
    for (int begin = range.x; begin < range.y; begin += BLOCK) {
        if (__syncthreads_count(done) == BLOCK) break;
        const int cursor = begin + lane;
        if (cursor < range.y) {
            const int id = point_list[cursor];
            shared_id[lane] = id; shared_bad[lane] = 0;
            if (id < 0 || id >= n) {
                shared_bad[lane] = 2;
            } else {
                shared_xy[lane] = means[id]; shared_conic[lane] = conic[id];
                const float2 xy = shared_xy[lane]; const float4 c = shared_conic[lane];
                if (!(isfinite(xy.x) && isfinite(xy.y) && isfinite(c.x) &&
                      isfinite(c.y) && isfinite(c.z) && isfinite(c.w))) shared_bad[lane] = 4;
                #pragma unroll
                for (int k = 0; k < 5; ++k) {
                    const float value = planes[int64_t(id) * 5 + k]; shared_plane[lane][k] = value;
                    if (!isfinite(value)) shared_bad[lane] = 4;
                }
            }
        }
        __syncthreads();
        const int count = min(BLOCK, range.y - begin);
        for (int j = 0; !done && j < count; ++j) {
            if (shared_bad[j]) { bad |= shared_bad[j]; done = true; break; }
            const int ordinal = begin - range.x + j + 1, id = shared_id[j];
            const float2 xy = shared_xy[j]; const float4 c = shared_conic[j];
            const float dx = xy.x - float(x), dy = xy.y - float(y);
            const float power = -0.5f * (c.x * dx * dx + c.z * dy * dy) - c.y * dx * dy;
            if (!isfinite(power)) { bad |= 8; done = true; break; }
            if (power > 0.f) continue;
            const float alpha = fminf(0.99f, c.w * __expf(power));
            if (!isfinite(alpha)) { bad |= 8; done = true; break; }
            if (alpha < 1.f / 255.f) continue;
            const float test_T = T * (1.f - alpha);
            if (!isfinite(test_T)) { bad |= 8; done = true; break; }
            if (test_T < 0.0001f) { done = true; break; }  // Stop contributor is NOT accepted.
            const float weight = alpha * T;
            const float z = -shared_plane[j][4] /
                (shared_plane[j][0] * rx + shared_plane[j][1] * ry + shared_plane[j][2] + 1.e-8f);
            if (!isfinite(z)) { bad |= 16; ++nonfinite_plane; }
            if (isfinite(z) && z > 0.f) {
                int median_slot = -1;
                if (T > .5f) { median_slot = before_ptr; before_ptr = (before_ptr + 1) % 2; }
                else if (below_count < 2) median_slot = 2 + below_count++;
                if (median_slot >= 0) {
                    selected_id[0][median_slot] = id; selected_z[0][median_slot] = z;
                    selected_w[0][median_slot] = weight; selected_order[0][median_slot] = ordinal;
                }
                // Stable descending weight insertion: equal weight keeps the
                // earlier traversal ordinal, not the smaller Gaussian ID.
                int insert = 4;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (insert == 4 && (selected_id[1][k] < 0 || weight > selected_w[1][k])) insert = k;
                }
                if (insert < 4) {
                    for (int k = 3; k > insert; --k) {
                        selected_id[1][k] = selected_id[1][k-1]; selected_z[1][k] = selected_z[1][k-1];
                        selected_w[1][k] = selected_w[1][k-1]; selected_order[1][k] = selected_order[1][k-1];
                    }
                    selected_id[1][insert] = id; selected_z[1][insert] = z;
                    selected_w[1][insert] = weight; selected_order[1][insert] = ordinal;
                }
            }
            T = test_T; last = ordinal;  // Invalid/negative plane does not remove RGB alpha.
        }
        __syncthreads();  // All lanes finish reading before shared staging is overwritten.
    }
    if (!inside) return;
    #pragma unroll
    for (int mode = 0; mode < 2; ++mode) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int64_t at = int64_t(pixel) * 8 + mode * 4 + k;
            out_ids[at] = selected_id[mode][k]; out_depth[at] = selected_z[mode][k];
            out_weight[at] = selected_w[mode][k]; out_order[at] = selected_order[mode][k];
        }
    }
    out_T[pixel] = T; out_last[pixel] = last;
    out_nonfinite_plane[pixel] = nonfinite_plane; out_status[pixel] = bad;
}
}  // namespace

std::vector<torch::Tensor> layer_select_cuda(
    const torch::Tensor& means, const torch::Tensor& conic, const torch::Tensor& planes,
    const torch::Tensor& ranges, const torch::Tensor& ids, int width, int height,
    float fx, float fy, float cx, float cy) {
    const c10::cuda::CUDAGuard guard(means.device());
    auto ints = means.options().dtype(torch::kInt32);
    const std::vector<int64_t> layers{height, width, 2, 4}, pixels{height, width};
    auto selected = torch::full(layers, -1, ints), depth = torch::zeros(layers, means.options());
    auto weight = torch::zeros_like(depth), ordinal = torch::zeros(layers, ints);
    auto transmittance = torch::ones(pixels, means.options()), last = torch::zeros(pixels, ints);
    auto nonfinite = torch::zeros(pixels, ints), status = torch::zeros(pixels, ints);
    layer_kernel<<<dim3((width+15)/16, (height+15)/16), dim3(16,16), 0,
                   at::cuda::getCurrentCUDAStream(means.get_device())>>>(
        reinterpret_cast<const float2*>(means.data_ptr<float>()),
        reinterpret_cast<const float4*>(conic.data_ptr<float>()), planes.data_ptr<float>(),
        reinterpret_cast<const int2*>(ranges.data_ptr<int>()), ids.data_ptr<int>(),
        int(means.size(0)), int(ids.numel()), width, height, fx, fy, cx, cy,
        selected.data_ptr<int>(), depth.data_ptr<float>(), weight.data_ptr<float>(), ordinal.data_ptr<int>(),
        transmittance.data_ptr<float>(), last.data_ptr<int>(), nonfinite.data_ptr<int>(), status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {selected, depth, weight, ordinal, transmittance, last, nonfinite, status};
}
