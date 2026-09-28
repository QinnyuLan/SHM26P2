// Forward-only layer selection over existing IBGS buffers. No autograd binding.
#include <torch/extension.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

std::vector<torch::Tensor> layer_select_cuda(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, int, int, float, float, float, float);

std::vector<torch::Tensor> select_layers(
    const torch::Tensor& means, const torch::Tensor& conic,
    const torch::Tensor& planes, const torch::Tensor& ranges,
    const torch::Tensor& ids, int64_t width, int64_t height,
    double fx, double fy, double cx, double cy) {
    TORCH_CHECK(width > 0 && height > 0 && width <= INT32_MAX && height <= INT32_MAX,
                "Positive int32 image dimensions required");
    TORCH_CHECK(width * height <= INT32_MAX, "Image index exceeds int32");
    TORCH_CHECK(std::isfinite(fx) && std::isfinite(fy) && fx > 0 && fy > 0 &&
                std::isfinite(cx) && std::isfinite(cy) &&
                std::isfinite(float(fx)) && std::isfinite(float(fy)) &&
                float(fx) > 0 && float(fy) > 0 &&
                std::isfinite(float(cx)) && std::isfinite(float(cy)), "Invalid FP32 camera");
    for (const auto& tensor : {means, conic, planes, ranges, ids}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == means.device(), "One CUDA device required");
        TORCH_CHECK(tensor.is_contiguous(), "Contiguous buffer views required");
        TORCH_CHECK(!tensor.requires_grad(), "Frozen geometry only; backward is unsupported");
    }
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 2, "Nx2 means required");
    const auto n = means.size(0);
    TORCH_CHECK(n <= INT32_MAX && ids.numel() <= INT32_MAX - 256, "Buffer exceeds int32 traversal ABI");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32 && conic.scalar_type() == torch::kFloat32 &&
                planes.scalar_type() == torch::kFloat32, "FP32 projected inputs required");
    TORCH_CHECK(ranges.scalar_type() == torch::kInt32 && ids.scalar_type() == torch::kInt32,
                "Int32 bit views of uint32 ledger required");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 2 && conic.sizes() == torch::IntArrayRef({n, 4}) &&
                planes.sizes() == torch::IntArrayRef({n, 5}), "Projected attribute shape mismatch");
    const auto tiles = ((width + 15) / 16) * ((height + 15) / 16);
    TORCH_CHECK(ranges.sizes() == torch::IntArrayRef({tiles, 2}) && ids.dim() == 1,
                "16x16 tile ledger shape mismatch");
    return layer_select_cuda(means, conic, planes, ranges, ids, int(width), int(height),
                             float(fx), float(fy), float(cx), float(cy));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("select_layers", &select_layers, "Frozen IBGS median4/top4 selection (CUDA forward only)");
}
