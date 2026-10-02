#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void fps_distance_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ centers,
    float* __restrict__ distances,
    int64_t point_count,
    int64_t center_count) {
  const int64_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t center_index = blockIdx.y;
  if (point_index >= point_count || center_index >= center_count) return;

  const float* point = points + point_index * 3;
  const float* center = points + centers[center_index] * 3;
  const float dx = point[0] - center[0];
  const float dy = point[1] - center[1];
  const float dz = point[2] - center[2];
  const float dy2 = dy * dy;
  const float dxy2 = __fmaf_rn(dx, dx, dy2);
  distances[center_index * point_count + point_index] =
      __fmaf_rn(dz, dz, dxy2);
}

__global__ void fps_valid_kernel(
    const float* __restrict__ points,
    uint8_t* __restrict__ valid,
    int64_t point_count) {
  const int64_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= point_count) return;
  const float* point = points + point_index * 3;
  const float y2 = point[1] * point[1];
  const float xy2 = __fmaf_rn(point[0], point[0], y2);
  const float norm2 = __fmaf_rn(point[2], point[2], xy2);
  valid[point_index] = static_cast<double>(norm2) > 1.0e-3;
}

torch::Tensor fps_distances(torch::Tensor points, torch::Tensor centers) {
  TORCH_CHECK(points.is_cuda(), "points must be CUDA resident");
  TORCH_CHECK(centers.is_cuda(), "centers must be CUDA resident");
  TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be FP32");
  TORCH_CHECK(centers.scalar_type() == torch::kInt64, "centers must be INT64");
  TORCH_CHECK(points.is_contiguous() && centers.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
              "points must have shape [N,3]");
  TORCH_CHECK(centers.dim() == 1, "centers must have shape [C]");

  const c10::cuda::CUDAGuard guard(points.device());
  auto output = torch::empty(
      {centers.size(0), points.size(0)}, points.options());
  constexpr int threads = 256;
  const dim3 blocks(
      (points.size(0) + threads - 1) / threads, centers.size(0));
  fps_distance_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      points.data_ptr<float>(), centers.data_ptr<int64_t>(),
      output.data_ptr<float>(), points.size(0), centers.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fps_valid(torch::Tensor points) {
  TORCH_CHECK(points.is_cuda(), "points must be CUDA resident");
  TORCH_CHECK(points.scalar_type() == torch::kFloat32, "points must be FP32");
  TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
              "points must have shape [N,3]");
  const c10::cuda::CUDAGuard guard(points.device());
  auto output = torch::empty({points.size(0)}, points.options().dtype(torch::kUInt8));
  constexpr int threads = 256;
  const int blocks = (points.size(0) + threads - 1) / threads;
  fps_valid_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      points.data_ptr<float>(), output.data_ptr<uint8_t>(), points.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fps_distances", &fps_distances,
             "PointLLM CUDA-FMA FPS distance matrix");
  module.def("fps_valid", &fps_valid, "PointLLM CUDA-FMA FPS validity mask");
}
