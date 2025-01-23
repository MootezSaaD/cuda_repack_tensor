#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fused_repack_merge_impl(
    torch::Tensor input,
    torch::Tensor mask,
    torch::Tensor output,
    torch::Tensor output_mask,
    torch::Tensor current_indices,
    torch::Tensor pruned_counts,
    int M
);

std::tuple<torch::Tensor, torch::Tensor> fused_repack_merge(
    torch::Tensor input, 
    torch::Tensor mask
) {
    CHECK_CUDA(input);
    CHECK_CUDA(mask);

    const int B = input.size(0);
    const int N = input.size(1);
    const int d = input.size(2);

    auto kept_counts = mask.sum(1, false, torch::kInt32);
    auto pruned_counts = N - kept_counts;

    // Initialize current_indices to 0 for each batch
    auto current_indices = torch::zeros_like(kept_counts);

    const int M = (kept_counts + pruned_counts.gt(0)).max().item<int>();

    auto output = torch::zeros({B, M, d}, input.options());
    auto output_mask = torch::zeros({B, M}, input.options().dtype(torch::kBool));

    fused_repack_merge_impl(
        input, mask, output, output_mask,
        current_indices, pruned_counts, M
    );

    return {output, output_mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_repack_merge", &fused_repack_merge, "Fused token repacking");
}