#include "torchjit/tensor.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <numeric>

namespace tjit {

std::vector<int64_t> Tensor::compute_default_strides(const std::vector<int64_t>& shape) {
    if (shape.empty()) return {};
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype)
    : shape_(shape), strides_(compute_default_strides(shape)), dtype_(dtype), storage_offset_(0) {
    numel_ = 1;
    for (int64_t d : shape_) numel_ *= d;
    size_t size_bytes = static_cast<size_t>(numel_) * dtype_size(dtype_);
    storage_ = std::make_shared<Storage>(size_bytes);
}

Tensor::Tensor(std::shared_ptr<Storage> storage, size_t storage_offset, 
               const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, 
               DType dtype)
    : storage_(std::move(storage)), storage_offset_(storage_offset), 
      shape_(shape), strides_(strides), dtype_(dtype) {
    numel_ = 1;
    for (int64_t d : shape_) numel_ *= d;
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype) {
    return Tensor(shape, dtype);
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, DType dtype) {
    Tensor t(shape, dtype);
    if (dtype == DType::Float32) {
        float* ptr = t.data_ptr<float>();
        for (int64_t i = 0; i < t.numel(); ++i) ptr[i] = 1.0f;
    }
    return t;
}

Tensor Tensor::from_blob(void* data, const std::vector<int64_t>& shape, DType dtype) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    size_t bytes = static_cast<size_t>(n) * dtype_size(dtype);
    // Non-owning storage
    auto storage = std::make_shared<Storage>(data, bytes, [](void*) {}, false);
    return Tensor(storage, 0, shape, compute_default_strides(shape), dtype);
}

Tensor Tensor::load_mmap(const std::string& filepath, const std::vector<int64_t>& shape, DType dtype) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Cannot open file: " + filepath);

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Cannot stat file: " + filepath);
    }
    size_t size_bytes = static_cast<size_t>(sb.st_size);

    void* addr = mmap(nullptr, size_bytes, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) throw std::runtime_error("mmap failed for: " + filepath);

    auto storage = std::make_shared<Storage>(addr, size_bytes, [size_bytes](void* p) {
        munmap(p, size_bytes);
    }, true);

    return Tensor(storage, 0, shape, compute_default_strides(shape), dtype);
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    int64_t new_numel = 1;
    for (int64_t d : new_shape) new_numel *= d;
    if (new_numel != numel_) {
        throw std::invalid_argument("view shape numel mismatch");
    }
    return Tensor(storage_, storage_offset_, new_shape, compute_default_strides(new_shape), dtype_);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    auto new_shape = shape_;
    auto new_strides = strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return Tensor(storage_, storage_offset_, new_shape, new_strides, dtype_);
}

Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) throw std::invalid_argument("Tensor shapes must match for add");
    Tensor out(shape_, dtype_);
    const float* a = data_ptr<float>();
    const float* b = other.data_ptr<float>();
    float* c = out.data_ptr<float>();
    for (int64_t i = 0; i < numel_; ++i) {
        c[i] = a[i] + b[i];
    }
    return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (dim() != 2 || other.dim() != 2) throw std::invalid_argument("2D matmul only currently");
    int64_t M = shape_[0];
    int64_t K = shape_[1];
    int64_t K2 = other.shape_[0];
    int64_t N = other.shape_[1];
    if (K != K2) throw std::invalid_argument("Matmul inner dimension mismatch");

    Tensor out({M, N}, dtype_);
    const float* A = data_ptr<float>();
    const float* B = other.data_ptr<float>();
    float* C = out.data_ptr<float>();

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
    return out;
}

Tensor Tensor::silu() const {
    Tensor out(shape_, dtype_);
    const float* src = data_ptr<float>();
    float* dst = out.data_ptr<float>();
    for (int64_t i = 0; i < numel_; ++i) {
        dst[i] = src[i] / (1.0f + std::exp(-src[i]));
    }
    return out;
}

Tensor Tensor::rms_norm(float eps) const {
    Tensor out(shape_, dtype_);
    int64_t last_dim = shape_.back();
    int64_t outer_rows = numel_ / last_dim;

    const float* src = data_ptr<float>();
    float* dst = out.data_ptr<float>();

    for (int64_t row = 0; row < outer_rows; ++row) {
        const float* r_src = src + row * last_dim;
        float* r_dst = dst + row * last_dim;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < last_dim; ++i) {
            sum_sq += r_src[i] * r_src[i];
        }
        float rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(last_dim)) + eps);
        for (int64_t i = 0; i < last_dim; ++i) {
            r_dst[i] = r_src[i] * rms;
        }
    }
    return out;
}

std::string Tensor::cid() const {
    return storage_ ? storage_->compute_cid() : "bafkreiempty";
}

void Tensor::save(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) throw std::runtime_error("Cannot write file: " + filepath);
    ofs.write(reinterpret_cast<const char*>(raw_data()), static_cast<std::streamsize>(numel_ * dtype_size(dtype_)));
}

} // namespace tjit
