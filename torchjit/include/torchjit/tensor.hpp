#pragma once

#include "storage.hpp"
#include <vector>
#include <string>
#include <memory>
#include <initializer_list>
#include <iostream>

namespace tjit {

class Tensor {
public:
    Tensor() = default;

    // Constructors
    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    Tensor(std::shared_ptr<Storage> storage, size_t storage_offset, 
           const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, 
           DType dtype = DType::Float32);

    // Factory functions
    static Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor from_blob(void* data, const std::vector<int64_t>& shape, DType dtype = DType::Float32);
    static Tensor load_mmap(const std::string& filepath, const std::vector<int64_t>& shape, DType dtype = DType::Float32);

    // Metadata
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    DType dtype() const { return dtype_; }
    size_t storage_offset() const { return storage_offset_; }
    std::shared_ptr<Storage> storage() const { return storage_; }

    void* raw_data() const {
        if (!storage_ || !storage_->data()) return nullptr;
        return static_cast<char*>(storage_->data()) + (storage_offset_ * dtype_size(dtype_));
    }

    template<typename T>
    T* data_ptr() const {
        return static_cast<T*>(raw_data());
    }

    // View operations (Zero memory allocation)
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;

    // Basic arithmetic kernels
    Tensor add(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor silu() const;
    Tensor rms_norm(float eps = 1e-6f) const;

    // Persistence & CID
    std::string cid() const;
    void save(const std::string& filepath) const;

private:
    std::shared_ptr<Storage> storage_;
    size_t storage_offset_ = 0;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t numel_ = 0;
    DType dtype_ = DType::Float32;

    static std::vector<int64_t> compute_default_strides(const std::vector<int64_t>& shape);
};

} // namespace tjit
