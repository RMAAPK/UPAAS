#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace tjit {

enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int8 = 4,
    Uint8 = 5
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int32:   return 4;
        case DType::Int8:    return 1;
        case DType::Uint8:   return 1;
        default: return 4;
    }
}

inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32:   return "int32";
        case DType::Int8:    return "int8";
        case DType::Uint8:   return "uint8";
        default: return "unknown";
    }
}

// Storage: Continuous memory buffer with optional memory-mapping (mmap)
class Storage {
public:
    using Deleter = std::function<void(void*)>;

    Storage(size_t size_bytes);
    Storage(void* data, size_t size_bytes, Deleter deleter, bool is_mmap = false);
    ~Storage();

    // Disable copy, enable move
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    void* data() const { return data_; }
    size_t size_bytes() const { return size_bytes_; }
    bool is_mmap() const { return is_mmap_; }

    // Content Identifier: cryptographic SHA-256 multihash
    std::string compute_cid() const;

private:
    void* data_ = nullptr;
    size_t size_bytes_ = 0;
    Deleter deleter_ = nullptr;
    bool is_mmap_ = false;
};

} // namespace tjit
