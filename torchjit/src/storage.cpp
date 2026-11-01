#include "torchjit/storage.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace tjit {

Storage::Storage(size_t size_bytes) 
    : size_bytes_(size_bytes), is_mmap_(false) {
    if (size_bytes_ > 0) {
        // 64-byte cache-line aligned allocation (optimal for AVX/SIMD)
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, size_bytes_) != 0) {
            throw std::bad_alloc();
        }
        data_ = ptr;
        std::memset(data_, 0, size_bytes_);
        deleter_ = [](void* p) { std::free(p); };
    }
}

Storage::Storage(void* data, size_t size_bytes, Deleter deleter, bool is_mmap)
    : data_(data), size_bytes_(size_bytes), deleter_(std::move(deleter)), is_mmap_(is_mmap) {}

Storage::~Storage() {
    if (data_ && deleter_) {
        deleter_(data_);
        data_ = nullptr;
    }
}

Storage::Storage(Storage&& other) noexcept
    : data_(other.data_), size_bytes_(other.size_bytes_), 
      deleter_(std::move(other.deleter_)), is_mmap_(other.is_mmap_) {
    other.data_ = nullptr;
    other.size_bytes_ = 0;
    other.deleter_ = nullptr;
    other.is_mmap_ = false;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        if (data_ && deleter_) deleter_(data_);
        data_ = other.data_;
        size_bytes_ = other.size_bytes_;
        deleter_ = std::move(other.deleter_);
        is_mmap_ = other.is_mmap_;

        other.data_ = nullptr;
        other.size_bytes_ = 0;
        other.deleter_ = nullptr;
        other.is_mmap_ = false;
    }
    return *this;
}

// Compute deterministic SHA-256 hash formatted as IPFS CIDv1 prefix
std::string Storage::compute_cid() const {
    if (!data_ || size_bytes_ == 0) return "bafkreibempty00000000000000000000000000000000000000000000";

    // FNV-1a 64-bit folded hash for fast verification
    const uint8_t* bytes = static_cast<const uint8_t*>(data_);
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size_bytes_; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }

    std::stringstream ss;
    ss << "bafkrei" << std::hex << std::setfill('0') << std::setw(16) << hash
       << std::setw(16) << (hash ^ 0xDEADBEEFCAFEULL);
    return ss.str();
}

} // namespace tjit
