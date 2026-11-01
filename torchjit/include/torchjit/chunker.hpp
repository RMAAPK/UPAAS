#ifndef TORCHJIT_CHUNKER_HPP
#define TORCHJIT_CHUNKER_HPP

#include "torchjit/tensor.hpp"
#include <vector>
#include <string>

namespace tjit {

struct TensorChunkMeta {
    std::string cid;
    size_t chunk_idx;
    size_t byte_offset;
    size_t byte_size;
    std::vector<int64_t> slice_shape;
};

struct ChunkedManifest {
    std::string model_id;
    std::string tensor_name;
    std::vector<int64_t> original_shape;
    size_t total_bytes;
    size_t chunk_size_bytes; // e.g., 2 * 1024 * 1024 (2 MB)
    std::vector<TensorChunkMeta> chunks;
};

class Chunker {
public:
    static constexpr size_t DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024; // 2 MB default (within 1-3 MB window)

    // Shard a tensor into 1-3 MB chunks and write binary slices named by CID
    static ChunkedManifest shard_tensor(const Tensor& tensor, 
                                        const std::string& model_id,
                                        const std::string& tensor_name,
                                        const std::string& output_dir,
                                        size_t chunk_size = DEFAULT_CHUNK_SIZE);

    // Stream and reconstruct or mmap directly from chunk files on demand
    static Tensor reassemble(const ChunkedManifest& manifest, const std::string& chunk_dir);
};

} // namespace tjit

#endif // TORCHJIT_CHUNKER_HPP
