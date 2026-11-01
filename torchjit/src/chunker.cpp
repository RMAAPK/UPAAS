#include "torchjit/chunker.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <cstring>

namespace fs = std::filesystem;

namespace tjit {

static std::string hash_buffer_cid(const uint8_t* data, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    std::stringstream ss;
    ss << "bafkreib" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return ss.str();
}

ChunkedManifest Chunker::shard_tensor(const Tensor& tensor,
                                     const std::string& model_id,
                                     const std::string& tensor_name,
                                     const std::string& output_dir,
                                     size_t chunk_size) {
    fs::create_directories(output_dir);
    size_t total_bytes = tensor.numel() * sizeof(float);
    const uint8_t* raw_data = reinterpret_cast<const uint8_t*>(tensor.data_ptr<float>());

    ChunkedManifest manifest;
    manifest.model_id = model_id;
    manifest.tensor_name = tensor_name;
    manifest.original_shape = tensor.shape();
    manifest.total_bytes = total_bytes;
    manifest.chunk_size_bytes = chunk_size;

    size_t offset = 0;
    size_t chunk_idx = 0;

    while (offset < total_bytes) {
        size_t current_chunk_size = std::min(chunk_size, total_bytes - offset);
        std::string cid = hash_buffer_cid(raw_data + offset, current_chunk_size);

        std::string chunk_filename = cid + ".tjit";
        fs::path chunk_path = fs::path(output_dir) / chunk_filename;

        std::ofstream out(chunk_path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to write tensor chunk: " + chunk_path.string());
        }
        out.write(reinterpret_cast<const char*>(raw_data + offset), current_chunk_size);
        out.close();

        TensorChunkMeta meta;
        meta.cid = cid;
        meta.chunk_idx = chunk_idx;
        meta.byte_offset = offset;
        meta.byte_size = current_chunk_size;
        
        manifest.chunks.push_back(meta);

        offset += current_chunk_size;
        chunk_idx++;
    }

    return manifest;
}

Tensor Chunker::reassemble(const ChunkedManifest& manifest, const std::string& chunk_dir) {
    Tensor target = Tensor::zeros(manifest.original_shape);
    uint8_t* target_ptr = reinterpret_cast<uint8_t*>(target.data_ptr<float>());

    for (const auto& chunk : manifest.chunks) {
        std::string chunk_filename = chunk.cid + ".tjit";
        fs::path chunk_path = fs::path(chunk_dir) / chunk_filename;

        std::ifstream in(chunk_path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Missing chunk on disk: " + chunk_path.string());
        }
        in.read(reinterpret_cast<char*>(target_ptr + chunk.byte_offset), chunk.byte_size);
        in.close();
    }

    return target;
}

} // namespace tjit
