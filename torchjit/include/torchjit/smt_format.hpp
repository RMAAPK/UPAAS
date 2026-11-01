#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

namespace coldness {

// Alignment Constants
constexpr size_t SMTF_HUGE_PAGE_ALIGNMENT = 2 * 1024 * 1024; // 2 MiB (Huge Page / Layer Block Boundary)
constexpr size_t SMTF_PAGE_ALIGNMENT      = 4 * 1024;        // 4 KiB (OS Page / O_DIRECT Tensor Boundary)

// Inductive alignment helper: align_up(x, A) = (x + A - 1) & ~(A - 1)
constexpr inline uint64_t align_up(uint64_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~static_cast<uint64_t>(alignment - 1);
}

#pragma pack(push, 1)

// SMTF v3 File Header: exactly 128 bytes (2 CPU cache lines)
struct SMTFHeaderV3 {
    char     magic[4];           // 'S', 'M', 'T', '3'
    uint16_t version;           // 3
    uint16_t quant_profile;     // 1: NVFP4, 2: Q4_K_M, 3: FP8_E4M3, 4: BF16, 5: INT8
    uint32_t model_id;          // Model identifier (e.g. hash of model name)
    uint32_t stage_idx;         // Pipeline Stage Index (0 .. N-1)
    uint32_t start_layer;       // First layer in this stage bundle
    uint32_t end_layer;         // Last layer in this stage bundle (inclusive)
    uint32_t num_tensors;       // Total tensors stored in this file
    uint64_t header_bytes;      // Size of header + tensor directory (unpadded)
    uint64_t data_bytes_total;  // Total file size including zeroed alignment padding
    uint8_t  blake3_root[32];   // Merkle root hash of all 2MB blocks
    uint8_t  distributor_sig[64];// Ed25519 signature from publisher
    // 4 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 32 + 64 = 140 -> let's pad to exact 256 bytes
    uint8_t  reserved[116];     // Padding to 256 bytes
};
static_assert(sizeof(SMTFHeaderV3) == 256, "SMTFHeaderV3 must be exactly 256 bytes");

// Individual Tensor Record inside the SMTF directory: exactly 128 bytes (2 CPU cache lines)
struct SMTFTensorDescriptor {
    uint32_t tensor_id;         // Sequence ID within bundle (4 bytes)
    uint16_t layer_idx;         // Layer index (2 bytes)
    int16_t  expert_id;         // -1 for attention/dense, 0..255 for MoE routed experts (2 bytes)
    uint16_t tensor_type;       // 0: ATTN_QKV, 1: ATTN_OUT, 2: MOE_GATE, 3: MOE_UP, 4: MOE_DOWN, 5: SHARED_EXP (2 bytes)
    uint16_t dtype;             // Quantization / Precision enum (2 bytes)
    uint32_t ndims;             // Number of dimensions (e.g., 2 or 3) (4 bytes)
    uint32_t shape[4];          // Up to 4D tensor dimensions (16 bytes)
    uint64_t file_offset;       // Strictly inductive aligned offset from start of file (8 bytes)
    uint64_t data_bytes;        // Exact payload byte size (8 bytes)
    uint8_t  blake3_hash[32];   // Checksum of tensor payload (32 bytes)
    uint8_t  reserved[48];      // Padding / quantization scales metadata (48 bytes)
    // 4 + 2 + 2 + 2 + 2 + 4 + 16 + 8 + 8 + 32 + 48 = 128 bytes
};
static_assert(sizeof(SMTFTensorDescriptor) == 128, "SMTFTensorDescriptor must be exactly 128 bytes");

#pragma pack(pop)

// Inductive Alignment Layout Calculator
// Rule verified by cold audit:
// offset_0 = align(header_end, SMTF_PAGE_ALIGNMENT)
// offset_{i+1} = align(offset_i + size_i, SMTF_PAGE_ALIGNMENT)
// (or SMTF_HUGE_PAGE_ALIGNMENT for major layer transitions)
class SMTFLayoutBuilder {
public:
    static std::vector<uint64_t> calculate_offsets(uint64_t header_total_bytes, 
                                                  const std::vector<uint64_t>& tensor_sizes,
                                                  const std::vector<bool>& is_layer_boundary) {
        if (tensor_sizes.size() != is_layer_boundary.size()) {
            throw std::invalid_argument("Size mismatch between tensor sizes and boundary flags");
        }

        std::vector<uint64_t> offsets;
        offsets.reserve(tensor_sizes.size());

        if (tensor_sizes.empty()) return offsets;

        // First tensor aligned to 2MB huge page after directory
        uint64_t current_offset = align_up(header_total_bytes, SMTF_HUGE_PAGE_ALIGNMENT);
        offsets.push_back(current_offset);

        for (size_t i = 0; i < tensor_sizes.size() - 1; ++i) {
            uint64_t end_of_current = current_offset + tensor_sizes[i];
            size_t req_align = is_layer_boundary[i + 1] ? SMTF_HUGE_PAGE_ALIGNMENT : SMTF_PAGE_ALIGNMENT;
            current_offset = align_up(end_of_current, req_align);

            // Invariant check: inductive non-overlap guarantee
            if (current_offset < end_of_current) {
                throw std::runtime_error("SMTF layout assertion failed: overlapping tensor offsets!");
            }
            offsets.push_back(current_offset);
        }

        return offsets;
    }
};

} // namespace coldness
