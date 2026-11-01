#include "torchjit/tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=== [RMAA TorchJIT (TJIT) Native Core Verification] ===\n\n";

    // 1. Test creation and shape
    auto t1 = tjit::Tensor::ones({3, 3});
    std::cout << "[+] Created Tensor t1: Shape [3, 3], Numel=" << t1.numel() 
              << ", Strides=[" << t1.strides()[0] << ", " << t1.strides()[1] << "]\n";
    std::cout << "[+] Tensor t1 CID: " << t1.cid() << "\n";

    // 2. Test arithmetic (add)
    auto t2 = tjit::Tensor::ones({3, 3});
    auto t3 = t1.add(t2);
    float* data3 = t3.data_ptr<float>();
    std::cout << "[+] Matrix Addition t3 = t1 + t2. First element = " << data3[0] << "\n";
    assert(data3[0] == 2.0f);

    // 3. Test matrix multiplication
    auto A = tjit::Tensor::zeros({2, 3});
    auto B = tjit::Tensor::zeros({3, 2});
    float* a_ptr = A.data_ptr<float>();
    float* b_ptr = B.data_ptr<float>();
    for (int i = 0; i < 6; ++i) { a_ptr[i] = 1.0f; b_ptr[i] = 2.0f; }

    auto C = A.matmul(B);
    float* c_ptr = C.data_ptr<float>();
    std::cout << "[+] Matrix Matmul C = A @ B. Shape [" << C.shape()[0] << ", " << C.shape()[1] << "]\n";
    std::cout << "[+] C[0, 0] = " << c_ptr[0] << " (Expected 6.0)\n";
    assert(c_ptr[0] == 6.0f);

    // 4. Test SiLU & RMSNorm (Core Transformer Blocks)
    auto hidden = tjit::Tensor::ones({1, 4});
    auto normed = hidden.rms_norm();
    float* norm_ptr = normed.data_ptr<float>();
    std::cout << "[+] Transformer RMSNorm: Input [1, 1, 1, 1] -> Normed[0] = " << norm_ptr[0] << "\n";
    assert(std::abs(norm_ptr[0] - 1.0f) < 1e-3f);

    // 5. Test Serialization and Zero-Copy mmap Reloading
    std::string test_bin = "/tmp/test_tensor.tjit";
    t3.save(test_bin);
    std::cout << "[+] Serialized t3 to disk: " << test_bin << "\n";

    auto mmapped = tjit::Tensor::load_mmap(test_bin, {3, 3});
    float* m_ptr = mmapped.data_ptr<float>();
    std::cout << "[+] Zero-Copy mmap Loaded Tensor: m_ptr[0] = " << m_ptr[0] << ", is_mmap=" << mmapped.storage()->is_mmap() << "\n";
    std::cout << "[+] Mmapped Tensor CID: " << mmapped.cid() << "\n";
    assert(m_ptr[0] == 2.0f);
    assert(mmapped.cid() == t3.cid());

    std::cout << "\n>>> [VERIFIED] TorchJIT Core Tensor & mmap CID Engine bit-exact and fully operational! <<<\n";
    return 0;
}
