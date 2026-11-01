#include "torchjit/tensor.hpp"
#include <cstring>

extern "C" {

void* tjit_tensor_zeros(int64_t dim, const int64_t* shape) {
    std::vector<int64_t> s(shape, shape + dim);
    auto t = new tjit::Tensor(tjit::Tensor::zeros(s));
    return static_cast<void*>(t);
}

void* tjit_tensor_ones(int64_t dim, const int64_t* shape) {
    std::vector<int64_t> s(shape, shape + dim);
    auto t = new tjit::Tensor(tjit::Tensor::ones(s));
    return static_cast<void*>(t);
}

void* tjit_tensor_load_mmap(const char* path, int64_t dim, const int64_t* shape) {
    std::vector<int64_t> s(shape, shape + dim);
    auto t = new tjit::Tensor(tjit::Tensor::load_mmap(path, s));
    return static_cast<void*>(t);
}

bool tjit_tensor_save(void* handle, const char* path) {
    auto t = static_cast<tjit::Tensor*>(handle);
    t->save(path);
    return true;
}

void* tjit_tensor_matmul(void* a, void* b) {
    auto tA = static_cast<tjit::Tensor*>(a);
    auto tB = static_cast<tjit::Tensor*>(b);
    auto out = new tjit::Tensor(tA->matmul(*tB));
    return static_cast<void*>(out);
}

void* tjit_tensor_add(void* a, void* b) {
    auto tA = static_cast<tjit::Tensor*>(a);
    auto tB = static_cast<tjit::Tensor*>(b);
    auto out = new tjit::Tensor(tA->add(*tB));
    return static_cast<void*>(out);
}

void* tjit_tensor_silu(void* a) {
    auto tA = static_cast<tjit::Tensor*>(a);
    auto out = new tjit::Tensor(tA->silu());
    return static_cast<void*>(out);
}

void* tjit_tensor_rmsnorm(void* a, float eps) {
    auto t = static_cast<tjit::Tensor*>(a);
    auto out = new tjit::Tensor(t->rms_norm(eps));
    return static_cast<void*>(out);
}

float* tjit_tensor_data_ptr(void* handle) {
    auto t = static_cast<tjit::Tensor*>(handle);
    return t->data_ptr<float>();
}

int64_t tjit_tensor_numel(void* handle) {
    return static_cast<tjit::Tensor*>(handle)->numel();
}

int64_t tjit_tensor_dim(void* handle) {
    return static_cast<tjit::Tensor*>(handle)->dim();
}

void tjit_tensor_shape(void* handle, int64_t* out_shape) {
    auto t = static_cast<tjit::Tensor*>(handle);
    const auto& s = t->shape();
    for (size_t i = 0; i < s.size(); ++i) {
        out_shape[i] = s[i];
    }
}

void tjit_tensor_cid(void* handle, char* out_buf, size_t max_len) {
    auto t = static_cast<tjit::Tensor*>(handle);
    std::string cid = t->cid();
    std::strncpy(out_buf, cid.c_str(), max_len);
}

void tjit_tensor_free(void* handle) {
    delete static_cast<tjit::Tensor*>(handle);
}

} // extern "C"
