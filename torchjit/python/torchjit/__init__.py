"""
TorchJIT Python Runtime
Zero-copy C++20 backend binding via ctypes.
"""
import ctypes
import os
from typing import Sequence, List

_so_path = os.path.join(os.path.dirname(__file__), "libtorchjit.so")
if not os.path.exists(_so_path):
    raise RuntimeError(f"TorchJIT backend library not found at: {_so_path}")

_lib = ctypes.CDLL(_so_path)

# ABI signatures matching src/c_api.cpp
_lib.tjit_tensor_zeros.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)]
_lib.tjit_tensor_zeros.restype = ctypes.c_void_p

_lib.tjit_tensor_ones.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)]
_lib.tjit_tensor_ones.restype = ctypes.c_void_p

_lib.tjit_tensor_free.argtypes = [ctypes.c_void_p]
_lib.tjit_tensor_free.restype = None

_lib.tjit_tensor_data_ptr.argtypes = [ctypes.c_void_p]
_lib.tjit_tensor_data_ptr.restype = ctypes.POINTER(ctypes.c_float)

_lib.tjit_tensor_numel.argtypes = [ctypes.c_void_p]
_lib.tjit_tensor_numel.restype = ctypes.c_int64

_lib.tjit_tensor_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.tjit_tensor_add.restype = ctypes.c_void_p

_lib.tjit_tensor_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.tjit_tensor_matmul.restype = ctypes.c_void_p

_lib.tjit_tensor_silu.argtypes = [ctypes.c_void_p]
_lib.tjit_tensor_silu.restype = ctypes.c_void_p

_lib.tjit_tensor_rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.tjit_tensor_rmsnorm.restype = ctypes.c_void_p

_lib.tjit_tensor_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.tjit_tensor_save.restype = ctypes.c_bool

_lib.tjit_tensor_load_mmap.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)]
_lib.tjit_tensor_load_mmap.restype = ctypes.c_void_p

_lib.tjit_tensor_cid.argtypes = [ctypes.c_void_p]
_lib.tjit_tensor_cid.restype = ctypes.c_char_p

class Tensor:
    def __init__(self, handle: int, shape: Sequence[int] = None):
        self._handle = handle
        self._shape = list(shape) if shape is not None else []
        if not self._handle:
            raise RuntimeError("Failed to allocate TorchJIT tensor.")

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.tjit_tensor_free(self._handle)
            self._handle = None

    @property
    def shape(self) -> List[int]:
        return self._shape

    def numel(self) -> int:
        return _lib.tjit_tensor_numel(self._handle)

    @property
    def cid(self) -> str:
        res = _lib.tjit_tensor_cid(self._handle)
        return res.decode('utf-8') if res else ""

    def add(self, other: 'Tensor') -> 'Tensor':
        return Tensor(_lib.tjit_tensor_add(self._handle, other._handle), shape=self.shape)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        out_shape = [self.shape[0], other.shape[1]]
        return Tensor(_lib.tjit_tensor_matmul(self._handle, other._handle), shape=out_shape)

    def silu(self) -> 'Tensor':
        return Tensor(_lib.tjit_tensor_silu(self._handle), shape=self.shape)

    def rms_norm(self, eps: float = 1e-6) -> 'Tensor':
        return Tensor(_lib.tjit_tensor_rmsnorm(self._handle, ctypes.c_float(eps)), shape=self.shape)

    def save(self, filepath: str) -> bool:
        return _lib.tjit_tensor_save(self._handle, filepath.encode('utf-8'))

    def __getitem__(self, idx: int) -> float:
        ptr = _lib.tjit_tensor_data_ptr(self._handle)
        return ptr[idx]

    def __setitem__(self, idx: int, val: float):
        ptr = _lib.tjit_tensor_data_ptr(self._handle)
        ptr[idx] = float(val)

def zeros(shape: Sequence[int]) -> Tensor:
    c_shape = (ctypes.c_int64 * len(shape))(*shape)
    handle = _lib.tjit_tensor_zeros(len(shape), c_shape)
    return Tensor(handle, shape=shape)

def ones(shape: Sequence[int]) -> Tensor:
    c_shape = (ctypes.c_int64 * len(shape))(*shape)
    handle = _lib.tjit_tensor_ones(len(shape), c_shape)
    return Tensor(handle, shape=shape)

def empty(shape: Sequence[int]) -> Tensor:
    return zeros(shape)

def add(a: Tensor, b: Tensor) -> Tensor:
    return a.add(b)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)

def silu(a: Tensor) -> Tensor:
    return a.silu()

def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x.rms_norm(eps)

def load_mmap(filepath: str, shape: Sequence[int]) -> Tensor:
    c_shape = (ctypes.c_int64 * len(shape))(*shape)
    handle = _lib.tjit_tensor_load_mmap(filepath.encode('utf-8'), len(shape), c_shape)
    return Tensor(handle, shape=shape)
