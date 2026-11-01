import sys
import os
import shutil

sys.path.append("/media/thealidev/RMAA HDD1/RMAAPK/TorchJIT/python")
import torchjit as torch
from torchjit.chunker import MicroSharder

def test_micro_sharding():
    test_dir = "/tmp/tjit_chunks"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    # Create a 6MB tensor (1.5 million float32 numbers)
    # With 2MB chunk size, it must shard cleanly into exactly 3 chunks of 1-3MB pieces!
    num_floats = 1572864 # 1.5 * 1024 * 1024 floats = 6 MB
    t = torch.zeros([num_floats])
    for i in range(10):
        t[i] = float(i + 1)

    bin_path = os.path.join(test_dir, "weights_l0.bin")
    t.save(bin_path)
    print(f"Saved test tensor: {os.path.getsize(bin_path) / (1024*1024):.2f} MB")

    # Shard into 2MB chunks
    manifest = MicroSharder.shard_tensor_file(bin_path, test_dir, chunk_size=2 * 1024 * 1024)
    print(f"Manifest Generated: {len(manifest['chunks'])} chunks")
    for ch in manifest['chunks']:
        print(f"  Chunk #{ch['index']}: CID={ch['cid']} | Size={ch['size'] / (1024*1024):.2f} MB")

    assert len(manifest['chunks']) == 3, f"Expected 3 chunks, got {len(manifest['chunks'])}"

    # Lazy stream back into TorchJIT tensor
    reconstructed = MicroSharder.lazy_mmap_tensor(test_dir, manifest, [num_floats])
    for i in range(10):
        assert reconstructed[i] == float(i + 1), f"Value mismatch at {i}: {reconstructed[i]} vs {i+1}"

    print("✅ Micro-Sharding 1-3 MB slice & lazy reconstruction verified successfully!")

if __name__ == "__main__":
    test_micro_sharding()
