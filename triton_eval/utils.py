"Some utils"
import torch
import os

def to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, device) for v in x]
    return x



def set_gpu_arch(gpu: str="h100"):
    """
    Set env variable for torch cuda arch list to build kernels for specified architectures
    """

    gpu_arch_mapping = {
        "l40s": ["Ada"],
        "h100": ["Hopper"],
        "a100": ["Ampere"],
        "l4": ["Ada"],
        "t4": ["Turing"],
        "a10g": ["Ampere"],
    }
    arch_list = gpu_arch_mapping[gpu.lower()]
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(
                f"Invalid architecture: {arch}. Must be one of {valid_archs}"
            )

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)


def read_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""