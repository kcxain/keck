import os
import requests

gt_model_code = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed'''

to_check_model_code = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)

N = 2048 * 2

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed'''


gpu_ip = os.environ.get("CUDA_SERVER_IP", "10.200.240.10")

# curl -X POST http://10.200.198.14:8000/compute_score   -F "code=123"   -F "timeout_sec=60"   -F "nvcc_flags=-O2 -arch=sm_80"
response = requests.post(
    url=f"http://{gpu_ip}:8000/check_torch",
    data={
        "gt_torch": gt_model_code,
        "to_check_torch": to_check_model_code,
    },
    proxies={"http": None, "https": None},
)
response.raise_for_status()
res_json = response.json()

print(res_json)
