"""
FastAPI 入口：只负责路由和请求处理
"""

from typing import Optional

import ray
from fastapi import FastAPI, Form

from app.ray_init import init_ray
from app.schemas import format_error
from app.services import ValidationPipeline


APP_NAME = "ray-cuda-exec"
app = FastAPI(title=APP_NAME)

# 初始化 Ray 和验证流水线
init_ray()
pipeline = ValidationPipeline()


@app.post("/compute_score")
async def compute_score(
    cuda_code: Optional[str] = Form(None),
    torch_code: Optional[str] = Form(None),
):
    """
    验证 CUDA 实现的正确性和性能

    Args:
        cuda_code: 包含 load_inline 的 CUDA 实现代码
        torch_code: PyTorch 基线实现代码
    """
    if cuda_code is None:
        return format_error("Cannot find cuda code block").to_dict()

    return await pipeline.compile_and_eval_async(cuda_code, torch_code or "")


@app.post("/check_torch")
async def check_torch(
    gt_torch: Optional[str] = Form(None),
    to_check_torch: Optional[str] = Form(None),
):
    """
    对比两个 PyTorch 实现的正确性

    Args:
        gt_torch: 基准 PyTorch 实现
        to_check_torch: 待检查的 PyTorch 实现
    """
    if to_check_torch is None:
        return format_error("Cannot find torch code block").to_dict()

    return await pipeline.check_torch_async(gt_torch or "", to_check_torch)


@app.get("/healthz")
async def healthz():
    """健康检查接口"""
    return {
        "ray_initialized": ray.is_initialized(),
        "cluster_resources": ray.cluster_resources(),
        "available_resources": ray.available_resources(),
    }
