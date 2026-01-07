import asyncio
from typing import Dict, Any

import ray

from app.config import settings
from app.schemas import ValidationResponse, compile_error, success
from app.services.compiler import compile_cuda_extension
from app.services.executor import exec_cuda_eval, exec_torch_check
from app.utils import parse_compile_msg


async def _async_get(ref: ray.ObjectRef, poll_interval: float = 0.01):
    """
    非阻塞等待 Ray ObjectRef 完成

    使用 ray.wait 轮询而不是 asyncio.to_thread(ray.get)，
    避免线程池开销，更高效地利用事件循环
    """
    while True:
        ready, _ = ray.wait([ref], timeout=0)
        if ready:
            return ray.get(ref)
        await asyncio.sleep(poll_interval)


class ValidationPipeline:
    """
    编排 CPU 编译任务和 GPU 执行任务

    - 编译任务：num_cpus=N, num_gpus=0（纯 CPU）
    - 执行任务：num_cpus=1, num_gpus=1（需要 GPU）
    """

    def __init__(self, compiler_cpus: int | None = None):
        cpus = compiler_cpus or settings.compile.compiler_cpus
        gpus = settings.execute.execute_gpus

        # 创建 Ray remote 函数
        self._compile_remote = ray.remote(num_cpus=cpus)(compile_cuda_extension)
        self._cuda_eval_remote = ray.remote(num_gpus=gpus)(exec_cuda_eval)
        self._torch_check_remote = ray.remote(num_gpus=1)(exec_torch_check)

    # ========== 同步 API ==========

    def compile_and_eval(self, cuda_code: str, pytorch_module: str) -> Dict[str, Any]:
        """同步版本：编译 + 执行"""
        # 1. 编译
        compile_ok, compile_result = ray.get(self._compile_remote.remote(cuda_code))

        if not compile_ok:
            msg = parse_compile_msg(compile_result.message)
            return compile_error(msg).to_dict()

        # 2. 执行
        eval_ok, eval_result = ray.get(
            self._cuda_eval_remote.remote(
                ext_filename=compile_result.ext_filename,
                ext_content=compile_result.ext_content,
                model_new_patch=compile_result.model_new_patch,
                pytorch_module=pytorch_module,
            )
        )

        if eval_ok:
            return success(eval_result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=eval_result
        ).to_dict()

    def check_torch(self, gt_torch: str, to_check_torch: str) -> Dict[str, Any]:
        """同步版本：Torch vs Torch 对比"""
        ok, result = ray.get(self._torch_check_remote.remote(gt_torch, to_check_torch))

        if ok:
            return success(result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=result
        ).to_dict()

    async def compile_and_eval_async(
        self, cuda_code: str, pytorch_module: str
    ) -> Dict[str, Any]:
        """
        异步版本：编译 + 执行

        使用 Ray 原生异步轮询，避免阻塞事件循环
        """
        # 1. 编译（CPU 任务）
        compile_future = self._compile_remote.remote(cuda_code)
        compile_ok, compile_result = await _async_get(compile_future)

        if not compile_ok:
            msg = parse_compile_msg(compile_result.message)
            return compile_error(msg).to_dict()

        # 2. 执行（GPU 任务）
        eval_future = self._cuda_eval_remote.remote(
            ext_filename=compile_result.ext_filename,
            ext_content=compile_result.ext_content,
            model_new_patch=compile_result.model_new_patch,
            pytorch_module=pytorch_module,
        )
        eval_ok, eval_result = await _async_get(eval_future)

        if eval_ok:
            return success(eval_result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=eval_result
        ).to_dict()

    async def check_torch_async(
        self, gt_torch: str, to_check_torch: str
    ) -> Dict[str, Any]:
        """异步版本：Torch vs Torch 对比"""
        future = self._torch_check_remote.remote(gt_torch, to_check_torch)
        ok, result = await _async_get(future)

        if ok:
            return success(result).to_dict()
        return ValidationResponse(
            formated=True, compiled=True, passed=False, msg=result
        ).to_dict()
