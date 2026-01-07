import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Any

import torch

from app.config import settings
from app.utils import parse_eval_msg


# 延迟加载模板内容
_CUDA_TEMPLATE: str = ""
_TORCH_TEMPLATE: str = ""


def _load_templates():
    """延迟加载模板文件"""
    global _CUDA_TEMPLATE, _TORCH_TEMPLATE
    if not _CUDA_TEMPLATE:
        template_dir = settings.templates_dir
        _CUDA_TEMPLATE = Path(f"{template_dir}/test_cuda.py").read_text()
        _TORCH_TEMPLATE = Path(f"{template_dir}/test_torch.py").read_text()


def exec_cuda_eval(
    ext_filename: str,
    ext_content: bytes,
    model_new_patch: str,
    pytorch_module: str,
) -> Tuple[bool, Any]:
    """
    在 GPU 上执行 CUDA 扩展 vs PyTorch 的正确性验证和性能测试
    
    Args:
        ext_filename: 编译好的 .so 文件名
        ext_content: .so 文件内容
        model_new_patch: 重写后的 model_new_patch.py 内容
        pytorch_module: PyTorch 基线实现（model.py）
        
    Returns:
        (success, result_or_error_msg)
    """
    _load_templates()
    cfg = settings.execute
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入所有需要的文件
        Path(os.path.join(tmpdir, ext_filename)).write_bytes(ext_content)
        Path(os.path.join(tmpdir, "model_new_patch.py")).write_text(model_new_patch)
        Path(os.path.join(tmpdir, "model.py")).write_text(pytorch_module)
        Path(os.path.join(tmpdir, "test.py")).write_text(_CUDA_TEMPLATE)
        
        test_log = ""
        try:
            result = subprocess.run(
                "python test.py",
                timeout=cfg.exec_timeout,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=tmpdir,
            )
            test_log = result.stdout.decode()
            
        except subprocess.TimeoutExpired:
            return False, "failed: test timed out"
        except Exception as e:
            return False, f"failed: test error: [{e}] log: [{test_log}]"
        finally:
            _cleanup_gpu()
        
        if result.returncode != 0:
            return False, _extract_error(test_log)
        
        if "#### Correctness check passed!" not in test_log:
            return False, f"failed: correctness check not passed. log: [{test_log[:500]}]"
        
        return True, parse_eval_msg(test_log)


def exec_torch_check(gt_torch: str, to_check_torch: str) -> Tuple[bool, str]:
    """
    在 GPU 上执行两个 PyTorch 实现的正确性对比
    
    Args:
        gt_torch: 基准 PyTorch 实现（model.py）
        to_check_torch: 待检查的 PyTorch 实现（model_new.py）
        
    Returns:
        (success, result_or_error_msg)
    """
    _load_templates()
    cfg = settings.execute
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model.py")).write_text(gt_torch)
        Path(os.path.join(tmpdir, "model_new.py")).write_text(to_check_torch)
        Path(os.path.join(tmpdir, "test.py")).write_text(_TORCH_TEMPLATE)
        
        test_log = ""
        try:
            result = subprocess.run(
                "python test.py",
                timeout=cfg.exec_timeout,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=tmpdir,
            )
            test_log = result.stdout.decode()
            
        except subprocess.TimeoutExpired:
            return False, "failed: test timed out"
        except Exception as e:
            return False, f"failed: test error: [{e}] log: [{test_log}]"
        finally:
            _cleanup_gpu()
        
        if result.returncode != 0:
            return False, _extract_error(test_log)
        
        if "#### Correctness check passed!" not in test_log:
            return False, f"failed: correctness check not passed"
        
        return True, test_log


def _cleanup_gpu():
    """清理 GPU 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _extract_error(test_log: str) -> str:
    """从测试日志中提取错误信息"""
    lines = test_log.strip().splitlines()
    filtered = [
        line for line in lines 
        if "AssertionError" in line or "Mismatch" in line or "Error" in line
    ]
    if filtered:
        return f"failed: test error: {chr(10).join(filtered[:5])}"
    return f"failed: test error: {chr(10).join(lines[:5])}"

