import glob
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

from app.config import settings
from app.schemas import CompileResult
from app.utils import rewrite_load_inline


def compile_cuda_extension(cuda_code: str) -> Tuple[bool, CompileResult]:
    """
    编译用户提供的 CUDA 扩展代码

    流程:
    1. 把 cuda_code 写入临时目录的 model_new.py
    2. 执行 python model_new.py 触发 load_inline 编译
    3. 收集生成的 .so 文件
    4. 用 AST 重写生成 model_new_patch.py（把 load_inline 改为 import）

    Returns:
        (success, CompileResult)
    """
    cfg = settings.compile

    with tempfile.TemporaryDirectory() as tmpdir:
        model_new_path = os.path.join(tmpdir, "model_new.py")
        Path(model_new_path).write_text(cuda_code)

        compile_log = ""
        try:
            # 设置编译环境变量
            env_overrides = {
                "CUDA_HOME": cfg.cuda_home,
                "TORCH_CUDA_ARCH_LIST": cfg.cuda_arch_list,
                "TORCH_EXTENSIONS_DIR": "build",
                "MAX_JOBS": cfg.max_jobs,
            }

            with patch.dict(os.environ, env_overrides):
                result = subprocess.run(
                    "python model_new.py",
                    timeout=cfg.compile_timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    cwd=tmpdir,
                )

            compile_log = result.stdout.decode()

            # 查找生成的 .so 文件
            so_files = glob.glob(f"{tmpdir}/build/**/*.so", recursive=True)
            if len(so_files) != 1:
                raise RuntimeError(
                    f"Expected 1 .so file, got {len(so_files)}: {so_files}"
                )

            # 读取编译产物
            ext_filename = os.path.basename(so_files[0])
            ext_content = Path(so_files[0]).read_bytes()

            # 生成 model_new_patch.py
            patch_path = os.path.join(tmpdir, "model_new_patch.py")
            rewrite_load_inline(src_path=model_new_path, dst_path=patch_path)
            model_new_patch = Path(patch_path).read_text()
            print(model_new_patch)
            return True, CompileResult(
                success=True,
                ext_filename=ext_filename,
                ext_content=ext_content,
                model_new_patch=model_new_patch,
                message="compile success",
            )

        except subprocess.TimeoutExpired:
            return False, CompileResult(
                success=False,
                message="failed: compilation timed out",
            )
        except Exception as e:
            return False, CompileResult(
                success=False,
                message=f"failed: compilation error: [{e}] log: [{compile_log}]",
            )
