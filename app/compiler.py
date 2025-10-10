import os
import glob
import tempfile
import subprocess
from typing import Dict, Tuple
from unittest.mock import patch


def _compile_ext(cuda_code: str) -> Tuple[bool, Dict]:
    ret = {
        "ext_filename": None,
        "ext_content": None,
        "msg": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(cuda_code)

        compile_log = ""
        success = True
        try:
            compile_cmd = "python model_new.py"
            # A100: 8.0
            with patch.dict(
                os.environ,
                {
                    "TORCH_CUDA_ARCH_LIST": "8.0",
                    "TORCH_EXTENSIONS_DIR": "build",
                    "MAX_JOBS": "1",
                },
            ):
                compile_result = subprocess.run(
                    compile_cmd,
                    timeout=180,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    cwd=tmpdir,
                )
            compile_log = compile_result.stdout.decode()
            so_files = glob.glob(f"{tmpdir}/build/**/*.so")
            assert len(so_files) == 1, f"should generate 1 .so file, got {so_files}"
            with open(so_files[0], "rb") as fin:
                bin_content = fin.read()
            ret["ext_filename"] = os.path.basename(so_files[0])
            ret["ext_content"] = bin_content
            ret["msg"] = "compile success"
            success = True
        except subprocess.TimeoutExpired:
            success = False
            ret["msg"] = "failed: compilation timed out"
        except Exception as e:
            success = False
            ret["msg"] = f"failed: compilation error: [{e}] log: [{compile_log}]"
        return success, ret
