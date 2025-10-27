from dataclasses import dataclass
import glob
import os
import re
import subprocess
import tempfile
import torch
from unittest.mock import patch
from typing import Dict, Optional, Tuple
import orjson
import ray
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from app.helper import _validate_cuda_code, parse_eval_msg

APP_NAME = "ray-cuda-exec"

app = FastAPI(title=APP_NAME)

if not ray.is_initialized():
    # this is for local ray cluster
    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "0",
            "BPEX_NO_WARN_ON_UNTUNED_CASE": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
    }
    try:
        # 首选带资源限制的初始化（适用于新启动的本地集群）
        ray.init(runtime_env=runtime_env, num_cpus=128, num_gpus=8)
    except ValueError as e:
        # 如果正在连接到已有集群，不能传 num_cpus/num_gpus，重试一次不传这些参数
        msg = str(e)
        if (
            "When connecting to an existing cluster, num_cpus and num_gpus must not be provided"
            in msg
        ):
            ray.init(runtime_env=runtime_env)
        else:
            # 不是这个已知问题则继续抛出
            raise


def json_response(data: dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=orjson.loads(orjson.dumps(data)), status_code=status_code
    )


with open("app/test_code_tmpl.py", "r") as fin:
    TEST_CODE_TMPL = fin.read()

with open("app/test_torch.py", "r") as fin:
    TEST_TORCH_TMPL = fin.read()


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


def _exec_eval(
    ext_filename: str, ext_content: bytes, cuda_code: str, pytorch_module: str
):
    """Compile and execute test code which checks output with cuda implementation and pytorch module
    :param ext_filename: the cuda extension filename, in the format as "cuda_module.cpython-xxx.so"
    :param ext_content: file content of the extension file
    :param cuda_code: file content of the python file containing inline cuda code
    :param pytorch_module: pytorch baseline implementation. Should have Model.forward(...) and get_inputs() api
    :return (status,msg): (True,stdout) for success, (False,stderr) for error
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, ext_filename), "wb") as fout:
            fout.write(ext_content)
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(cuda_code)
        with open(os.path.join(tmpdir, "model.py"), "w") as fout:
            fout.write(pytorch_module)
        with open(os.path.join(tmpdir, "test.py"), "w") as fout:
            fout.write(TEST_CODE_TMPL)
        # with open(os.path.join(tmpdir, "profiler.py"), "w") as fout:
        #     fout.write(PROFILER_CODE)

        test_log = ""
        try:
            test_cmd = "python test.py"
            test_result = subprocess.run(
                test_cmd,
                timeout=60,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=tmpdir,
            )
            test_log = test_result.stdout.decode()
        except subprocess.TimeoutExpired:
            # 超时
            return False, "failed: test timed out"
        except Exception as e:
            # 错误
            return False, f"failed: test error: [{e}] log: [{test_log}]"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                # logger.info(
                #     f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                # )
        if test_result.returncode != 0:
            # assert error, 输出可能过长
            lines = test_log.strip().splitlines()
            filtered = []
            for line in lines:
                if "AssertionError" in line or "Mismatch" in line:
                    filtered.append(line)
            short_log = "\n".join(filtered)
            if filtered == []:
                short_log = lines[:5]
            return False, f"failed: test error: {short_log}"
    assert "#### Correctness check passed!" in test_log
    return True, parse_eval_msg(test_log)


def parse_compile_msg(msg: str, keep_levels=5) -> str:
    # 1. 提取关键行
    key_lines = []
    for line in msg.splitlines():
        if re.search(
            r"(error:|FAILED:|RuntimeError:|CalledProcessError:)", line, re.IGNORECASE
        ):
            key_lines.append(line.strip())

    if not key_lines:
        return "\n".join(msg.splitlines()[:10])

    # 2. 简化路径，保留最后 keep_levels 层
    def simplify_path_keep_levels(text: str, levels=keep_levels):
        # 匹配所有类似 /a/b/c/.../file.ext 的路径
        def replacer(m):
            full_path = m.group(0)
            parts = full_path.strip("/").split("/")
            if len(parts) > levels:
                parts = parts[-levels:]  # 只保留最后几层
            return "/".join(parts)

        # 匹配 .cu/.cpp/.c/.h/.py/.so/.o 文件
        text = re.sub(
            r"(/[A-Za-z0-9_\-./]+/)*[A-Za-z0-9_\-]+\.(?:cu|cpp|c|cc|h|py|so|o|d)",
            replacer,
            text,
        )
        # 去掉多余引号和转义
        text = text.replace("\\", "").replace("'", "").replace('"', "")
        return text.strip()

    simplified = [simplify_path_keep_levels(l) for l in key_lines]

    # 3. 去重并压缩空格
    seen = set()
    cleaned = []
    for line in simplified:
        line = re.sub(r"\s+", " ", line)
        if line not in seen:
            seen.add(line)
            cleaned.append(line)

    return "\n".join(cleaned[:5])


def compile_and_eval(cuda_code: str, pytorch_module: str):
    remote_compile_ext = ray.remote(num_cpus=8)(_compile_ext)
    compile_succ, compile_ret = ray.get(remote_compile_ext.remote(cuda_code=cuda_code))
    compile_msg = compile_ret["msg"]
    compile_msg = parse_compile_msg(compile_msg)
    if not compile_succ:
        return {
            "formated": True,
            "compiled": False,
            "passed": False,
            "msg": compile_msg,
        }

    ext_filename = compile_ret["ext_filename"]
    ext_content = compile_ret["ext_content"]
    run_kwargs = dict(
        ext_filename=ext_filename,
        ext_content=ext_content,
        cuda_code=cuda_code,
        pytorch_module=pytorch_module,
    )
    gpu_eval_task = ray.remote(num_gpus=1)(_exec_eval)
    eval_future = gpu_eval_task.remote(**run_kwargs)
    status, eval_content = ray.get(eval_future)
    return {
        "formated": True,
        "compiled": True,
        "passed": status,
        "msg": eval_content,
    }


@dataclass
class Result:
    formated: bool
    compiled: bool
    passed: bool
    msg: str | dict


@app.post("/compute_score")
async def compute_score(
    cuda_code: Optional[str] = Form(None), torch_code: Optional[str] = Form(None)
):
    # cuda_code = _extract_cuda_code(cuda_code)

    if cuda_code is None:
        res = Result(
            formated=False,
            compiled=False,
            passed=False,
            msg="Cannot find cuda code block",
        )
        return json_response(res)
    else:
        validate_ret, validate_msg = _validate_cuda_code(cuda_code)
        if not validate_ret:
            res = Result(
                formated=True,
                compiled=False,
                passed=False,
                msg=validate_msg,
            )
            return json_response(res)

        res = compile_and_eval(cuda_code, torch_code)
        return json_response(res)


def _exec_torch(gt_torch: str, to_check_torch: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "model_new.py"), "w") as fout:
            fout.write(to_check_torch)
        with open(os.path.join(tmpdir, "model.py"), "w") as fout:
            fout.write(gt_torch)
        with open(os.path.join(tmpdir, "test.py"), "w") as fout:
            fout.write(TEST_TORCH_TMPL)

        test_log = ""
        try:
            test_cmd = "python test.py"
            test_result = subprocess.run(
                test_cmd,
                timeout=60,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=tmpdir,
            )
            test_log = test_result.stdout.decode()
        except subprocess.TimeoutExpired:
            # 超时
            return False, "failed: test timed out"
        except Exception as e:
            # 错误
            return False, f"failed: test error: [{e}] log: [{test_log}]"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                # logger.info(
                #     f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                # )
        if test_result.returncode != 0:
            # assert error, 输出可能过长
            lines = test_log.strip().splitlines()
            filtered = []
            for line in lines:
                if "AssertionError" in line or "Mismatch" in line:
                    filtered.append(line)
            short_log = "\n".join(filtered)
            if filtered == []:
                short_log = lines[:5]
            return False, f"failed: test error: {short_log}"
    assert "#### Correctness check passed!" in test_log
    return True, test_log


@app.post("/check_torch")
async def check_torch(
    gt_torch: Optional[str] = Form(None), to_check_torch: Optional[str] = Form(None)
):
    if to_check_torch is None:
        res = Result(
            formated=False,
            compiled=False,
            passed=False,
            msg="Cannot find torch code block",
        )
        return json_response(res)
    else:
        run_kwargs = dict(gt_torch=gt_torch, to_check_torch=to_check_torch)
        gpu_eval_task = ray.remote(num_gpus=1)(_exec_torch)
        eval_future = gpu_eval_task.remote(**run_kwargs)
        status, eval_content = ray.get(eval_future)
        res = {
            "formated": True,
            "compiled": True,
            "passed": status,
            "msg": eval_content,
        }
        return json_response(res)


@app.get("/healthz")
async def healthz():
    info = {
        "ray_initialized": ray.is_initialized(),
        "cluster_resources": ray.cluster_resources(),
        "available_resources": ray.available_resources(),
    }
    return json_response(info)
