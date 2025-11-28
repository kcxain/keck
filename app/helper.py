import re
import torch
import ast


def _extract_cuda_code(text: str):
    codeblock_seps = ["python"]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    codeblock_start = f"```({languages_pattern})"
    pattern = re.compile(codeblock_start + r"\n(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)
    matches = list(pattern.finditer(text))

    if matches:
        last_match = matches[-1]
        # language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        return code_content
    return None


def _validate_cuda_code(code: str):
    try:
        tree = ast.parse(code)
        code_without_comments = ast.unparse(tree)
    except SyntaxError:
        return False, "Invalid Python syntax"

    all_ops = set(torch.ops.aten.__dict__.keys())
    allowed_ops = set([
        "empty",
        "empty_like",
        "empty_strided",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "numel",
        "view",
        "copy",
        "dim",
        "eye",
        "full",
        "full_like",
        "mode",
        "new_empty",
        "new_empty_strided",
        "new_full",
        "new_ones",
        "new_zeros",
        "randn",
        "rand",
    ])
    forbidden_ops = all_ops - allowed_ops
    pattern = re.compile(
        pattern="(torch::|aten::|torch\.)(" + "|".join(forbidden_ops) + ")\(",
        flags=re.DOTALL,
    )
    matched = re.search(pattern, code_without_comments)
    if matched is not None:
        return False, f"Using {matched.group(0)[:-1]} is forbidden"
    return True, "success"




def parse_eval_msg(msg: str):
    # 解析结果
    speedup_match = re.search(
        r"Torch time:\s*([\d.]+)s,\s*CUDA time:\s*([\d.]+)s,\s*Speedup:\s*([\d.]+)x",
        msg,
    )
    assert speedup_match is not None, f"cannot find speedup info from test log: {msg}"
    speed_up_dict = {}
    speed_up_dict["torch_time"] = float(speedup_match.group(1))
    speed_up_dict["cuda_time"] = float(speedup_match.group(2))
    speed_up_dict["speedup"] = float(speedup_match.group(3))

    torch_match = re.search(
        r"#### Benchmark Torch Start\s*(.*?)\s*#### Benchmark Troch End", msg, re.S
    )
    cuda_match = re.search(
        r"#### Benchmark CUDA Start\s*(.*?)\s*#### Benchmark CUDA End", msg, re.S
    )
    assert torch_match is not None, (
        f"cannot find torch benchmark info from test log: {msg}"
    )
    assert cuda_match is not None, (
        f"cannot find cuda benchmark info from test log: {msg}"
    )
    torch_profiler = torch_match.group(1).strip()
    cuda_profiler = cuda_match.group(1).strip()
    return {
        "torch_profile":{
            "torch": torch_profiler,
            "cuda": cuda_profiler,
        },
        "torch_time":speed_up_dict["torch_time"],
        "cuda_time":speed_up_dict["cuda_time"],
        "speedup":speed_up_dict["speedup"],
    }