import torch
import torch.nn.functional as F
import ast
from pathlib import Path
import gc
import time
from contextlib import contextmanager


def rewrite_cuda_model_code(src_path, dst_path):
    """Replace "op = load_inline" with "import op" to separate compilation and execution"""
    model_src = Path(src_path).read_text()
    tree = ast.parse(model_src)
    for i, node in enumerate(tree.body):
        if (
            isinstance(node, ast.Assign)
            and isinstance(call := node.value, ast.Call)
            and (
                (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "load_inline"
                )
                or (isinstance(call.func, ast.Name) and call.func.id == "load_inline")
            )
        ):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            ext_alias = node.targets[0].id
            for kw in call.keywords:
                if kw.arg == "name":
                    assert isinstance(kw.value, ast.Constant)
                    ext_name = kw.value.value
                    break
            else:
                raise RuntimeError("Cannot find extension name from model_new.py")
            tree.body[i] = ast.parse(f"import {ext_name} as {ext_alias}").body[0]
    model_src = ast.unparse(tree)
    Path(dst_path).write_text(model_src)


rewrite_cuda_model_code(src_path="model_new.py", dst_path="model_new_patch.py")
from model import Model, get_inputs, get_init_inputs
from model_new_patch import ModelNew


def transform_tensors(tensors, fn):
    if not isinstance(tensors, (list, tuple)):
        return tensors
    outputs = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = fn(tensor)
        elif isinstance(tensor, (list, tuple)):
            tensor = transform_tensors(tensor, fn)
        elif isinstance(tensor, dict):
            tensor = {k: transform_tensors(v, fn) for k, v in tensor.items()}
        outputs.append(tensor)
    return outputs


def check_equal(actual, expected):
    assert isinstance(actual, (list, tuple)) == isinstance(expected, (list, tuple))
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
        expected = [expected]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-2, rtol=1e-2)


def benchmark_model(model, inputs, warmup_runs=1, benchmark_runs=2):
    """Benchmark model execution time"""
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*inputs)
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(benchmark_runs):
            _ = model(*inputs)
            torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) / benchmark_runs

    # torch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            _ = model(*inputs)
    # 打印的是表格
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # json_summary = profile_to_json(prof, top_k=10)
    torch.cuda.synchronize()
    return avg_time


# @contextmanager
# def block_torch_functional(excludes=None):
#     if excludes is None:
#         excludes = set()
#     originals = {}
#     for name in dir(F):
#         attr = getattr(F, name)
#         if callable(attr) and not name.startswith("_") and name not in excludes:
#             originals[name] = attr

#             def wrapper(*args, __name=name, **kwargs):
#                 raise RuntimeError(
#                     f"Function {F.__name__}.{__name} is not allowed in this context."
#                 )

#             setattr(F, name, wrapper)
#     try:
#         yield
#     finally:
#         for name, attr in originals.items():
#             setattr(F, name, attr)


init_inputs = get_init_inputs()

if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]

torch_model = Model(*init_inputs).cuda()
cuda_model = ModelNew(*init_inputs).cuda()

cuda_model.load_state_dict(torch_model.state_dict())

torch_inputs = get_inputs()
if not isinstance(torch_inputs, (list, tuple)):
    torch_inputs = [torch_inputs]

torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

# 正确性
with torch.no_grad():
    # CUDA outputs
    #   with block_torch_functional():
    cuda_outputs = cuda_model(*cuda_inputs)
    torch.cuda.synchronize()
    # Torch outputs
    torch_outputs = torch_model(*torch_inputs)
    torch.cuda.synchronize()
    check_equal(cuda_outputs, torch_outputs)


print("#### Correctness check passed!")
print("#### Benchmark Torch Start")
torch_time = benchmark_model(torch_model, torch_inputs)
print("#### Benchmark Troch End")
# with block_torch_functional():
print("#### Benchmark CUDA Start")
cuda_time = benchmark_model(cuda_model, cuda_inputs)
print("#### Benchmark CUDA End")
speedup = torch_time / cuda_time if cuda_time > 0 else 0
print(
    f"Torch time: {torch_time:.6f}s, CUDA time: {cuda_time:.6f}s, Speedup: {speedup:.2f}x"
)
del cuda_model, cuda_inputs, torch_model, torch_inputs
if "cuda_outputs" in locals():
    del cuda_outputs
if "torch_outputs" in locals():
    del torch_outputs
gc.collect()
torch.cuda.empty_cache()
