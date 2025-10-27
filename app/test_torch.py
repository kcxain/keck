import torch
import gc

from model import Model, get_inputs, get_init_inputs
from model_new import Model as ModelNew


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


init_inputs = get_init_inputs()

if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]

torch_model = Model(*init_inputs).cuda()
check_model = ModelNew(*init_inputs).cuda()

torch_inputs = get_inputs()
if not isinstance(torch_inputs, (list, tuple)):
    torch_inputs = [torch_inputs]

torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
check_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

# 正确性
with torch.no_grad():
    gt_outputs = torch_model(*torch_inputs)
    check_outputs = check_model(*check_inputs)
    check_equal(check_outputs, gt_outputs)

print("#### Correctness check passed!")
gc.collect()
torch.cuda.empty_cache()
