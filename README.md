# Keck: Multi Kernel Checker


## 特性

- **编译与执行分离**：CUDA 编译在 CPU 上完成，不占用 GPU 资源
- **多 GPU 并行**：基于 Ray 调度，支持 多 卡 A100 并行验证
- **异步非阻塞**：FastAPI 异步处理，高并发场景下不阻塞
- **配置灵活**：通过环境变量控制编译并行度、超时时间等

## 项目结构

```
app/
├── __init__.py          # 模块说明
├── config.py            # 配置管理（支持环境变量）
├── schemas.py           # 数据模型定义
├── ray_init.py          # Ray 集群初始化
├── main.py              # FastAPI 路由入口
├── services/            # 业务逻辑层
│   ├── compiler.py      # CUDA 编译（纯 CPU，不占 GPU）
│   ├── executor.py      # GPU 执行验证
│   └── pipeline.py      # 流水线编排（编译 → 执行）
├── utils/               # 工具函数
│   ├── ast_utils.py     # AST 重写（load_inline → import）
│   └── message_parser.py # 日志/消息解析
└── templates/           # 验证脚本模板
    ├── test_cuda.py     # CUDA vs PyTorch 验证
    └── test_torch.py    # PyTorch vs PyTorch 验证
```

## 安装

```bash
uv sync
```

或使用 pip：

```bash
pip install -r requirements.txt
```

## 启动服务

### 1. 启动 Ray 集群

单节点模式（使用所有 GPU）：

```bash
ray start --head
```

### 2. 启动 API 服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 32
```

或使用启动脚本：

```bash
./server.sh
```

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `COMPILER_CPUS` | `4` | 单个编译任务占用的 CPU 核数 |
| `COMPILE_TIMEOUT` | `180` | 编译超时时间（秒） |
| `EXEC_TIMEOUT` | `60` | 执行超时时间（秒） |
| `TORCH_CUDA_ARCH_LIST` | `8.0` | 目标 CUDA 架构（A100 = 8.0） |
| `MAX_JOBS` | `1` | nvcc 并行编译任务数 |

示例：

```bash
COMPILER_CPUS=8 COMPILE_TIMEOUT=300 uvicorn app.main:app --port 8000
```

## API 接口

### POST `/compute_score`

验证 CUDA 实现的正确性和性能。

**请求参数**（form-data）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `cuda_code` | string | 是 | 包含 `load_inline` 的 CUDA 实现代码 |
| `torch_code` | string | 是 | PyTorch 基线实现代码 |

**响应示例**：

```json
{
  "formated": true,
  "compiled": true,
  "passed": true,
  "msg": {
    "torch_profile": {
      "torch": "...",
      "cuda": "..."
    },
    "torch_time": 0.001234,
    "cuda_time": 0.000456,
    "speedup": 2.71
  }
}
```

**错误响应**：

```json
{
  "formated": true,
  "compiled": false,
  "passed": false,
  "msg": "failed: compilation error: ..."
}
```

### POST `/check_torch`

对比两个 PyTorch 实现的正确性。

**请求参数**（form-data）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `gt_torch` | string | 是 | 基准 PyTorch 实现 |
| `to_check_torch` | string | 是 | 待检查的 PyTorch 实现 |

### GET `/healthz`

健康检查接口，返回 Ray 集群状态。

**响应示例**：

```json
{
  "ray_initialized": true,
  "cluster_resources": {"CPU": 128, "GPU": 8},
  "available_resources": {"CPU": 120, "GPU": 6}
}
```

## 代码规范

### CUDA 代码要求

`cuda_code` 需要包含以下结构：

```python
import torch
from torch.utils.cpp_extension import load_inline

# CUDA 扩展定义
cuda_source = """
__global__ void my_kernel(...) { ... }
"""

# 使用 load_inline 编译
op = load_inline(
    name="my_extension",
    cpp_sources=[...],
    cuda_sources=[cuda_source],
    functions=["my_function"],
)

class ModelNew(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...
    
    def forward(self, x):
        return op.my_function(x)
```

### PyTorch 基线代码要求

`torch_code` 需要包含以下接口：

```python
import torch

class Model(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...
    
    def forward(self, x):
        ...

def get_init_inputs():
    """返回模型初始化参数"""
    return []

def get_inputs():
    """返回模型前向输入"""
    return [torch.randn(32, 64)]
```

## 架构说明

### 请求处理流程

```
请求 → FastAPI → ValidationPipeline
                      │
                      ├─→ [CPU] compile_cuda_extension
                      │         └─ 编译 CUDA 扩展 (.so)
                      │         └─ 生成 model_new_patch.py
                      │
                      └─→ [GPU] exec_cuda_eval
                                └─ 加载 .so 和模型
                                └─ 正确性检查
                                └─ 性能 benchmark
```

### 并行模型

- **编译任务**：`ray.remote(num_cpus=N, num_gpus=0)` - 纯 CPU，多个编译任务可并行
- **执行任务**：`ray.remote(num_cpus=1, num_gpus=1)` - 每个任务独占一张 GPU

在 8 卡 A100 上：
- 最多 8 个执行任务并行（每卡一个）
- 编译任务数受 CPU 核数限制（如 128 核 / 4 = 32 个并行编译）

## 开发

### 添加新的验证类型

1. 在 `app/templates/` 下添加新的验证脚本
2. 在 `app/services/executor.py` 中添加执行函数
3. 在 `app/services/pipeline.py` 中添加流水线方法
4. 在 `app/main.py` 中添加路由

### 修改配置

编辑 `app/config.py` 或通过环境变量覆盖。

## License

MIT
