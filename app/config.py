import os
from dataclasses import dataclass, field


@dataclass
class RayConfig:
    """Ray 集群配置"""

    num_cpus: int = field(default_factory=lambda: int(os.getenv("NUM_CPUS", "64")))
    num_gpus: int = field(default_factory=lambda: int(os.getenv("NUM_GPUS", "8")))
    env_vars: dict = field(
        default_factory=lambda: {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "0",
            "BPEX_NO_WARN_ON_UNTUNED_CASE": "1",
        }
    )


@dataclass
class CompileConfig:
    """CUDA 编译配置"""

    # 编译任务占用的 CPU 核数
    compiler_cpus: int = field(
        default_factory=lambda: int(os.getenv("COMPILER_CPUS", "4"))
    )
    # 编译超时时间（秒）
    compile_timeout: int = field(
        default_factory=lambda: int(os.getenv("COMPILE_TIMEOUT", "180"))
    )
    # CUDA 安装路径
    cuda_home: str = field(
        default_factory=lambda: os.getenv(
            "CUDA_HOME", "/tools/cluster-software/cuda-cudnn/cuda-12.4.1-9.1.1"
        )
    )
    # 目标 CUDA 架构（A100 = 8.0）
    cuda_arch_list: str = field(
        default_factory=lambda: os.getenv("TORCH_CUDA_ARCH_LIST", "8.0")
    )
    # nvcc 并行编译任务数
    max_jobs: str = field(default_factory=lambda: os.getenv("MAX_JOBS", "1"))


@dataclass
class ExecuteConfig:
    """GPU 执行配置"""

    # 执行超时时间（秒）
    exec_timeout: int = field(
        default_factory=lambda: int(os.getenv("EXEC_TIMEOUT", "120"))
    )
    # benchmark warmup 次数
    warmup_runs: int = 1
    # benchmark 正式运行次数
    benchmark_runs: int = 1
    # GPU 资源占用（1 = 独占一张卡，0.5 = 两个任务共享一张卡）
    execute_gpus: float = field(
        default_factory=lambda: float(os.getenv("EXECUTE_GPUS_PER_TASK", "1"))
    )


@dataclass
class Settings:
    """全局配置"""

    ray: RayConfig = field(default_factory=RayConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    execute: ExecuteConfig = field(default_factory=ExecuteConfig)

    # 模板文件路径
    templates_dir: str = "app/templates"


# 全局单例
settings = Settings()
