## Ray CUDA Execution Service

A FastAPI + Ray service that receives CUDA (*.cu) programs over HTTP, compiles and runs them on a multi-GPU server, and returns stdout/stderr and wall-clock runtime. Scheduling ensures one job per GPU at a time.


### Install
```bash
uv sync
```

### Start Ray and API server
- Single node, all GPUs:
```bash
ray start --head
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 32
```