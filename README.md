## Ray CUDA Execution Service

A FastAPI + Ray service that receives CUDA (*.cu) programs over HTTP, compiles and runs them on a multi-GPU server, and returns stdout/stderr and wall-clock runtime. Scheduling ensures one job per GPU at a time.

### Requirements
- CUDA toolkit (nvcc, runtime) installed on GPU server
- NVIDIA drivers and nvidia-smi
- Python 3.10+

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Start Ray and API server
- Single node, all GPUs:
```bash
ray start --head
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Remote servers can call the API at `http://<server_ip>:8000/run`.

- To form a Ray cluster (optional):
  - Head node:
    ```bash
    ray start --head
    ```
  - Worker nodes:
    ```bash
    ray start --address '<head_ip>:6379'
    ```

### API
POST `/run`
- Form fields:
  - `code`: file upload of a `.cu` CUDA source or plain text
  - `timeout_sec` (optional, default 120)
  - `nvcc_flags` (optional string, e.g. `-O2 -arch=sm_80`)

Response JSON:
```json
{
  "stdout": "...",
  "stderr": "...",
  "return_code": 0,
  "compile_stderr": "",
  "compile_return_code": 0,
  "runtime_sec": 0.123,
  "node_id": "...",
  "gpu_index": "0"
}
```

### Security
- This runs untrusted code. Use in a controlled environment. Consider container sandboxing (Docker) and network isolation.

### Notes
- Scheduling uses Ray actors with `num_gpus=1`, ensuring only one job runs per GPU simultaneously.
- NVCC compilation is performed per request in a temporary directory.
