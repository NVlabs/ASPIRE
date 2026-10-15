# Host setup

What this repository assumes about the machine, and how to provide it. Nothing
here is installed by the bootstraps -- they check for it and fail if it is
missing, sometimes indirectly.

Everything here applies to any machine that runs the provider stack.

## uv

Every bootstrap begins with it and reports only `error: uv is required`. Install
it and put it on `PATH`; on this bench it lives in `~/.local/bin`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"        # add to your shell profile
```

## SAM3 model weights need a Hugging Face token

`scripts/bootstrap_sam3.sh` succeeds without one, and the failure appears later,
when the service first loads a model: `third_party/sam3/sam3/model_builder.py`
calls `hf_hub_download(repo_id=SAM3_MODEL_ID, ...)` against a **gated**
repository. Without credentials the download 401s.

```bash
huggingface-cli login          # or: export HF_TOKEN=...
```

Request access to the SAM3 model on Hugging Face first, or the token will not
help. Weights cache under `~/.cache/huggingface/`, so this is once per machine.

## NVIDIA driver and CUDA

The provider stack needs a working driver; `nvidia-smi` should list a GPU. cuRobo
builds CUDA extensions during `scripts/bootstrap_curobo.sh`, but pulls its own
`nvidia-cuda-nvcc` wheel, so a system CUDA toolkit is **not** required.

## Checking it worked

```bash
cap-harness doctor --providers sam3,pyroki,curobo --unit
```

Then run one example program from [`examples/`](../examples/README.md).

## BEHAVIOR-1K (Isaac Sim) hosts

The BEHAVIOR runtime needs, beyond the items above: Python 3.11 (uv downloads it), a CUDA 12.x
toolkit with `nvcc` to compile cuRobo against the cu128 torch build (a CUDA 13 toolkit alone is
rejected; a user-local runfile install under `~/cuda-12.8` is enough, pointed at with
`CAP_HARNESS_CUDA_HOME`), glibc 2.35 or newer for the Isaac Sim wheels, an NVIDIA Vulkan ICD for
headless rendering, about 12 GB for the venv and 40 GB for the datasets, and 32 GB of RAM. The
Omniverse EULA is accepted by OmniGibson itself at launch; the BEHAVIOR Data Bundle terms are
accepted explicitly with `scripts/bootstrap_behavior.sh --accept-dataset-tos`. The first launch on
a host compiles shaders into `OMNIGIBSON_APPDATA_PATH` for several minutes; keep that directory on
local disk. Isaac chooses its GPU with `OMNIGIBSON_GPU_ID`, so on a single-GPU bench the simulator
shares the card with SAM3 and Contact-GraspNet and only one Isaac process can run at a time.
