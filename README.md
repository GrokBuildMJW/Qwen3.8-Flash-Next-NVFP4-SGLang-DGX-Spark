# Qwen3.8-Flash-Next NVFP4 + NEXTN on one DGX Spark (SGLang)

Serving recipe for **Qwen3.8-Flash-Next NVFP4** on one NVIDIA DGX Spark (GB10 / SM121) using **SGLang** (`lmsysorg/sglang:qwen38flashnext`).

This repository is a **serving config for the [Ironclad AI](https://github.com/GrokBuildMJW/ironclad-ai) use case**: a local OpenAI-compatible backend for the agent orchestrator. Ironclad clients call `"model": "Qwen3.8-Flash-Next"` with thinking off, the `qwen3_coder` tool parser, and the `qwen3` reasoning parser. The live pin matches Ironclad's packaged `spark` orchestrator profile: **65,536-token context**, **eight concurrent requests**, weights revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

It is a production pin plus launcher, not a training or fine-tune tree. Throughput A/B tables for this engine live in the upstream GB10 recipe this tree vendors patches from; this repo records **what Ironclad actually serves**.

Sibling (previous production / rollback): [Qwen3.8-27B NVFP4 on vLLM](https://github.com/GrokBuildMJW/Qwen3.8-27B-NVFP4-vLLM-DGX-Spark). One GPU occupant — do not run both at once.

## Pinned stack

Recorded from the live Ironclad backend on one GB10 (container `qwen38-flash-next`, serving since 2026-09-03, documented 2026-09-07).

| Piece | Value |
|---|---|
| Target | `RadixArk/Qwen3.8-Flash-Next-NVFP4` served as `Qwen3.8-Flash-Next` |
| Revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Architecture | `qwen4_exp` — ~180B multimodal MoE (125B compute + 51B PLE n-gram + MTP), ~6B active/token, NVFP4 routed experts |
| Image | `lmsysorg/sglang:qwen38flashnext` |
| Image digest | `sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1` |
| Engine commit | `d91c3682b0b429e4c70df63cd57f819588ce29b0` |
| Topology | 1x GB10, TP=1, `--context-length 65536` (native card window is 262144) |
| Speculation | NEXTN 3/1/4, draft left unquant (`--speculative-draft-model-quantization unquant`) |
| PLE | file-backed mmap on NVMe (`--ple-offload-embedding`, ~48 GiB backing file) |
| GB10 patches | [shantanugoel/qwen38-flash-next-sglang-dgx-spark](https://github.com/shantanugoel/qwen38-flash-next-sglang-dgx-spark) `@ 6a81afc` |
| CPU pin | Docker `--cpuset-cpus 0-19` (full GB10; not the 27B X5-only pin) |
| Memory cap | `--memory 116g --memory-swap 116g` |
| Listen | `0.0.0.0:8000` → container `:30000` |

The 27B vLLM pin used 65,536 context and 24 sequences. Flash-Next keeps the same context for the Ironclad agent loop and drops concurrency to **8**, which is the MPR fan-out width Ironclad is calibrated for.

## Ironclad contract

These flags are the ones Ironclad's `spark` profile is calibrated against. Changing the served name, context, or concurrency without re-running orchestrator calibration is a different backend.

| Ironclad setting | This pin |
|---|---|
| `llm.profiles.spark.model` | `Qwen3.8-Flash-Next` |
| `model_identity.provider` | `sglang` |
| `model_identity.provider_version` | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| `context_window_tokens` | `65536` |
| `max_concurrent_requests` | `8` |
| `default_thinking_policy` | disabled (`chat_template_kwargs.enable_thinking=false`) |
| Tool parser | `qwen3_coder` |
| Reasoning parser | `qwen3` |
| Sampling (checkpoint default) | `temperature=1.0`, `top_p=0.95`, `top_k=20` |

## Why the GB10 patches exist

The NVFP4 checkpoint is ~135 GB; a Spark has 128 GB unified memory. Stock `--ple-offload-embedding` still pins the 48 GiB PLE n-gram table in the same UMA pool, so the model does not fit. A token only reads 16 PLE rows (~2.5 KB). File-backed `torch.from_file` keeps the table on NVMe; CUDA graphs still work because GB10 walks the host page tables.

Stock SGLang also gates the TRT-LLM sparse decode kernel to SM100. GB10 is SM121, so QSA otherwise falls into an FA4 path that does not compile. The vendored patches widen that gate and mmap the PLE table.

First boot copies the PLE table into the mmap file (~45–60 min). Later boots reuse shards already on disk (~10 min) via `patches/ple_reuse.py`.

## Serve (Ironclad pin)

On a DGX Spark / GB10 with Docker GPU passthrough, Hugging Face access to the checkpoint, and ~230 GB free NVMe:

```bash
cp .env.example .env   # set HF_TOKEN
./scripts/prepare.sh   # pull the pinned image, patch two in-image files, download weights
./scripts/serve.sh     # :8000, unless-stopped
./scripts/wait_ready.sh
./scripts/smoke.sh
```

Equivalent production flags (after `prepare.sh` has written `build/`):

```bash
docker run -d --name qwen38-flash-next --init --restart unless-stopped \
  --user "$(id -u):$(id -g)" --group-add video --group-add render \
  --gpus all --ipc=host --shm-size 16g \
  --memory 116g --memory-swap 116g --cpuset-cpus 0-19 \
  --workdir /tmp \
  -p 0.0.0.0:8000:30000 \
  -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e HF_TOKEN \
  -e HF_HOME=/huggingface \
  -e SGLANG_QWEN4_PLE_MMAP_DIR=/ple \
  -v "$HOME/.cache/huggingface:/huggingface" \
  -v "$PWD/data/sglang-cache:/tmp/.cache/sglang" \
  -v "$PWD/data/ple:/ple" \
  -v "$PWD/build/qwen4_exp.py:<in-image qwen4_exp.py>:ro" \
  -v "$PWD/build/qwen_sparse_attn_backend.py:<in-image qsa backend>:ro" \
  lmsysorg/sglang:qwen38flashnext \
  sglang serve \
    --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
    --served-model-name Qwen3.8-Flash-Next \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-track-interval 64 \
    --max-mamba-cache-size 40 \
    --mamba-ssm-dtype float32 \
    --chunked-prefill-size 4096 \
    --max-running-requests 8 \
    --max-total-tokens 524288 \
    --context-length 65536 \
    --mem-fraction-static 0.95 \
    --allow-auto-truncate \
    --ple-offload-embedding \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --prefill-attention-backend triton \
    --decode-attention-backend trtllm_mha \
    --disable-prefill-cuda-graph \
    --disable-flashinfer-autotune \
    --enable-metrics \
    --enable-cache-report \
    --enable-gdn-replayssm-spec \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant
```

`prepare.sh` resolves the two in-image Python paths and writes them to `build/path_qwen4_exp.txt` and `build/path_qsa.txt`. Do not guess those paths; they move between image builds.

Clients call `http://127.0.0.1:8000/v1` with `"model": "Qwen3.8-Flash-Next"`. For instruct / tool-heavy agent turns send `chat_template_kwargs.enable_thinking=false`.

Default bind is `0.0.0.0` because Ironclad talks to this endpoint over the LAN. Restrict with host firewall rules. Loopback-only: `BIND_ADDR=127.0.0.1 ./scripts/serve.sh`.

Stop: `./scripts/stop.sh`.

## Deltas vs the upstream GB10 recipe

[shantanugoel/qwen38-flash-next-sglang-dgx-spark](https://github.com/shantanugoel/qwen38-flash-next-sglang-dgx-spark) is the GB10 patch source (`@ 6a81afc`). This tree keeps those patches and changes the serve contract to the Ironclad pin:

| Knob | Upstream recipe default | This pin (live) |
|---|---|---|
| Listen | `127.0.0.1:30000` | `0.0.0.0:8000` |
| Served name | `qwen38-flash-next-nvfp4-mtp` | `Qwen3.8-Flash-Next` |
| Context | 262144 | **65536** |
| Max running | 4 | **8** |
| Mamba cache slots | 20 | **40** |
| Restart | none | `unless-stopped` |

262k is the native rope limit and still boots. Ironclad is calibrated at 64k / 8 slots; raising context without dropping concurrency starves the KV / GDN pool on one GB10. `--max-mamba-cache-size 40` is what makes eight overlapping GDN states fit.

## Shared launch contract

Unchanged from the live container:

- `--quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass`
- `--page-size 64 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --mamba-ssm-dtype float32`
- `--chunked-prefill-size 4096 --max-total-tokens 524288 --mem-fraction-static 0.95`
- `--ple-offload-embedding`
- `--prefill-attention-backend triton --decode-attention-backend trtllm_mha`
- `--disable-prefill-cuda-graph --disable-flashinfer-autotune`
- `--enable-gdn-replayssm-spec` + NEXTN 3/1/4 + unquant draft
- `--enable-auto-tool-choice` is **not** set; Ironclad sends tools on the request. Parser remains `qwen3_coder`.
- Docker `--gpus all --ipc=host --shm-size 16g --memory 116g --memory-swap 116g --init`
- Non-root container user (host uid:gid) plus `video` / `render` groups

## Smoke

`scripts/smoke.sh` checks `/health`, `/v1/models` (id must be `Qwen3.8-Flash-Next`), greedy `12*17` → `204` with thinking off, and a structured `list_dir` tool call. That is the same generate + tool-call round-trip Ironclad's orchestrator calibration exercises.

Verified against the live pin on 2026-09-07: `/health` 200, `/v1/models` reports `Qwen3.8-Flash-Next` with `max_model_len: 65536`.

## What this is not

- Not a quality leaderboard. No GSM8K / HumanEval / SWE / Terminal-Bench numbers here.
- Not vLLM, not llama.cpp, not a dual-Spark TP2 recipe.
- Not a claim that NEXTN output matches speculation-off byte for byte.
- Not a dual-instance recipe. Flash-Next fills this box; the 27B vLLM rollback cannot run alongside it.
- Not a re-publish of the upstream recipe's tok/s tables. Use that repo if you want their decode/agentic benches at 262k / 4 running.

## License

MIT for the scripts and notes in this tree. Vendored GB10 patches keep their upstream notices (see `NOTICE`). Model weights follow their own cards (Qwen / RadixArk) and are not stored here.
