# Capacity Estimate

## Goal
Estimate how many concurrent users a single instance of this deployment can serve,
and how many instances are needed for 100 concurrent users.

---

## 1. Model Memory Footprint

| Component                        | Calculation                                           | Size       |
|----------------------------------|-------------------------------------------------------|------------|
| Llama 3.2 3B parameters          | 3.21B params                                          | —          |
| fp16 weights (no 4-bit on M1)    | 3.21B × 2 bytes/param                                | ~6.40 GB   |
| LoRA adapter (rank=8, q+v proj)  | ~4M params × 4 bytes (fp32)                          | ~16 MB     |
| KV cache (per request, 512 tok)  | 2 × 28 layers × 32 heads × 128 dim × 512 × 2 bytes  | ~235 MB    |
| Activation / framework overhead  | empirical estimate                                    | ~500 MB    |
| **Total per instance**           |                                                       | **~7.2 GB**|

> M1 MacBook Air 8GB unified memory: one instance fits with ~800MB headroom.
> In production (Linux server), a 16GB RAM node comfortably hosts one instance.
> Note: 4-bit quantization (bitsandbytes) is CUDA-only and not available on M1.
> A quantized deployment on a CUDA server would reduce weights to ~1.6GB (~2.4GB total).

---

## 2. Inference Latency

| Metric                        | Estimated (M1 MPS)  | Measured (fill after Phase 4) |
|-------------------------------|---------------------|-------------------------------|
| Time to first token (TTFT)    | ~2–4 s              | TBD                           |
| Tokens per second             | ~10–20 tok/s        | TBD                           |
| Avg response length           | 200 tokens          | TBD                           |
| Total response time           | ~10–20 s            | TBD                           |

---

## 3. Requests Per Second (RPS) — Single Instance

```
RPS = 1 / avg_response_time_seconds
    = 1 / 15s
    = ~0.067 RPS
```

This is a **sequential baseline** (one request at a time, no batching).

With SSE streaming, the server holds the connection open for the full generation duration.
Each active generation occupies the model exclusively in this single-instance setup.

---

## 4. Concurrent Users — Single Instance

```
Concurrent users one instance can serve =
    RPS × avg_response_time = 0.067 × 15 = ~1 user at a time
```

With a small request queue (3–5 slots), practical concurrency ≈ **3–5 users** before
latency degrades beyond acceptable thresholds (>60s wait).

---

## 5. Instances Needed for 100 Concurrent Users

```
Instances = ceil(target_concurrent_users / users_per_instance)
           = ceil(100 / 4)
           = 25 instances
```

Each instance requires ~7.2 GB RAM (fp16, M1) or ~2.4 GB (4-bit, CUDA server).

**CUDA production fleet (4-bit quantized):**
- 25 instances × 2.4 GB = ~60 GB total RAM across fleet
- AWS g4dn.xlarge (16GB GPU RAM, ~$0.50/hr) → 6 instances/node
- Nodes needed: ceil(25 / 6) = **5 nodes**
- Estimated cost: ~$2.50/hr for 100 concurrent users

---

## 6. Storage Architecture

| Data Type      | Storage Location        | Reason                                                    |
|----------------|-------------------------|-----------------------------------------------------------|
| Model weights  | Local volume / S3       | Large (GBs), infrequently read, versioned by image tag    |
| Training logs  | MLflow tracking server  | Structured metrics, queryable, UI-accessible              |
| Request logs   | Append-only log files   | High write frequency, audit trail, separate lifecycle     |

Model weights are immutable once merged — stored once per version, never overwritten.
Training logs are small structured data — MLflow handles indexing and visualization.
Request logs are high-volume append-only writes — kept separate so they never
interfere with model serving I/O or inflate the MLflow database.

---

## 7. Real Numbers (fill after Phase 4 evaluation)

- Measured TTFT: ___ s
- Measured tok/s: ___
- Measured avg response time (200 tok): ___ s
- Revised RPS: ___
- Revised concurrent users / instance: ___
- Revised instances for 100 users: ___