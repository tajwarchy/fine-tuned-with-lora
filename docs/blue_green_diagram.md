# Blue-Green Deployment Architecture

## Concept

Blue-green deployment runs two identical production environments (blue = v1, green = v2).
Traffic is live on one at a time. The other is idle or staging.
Switching is instant — change one env var and restart the router.
Rollback is equally instant — switch back.

## Our Implementation

In this project, "traffic switch" = changing the MODEL_VERSION environment variable
in docker-compose.yml and running `docker-compose up -d`.
Each version is a separate, immutable Docker image tag.

---

## Diagram

```
                   ┌─────────────────────────────────────┐
                   │         docker-compose.yml          │
                   │      MODEL_VERSION=v1  (or v2)      │
                   └────────────────┬────────────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │        Nginx / Router        │
                     │   (or direct port mapping)   │
                     └──────┬───────────────┬───────┘
                            │               │
         ┌──────────────────▼──┐       ┌───▼──────────────────┐
         │   myapp:v1 (BLUE)   │       │   myapp:v2 (GREEN)   │
         │   port 8000         │       │   port 8001           │
         │   base model        │       │   fine-tuned model    │
         │   STATUS: LIVE ✅   │       │   STATUS: IDLE ⏸     │
         └─────────────────────┘       └──────────────────────┘
```

---

## Traffic Switch (v1 → v2)

```
Step 1: Build and verify v2 image locally
        docker build -t myapp:v2 .

Step 2: Update docker-compose.yml
        MODEL_VERSION=v2

Step 3: Bring up v2 alongside v1
        docker-compose up -d

Step 4: Smoke test v2 endpoint
        curl http://localhost:8001/health

Step 5: Switch live traffic to v2
        (update port mapping or router config)

Step 6: v1 container remains running for rollback window (15–30 min)
```

---

## Rollback Path (v2 → v1)

```
Step 1: MODEL_VERSION=v1 in docker-compose.yml
Step 2: docker-compose up -d
Step 3: v1 is live again — v2 stopped
Total rollback time: < 30 seconds
```

---

## Immutable Infrastructure Principle

Every Docker image is built once and never modified after the fact.
- `myapp:v1` always serves the base model. Always.
- `myapp:v2` always serves the fine-tuned model. Always.
- No `docker exec` to swap weights, no in-place file edits, no mutations.

**Why this matters:**
- Rollback is guaranteed — v1 image is identical to what was tested
- No "works on my machine" — the image is the environment
- Audit trail is clear — image tag = exact model version = exact code version

---

## SSE vs Regular REST

| Property              | REST (blocking)               | SSE (streaming)                        |
|-----------------------|-------------------------------|----------------------------------------|
| Connection lifecycle  | Open → response → close       | Open → stream tokens → close           |
| Server resource usage | Thread held for full gen time | Thread held + connection open per user |
| UX                    | User waits, then sees all     | User sees tokens as they arrive        |
| Timeout risk          | High for long generations     | Low — data flows continuously          |
| Infrastructure impact | Easier to scale horizontally  | Requires sticky connections or careful LB config |

SSE is strictly better for UX with LLMs because generation takes 10–30s.
Without streaming, the user stares at a blank screen. With SSE, they read along.
The tradeoff is that each active SSE connection holds a server thread open for the
full generation duration — this is why capacity estimation (concurrent connections
vs RPS) matters more for streaming LLM servers than for typical REST APIs.