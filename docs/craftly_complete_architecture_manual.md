# Craftly Complete Architecture Manual

This file is the single source of truth for the current Craftly system. A person
should be able to read this file and understand what exists, why it exists, when
it runs, how it works, and what to inspect after it runs.

Rule for future work: every new production feature must update this file with:

- what the feature does
- why it exists
- when it should run
- how it is triggered
- what files, APIs, tables, or artifacts it uses
- how to verify that it works

The old numbered phase architecture has been removed. Production code belongs
under `src/craftly`.

## Operating Roadmap

This roadmap is the default rule for future Craftly work:

- Scratch-origin model development. Craftly must not import external model
  weights, adapt third-party checkpoints, or add LoRA/QLoRA/adapter paths.
  After the 1B foundation proof passes, Craftly-owned weights may enter a
  separately governed full-parameter instruction, tool-use, or execution-
  grounded continuation stage with executable regression gates.
- Production-grade implementation only. New code must use explicit validation,
  fail-fast dependency checks, secure defaults, durable artifacts, and real
  tests. It must not claim readiness without evidence.
- Coding-agent performance comes first. Repository indexing, context packing,
  TaskGraph execution, patch generation, hardened tool execution, verifier
  loops, and benchmark feedback are higher priority than decorative
  architecture.
- Cybersecurity data and tooling stay governed. Allowed work includes approved
  sources, defensive analysis, authorized labs/CTFs/evaluation material,
  vulnerability detection, patch generation, and verification. Craftly must not
  add autonomous live-target attack workflows.
- Data quality comes before data scale. Source reputation, license/provenance,
  contamination filtering, deduplication, task extraction, human-review queues,
  and eval holdouts must happen before tokenizer/sharding/training.
- Production status must be evidence-based. Local smoke, Kaggle/Colab
  validation, and real cluster production are different statuses. Dependencies
  such as Redis, Postgres, S3/MinIO, Qdrant, Docker, CUDA, benchmark files, and
  scanner CLIs must be checked explicitly.
- Keep the architecture consolidated. Do not reintroduce phase explosion or many
  tiny wrapper files unless separation is required for security, testing,
  deployment, or clear ownership.

## Current Status

Craftly is now a consolidated agentic coding and defensive cybersecurity AI
architecture. It has:

- a FastAPI gateway
- JWT auth and quota middleware
- repository indexing and context packing
- durable TaskGraph execution
- concurrent role-separated workers and durable typed agent communication
- run-scoped immutable-evidence blackboard and bounded reflection
- failure signature clustering and verified repair linkage
- tool execution and Docker sandbox contracts
- patch preview/apply/rollback/verify flows
- defensive verification and benchmark harnesses
- scratch-only model foundation contracts
- tokenizer, token-sharding, and scratch pretraining loops
- production data platform for approved cyber/coding data collection
- Postgres/Redis/MinIO/Kubernetes deployment assets
- data quality, contamination, review, and benchmark feedback loops

Craftly is scratch-origin. Borrowed-model training and borrowed-model quality
baselines are not part of the architecture. The `mock` backend exists only as a
plumbing test double. A future full-parameter continuation stage may update
qualified Craftly-owned weights, but cannot use borrowed weights or adapters.

## Architecture Map

```text
User / Client
  |
  v
Gateway Layer
  |
  +--> Identity & Quota
  +--> Project / Session APIs
  +--> Repository Indexing
  +--> Context Builder
  +--> Agent Runtime
  +--> TaskGraph Runtime
  |      +--> Concurrent Worker Pool
  |      +--> Typed Agent Message History
  |      +--> Shared Blackboard
  |      +--> Negotiation / Reflection
  |      +--> Failure Intelligence
  +--> Tool / Sandbox Runtime
  +--> Patch Manager
  +--> Verifier
  +--> Evaluation
  +--> Model Foundation Status
  +--> Data Platform Status

Learning / Data Platform
  |
  +--> Approved Source Registry
  +--> Crawl Frontier: SQLite local, Postgres distributed
  +--> Async Workers
  +--> Quality Gate
  +--> Contamination Gate
  +--> Quality Inspection
  +--> Source Quality Scoring
  +--> Task Extraction
  +--> Automated/Human Review Queue
  +--> Benchmark Feedback
  +--> Dataset Version Manifest
  +--> Object Storage: local or S3/MinIO
  +--> Tokenizer + Token Shards
  +--> Scratch Pretraining Loop
```

## Source Layout

```text
src/craftly/
  agents/        Agent routing helpers.
  context/       Workspace context helpers.
  db/            SQLite local store, Postgres schema, migrations.
  evaluation/    Benchmarks, release gate, scorecard hooks.
  gateway/       FastAPI application and HTTP endpoints.
  guardrails/    Critic/security/verifier policy contracts.
  identity/      JWT auth and quota/rate limiting.
  indexing/      Repository indexer, AST facts, vector search, context builder.
  learning/      Data collection, quality, review, versioning, training data platform.
  memory/        Unified working/project/episodic/semantic/user/fix memory.
  model_ops/     Scratch model specs, tokenizer, shards, pretraining, GPU smoke.
  patches/       Patch preview/apply/rollback/verify.
  planning/      Intent and planning engine.
  runtime/       Agent runtime and durable TaskGraph.
  shared/        Strict schemas and shared contracts.
  tools/         Local command runner and Docker sandbox runner.
  verifier/      Verification runtime and secret scan.
deploy/
  dev/           Local development compose.
  prod/          Production compose with Postgres, Redis, MinIO, crawler workers.
  k8s/           Kubernetes API, Postgres, Redis, MinIO, workers, HPA, policies.
  gpu/           GPU smoke/pretraining profile scripts.
config/
  data_sources.ultimate.json
scripts/
  runtime checks, Docker repair, security tools install, data platform helpers.
tests/
  release-gated unit and smoke tests.
```

## Gateway Layer

Files:

- `src/craftly/gateway/api.py`

Purpose:

The gateway is the single HTTP entrypoint. It exposes health, metrics, auth,
model status, data-platform status, project/session APIs, indexing APIs,
TaskGraph APIs, tool execution, sandbox execution, patch operations, verifier
operations, and agent run operations.

When it runs:

- during local API development
- in Docker Compose production
- in Kubernetes deployment

How to run:

```powershell
python -m uvicorn src.craftly.gateway.api:app --host 127.0.0.1 --port 8090
```

Important endpoints:

- `GET /health/ready`: readiness status
- `GET /metrics`: Prometheus-style metrics
- `GET /v1/auth/status`: auth settings status
- `POST /v1/auth/token`: token issue endpoint, gated in production
- `POST /v1/projects`: register a repository
- `POST /v1/projects/{project_id}/index`: index repository files
- `GET /v1/projects/{project_id}/symbols`: inspect extracted symbols
- `POST /v1/context/build`: build ranked context pack
- `POST /v1/context/vector-search`: vector search with backend selector
- `GET /v1/context/vector-capabilities`: vector backend readiness
- `POST /v1/memory/ingest`: ingest verified memory
- `POST /v1/memory/retrieve`: retrieve ranked memory
- `POST /v1/agent/runs`: create a durable agent run
- `GET /v1/taskgraphs/{task_graph_id}`: inspect TaskGraph
- `POST /v1/taskgraphs/{task_graph_id}/advance`: advance next ready task
- `POST /v1/tools/execute`: bounded command execution
- `POST /v1/sandbox/run`: Docker sandbox run
- `POST /v1/patches/preview`: preview file changes
- `POST /v1/patches/verify`: preview/apply/verify/rollback loop
- `POST /v1/verifier/run`: run verifier checks
- `GET /v1/model/foundation/status`: scratch model foundation status
- `GET /v1/data/platform/status`: latest local dataset version/dashboard status

Why it exists:

The rest of the system should not expose random scripts as product APIs. The
gateway gives one stable surface for UI, CLI, tests, and future services.

How to verify:

```powershell
python -m src.craftly.evaluation.release_gate
```

## Identity And Quota

Files:

- `src/craftly/identity/auth.py`
- `src/craftly/identity/quota.py`

Purpose:

Protect API routes with JWT auth and rate-limit users with a regenerative token
bucket. Redis is used when `CRAFTLY_REDIS_URL` is set. Local in-memory quota is
development-only fallback.

When it runs:

- every protected API request
- before gateway handlers execute

Important environment:

```bash
CRAFTLY_AUTH_ENABLED=1
CRAFTLY_JWT_SECRET=<long-random-secret>
CRAFTLY_ALLOW_TOKEN_ISSUE=0
CRAFTLY_TOKEN_ISSUE_KEY=<only-if-token-issue-enabled>
CRAFTLY_QUOTA_ENABLED=1
CRAFTLY_REDIS_URL=redis://redis:6379/0
```

Why it exists:

Without auth, anyone can call the AI backend. Without quota, one user can
exhaust the system. Token issuance is deliberately blocked in production unless
explicitly enabled.

How to verify:

- `tests/test_craftly_production_hardening.py`
- `GET /v1/auth/status`

## Observability

Files:

- `src/craftly/observability.py`

Purpose:

Provide structured JSON request logs and Prometheus-style metrics. Every request
records method, path, status, duration, and user id when available.

When it runs:

- middleware around every gateway request

How to inspect:

```powershell
Invoke-RestMethod http://127.0.0.1:8090/metrics
```

Why it exists:

For production, the system must show live health and request behavior. Debugging
large agent/data jobs without metrics is too slow.

## Repository Intelligence

Files:

- `src/craftly/indexing/repository_indexer.py`
- `src/craftly/indexing/context_builder.py`
- `src/craftly/indexing/vector_index.py`

Purpose:

Turn a repository into searchable, structured context. The indexer walks files,
chunks code, extracts Python AST facts, extracts import/dependency hints for
multiple languages, and stores chunks in the local store.

What it extracts:

- file path
- language
- content hash
- chunk boundaries
- token estimate
- Python functions/classes
- signatures
- imports
- calls
- decorators
- docstrings
- state mutations
- generic dependency hints for JS/TS/Go/Rust/Java/C/C++/Bash

When it runs:

- after a project is created
- before context building or agent work

Command path:

```powershell
python -m uvicorn src.craftly.gateway.api:app --host 127.0.0.1 --port 8090
```

Then call:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8090/v1/projects/<project_id>/index
Invoke-RestMethod http://127.0.0.1:8090/v1/projects/<project_id>/symbols
```

Why it exists:

Agentic coding quality depends heavily on repo understanding. The model should
not rely only on raw file text; it needs ranked, structured code context.

## Context Builder

Files:

- `src/craftly/indexing/context_builder.py`
- `src/craftly/indexing/vector_index.py`
- `src/craftly/indexing/repository_indexer.py`

Purpose:

`HybridRAGEngine` in `context_builder.py` is the only retrieval orchestration
authority. It combines lexical/symbol candidates, Qdrant semantic candidates,
AST/call/dependency expansion and verified memory. Reciprocal Rank Fusion uses
`k=60`; MMR uses `lambda=0.75`; the final pack is evidence-bound and token
budgeted. `ContextBuilder` is only a compatibility name for this same class,
not a second implementation.

When it runs:

- before agent planning
- when the user asks a repo-specific coding/debugging/security question

API:

- `POST /v1/context/build`
- `POST /v1/context/vector-search` (deprecated compatibility route)
- `GET /v1/context/vector-capabilities`

Vector backend details:

- `local_hashing`: built-in deterministic hashed embedding backend. It scans
  indexed chunks exactly and is the default for local development, tests, and
  small/medium repositories.
- `faiss`: explicit FAISS adapter contract. Use when `faiss` is installed and a
  large local ANN sidecar index is desired.
- `hnsw`: explicit HNSW adapter contract. Use when `hnswlib` is installed and a
  fast local approximate index is desired.
- `qdrant`: the only production semantic backend. It requires server-owned
  Qdrant/embedding configuration and a hash-bound scratch embedding manifest.
- `pgvector`: Postgres-native vector search contract. Requires
  `CRAFTLY_DATABASE_URL` or `postgres_dsn`.

Backend policy:

- local/dev/smoke: `local_hashing`
- production: Qdrant only, with mandatory organization/project/revision filters
- degraded production: explicit lexical plus graph retrieval
- `pgvector`, FAISS and HNSW are capability contracts and do not qualify as
  production semantic retrieval in the current release

The API rejects unavailable production backends with explicit configuration or
dependency errors instead of silently pretending they are active.

Why it exists:

Long context is expensive. The system must choose the most relevant repo
context instead of dumping all files.

## Planning And TaskGraph Runtime

Files:

- `src/craftly/planning/engine.py`
- `src/craftly/runtime/taskgraph.py`
- `src/craftly/runtime/engine.py`
- `src/craftly/runtime/collaboration.py`

Purpose:

Create and execute durable task graphs for agentic work. The graph is a DAG,
not a decorative linear checklist:

```text
understand -> planner -> retrieve_context -> edit
                                           |-> test -> critic_review --|
                                           |-> security_review --------|-> verify -> summarize
                                           `-> performance_review -----|
```

`AgentWorkerPool` atomically claims every dependency-ready node up to the
configured concurrency limit. A task claim has a worker ID and expiring lease.
Long-running handlers renew the lease in the background. A crashed worker's
expired lease consumes one retry attempt; the task is requeued only while its
bounded retry budget remains. Each handler has an absolute timeout. Cancellation
marks all queued/running nodes cancelled and stops in-flight asyncio handlers.

Role ownership is fixed:

| Task | Role | Responsibility boundary |
|---|---|---|
| understand, planner, retrieve_context | architect | requirements, plan, context; no executable patch |
| edit | coder | code or patch proposal |
| test | tester | measured execution evidence |
| security_review | security reviewer | defensive risk challenge and review |
| critic_review, performance_review | critic | flaws and confidence; no final solution |
| verify | verifier | success-criteria decision only |
| summarize | orchestrator | evidence-backed response assembly |

Registering a handler under the wrong role fails before execution.

### Typed Agent Communication

All packets are Pydantic-validated and persisted in `agent_messages`. The only
message kinds are:

- `proposal`
- `evidence`
- `challenge`
- `review`
- `decision`

Packets carry run, graph, task, correlation, sender/recipient role, bounded JSON
payload, evidence references, and timestamp. Payloads are capped at 64 KiB,
common secret keys/values are redacted, duplicate evidence references are
rejected, and sender role capabilities are validated. Architects cannot publish
patch artifacts, critics cannot issue decisions, and only verifiers can publish
verification decisions.

### Shared Blackboard

`blackboard_entries` is temporary run-scoped coordination state. It does not
duplicate Unified Memory.

- kinds: fact, artifact, decision, question, evidence
- evidence is immutable after insertion
- mutable updates require `expected_version`
- conflicting writes fail with a version conflict
- every entry can reference its source message
- only accepted verifier decisions may be promoted to Unified Memory

Working drafts, raw thoughts, failed guesses, and unverified output remain in
run history and are never promoted as long-term memory.

### Negotiation And Reflection

The strict flow is:

```text
worker proposal
  -> peer challenge/review
  -> evidence response
  -> critic confidence and flaw report
  -> verifier decision
  -> revise when rejected or confidence < 0.85
```

Revision always checks:

1. What assumptions are wrong?
2. What can fail?
3. What security risks exist?
4. What was not verified?

There are at most three revision cycles. Acceptance requires peer acceptance,
critic confidence at or above the configured threshold, verifier acceptance,
and at least one verification evidence reference. This prevents infinite
reasoning loops and evidence-free confidence claims.

### Verified Repository Execution

`src/craftly/runtime/execution.py` is the single orchestration boundary that
turns the collaboration primitives into a real coding-agent run. It does not
replace TaskGraph, patch, tool, scanner, indexing, or verifier ownership.

The worker sequence is:

```text
Architect
  validates a structured plan; executable code is forbidden
Coder
  returns validated complete-file edits and candidate test argv
Stage transaction
  copies the bounded repository without links/generated directories
  previews the same patch for the original and applies it only to the stage
Tester || Security Reviewer || Performance Reviewer
  run concurrently after the patch exists
Critic
  receives all three evidence sets and returns confidence plus concrete flaws
Verifier
  checks objective criteria without generating a solution
Orchestrator
  returns a typed evidence-backed summary
```

Every model-produced control artifact is JSON parsed into a strict Pydantic
schema. Markdown-wrapped JSON, missing fields, contradictory final status,
duplicate patch paths, path escapes, oversized patches, and role mixing fail
the task. Raw private chain-of-thought is not requested or persisted.

The original repository mutation invariant is:

```text
copy original -> apply candidate to stage -> execute evidence gates
  -> accepted: atomic apply to original
  -> rejected: original preview record becomes rolled_back
```

Atomic patch writes use a same-directory temporary file, `fsync`, and
`os.replace`. If a multi-file apply raises, already-written files are restored.
Rollback restores existing files and removes files created by the rejected
patch, including empty parent directories. Apply and rollback are idempotent
for their terminal states.

Strict execution performs dependency preflight before calling the model:

- private non-mock Craftly scratch serving backend;
- reachable Docker engine;
- Semgrep and/or CodeQL when configured as required;
- allowlisted sandbox image;
- no host execution fallback.

Tests run in a network-disabled, read-only-root container with one CPU, 512 MiB
RAM, dropped capabilities, unprivileged UID/GID, and bounded tmpfs workspace.
The security reviewer compares finding fingerprints from the original baseline
with the staged patch and rejects newly introduced findings. A missing required
scanner is evidence of failure, not a pass.

On rejection, critic flaws, incorrect assumptions, failure modes, security
risks, missing evidence, scanner issues, and verifier failures become the next
revision context. The initial attempt plus at most three revisions are allowed.
Each revision is a distinct durable run and TaskGraph. Repeated normalized
failures enter Failure Intelligence; only a later verified patch can link and
promote the resolved experience.

Gateway API:

```text
POST /v1/agent/execute
scope: agents:execute
```

The synchronous response contains every attempt, test evidence, scanner
evidence, critic review, verifier decision, final patch status, confidence, and
whether the original workspace was changed. Metrics include execution count,
duration, and revision attempts.

### Repository Agent Scorecard

`src/craftly/evaluation/agent_scorecard.py` runs the same Gateway execution
service against isolated copies of approved real repositories. It never runs a
weaker scoring-only implementation.

Each JSONL task supplies:

```json
{
  "task_id": "auth-debug-001",
  "repository": "auth-debug-001",
  "prompt": "fix login",
  "category": "debugging",
  "verification_commands": [["python", "-m", "pytest", "-q"]],
  "expected_changed_files": ["src/auth.py"],
  "required_substrings": {"src/auth.py": ["compare_digest"]},
  "forbidden_substrings": {"src/auth.py": ["== supplied_token"]},
  "short_prompt": true,
  "run_semgrep": true,
  "run_codeql": false
}
```

Strict suite rules:

- 50-100 unique tasks;
- at least 10 each for coding, debugging, security, patch, and long-context;
- at least 10 short-prompt tasks;
- every task has test commands, expected files, and content assertions;
- every task enables Semgrep or CodeQL;
- repository paths stay under `CRAFTLY_SCORECARD_REPO_ROOTS`;
- a non-mock Craftly scratch backend is mandatory.

Tasks run with bounded concurrency. The runner hashes every source repository
before and after execution, stages a separate worktree, and writes each complete
execution report under `tasks/<task_id>/execution_report.json`. Aggregate JSON
and Markdown expose:

- architecture reliability;
- workflow completion;
- security detection/fix score;
- short-prompt understanding;
- sandbox/test pass rate;
- regression count;
- confidence and objective score per task.

Production pass requires at least 95% source/reliability integrity, at least 80%
workflow, security, short-prompt and sandbox/test rates, and zero accepted
regressions. `CRAFTLY_AGENT_SCORECARD_REPORT` is consumed by production
readiness. Without a measured strict 50-100 task report the subsystem remains
`built_not_cluster_proven` in development and blocks production.

```powershell
$env:CRAFTLY_SCORECARD_REPO_ROOTS = "D:\approved-agent-eval-repos"
python -m src.craftly.evaluation.agent_scorecard `
  --tasks D:\approved-agent-eval-repos\tasks.jsonl `
  --repository-root D:\approved-agent-eval-repos `
  --output-dir artifacts\craftly\agent-scorecard `
  --policy-mode strict --concurrency 4
```

### Failure Intelligence

Runtime exceptions and timeouts are secret-redacted and normalized by replacing
volatile paths, addresses, UUIDs, locations, and large numbers. A SHA-256
cluster key groups equivalent failures. Each cluster tracks occurrence count,
last seen time, task kind, root cause, patch, and verification reference.

A failure becomes a `pending_review` scratch-training dataset candidate only
when:

- it repeated at least the configured threshold;
- a real patch record is linked;
- root cause is recorded;
- verification passed.

Unresolved failures and unverified patches cannot enter the learning candidate
path.

### Postgres Production Proof

Migration `0005_agent_collaboration.sql` owns task leases, messages, blackboard
entries, and failure records. The live proof command creates a namespaced
project/run, races two Postgres claimers with `FOR UPDATE SKIP LOCKED`, verifies
there is exactly one winner, verifies stale blackboard CAS rejection, reads a
durable message back, and removes all proof rows through project cascade.

```powershell
python -m src.craftly.runtime.collaboration --postgres-proof `
  --database-url "$env:CRAFTLY_DATABASE_URL" `
  --output-dir artifacts\craftly\agent-collaboration-proof
```

The production readiness gate requires the measured report. Merely setting a
Postgres URL is insufficient.

When it runs:

- when an agent run is created
- when the client advances tasks
- when a task is completed or failed

APIs:

- `POST /v1/agent/runs`
- `GET /v1/taskgraphs/{task_graph_id}`
- `POST /v1/taskgraphs/{task_graph_id}/advance`
- `POST /v1/taskgraphs/{task_graph_id}/cancel`
- `POST /v1/tasks/{task_id}/complete`
- `POST /v1/tasks/{task_id}/fail`
- `POST /v1/agent/messages`
- `GET /v1/agent/runs/{run_id}/messages`
- `PUT /v1/agent/blackboard`
- `GET /v1/agent/runs/{run_id}/blackboard`
- `GET /v1/projects/{project_id}/failure-clusters`

Why it exists:

Agent work must be durable and inspectable. A hidden single prompt loop is hard
to debug. TaskGraph state shows what failed: intent understanding, planning,
context retrieval, editing, testing, critic review, security review,
performance review, verification, or summary.

## Tool Runtime And Sandbox

Files:

- `src/craftly/tools/`

Purpose:

Run bounded commands and sandboxed code execution. The Docker sandbox contract
uses hardened defaults such as no network, memory cap, read-only root, tmpfs,
and dropped capabilities when Docker is available.

When it runs:

- for tests
- for compile commands
- for patch verification
- for defensive sandbox checks

APIs:

- `POST /v1/tools/execute`
- `POST /v1/sandbox/run`

Why it exists:

Coding agents need tools. They should not execute arbitrary commands without
limits.

## Patch Manager

Files:

- `src/craftly/patches/service.py`

Purpose:

Preview, apply, verify, and rollback file edits. It protects project root
boundaries and rejects writes into `.git`.

Patch acceptance standard:

- preview before apply
- path must stay inside project root
- `.git` writes are blocked
- patch applies cleanly
- verification commands pass
- secret scan is clean
- rollback remains possible

APIs:

- `POST /v1/patches/preview`
- `POST /v1/patches/verify`
- `POST /v1/patches/{patch_id}/apply`
- `POST /v1/patches/{patch_id}/rollback`

Why it exists:

Coding AI should not blindly overwrite a repository. Every edit needs a
reversible lifecycle.

## Verifier And Guardrails

Files:

- `src/craftly/verifier/runtime.py`
- `src/craftly/guardrails/service.py`

Purpose:

Verifier runs configured commands and secret scans. Guardrails provide simple
critic/security policy contracts for defensive review.

When it runs:

- after patch preview/apply
- before patch acceptance
- during release tests

Why it exists:

The model can be wrong. Verification is the systemâ€™s practical truth source.

## Evaluation Service

Files:

- `src/craftly/evaluation/benchmarks.py`
- `src/craftly/evaluation/release_gate.py`
- `src/craftly/evaluation/service.py`

Purpose:

Run local benchmark and smoke tests. The release gate is the native "can we
ship this code?" check.

Command:

`python -m src.craftly.learning.ablation_runner --help` presents the
authoritative scientific command surface. The older flag-only data-mix command
remains compatible when invoked with `--inputs` or `--base-run-dir`.

```powershell
python -m src.craftly.evaluation.release_gate
```

Current release gate covers:

- gateway flows
- project/session/index/context APIs
- TaskGraph lifecycle
- patch verify/rollback
- security benchmark harness
- production hardening checks
- data platform pipeline
- scratch decoder smoke paths

Why it exists:

Without a release gate, architecture quality is only a claim. The gate makes it
measurable.

## Database Layer

Files:

- `src/craftly/db/local_store.py`
- `src/craftly/db/schema.sql`
- `src/craftly/db/migrations/`
- `src/craftly/db/migration_runner.py`

Purpose:

Local SQLite is used for development. Postgres schema and migrations define the
production contract.

Main application tables:

- projects
- sessions
- runs
- task_graphs
- tasks
- workspace_files
- code_chunks
- patches
- evaluations
- memory_entries
- learning_candidates

Data platform tables:

- data_sources
- dataset_versions
- data_quality_events

When it runs:

- local store runs during local API/tests
- migrations run during production bootstrap

Command:

```powershell
python -m src.craftly.db.migration_runner --database-url $env:CRAFTLY_DATABASE_URL
```

## Unified Memory

Files:

- `src/craftly/memory/system.py`

Purpose:

Store and retrieve typed, ranked memory without polluting future context. The
memory manager supports six layers:

- `working`: current session/task context only; in-process TTL.
- `project`: repository facts, architecture decisions, module paths, stack.
- `episodic`: solved workflow traces and successful run outcomes.
- `semantic`: durable technical concepts and reusable knowledge.
- `user`: durable user preferences and operating constraints.
- `verified_fix`: failure -> fix -> verified outcome records.

Anti-pollution policy:

- allowed: verified fixes, passed benchmarks, security findings, successful
  plans, project facts, user preferences
- rejected: raw thoughts, failed guesses, transient outputs

Retrieval ranking:

```text
Final Score =
  0.4 * vector_similarity
  + 0.3 * success_rate
  + 0.2 * recency_weight
  + 0.1 * usage_count_weight
```

This formula is implemented in `memory_rank_score`.

When it runs:

- after a bug is fixed and verified
- after a benchmark or security finding is confirmed
- when project facts or user preferences should persist
- before similar future tasks

APIs:

- `POST /v1/memory/ingest`
- `POST /v1/memory/retrieve`

Why it exists:

Repeated failures are expensive, but bad memory is worse than no memory. The
layered manager preserves useful evidence while preventing context pollution
from guesses and transient outputs.

## Model Foundation And Scratch Training

Files:

- `src/craftly/model_ops/foundation.py`
- `src/craftly/model_ops/torch_decoder.py`
- `src/craftly/model_ops/tokenizer_pipeline.py`
- `src/craftly/model_ops/data_loader.py`
- `src/craftly/model_ops/pretrain_loop.py`
- `deploy/gpu/`

Purpose:

Define Craftly-owned scratch model contracts and executable training primitives.
The system supports architecture planning for 7B, 32B, 70B, and 100B-class
decoder models, but local tests use tiny smoke configs.

What exists:

- scratch decoder architecture specs
- parameter estimates
- pretraining readiness contract
- tokenizer contract
- BPE tokenizer training
- token shard creation
- streaming dataloader
- checkpoint-resumable pretraining loop
- gradient accumulation
- mixed precision on CUDA
- validation loss support
- checkpoint manifests with hashes
- GPU smoke scripts

Commands:

```bash
pip install -r requirements-linux-gpu.txt
python deploy/gpu/run_scratch_gpu_smoke.py --device cuda --steps 2 --sequence-length 64
python -m src.craftly.model_ops.tokenizer_pipeline --input data/training/clean.jsonl --tokenizer-out artifacts/craftly/tokenizer/tokenizer.json --shards-out artifacts/craftly/shards --vocab-size 128000 --sequence-length 128
python -m src.craftly.model_ops.pretrain_loop --device cuda --manifest artifacts/craftly/shards/manifest.json --steps 100 --batch-size 2 --sequence-length 128 --gradient-accumulation-steps 4 --dtype bf16
python deploy/gpu/run_pretraining_pipeline.py --input data/training/clean.jsonl --device cuda --steps 100 --sequence-length 128
```

Why it exists:

The user wants scratch training, not borrowed external checkpoints. These
modules prepare that path without pretending a laptop can train a large model.

## Data Platform Overview

The data platform is the most important current subsystem for future model
quality. It exists to collect, filter, review, version, and prepare approved
coding/security data.

Full flow:

```text
Approved source registry
  -> production readiness check
  -> capacity planning
  -> crawl frontier
  -> async workers
  -> raw JSONL shards
  -> quality gate
  -> clean JSONL shards
  -> contamination gate
  -> quality inspection
  -> source quality scoring
  -> task extraction
  -> automated policy screen
  -> independent human review and adjudication
  -> Dataset Trust Authority promotion
  -> promoted task JSONL
  -> benchmark/data feedback
  -> tokenizer training
  -> token shard creation
  -> dataset version manifest
  -> object storage upload
  -> dashboard
```

Safety boundary:

The data platform is for approved public sources, licensed repositories,
defensive security references, documentation, benchmark corpora, and approved
mirrors. It does not run exploits or collect unauthorized targets.

## Approved Source Registry

Files:

- `src/craftly/learning/source_registry.py`
- `config/data_sources.ultimate.json`

Purpose:

Define which sources may be crawled. Every source declares:

- name
- seed URLs
- allowed domains
- license
- category

When it runs:

- before any crawl
- before run-plan generation
- before production readiness checks

What it blocks:

- URL outside allowed domain
- unsupported URL scheme
- empty domain allowlist
- unapproved license warnings
- duplicate seed warnings

Commands:

```bash
python -m src.craftly.learning.source_registry --sources config/data_sources.ultimate.json
```

Why it exists:

Best data starts with best sources. Crawling random internet pages creates
noise, legal risk, and model contamination.

Canonical registry structure:

- `sources`: crawl-ready, license-gated sources used directly by the crawler.
- `vulnerability_adapters`: official API-backed vulnerability feeds consumed by
  adapter code, not by the web crawler.
- `review_required_sources`: valuable sources that must be license-approved
  before being promoted into `sources`.

The ultimate registry currently contains 17 crawl-ready sources, 76 seed URLs,
and covers:

- OWASP Cheat Sheets, Top 10, ASVS, WSTG, API Security
- CISA KEV catalog
- NIST SSDF and cryptographic standards
- Python, Rust, Go, Node.js, TypeScript
- FastAPI, Django, pytest, PostgreSQL
- Docker, Kubernetes, Git, OpenSSF Scorecard

High-value sources kept in `review_required_sources` include MITRE CWE/CAPEC,
GitHub CodeQL docs, Semgrep docs/rules, RustSec, Go vulnerability database,
PyPA advisory database, and real permissive-license security patch repositories.

Use the ultimate registry for every local, Kaggle, Colab, and production data
run. Keep source balancing enabled, because large documentation sources can
otherwise dominate training rows.

## Data Source Governance

Files:

- `src/craftly/learning/governance.py`
- `src/craftly/learning/resource_catalog.py`
- `config/data_sources.ultimate.json`

Purpose:

Provide an auditable legal/source approval workflow before high-value but
license-sensitive sources become training data. The ultimate registry separates
three categories:

- `sources`: directly crawlable and license-gated.
- `vulnerability_adapters`: official API-backed vulnerability feeds.
- `review_required_sources`: valuable sources that need approval first.
- `training_resources`: 45 external cybersecurity, security-evaluation, and
  agentic-coding resources supplied by the project owner.
- `training_priority_groups`: top six priority groups used to order serious
  data work.

What it stores:

- source approval requests
- approval/rejection decisions
- human-review queue items
- reviewer decisions and reasons

Commands:

```bash
python -m src.craftly.learning.governance --store artifacts/craftly/governance report
python -m src.craftly.learning.governance --store artifacts/craftly/governance submit-source --source-name portswigger-web-security-academy --category authorized_security_testing_labs --url https://portswigger.net/web-security --license review-required --evidence-url https://portswigger.net/web-security --requested-by security-team --justification "High-value authorized web security education source"
python -m src.craftly.learning.resource_catalog --catalog config/data_sources.ultimate.json --output artifacts/craftly/resource_catalog_report.json
```

Why it exists:

Cybersecurity data can be high-value and legally sensitive at the same time.
This module prevents accidental ingestion of unclear sources while still giving
the project a path to approve excellent security-testing education, advisory
databases, and real patch repositories.

## Vulnerability Database Adapters

Files:

- `src/craftly/learning/vulnerability_adapters.py`

Supported adapters:

- CISA KEV
- NVD CVE 2.0
- OSV
- Go vulnerability database
- GitHub Advisory Database

What they output:

Normalized defensive JSONL records with source, vulnerability ID, summary,
details, affected packages, CWE IDs, references, severity, license, provenance,
content hash, and text.

Commands:

```bash
python -m src.craftly.learning.vulnerability_adapters --adapter cisa-kev --output artifacts/craftly/vulns/cisa-kev.jsonl --max-records 100
python -m src.craftly.learning.vulnerability_adapters --adapter nvd-cve --output artifacts/craftly/vulns/nvd.jsonl --max-records 100
python -m src.craftly.learning.vulnerability_adapters --adapter go-vuln --output artifacts/craftly/vulns/go.jsonl --max-records 100
```

Why it exists:

Official vulnerability feeds produce cleaner cybersecurity training data than
random scraping because they include structured IDs, references, affected
packages, timestamps, and provenance.

## Crawl Frontier And Data Engine

Files:

- `src/craftly/learning/data_engine.py`
- `src/craftly/learning/supervisor.py`

Purpose:

Perform large allowlisted crawls with persistent state.

Local mode:

- SQLite frontier
- good for laptop/dev/small runs

Production mode:

- Postgres frontier
- row locks with `FOR UPDATE SKIP LOCKED`
- many workers can claim URLs without duplicate work

What it does:

- seeds URLs from registry
- claims queued URLs
- respects robots rules
- enforces per-domain throttling
- fetches pages
- extracts text from HTML
- writes raw JSONL shards
- evaluates quality
- writes clean JSONL shards
- records provenance
- deduplicates by content hash
- discovers new links inside allowed domains
- retries failed URLs
- tracks done/failed/queued states

Commands:

```bash
python -m src.craftly.learning.data_engine --sources config/data_sources.ultimate.json --frontier artifacts/craftly/data-engine/frontier.sqlite3 --raw-output-dir artifacts/craftly/data-engine/raw --clean-output-dir artifacts/craftly/data-engine/clean --max-docs 1000 --workers 8 --max-depth 1
python -m src.craftly.learning.data_engine --sources config/data_sources.ultimate.json --frontier-backend postgres --postgres-dsn "$CRAFTLY_DATABASE_URL" --raw-output-dir artifacts/craftly/data-engine/raw --clean-output-dir artifacts/craftly/data-engine/clean --max-docs 1000000 --workers 64
python -m src.craftly.learning.supervisor --sources config/data_sources.ultimate.json --postgres-dsn "$CRAFTLY_DATABASE_URL" --raw-output-dir artifacts/craftly/data-engine/raw --clean-output-dir artifacts/craftly/data-engine/clean --object-store-uri s3://craftly-datasets/pretraining --worker-replicas 8 --async-workers 64
```

The supervisor repeatedly launches bounded crawl cycles, writes heartbeat and
status JSON files, applies readiness checks before starting, and stops after too
many failures. In production it runs beside distributed workers and object
storage, not as a local-only script.

Why it exists:

Million-scale data requires resume, retry, dedup, and distributed URL claiming.
A simple script cannot safely collect large corpora.

## Quality Gate

Files:

- `src/craftly/learning/quality.py`

Purpose:

Reject bad rows and add metadata to accepted rows.

Checks:

- minimum text length
- maximum text length
- allowed license
- secret-like content
- email-like PII
- duplicate content

Adds:

- labels such as `defensive_security`, `code`, `tests`
- quality score
- language hint
- data type
- content hash

When it runs:

- inside the data engine
- during standalone JSONL filtering

Why it exists:

Low-quality data trains low-quality behavior. The quality gate is the first
hard filter.

## Contamination Gate

Files:

- `src/craftly/learning/benchmark_contamination_filter.py`

Purpose:

Block benchmark and holdout leakage before tokenizer/shard creation.

Default patterns include:

- HumanEval
- MBPP
- SWE-bench
- CyberSecEval
- common benchmark marker strings

When it runs:

- after clean shards are written
- before tokenizer training
- before dataset version promotion

Why it exists:

If benchmark prompts leak into training, evaluation becomes fake.

## Quality Inspection

Files:

- `src/craftly/learning/quality_inspector.py`

Purpose:

Summarize clean data quality after filtering.

Reports:

- row count
- average/min/max quality score
- distribution by label
- distribution by language
- distribution by data type
- distribution by license
- distribution by source

When it runs:

- inside `data_pipeline`
- manually after a dataset run

Command:

```bash
python -m src.craftly.learning.quality_inspector --input artifacts/craftly/data-pipeline/clean/clean-000000.jsonl --output artifacts/craftly/data-pipeline/reports/quality_report.json
```

Why it exists:

After crawling, you need to know what you actually collected. Counts alone are
not enough.

## Source Quality Scoring

Files:

- `src/craftly/learning/quality.py`
- `src/craftly/learning/quality_inspector.py`
- `src/craftly/learning/source_quality.py`

Purpose:

Score each source based on accepted rows, quality score, code coverage, and
defensive security coverage.

The row-level quality classifier now scores more than length. It records:

- `component_scores.length`
- `component_scores.security_signal`
- `component_scores.agentic_signal`
- `component_scores.code_signal`
- `component_scores.test_signal`
- `component_scores.structure`
- `component_scores.low_noise`
- `component_scores.lexical_diversity`
- `risk_flags`
- inferred `language_hint`
- inferred `data_type`

Supported language and artifact signals include Python, Rust, Go, JavaScript,
TypeScript, Java, C/C++, Bash, Solidity, Docker/Kubernetes/config material,
patches, tests, debug traces, CVE/CWE references, and defensive security
documentation.

Hard rejects are reserved for unsafe or unusable rows: too short, too large,
disallowed license, secret-like content, email-like PII, duplicate content, very
low text signal, or extremely degenerate repetition. Weaker signals such as
boilerplate or low lexical diversity become risk flags so useful technical
references are not thrown away too aggressively.

Actions:

- `promote`: source is strong
- `watch`: source is usable but needs monitoring
- `demote`: source is noisy or low value

When it runs:

- inside `data_pipeline`
- after quality inspection

Why it exists:

Large crawls should improve over time. Good sources should get more crawl
budget; noisy sources should lose budget.

## Task Extraction

Files:

- `src/craftly/learning/task_extraction.py`

Purpose:

Convert clean corpus rows into task candidates for agentic coding and defensive
security training/evaluation.

Task types:

- `security_vulnerability_identification`
- `security_patch_generation`
- `secure_code_review`
- `regression_test_generation`
- `debugging_from_error_trace`
- `implementation_planning`

How it works:

- extracts fenced code blocks and diff blocks when present
- detects vulnerability categories such as SQL injection, XSS, SSRF, command
  injection, deserialization, weak crypto, path traversal, hardcoded secrets,
  buffer overflow, and vulnerability taxonomy references
- converts runtime traces and compile errors into debugging tasks
- converts code artifacts into secure code-review tasks
- converts test-heavy rows into regression-test generation tasks
- builds prompts that preserve source URL/provenance
- attaches `success_criteria` and `negative_constraints` to each task
- deduplicates task prompts
- writes JSONL task candidates

When it runs:

- after contamination and quality inspection

Output:

- `tasks/tasks.jsonl`

Why it exists:

Raw documents are not enough for agentic learning. The model needs tasks,
prompts, context, and verifiable work patterns.

## Automated Task Policy Screen

Files:

- `src/craftly/learning/review.py`

Purpose:

Screen extracted task candidates before independent human review. This module
does not approve training data and cannot promote a dataset version.

Outputs:

- review decisions JSONL
- automated-pass candidate JSONL
- persisted review report JSON

Decision statuses:

- `automated_pass`
- `needs_human_review`
- `rejected`

Automated policy checks:

- prompt length
- defensive/safe wording
- source URL present
- language present
- useful task type
- high-risk action terms

Command:

```bash
python -m src.craftly.learning.review --input artifacts/craftly/data-pipeline/tasks/tasks.jsonl --decisions-out artifacts/craftly/data-pipeline/reports/task_review_decisions.jsonl --automated-pass-out artifacts/craftly/data-pipeline/tasks/automated_pass_tasks.jsonl --report-out artifacts/craftly/data-pipeline/reports/task_review_report.json
```

Why it exists:

Task extraction can create noisy prompts. At most three ranked tasks are emitted
per source row by default. Security categories are evidence-weighted, capped at
three, and do not treat one navigation-menu mention as category proof.
Automated-pass tasks must still pass independent review, benchmark, evidence,
and Dataset Trust Authority promotion gates.

## Benchmark And Data Feedback

Files:

- `src/craftly/learning/feedback.py`

Purpose:

Combine benchmark results, quality reports, and review reports into
recommendations.

Recommendations can say:

- quality is too low
- task extraction is too noisy
- approved task diversity is too narrow
- security/agentic component scores are weak
- benchmark score is below promotion threshold
- dataset can be promoted

Command:

```bash
python -m src.craftly.learning.feedback --output artifacts/craftly/data-pipeline/reports/feedback_report.json --quality-report artifacts/craftly/data-pipeline/reports/quality_report.json --review-report artifacts/craftly/data-pipeline/reports/task_review_report.json
```

Why it exists:

Data quality must be tied to model/evaluation outcomes. If benchmark score drops
or task review approval is weak, the dataset should not be promoted blindly.

## Production Dataset Pack

Files:

- `src/craftly/learning/production_dataset.py`

Purpose:

Turn crawled and cleaned JSONL rows into a governed production training corpus
inside `data/production`. This is the final gate before tokenizer training,
sharding, and scratch pretraining.

It runs these stages:

1. License allowlist filtering.
2. Dataset quality scoring and metadata injection.
3. Benchmark contamination filtering.
4. Exact and near-duplicate removal.
5. Source quality scoring.
6. Source reputation scoring.
7. Source budget planning for the next crawl.
8. Training data gate promotion.
9. Verified patch/task row normalization.
10. Human-review approved high-value row promotion.
11. Benchmark holdout separation.
12. Train/validation/test split.
13. Dataset validation.
14. Dataset version manifest writing.

Required production evidence:

- 100k to 1M+ promoted clean records.
- explicit `license` per row.
- provenance metadata per row.
- contamination-clean report.
- source reputation and source budget reports.
- near-duplicate report.
- train/validation/test split manifest.
- benchmark holdout separation report.
- verified patch/task dataset report.
- human-review approved high-value row report.
- dataset version manifest.

Command:

```bash
python -m src.craftly.learning.production_dataset \
  --input artifacts/craftly/data-runs/first-serious-run/clean/*.jsonl \
  --output-dir data/production/craftly-corpus-v1 \
  --dataset-id craftly-corpus-v1 \
  --source-registry config/data_sources.ultimate.json \
  --benchmark-holdout data/eval/humaneval.jsonl \
  --benchmark-holdout data/eval/mbpp.jsonl \
  --verified-patch artifacts/craftly/verified-patches/verified_patch_tasks.jsonl \
  --human-review-approved artifacts/craftly/review/approved_high_value.jsonl \
  --min-promoted-records 100000 \
  --min-verified-patch-records 100 \
  --min-human-review-approved-records 100 \
  --min-train-records 90000
```

Outputs:

- `data/production/<dataset>/final/train.jsonl`
- `data/production/<dataset>/final/val.jsonl`
- `data/production/<dataset>/final/test.jsonl`
- `data/production/<dataset>/final/holdout.jsonl`
- `data/production/<dataset>/review/human_review_queue.jsonl`
- `data/production/<dataset>/reports/*.json`
- `data/production/<dataset>/dataset_version_manifest.json`
- `data/production/<dataset>/dataset_version_manifest.md`

Production behavior:

- Missing or insufficient real data fails the command.
- Benchmark leakage is removed before final split.
- Dev-smoke mode is allowed only for code-path validation and cannot be treated
  as production data proof.

## Dataset Versioning And Ledger

Files:

- `src/craftly/learning/versioning.py`

Purpose:

Create immutable dataset version manifests and append them to a ledger.

Manifest includes:

- dataset id
- version id
- source registry report
- crawl report
- contamination report
- quality report
- source quality report
- task extraction report
- review report
- feedback report
- tokenizer path
- token shard manifest
- artifact hashes
- uploaded object URIs

Outputs:

- `versions/<version_id>.json`
- `versions/ledger.jsonl`

Why it exists:

You need to know exactly which data created which training run. Without
versioning, model regressions cannot be traced.

## Object Storage

Files:

- `src/craftly/learning/storage.py`

Purpose:

Upload dataset artifacts to local storage or S3/MinIO.

Supported:

- `local://...`
- `file://...`
- `s3://bucket/prefix`

Production:

- use S3/MinIO
- upload clean shards, task files, reports, tokenizer, shard manifest, token
  shards, and version manifest
- S3 uploads retry with backoff

Why it exists:

Large datasets cannot live only on a laptop filesystem. Object storage is the
durable artifact layer.

## Data Dashboard

Files:

- `src/craftly/learning/dashboard.py`

Purpose:

Render a simple HTML dashboard for a dataset run.

Shows:

- dataset id
- version id
- sources
- seed URLs
- fetched/accepted/rejected rows
- contamination hits
- average quality score
- source score count
- extracted tasks
- approved tasks
- feedback item count
- uploaded object count

Output:

- `dashboard.html`

Why it exists:

A dataset run should be inspectable without reading raw JSON files.

## Production Readiness Gate

Files:

- `src/craftly/learning/production_check.py`

Purpose:

Fail fast before a serious production data run if unsafe local-only settings are
used.

Checks:

- source registry is valid
- production run uses Postgres frontier
- production run uses S3/MinIO object storage
- contamination gate is enabled
- data-platform migration exists
- worker scale is high enough

Command:

```bash
python -m src.craftly.learning.production_check --sources config/data_sources.ultimate.json --frontier-backend postgres --postgres-dsn "$CRAFTLY_DATABASE_URL" --object-store-uri s3://craftly-datasets/pretraining --production --worker-replicas 8 --async-workers 64
```

Why it exists:

Production data jobs are expensive. The system should block obviously unsafe
configuration before crawling begins.

## Capacity Planner

Files:

- `src/craftly/learning/capacity.py`

Purpose:

Estimate storage, bandwidth, days to completion, and recommended worker replica
count.

Command:

```bash
python -m src.craftly.learning.capacity --target-documents 1000000000 --target-days 30 --worker-replicas 32 --async-workers-per-replica 32
```

Example meaning:

If 1B documents at 64KB average are targeted, raw storage is about 64TB. If
workers are too few, the planner recommends how many replicas are needed for
the target schedule.

Why it exists:

"Billion-scale" is not just code. It is storage, bandwidth, workers, and time.

## First Serious Run Planner

Files:

- `src/craftly/learning/run_plan.py`

Purpose:

Prepare a serious 100k-1M run before executing it.

It does:

- merge one or more source registries
- validate the merged registry
- run production readiness checks
- calculate capacity plan
- write `run_plan.json`
- write `commands.ps1`

Command:

```bash
python -m src.craftly.learning.run_plan --sources config/data_sources.ultimate.json --output-dir artifacts/craftly/data-runs/first-serious-run --target-documents 1000000 --target-days 7 --postgres-dsn "$CRAFTLY_DATABASE_URL" --object-store-uri s3://craftly-datasets/pretraining --worker-replicas 8 --async-workers 64
```

Why it exists:

Before collecting a real dataset, you need a reproducible plan and exact
commands. This prevents ad hoc data runs.

## End-To-End Data Pipeline

Files:

- `src/craftly/learning/data_pipeline.py`

Purpose:

Run the full data pipeline in one command.

Order:

1. Load and validate source registry.
2. Build SQLite or Postgres frontier.
3. Crawl approved URLs.
4. Write raw and clean JSONL shards.
5. Run contamination gate.
6. Write quality report.
7. Write source quality report.
8. Extract tasks.
9. Review tasks.
10. Write approved tasks.
11. Build benchmark/data feedback report.
12. Train tokenizer.
13. Build token shards.
14. Optionally run scratch pretraining loop.
15. Write dataset version manifest.
16. Append dataset ledger.
17. Upload artifacts to object storage.
18. Write dashboard.

Command:

```bash
python -m src.craftly.learning.data_pipeline --sources config/data_sources.ultimate.json --dataset-id craftly-defensive-coding-corpus --work-dir artifacts/craftly/data-pipeline --frontier-backend postgres --postgres-dsn "$CRAFTLY_DATABASE_URL" --object-store-uri s3://craftly-datasets/pretraining --object-store-endpoint-url "$S3_ENDPOINT_URL" --max-docs 1000000 --workers 64 --max-depth 2 --vocab-size 128000 --sequence-length 2048 --shard-token-count 1000000 --skip-train
```

When to use `--skip-train`:

- during collection
- during quality inspection
- before GPU is available
- before final data approval

When to remove `--skip-train`:

- only after dataset quality, contamination, review, and feedback reports are
  acceptable
- only when GPU training hardware is ready

## Distributed Data Workers

Files:

- `src/craftly/learning/worker.py`
- `deploy/k8s/data-worker.yaml`
- `deploy/k8s/data-worker-hpa.yaml`

Purpose:

Run long-lived distributed crawlers against the same Postgres frontier.

Why Postgres matters:

Multiple workers can claim URLs safely using row locks. This prevents many
machines from crawling the same URL.

Local start:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_data_platform_local.ps1 -WorkerScale 2
```

Kubernetes:

```bash
kubectl apply -f deploy/k8s/data-worker.yaml
kubectl apply -f deploy/k8s/data-worker-hpa.yaml
```

Why it exists:

Million/billion-scale collection needs multiple machines and long-running
workers.

## Production Deployment

Files:

- `deploy/prod/docker-compose.yml`
- `deploy/k8s/api.yaml`
- `deploy/k8s/postgres-redis.yaml`
- `deploy/k8s/minio.yaml`
- `deploy/k8s/data-worker.yaml`
- `deploy/k8s/data-worker-hpa.yaml`
- `deploy/k8s/data-network-policy.yaml`
- `deploy/k8s/data-pipeline-job.yaml`
- `deploy/k8s/secrets.example.yaml`

Local production-like data platform:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_data_platform_local.ps1 -WorkerScale 4
```

Kubernetes deployment order:

```bash
kubectl apply -f deploy/k8s/secrets.example.yaml
kubectl apply -f deploy/k8s/postgres-redis.yaml
kubectl apply -f deploy/k8s/minio.yaml
kubectl apply -f deploy/k8s/api.yaml
kubectl apply -f deploy/k8s/data-worker.yaml
kubectl apply -f deploy/k8s/data-worker-hpa.yaml
kubectl apply -f deploy/k8s/data-network-policy.yaml
kubectl apply -f deploy/k8s/data-pipeline-job.yaml
```

What each piece does:

- Postgres: distributed frontier and production database.
- Redis: quota backend.
- MinIO: S3-compatible dataset artifact storage.
- Data workers: long-running crawler workers.
- HPA: scales workers from 4 to 64 replicas.
- NetworkPolicy: restricts data platform network flows.
- Data pipeline job: batch job for full pipeline processing.

## Recommended Next Operational Step

Do not start with 1M documents immediately. Run a 100-500 document real smoke
first:

1. Start local data platform.
2. Run DB migrations.
3. Run production readiness check.
4. Generate run plan.
5. Run `data_pipeline` with `--max-docs 200 --skip-train`.
6. Inspect:
   - `dashboard.html`
   - `reports/quality_report.json`
   - `reports/source_quality_report.json`
   - `reports/contamination_report.json`
   - `reports/task_review_report.json`
   - `reports/feedback_report.json`
   - `tasks/automated_pass_tasks.jsonl`
7. Confirm that `human_approved` is zero unless durable independent review
   evidence is supplied through the Dataset Trust Authority.
8. If every hard gate passes, scale to 5k, then 100k, then 1M.

## Real Approved-Source GPU Training Run

Files:

- `deploy/gpu/run_real_data_training_pipeline.py`
- `src/craftly/learning/data_pipeline.py`
- `src/craftly/evaluation/checkpoint_eval.py`

Purpose:

Run the first serious end-to-end training path from approved internet sources:

1. Load approved source registry.
2. Crawl only allowlisted domains.
3. Respect robots policy by default.
4. Deduplicate by content hash.
5. Run quality filtering.
6. Run contamination detection.
7. Extract coding/security tasks.
8. Review extracted tasks.
9. Train Craftly tokenizer.
10. Build token shards.
11. Run scratch-only GPU pretraining.
12. Save checkpoint manifest.
13. Evaluate checkpoint integrity and training stability.
14. Run built-in defensive/security/coding benchmark harness.
15. Write JSON and Markdown reports.

Why this exists:

Toy data proves plumbing. Real approved-source data proves that the architecture
can ingest web data, filter it, convert it into training shards, train on GPU,
and produce auditable evidence after the checkpoint.

Kaggle T4 smoke command:

```bash
python deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --output-dir artifacts/craftly/real-data-smoke \
  --max-docs 200 \
  --min-clean-records 25 \
  --workers 8 \
  --max-depth 1 \
  --delay-seconds 0.2 \
  --vocab-size 8000 \
  --sequence-length 64 \
  --validation-fraction 0.05 \
  --train-steps 200 \
  --train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --dtype fp16 \
  --device cuda
```

First 10k-record command:

```bash
python deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --output-dir artifacts/craftly/real-data-10k \
  --max-docs 10000 \
  --min-clean-records 10000 \
  --workers 16 \
  --max-depth 2 \
  --delay-seconds 0.5 \
  --vocab-size 128000 \
  --sequence-length 128 \
  --validation-fraction 0.02 \
  --train-steps 1000 \
  --train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --early-stopping-patience 8 \
  --max-source-fraction 0.35 \
  --dtype fp16 \
  --device cuda
```

Top-class balanced 20k-record command:

```bash
python deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --work-dir artifacts/craftly/real-data-20k-top-class-balanced \
  --target-records 20000 \
  --max-docs 50000 \
  --max-bytes-per-doc 300000 \
  --workers 24 \
  --max-depth 2 \
  --delay-seconds 0.35 \
  --vocab-size 128000 \
  --sequence-length 256 \
  --validation-fraction 0.02 \
  --steps 5000 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --validation-interval 100 \
  --early-stopping-patience 10 \
  --max-source-fraction 0.30 \
  --dtype fp16 \
  --device cuda
```

Kaggle memory safety:

- Exit code `137` means the Kaggle/Linux runtime killed the process, usually
  because host RAM was exhausted.
- The tokenizer and token-shard builders stream JSONL line by line and should
  not load full corpus shards into memory.
- Source balancing is a two-pass streaming process: first count source rows,
  then write capped rows without keeping the full corpus in memory.
- `deploy/gpu/run_real_data_training_pipeline.py` defaults to
  `--max-bytes-per-doc 300000` so huge documentation pages do not dominate RAM.
- If Kaggle still kills the run, reduce `--workers`, `--max-docs`,
  `--sequence-length`, or `--batch-size`, and use a fresh `--work-dir`.

100k-record command:

```bash
python deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --output-dir artifacts/craftly/real-data-100k \
  --max-docs 100000 \
  --min-clean-records 100000 \
  --workers 32 \
  --max-depth 3 \
  --delay-seconds 0.5 \
  --vocab-size 128000 \
  --sequence-length 128 \
  --validation-fraction 0.02 \
  --train-steps 10000 \
  --train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --dtype fp16 \
  --device cuda
```

Important:

- Kaggle is good for a real smoke or small training run.
- Production-scale 10k-100k+ collection is better on a long-running VM or
  Kubernetes worker setup.
- The crawler is defensive and allowlist-based. It is not a general web
  scraper for unauthorized collection.
- `--min-clean-records` intentionally blocks the run if the crawl does not
  produce enough accepted records.
- For short smoke runs, validation must be scheduled inside the run length.
  `deploy/gpu/run_real_data_training_pipeline.py` defaults to
  `--validate-every 25`, and clamps validation cadence to `--train-steps` so a
  50-step run can produce validation losses.
- Use a fresh `--output-dir` for each run. The pipeline now overwrites its own
  `raw-*.jsonl` and `clean-*.jsonl` shards at startup to prevent stale partial
  JSONL lines from interrupted runs contaminating the next run.
- JSONL readers report the exact file and line number if a malformed row is
  encountered.
- JSONL readers stream by physical newline instead of Python `splitlines()` so
  Unicode line separators inside JSON strings do not split valid records.
- Non-text HTTP responses such as `image/png`, fonts, archives, media, and PDFs
  are rejected before raw/clean JSONL writing.
- A `.pipeline.lock` file prevents two processes from writing to the same
  `--output-dir` at the same time. If a lock error appears, use a fresh output
  directory or wait for the existing run to finish.

Primary outputs:

- `reports/real_data_training_report.json`
- `reports/pipeline_report.json`
- `reports/quality_report.json`
- `reports/training_quality_report.json`
- `reports/source_quality_report.json`
- `reports/source_balance_report.json`
- `reports/contamination_report.json`
- `reports/task_review_report.json`
- `reports/feedback_report.json`
- `reports/checkpoint_eval/checkpoint_eval_report.json`
- `reports/checkpoint_eval/benchmarks/built_in_security_benchmark.md`
- `tokenizer/tokenizer.json`
- `shards/manifest.json`
- `train/checkpoint_manifest.json`
- `train/best_checkpoint_manifest.json`

Checkpoint evaluation:

`src/craftly/evaluation/checkpoint_eval.py` verifies:

- checkpoint files exist
- checkpoint file hashes match the checkpoint manifest
- validation-best checkpoint selection is being used when available
- training loss is finite and non-exploding
- validation loss is finite when validation batches exist
- built-in defensive/security/coding benchmark harness passes

Best checkpoint and early stopping:

- `src/craftly/model_ops/pretrain_loop.py` writes both:
  - `train/checkpoint_manifest.json` for the final checkpoint
  - `train/best_checkpoint_manifest.json` for the best validation checkpoint
- `training.best_validation_loss` and `training.best_validation_step` record the
  selected checkpoint.
- `--early-stopping-patience` stops training after repeated validation checks
  fail to improve by `--early-stopping-min-delta`.
- The real-data GPU runner defaults to `--early-stopping-patience 8`.
- Checkpoint evaluation uses `best_checkpoint_manifest` when present.

Source balancing:

- `src/craftly/learning/source_balancing.py` creates
  `balanced/balanced-clean-000000.jsonl` before tokenizer and shard training.
- The default `--max-source-fraction 0.35` prevents a single source such as
  `git-documentation` from dominating model training.
- `reports/source_balance_report.json` records input rows, output rows, and
  capped sources.
- Disable only for diagnostic runs with `--no-source-balancing`.

Scratch instruction mix:

- `src/craftly/learning/mixer.py` now owns the production scratch data mixer.
- The real-data pipeline runs it after license filtering, benchmark
  contamination filtering, near-deduplication, training-data promotion, and
  source balancing.
- Raw rows are converted into explicit scratch training records with
  `<|thought_start|>`, analysis target, correct answer, `<|patch_start|>`,
  code patch or patch plan, tests, and verification result.
- Default token ratio target:
  - 40% `instruction_security_coding`
  - 30% `verified_patch_tests`
  - 20% `high_quality_docs_code`
  - 10% `debugging_error_logs`
- Verified patch/test examples are classified before general security/coding
  rows, so real fixes and regression evidence get priority.
- `reports/instruction_mix_report.json` records input paths, output JSONL,
  bucket rows, estimated token ratios, rejected rows, and recommendations for
  under-supplied buckets.
- The dataset version manifest includes both the instruction mix report and
  `mixed/scratch-instruction-mix.jsonl`.
- Disable only for diagnosis with `--no-instruction-mix`; production mode
  requires instruction mixing.

Checkpoint pass/fail gate:

- `--checkpoint-compare-prompt-suite` connects the expanded prompt suite
  directly into the real-data pipeline.
- The pipeline evaluates the best checkpoint with deterministic generation and
  writes `reports/checkpoint_compare/checkpoint_comparison_report.json`.
- `--checkpoint-compare-min-score` blocks a run when the checkpoint output
  quality is below the required minimum.
- A regressed checkpoint also blocks the run. This keeps "training completed"
  separate from "checkpoint should be promoted."

Expanded built-in benchmark:

- `src/craftly/evaluation/benchmarks.py` includes SQL injection, hardcoded
  secrets, command injection, path traversal, XSS, weak crypto, insecure random,
  unsafe deserialization, C buffer copy, SSRF, unsafe YAML loading, unsafe JWT
  settings, open redirect, Node.js command execution, TypeScript DOM XSS, Go SQL
  injection, Rust command execution, Java deserialization, Solidity reentrancy
  shape, Docker hardening, Kubernetes privileged containers, GitHub Actions
  script injection, debugging trace shape, regression-test shape, and patch-shape
  checks.
- This is still a lightweight built-in gate, not a replacement for SWE-Bench or
  a full security benchmark suite.

Training safety preflight:

- the pretraining loop checks that shards can produce at least one batch before
  allocating the model
- the scratch model vocabulary is automatically expanded to match the tokenizer
  vocabulary and the highest token ID found in train/validation shards
- this prevents CUDA device-side asserts from tokenizer/model vocabulary
  mismatch
- the pretraining report includes `model_config` so the exact executable model
  shape is auditable after each run

Standalone checkpoint eval:

```bash
python -m src.craftly.evaluation.checkpoint_eval \
  --checkpoint-manifest artifacts/craftly/real-data-smoke/train/checkpoint_manifest.json \
  --training-report artifacts/craftly/real-data-smoke/train/pretrain_report.json \
  --output-dir artifacts/craftly/real-data-smoke/reports/checkpoint_eval
```

## Checkpoint Learning Comparison

Files:

- `src/craftly/model_ops/checkpoint_compare.py`
- `deploy/gpu/run_checkpoint_comparison.py`

Purpose:

Measure whether a newer scratch checkpoint is behaviorally better than an older
checkpoint on the same deterministic prompt suite. This answers whether the
model learned more than before, not just whether training loss moved.

How it works:

1. Load a baseline checkpoint manifest.
2. Load a candidate checkpoint manifest.
3. Load the exact tokenizer used by the training run.
4. Run the same fixed coding/security/debugging prompts through both models.
5. Score outputs locally with deterministic heuristics:
   - expected security/coding terms found
   - forbidden unsafe terms avoided
   - output is non-empty
   - output has basic structure signals
   - repetition is not excessive
6. Write JSON and Markdown reports with per-task deltas.

Default prompt categories:

- SQL injection finding and safe patch
- XSS review and safe fix
- Python traceback debugging
- FastAPI JWT middleware planning
- empty-password patch and regression tests

Compare final vs best checkpoint from a real data run:

```bash
python deploy/gpu/run_checkpoint_comparison.py \
  --training-report artifacts/craftly/real-data-20k-v3-kaggle-safe/reports/real_data_training_report.json \
  --output-dir artifacts/craftly/real-data-20k-v3-kaggle-safe/reports/checkpoint_compare \
  --device cuda \
  --max-new-tokens 96
```

Compare two explicit checkpoint manifests:

```bash
python deploy/gpu/run_checkpoint_comparison.py \
  --baseline-manifest artifacts/craftly/run-a/train/best_checkpoint_manifest.json \
  --candidate-manifest artifacts/craftly/run-b/train/best_checkpoint_manifest.json \
  --tokenizer artifacts/craftly/run-b/tokenizer/tokenizer.json \
  --output-dir artifacts/craftly/checkpoint-compare/run-a-vs-run-b \
  --device cuda
```

Outputs:

- `checkpoint_comparison_report.json`
- `checkpoint_comparison_report.md`

Interpretation:

- `status=improved`: candidate scored higher without meaningful regressions.
- `status=neutral`: candidate is not clearly better yet.
- `status=regressed`: candidate should not be promoted without inspection.
- `score_delta`: average score movement across the prompt suite.
- `pass_delta`: number of tasks crossing the pass threshold.

## Scratch Training Control Plane

Files:

- `config/eval_schedule.json`
- `config/mix_ratios.json`
- `src/craftly/evaluation/eval_runner.py`
- `src/craftly/learning/mixer.py`
- `src/craftly/learning/ablation_runner.py`
- `src/craftly/model_ops/pretrain_loop.py`
- `src/craftly/model_ops/tokenizer_pipeline.py`

Purpose:

This layer controls checkpoint promotion, data composition, tokenizer/shard
preparation, scratch pretraining, and scientific model selection. Every model
originates from random Craftly initialization. A separately governed future
stage may continue training Craftly-owned weights with full-parameter
instruction, tool-use, or execution-grounded objectives only after the 1B
foundation gate. Borrowed checkpoints and adapter training remain prohibited.

Checkpoint eval loop:

1. Load a scratch checkpoint manifest.
2. Load `config/eval_schedule.json`.
3. Run deterministic evaluation with `temperature=0` and a fixed seed.
4. Execute built-in defensive security checks, optional JSONL benchmark
   adapters, MCQ-style scored rows, static benchmark rows, and generation
   suites when a tokenizer is supplied.
5. Required missing benchmark files fail the report.
6. Optional missing benchmark files are marked `skipped`.
7. Compare aggregate scores against a previous report when supplied.
8. Flag score drops over 3 percent as warnings and over 5 percent as failures.
9. Write `eval_report.json` and `eval_report.md`.

Command:

```powershell
python -m src.craftly.evaluation.eval_runner `
  --checkpoint-manifest artifacts\\craftly\train\checkpoint_manifest.json `
  --schedule config\eval_schedule.json `
  --output-dir artifacts\\craftly\eval `
  --tokenizer-path artifacts\\craftly\tokenizer\tokenizer.json `
  --device cpu
```

Data mixing controller:

1. Read clean JSONL rows that already passed license, contamination, quality,
   and dedup gates.
2. Classify each row into `general`, `code`, `cybersecurity`, or `agentic`.
3. Exclude `eval_holdout` and `benchmark_holdout` rows from training output.
4. Estimate token counts with the tokenizer when available.
5. Sample rows according to the configured experiment ratios:
   - `baseline_70_15_15`
   - `domain_heavy_55_15_30`
   - `domain_extreme_40_10_50`
6. Write mixed JSONL and, when a tokenizer is supplied, compatible token
   shards for the pretraining loop.
7. Write `mix_manifest.json`.

Command:

```powershell
python -m src.craftly.learning.mixer `
  --inputs data\training\clean.jsonl `
  --config config\mix_ratios.json `
  --experiment domain_heavy_55_15_30 `
  --output-dir artifacts\\craftly\mix-domain-heavy `
  --tokenizer-path artifacts\\craftly\tokenizer\tokenizer.json
```

Ablation and scientific experiment authority:

Legacy invocation prepares each configured data mix and writes
`ablation_report.json` with status `prepared_not_evaluated`. That report cannot
promote a tokenizer, architecture, or checkpoint. The same module owns the
authoritative `plan -> run/resume -> inspect/compare -> decide -> promote`
scientific state machine. It binds governed data/splits, optimizer and
evaluation policy, model contracts, seeds, token/FLOP budgets, Git/container
identity, checkpoints, tokenizer audits, and executable evaluations. The plan
also writes `arm_execution_requests.json`: an exact scheduler-ready contract
containing the model and dataloader seed, optimizer-step count, world size,
tokens per optimizer step, canonical `6 * active parameters * tokens` FLOPs,
and all immutable input hashes.

Command:

```powershell
python -m src.craftly.learning.ablation_runner `
  --mix-config config\mix_ratios.json `
  --base-run-dir artifacts\\craftly\data-pipeline `
  --output-dir artifacts\\craftly\mix-ablation
```

Controlled tokenizer selection starts with:

```powershell
python -m src.craftly.evaluation.qualification_campaign plan `
  --config config\defensive_checkpoint_qualification.json `
  --output-dir artifacts\craftly\qualification-plan

python -m src.craftly.learning.ablation_runner plan `
  --campaign tokenizer-selection-v1 `
  --dataset-manifest data\production\craftly-foundation-v1\dataset_version_manifest.json `
  --split-manifest data\production\craftly-foundation-v1\split_manifest.json `
  --optimizer-policy config\training_profiles.json `
  --evaluation-manifest artifacts\craftly\qualification-plan\qualification_campaign_plan.json `
  --tokenizer-manifest 32000=artifacts\craftly\tokenizers\32k\tokenizer_manifest.json `
  --tokenizer-manifest 64000=artifacts\craftly\tokenizers\64k\tokenizer_manifest.json `
  --tokenizer-manifest 128000=artifacts\craftly\tokenizers\128k\tokenizer_manifest.json `
  --container-digest "craftly-training@sha256:<64-hex-digest>" `
  --output-dir artifacts\craftly\experiments\tokenizer-selection-v1
```

Each planned arm must be executed through the existing trainer/workspace and
must not hand-author an arm-evidence object. Run HumanEval/MBPP through the
executable Docker benchmark authority and the governed 50-task repository pack
through the strict agent scorecard. `assemble-evaluation` accepts them only when
both identify the same checkpoint and tokenizer and when every task-suite hash
matches the experiment manifest. `admit-arm` then derives all loss, score,
FLOP, router, reload, tokenizer, and collapse fields from the bound reports:

```powershell
python -m src.craftly.learning.ablation_runner assemble-evaluation `
  --experiment-dir EXPERIMENT_DIR `
  --code-benchmark-report CODE_EVAL\benchmark_suites_report.json `
  --repository-scorecard-report SCORECARD\agent_scorecard.json `
  --output ARM_REPORTS\scientific_evaluation_report.json

python -m src.craftly.learning.ablation_runner admit-arm `
  --experiment-dir EXPERIMENT_DIR `
  --arm-id ARM_ID `
  --training-report ARM_REPORTS\training_report.json `
  --evaluation-report ARM_REPORTS\scientific_evaluation_report.json `
  --generation-audit ARM_REPORTS\generation_audit.json `
  --tokenizer-audit ARM_REPORTS\tokenizer_audit_report.json
```

The experiment manifest binds the protected configuration, protected manifest,
and each required suite file independently. Any suite drift invalidates the
whole chain. Missing arms remain `blocked`; the authority never invents a score.
Tokenizer manifests bind the actual tokenizer and shard hashes to the same
family-safe dataset. Tokenizer selection compares 32K, 64K, and 128K and
promotes the smallest vocabulary within one downstream aggregate point of the
best result. A larger vocabulary outside that boundary must also have
non-overlapping downstream confidence evidence.

Architecture and scaling campaigns pass one
`--tokenizer-manifest selected=<path>` argument. Model parameter accounting is
then regenerated for the evidence-selected vocabulary. Scaling campaigns use
crossed token budgets at each model size, additive parameter/data power-law
fitting, leave-one-shape-out error, bootstrap confidence intervals, predicted
7B/32B FLOPs, and compute-optimal parameter/token estimates. A decision locks
the experiment permanently; changed evidence requires a new preregistered run.

After tokenizer, architecture, and scaling campaigns each produce a promoted,
replay-valid chain, `advance-7b` checks that all three share one dataset, split,
and evaluation lineage; that the architecture/scaling runs use the promoted
tokenizer bytes; and that scaling explicitly selected 7B. It emits either a
sealed `authorized` decision targeting `7b` or `7b_moe`, or a sealed `blocked`
decision with reasons. Both 7B training profiles require the matching decision
path and SHA-256 before job admission. This is scientific authorization only;
production qualification remains the final release authority.

Scratch-only training rule:

1. Raw crawl rows never train directly.
2. Only promoted rows from the data gate enter tokenizer training, sharding, or
   scratch pretraining.
3. Before the 1B proof, instruction-like, repair-like, and safety-related
   examples are ordinary causal-pretraining curriculum text.
4. After the 1B proof, only a separately governed Craftly-owned full-parameter
   continuation stage may introduce instruction/tool/execution objectives.
5. There is no adapter-based or external-checkpoint tuning path in Craftly.
6. Checkpoints are promoted only through validation loss, benchmark gates,
   security gates, and regression comparison.

Scratch pretraining command:

```powershell
python -m src.craftly.learning.data_pipeline `
  --sources config\data_sources.ultimate.json `
  --dataset-id craftly-defensive-coding-corpus `
  --work-dir artifacts\\craftly\data-pipeline `
  --vocab-size 128000 `
  --sequence-length 2048 `
  --shard-token-count 1000000
```

Security boundary:

- Allowed: authorized labs, CTF/eval data, vulnerability metadata, defensive
  analysis, secure patch generation, and reviewed educational material.
- Blocked: autonomous exploit execution, malware collection, live-target attack
  workflows, credential theft instructions, and unsupervised harmful misuse data.

## Production Readiness Hardening

Files:

- `alembic.ini`
- `src/craftly/db/alembic/env.py`
- `src/craftly/db/alembic/versions/0001_initial.py`
- `src/craftly/db/alembic/versions/0002_data_platform.py`
- `src/craftly/learning/storage.py`
- `src/craftly/learning/dataset_validation.py`
- `src/craftly/deployment/k8s_validate.py`
- `src/craftly/evaluation/benchmark_suites.py`
- `src/craftly/security/audit.py`
- `deploy/gpu/run_10k_training_validation.py`
- `deploy/prod/prometheus.yml`
- `deploy/prod/grafana-dashboard.json`
- `deploy/prod/otel-collector.yaml`

Purpose:

This layer turns the architecture from local MVP code into a deployable
production candidate. It does not magically prove billion-scale operation on a
single laptop; it provides the gates and commands that must pass on real
infrastructure before production promotion.

Postgres migration strategy:

- The existing SQL migration runner remains available for lightweight Docker
  and CI workflows.
- Alembic is now configured for standard production migration operations.
- The Alembic version scripts reuse the same SQL migration files, keeping one
  source of truth.
- Migrations are forward-only to avoid destructive rollback surprises.

Commands:

```powershell
python -m src.craftly.db.migration_runner --database-url postgresql://craftly:pass@localhost:5432/craftly --dry-run
alembic upgrade head
```

Object storage lifecycle:

- Supports local storage and S3-compatible storage such as MinIO.
- Verifies write, head, download, checksum match, list, and delete.
- Writes `object_store_lifecycle_report.json`.

Commands:

```powershell
python -m src.craftly.learning.storage `
  --uri local://artifacts/craftly/object-store `
  --work-dir artifacts/craftly/object-store-lifecycle

python -m src.craftly.learning.storage `
  --uri s3://craftly-datasets/pretraining `
  --endpoint-url http://localhost:9000 `
  --work-dir artifacts/craftly/s3-lifecycle
```

Kubernetes validation:

- Loads every YAML manifest under `deploy/k8s`.
- Checks workload resources, probes, secret references, privileged containers,
  privilege escalation, PVCs, HPA presence, services, and network policy.
- Optional `--kubectl-dry-run` performs server-side validation against a real
  cluster.
- Placeholder secrets in `secrets.example.yaml` are warnings, not blockers.

Commands:

```powershell
python -m src.craftly.deployment.k8s_validate --output-dir artifacts/craftly/k8s-validation
python -m src.craftly.deployment.k8s_validate --kubectl-dry-run --output-dir artifacts/craftly/k8s-validation
```

Long-running crawler supervision:

- `src/craftly/learning/supervisor.py` runs supervised crawl cycles against a
  Postgres frontier.
- It writes heartbeat and status JSON for external monitoring.
- Docker Compose and Kubernetes include crawler worker and supervisor services.
- The data dashboard summarizes crawl, quality, license, contamination,
  source reputation, budget, task extraction, review, upload, and feedback
  status.

Large dataset validation:

- `src/craftly/learning/dataset_validation.py` streams JSONL files and does not
  load the full corpus into memory.
- It checks record count, duplicate fraction, average text length, category
  coverage, license presence, quality metadata, and holdout/train split.
- Use it before tokenizer training and before promoting a dataset version.

Command:

```powershell
python -m src.craftly.learning.dataset_validation `
  --inputs artifacts/craftly/data-pipeline/clean/clean-000000.jsonl `
  --output-dir artifacts/craftly/dataset-validation `
  --min-records 100000 `
  --max-duplicate-fraction 0.02
```

10k-step GPU validation:

- `deploy/gpu/run_10k_training_validation.py` enforces at least 10,000 training
  steps.
- It runs the scratch pretraining loop and checkpoint eval.
- Use this on Kaggle/Colab T4/A100/L4/P100-compatible PyTorch builds or on a
  real GPU node.
- On Kaggle/Colab, install `requirements-kaggle-smoke.txt` first. It avoids
  `vllm`, `deepspeed`, and torch reinstallations that can break the hosted CUDA
  runtime. Use `requirements-linux-gpu.txt` only on controlled GPU machines or
  containers.
- Long-running GPU/data commands emit structured progress to stdout and to
  `progress.jsonl`. Each event has `stage`, `status`, `ts_unix`, and metrics
  such as fetched docs, accepted rows, train loss, validation loss, trained
  tokens, checkpoint paths, and final report paths.

Command:

```bash
python deploy/gpu/run_10k_training_validation.py \
  --manifest artifacts/craftly/shards/manifest.json \
  --device cuda \
  --steps 10000 \
  --sequence-length 128 \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

Live progress example:

```bash
PYTHONUNBUFFERED=1 python -u deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --work-dir artifacts/craftly/kaggle-real-data-smoke \
  --target-records 1000 \
  --max-docs 3000 \
  --steps 200 \
  --device cuda \
  --dtype fp16 \
  --progress-every-docs 10 \
  --progress-every-steps 10

tail -n 80 artifacts/craftly/kaggle-real-data-smoke/progress.jsonl
```

Kaggle validation preset:

```bash
%%bash
cd /kaggle/working/craftly-agentic-ai
PYTHONUNBUFFERED=1 python -u deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --work-dir artifacts/craftly/real-data-validation-v1 \
  --kaggle-validation \
  --steps 1000 \
  --sequence-length 128 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --validation-interval 100 \
  --validation-batches 4 \
  --early-stopping-patience 5 \
  --checkpoint-compare-prompt-suite artifacts/craftly/learning-validation-v1/expanded_eval_suite.jsonl \
  --checkpoint-compare-min-score 0.20 \
  --progress-to-stdout
```

Kaggle notebooks may buffer `%%bash` output until the process exits. For real
live progress, run the job in the background and tail the progress file in a
second cell.

Cell 1:

```bash
%%bash
cd /kaggle/working/craftly-agentic-ai
git pull origin master
mkdir -p artifacts/craftly/real-data-10k-strict-v1
PYTHONUNBUFFERED=1 nohup python -u deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --work-dir artifacts/craftly/real-data-10k-strict-v1 \
  --target-records 10000 \
  --min-training-rows 5000 \
  --min-train-tokens 2000000 \
  --max-docs 16000 \
  --max-bytes-per-doc 250000 \
  --workers 6 \
  --max-depth 2 \
  --delay-seconds 0.35 \
  --steps 10000 \
  --sequence-length 128 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --validation-interval 250 \
  --validation-batches 8 \
  --early-stopping-patience 12 \
  --min-training-quality-score 0.62 \
  --min-training-average-quality-score 0.62 \
  --min-source-reputation-score 0.50 \
  --eval-holdout-fraction 0.02 \
  --max-source-fraction 0.25 \
  --checkpoint-compare-prompt-suite artifacts/craftly/learning-validation-v1/expanded_eval_suite.jsonl \
  --checkpoint-compare-min-score 0.20 \
  --progress-path artifacts/craftly/real-data-10k-strict-v1/progress.jsonl \
  --progress-to-stdout \
  --progress-every-docs 10 \
  --progress-every-steps 25 \
  > artifacts/craftly/real-data-10k-strict-v1/run.log 2>&1 &
echo $! > artifacts/craftly/real-data-10k-strict-v1/run.pid
cat artifacts/craftly/real-data-10k-strict-v1/run.pid
```

Cell 2:

```bash
%%bash
cd /kaggle/working/craftly-agentic-ai
tail -f artifacts/craftly/real-data-10k-strict-v1/progress.jsonl
```

When the job finishes:

```bash
%%bash
cd /kaggle/working/craftly-agentic-ai
tail -n 80 artifacts/craftly/real-data-10k-strict-v1/run.log
cat artifacts/craftly/real-data-10k-strict-v1/reports/real_data_training_report.json

python deploy/gpu/run_checkpoint_comparison.py \
  --training-report artifacts/craftly/real-data-10k-strict-v1/reports/real_data_training_report.json \
  --output-dir artifacts/craftly/real-data-10k-strict-v1/reports/checkpoint_compare \
  --device cuda
```


Inspect the run and get the next recommended action:

```bash
%%bash
cd /kaggle/working/craftly-agentic-ai
python deploy/gpu/inspect_real_data_run.py \
  --work-dir artifacts/craftly/real-data-validation-v1
```
Benchmark suite adapters:

- `swe_bench_style`: local SWE-Bench-like JSONL files.
- `human_eval_style`: local HumanEval-like rows.
- `mbpp_style`: local MBPP-like rows.
- `cyberseceval_style`: local CyberSecEval-like rows.
- `custom_security`: Craftly-owned security benchmark rows.
- Benchmark files are local/explicit. The system does not automatically
  download or mix protected eval data into training.

Command:

```powershell
python -m src.craftly.evaluation.benchmark_suites `
  --suite swe swe_bench_style data/eval/swe_style.jsonl `
  --suite cyber cyberseceval_style data/eval/cyber.jsonl `
  --output-dir artifacts/craftly/benchmark-suites
```

Security audit:

- Scans production source and deployment assets.
- Checks hardcoded secret patterns, SSRF sinks, path traversal sinks, risky
  process execution sinks, dependency version bounds, optional Bandit output,
  and Kubernetes manifest status.
- Tests and deliberately vulnerable benchmark fixtures are excluded from the
  default production-source scan to reduce false positives.

Commands:

```powershell
python -m src.craftly.security.audit --no-bandit --output-dir artifacts/craftly/security-audit
python -m src.craftly.security.audit --output-dir artifacts/craftly/security-audit
```

Monitoring:

- Prometheus scrapes `/metrics`.
- Grafana dashboard is defined in `deploy/prod/grafana-dashboard.json`.
- Optional OpenTelemetry tracing is enabled when
  `CRAFTLY_OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- Docker Compose includes Prometheus, Grafana, and an OTel collector under the
  `monitoring` profile.

Command:

```powershell
docker compose --env-file deploy/prod/.env.example -f deploy/prod/docker-compose.yml --profile monitoring up
```

## Strict Training Data Quality Gate

The data pipeline now has a promotion gate between deduplication/source
reputation and tokenizer/shard construction. Its purpose is to keep weak
internet text out of scratch pretraining and to preserve high-value rows for
review instead of silently discarding them.

Pipeline order:

1. Crawl approved sources.
2. Apply license filtering.
3. Remove benchmark contamination.
4. Remove exact and near duplicates.
5. Scan contamination patterns.
6. Inspect quality and source quality.
7. Extract security/coding tasks and automated review decisions.
8. Score source reputation and allocate future source budgets.
9. Promote rows through `training_data_gate.py`.
10. Balance sources, train tokenizer, build shards, and train/evaluate.

The gate writes:

- `gated/training-promoted.jsonl`: rows allowed into training.
- `gated/eval-holdout.jsonl`: promoted rows reserved for local validation.
- `gated/human-review-queue.jsonl`: high-value rows that need review.
- `reports/training_data_gate_decisions.jsonl`: one decision per scanned row.
- `reports/training_data_gate_report.json`: aggregate promotion/rejection
  report.

Gate scoring uses:

- Row quality score.
- Source reputation score.
- Patch/debug/security priority labels.
- Boilerplate and repeated-line risk flags.
- Holdout sampling controlled by `--eval-holdout-fraction`.

Default thresholds:

- `--min-training-quality-score 0.62` in the Kaggle real-data entrypoint.
- `--min-training-average-quality-score 0.62` for the actual tokenizer/training corpus.
- `--min-source-reputation-score 0.50` in the Kaggle real-data entrypoint.
- `--eval-holdout-fraction 0.02`

For strict Kaggle validation on public approved sources, start with:

```bash
python deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --work-dir artifacts/craftly/real-data-10k-strict-v1 \
  --target-records 10000 \
  --min-training-rows 5000 \
  --min-train-tokens 2000000 \
  --max-docs 16000 \
  --steps 10000 \
  --sequence-length 128 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --validation-interval 250 \
  --validation-batches 8 \
  --early-stopping-patience 12 \
  --min-training-quality-score 0.62 \
  --min-training-average-quality-score 0.62 \
  --min-source-reputation-score 0.50 \
  --max-source-fraction 0.25 \
  --progress-every-steps 25
```

For production dataset builds, use stricter defaults and inspect the gate
report before accepting a dataset version. A serious run should have:

- High promoted count.
- Low boilerplate rejection after source allowlist tuning.
- Non-empty human review queue for high-value security rows.
- Eval holdout separated from training.
- Source mix controlled by `config/mix_ratios.json`.

## Model Quality Build Blocks

These modules are the current production path for improving actual model
quality before large GPU training.

### Strong Real-Corpus Tokenizer

Module:

- `src/craftly/model_ops/tokenizer_pipeline.py`

Purpose:

- Train a code/security optimized BPE tokenizer on promoted real corpus rows.
- Inject deterministic stress samples for indentation, hex dumps, compile
  errors, memory markers, and Craftly control tokens.
- Build token shards from the same corpus.
- Write `tokenizer_audit_report.json` with special-token coverage, vocab size,
  sample token counts, source row/character counts, and shard manifest.

Command:

```bash
python -m src.craftly.model_ops.tokenizer_pipeline \
  --real-corpus-audit \
  --input artifacts/craftly/real-data-5k-quality-gated/gated/training-promoted.jsonl \
  --output-dir artifacts/craftly/real-tokenizer-v1 \
  --vocab-size 128000 \
  --min-frequency 2 \
  --shard-token-count 1000000 \
  --sequence-length 128 \
  --validation-fraction 0.02
```

### Verified Security Patch Dataset

Module:

- `src/craftly/learning/verified_patch_dataset.py`

Purpose:

- Read approved local Git repositories.
- Find security-relevant commits.
- Extract parent commit, patch, changed files, before/after snippets, and
  vulnerability categories.
- Verify the patch applies cleanly to the parent commit using `git apply
  --check` in an isolated temporary clone.
- Write scratch-training JSONL records with repository context, patch text,
  provenance, and verification metadata.

Command:

```bash
python -m src.craftly.learning.verified_patch_dataset \
  --repo /path/to/approved/permissive/repo \
  --license mit \
  --output artifacts/craftly/verified-patches/security_patches.jsonl \
  --max-commits-per-repo 500
```

Only use repositories with approved licenses. This does not run exploit code or
attack live targets.

### Repository Indexing + Verified Patch Loop

Module:

- `src/craftly/patches/verified_loop.py`

Purpose:

- Index the repository.
- Build pre-patch context with changed files pinned.
- Preview/apply patch edits.
- Run configured commands, secret scan, and optional Semgrep/CodeQL checks.
- Reindex and build post-patch context.
- Roll back by default unless `--apply-on-accept` is provided.

Edits file format:

```json
{
  "edits": [
    {
      "path": "src/auth.py",
      "new_content": "def login(user, password):\n    return bool(user) and bool(password)\n"
    }
  ]
}
```

Command:

```bash
python -m src.craftly.patches.verified_loop \
  --repo /path/to/repo \
  --goal "fix authentication validation" \
  --edits-json artifacts/craftly/patch-edits.json \
  --command "python -m pytest" \
  --output artifacts/craftly/patch-loop/report.json
```

### Benchmark Pack

Module:

- `src/craftly/evaluation/benchmark_pack.py`

Purpose:

- Run local HumanEval-like, MBPP-like, SWE-Bench-like, CyberSecEval-like, and
  Craftly-owned security suites through one strict report.
- Required benchmark files fail if missing in strict mode.
- Optional custom security suite is skipped if omitted.
- Reports are explicit measured/skipped/failed states, never fake passes.

Command:

```bash
python -m src.craftly.evaluation.benchmark_pack \
  --human-eval data/eval/humaneval.jsonl \
  --mbpp data/eval/mbpp.jsonl \
  --swe-bench data/eval/swe_bench_style.jsonl \
  --cyberseceval data/eval/cyberseceval_style.jsonl \
  --custom-security data/eval/craftly_security.jsonl \
  --output-dir artifacts/craftly/benchmark-pack
```

Current benchmark pack validates local suite adapters and static expected-term
contracts. Full model-generation benchmark scoring requires connecting the
trained Craftly checkpoint generation runner to these suites.

## Transformer Core

Module:

- `src/craftly/model_ops/torch_decoder.py`

Current implemented transformer capabilities:

- Decoder-only scratch LM.
- RMSNorm.
- SwiGLU MLP.
- RoPE with scaling factor.
- Grouped-query attention.
- PyTorch SDPA attention path. On supported CUDA/PyTorch builds this can use
  FlashAttention or memory-efficient kernels through PyTorch's SDPA dispatcher.
- Eager attention fallback for portability and debugging.
- Optional sliding-window attention mask through `attention_window`.
- KV-cache inference with `past_key_values`.
- Greedy/top-k sampling generation API.
- Gradient checkpointing support.
- Logit soft-cap option.
- Finite-loss and finite-gradient checks in pretraining.
- Export directory with `model.pt`, `config.json`, and serving compatibility
  metadata, including native Craftly support status, KV-cache contract,
  vLLM/TensorRT conversion blockers, and deterministic generation defaults.
- Shape-valid scratch profiles: `tiny`, `1b`, `7b`, `32b`, `62b`.
- Cluster training plan generator for FSDP, DeepSpeed ZeRO-2/ZeRO-3, and
  Megatron-style launch contracts. This validates the shard manifest, global
  batch math, token throughput target, node/GPU counts, and required environment
  before a large run is attempted.

Important boundary:

- The code supports large-profile construction and cluster-oriented training
  flags.
- Actual 62B training, DeepSpeed/Megatron/FSDP scaling, vLLM/TensorRT serving,
  and 10k+ step multi-GPU validation still require real Linux CUDA cluster
  hardware. Do not mark those as production-proven until cluster release gates
  have run.
- Native PyTorch FSDP runtime is wired into the pretraining loop for `torchrun`
  execution: it initializes distributed state, maps local CUDA ranks, wraps
  decoder blocks, uses mixed precision when requested, and writes rank-safe full
  checkpoints. DeepSpeed and Megatron paths are launch/readiness contracts until
  their dedicated engine adapters pass cluster release gates.

Tiny transformer smoke:

```powershell
python -m unittest tests.test_craftly_scratch_decoder
```

Pretraining loop with memory-efficient settings:

```bash
python -m src.craftly.model_ops.pretrain_loop \
  --manifest artifacts/craftly/real-tokenizer-v1/shards/manifest.json \
  --output-dir artifacts/craftly/pretrain-eager-gc \
  --device cuda \
  --dtype fp16 \
  --model-profile tiny \
  --attention-impl auto \
  --gradient-checkpointing \
  --steps 1000 \
  --batch-size 2 \
  --sequence-length 128 \
  --gradient-accumulation-steps 8
```

62B config dry contract:

```python
from src.craftly.model_ops.torch_decoder import model_profile
profile = model_profile("62b")
print(profile.parameter_estimate(), profile.model_dump())
```

Cluster training plan:

```bash
python -m src.craftly.model_ops.pretrain_loop \
  --cluster-plan-only \
  --distributed-strategy fsdp \
  --manifest artifacts/craftly/real-tokenizer-v1/shards/manifest.json \
  --output-dir artifacts/craftly/cluster-runs/Craftly-62b \
  --cluster-plan-out artifacts/craftly/cluster-runs/Craftly-62b/cluster_training_plan.json \
  --model-profile 62b \
  --num-nodes 8 \
  --gpus-per-node 8 \
  --sequence-length 8192 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --steps 10000 \
  --dtype bf16 \
  --attention-impl auto \
  --gradient-checkpointing
```

What the plan proves:

- the training manifest exists
- the selected scratch model profile is shape-valid
- global batch size and tokens per optimizer step are explicit
- required distributed environment variables are visible before launch
- warnings appear when hardware is too small for the selected profile

What the plan does not prove:

- that the cluster has enough GPU memory
- that NCCL networking is healthy
- that DeepSpeed/Megatron/FSDP actually completed a checkpoint on your cluster
- that exported weights are vLLM/TensorRT native without adapter/conversion work

Those are cluster release-gate tasks, not laptop/Kaggle smoke tasks.

## Production Readiness Contract

Module:

- `src/craftly/production_readiness.py`
- `src/craftly/shared/config_contracts.py`
- `config/mix_ratios.json`
- `config/eval_schedule.json`
- `config/active_model_profile.json`
- `config/security_audit_excludes.json`
- `config/verifier_policy.json`

Purpose:

Craftly must never silently claim production readiness when it is only locally
smoke-tested. The readiness contract classifies each subsystem using explicit
machine-readable states:

- `production_ready`
- `production_ready_requires_external_service`
- `built_not_cluster_proven`
- `blocked_missing_dependency`
- `not_implemented`

Covered subsystems:

- auth and quota
- native Craftly serving
- Postgres, object storage, Qdrant, and OpenTelemetry config
- Semgrep, CodeQL, Bandit, pip-audit, Docker, and kubectl availability
- CUDA pretraining runtime
- FSDP, DeepSpeed/Megatron, vLLM/TensorRT
- required benchmark files

Development readiness report:

```powershell
python -m src.craftly.production_readiness --mode dev --output-dir artifacts\\craftly\production-readiness
```

Production readiness report:

```powershell
python -m src.craftly.production_readiness --mode production --benchmark-dir data\eval --output-dir artifacts\\craftly\production-readiness
```

Production mode is expected to fail until real external infrastructure is
configured. That is intentional. A missing Redis, Postgres, S3/MinIO, Qdrant,
Semgrep, CodeQL, Docker, kubectl, CUDA runtime, or benchmark suite must be a
visible blocker, not a hidden warning.

Live production proof:

`production_readiness` tells you whether configuration claims are valid.
`production_proof` actually touches the running dependencies and writes measured
evidence. It verifies:

- live Postgres migration apply in strict mode
- Redis regenerative quota execution through the Lua-backed quota store
- local or S3/MinIO object-store write/read/delete lifecycle
- Qdrant temporary collection create/upsert/query/delete lifecycle
- authenticated native Craftly serving identity and checkpoint readiness
- bounded OpenAI-compatible normal and SSE streaming load tests
- executable benchmark, strict 50-task scorecard, and active-profile hash replay
- security audit report

Validation mode:

```powershell
python -m src.craftly.deployment.production_proof `
  --object-store-uri local://artifacts/craftly/proof-object-store `
  --output-dir artifacts\\craftly\production-proof-validation
```

Validation mode is for local/Kaggle proof of code paths. Missing external
services are marked `skipped`, not `passed`.

Strict production mode:

```powershell
python -m src.craftly.deployment.production_proof `
  --strict `
  --postgres-url "$env:CRAFTLY_DATABASE_URL" `
  --apply-postgres-migrations `
  --redis-url "$env:CRAFTLY_REDIS_URL" `
  --object-store-uri "$env:CRAFTLY_OBJECT_STORE_URI" `
  --object-store-endpoint-url "$env:CRAFTLY_OBJECT_STORE_ENDPOINT_URL" `
  --qdrant-url "$env:CRAFTLY_QDRANT_URL" `
  --serving-url "$env:CRAFTLY_SERVING_URL" `
  --serving-api-key "$env:CRAFTLY_MODEL_API_KEY" `
  --load-test-requests 100 `
  --load-test-streaming-requests 20 `
  --executable-benchmark-report artifacts\craftly\executable-eval\benchmark_suites_report.json `
  --scorecard-report artifacts\craftly\agent-scorecard\agent_scorecard.json `
  --active-model-profile C:\CraftlyGovernance\active-model-profile.json `
  --run-security-audit `
  --strict-security-tools `
  --output-dir artifacts\\craftly\production-proof
```

Strict mode fails if required dependencies are missing or unhealthy. This is the
command to use after starting the production Docker Compose stack or a
Kubernetes deployment. Service URLs reject embedded credentials,
query/fragment ambiguity, and unapproved remote HTTP. Use
`--allow-insecure-service-host HOST` only for an explicitly approved private
HTTP host; HTTPS remains the default.

Configuration contract layer:

- All production-critical JSON configs are validated through strict Pydantic
  contracts before runtime use.
- `mix_ratios.json` enforces exact ratio sums, protected holdout policies,
  source budget metadata, scratch instruction mix ratios, and minimum bucket
  requirements.
- `eval_schedule.json` includes benchmark-level minimum scores, protected
  holdout flags, repetition-collapse thresholds, promotion policy, safety
  targets, and regression thresholds.
- `active_model_profile.json` must declare scratch-only status, dev-only
  status, checkpoint/tokenizer paths when applicable, and explicit production
  blockers for local test-double profiles.
- `security_audit_excludes.json` requires reason, owner, risk category, and
  explicit approved executable-sink classes before an excluded file may contain
  executable sink strings.
- `verifier_policy.json` defines default and production profiles, allowed
  command roots, timeouts, scanner selections, fail-closed behavior, and
  production-readiness flags.
- Invalid configs fail before execution: bad ratio sums, duplicate benchmark
  names, unprotected required benchmark paths, non-scratch model profiles,
  unsafe verifier shell shapes, and ungoverned audit excludes are rejected.

## Production Training Guardrails

The scratch pretraining loop now records production traceability metadata in
each checkpoint:

- optimizer state
- scheduler state
- model config
- training args
- tokenizer path and SHA-256 hash
- shard manifest path and SHA-256 hash
- git commit
- environment report
- distributed world size

Production mode:

```bash
python -m src.craftly.model_ops.pretrain_loop \
  --production \
  --manifest artifacts/craftly/shards/manifest.json \
  --output-dir artifacts/craftly/production-pretrain \
  --device cuda \
  --model-profile 7b \
  --dtype bf16 \
  --validate-every 100 \
  --checkpoint-every 500 \
  --steps 10000
```

Production mode rejects:

- `model-profile=tiny` unless `--dev-smoke` is explicitly set
- missing shard manifest
- missing tokenizer asset
- validation schedules that never run
- disabled checkpointing
- non-finite or catastrophic loss
- incompatible checkpoint resume shape

The data pipeline also has production validation. When `--production` is used
through `deploy/gpu/run_real_data_training_pipeline.py`, it requires Postgres
frontier storage, non-local object storage, license filtering, benchmark
contamination filtering, near-dedup, source balancing, training-data gate, and
in-run validation.

## Native Scratch Serving

Module:

- `src/craftly/model_ops/native_serving.py`

Purpose:

Serve a Craftly-owned scratch checkpoint directly before vLLM/TensorRT
conversion exists. This is not a mock backend. It loads the checkpoint manifest,
`model.pt`, tokenizer, validates tokenizer/model compatibility, and exposes an
OpenAI-compatible chat endpoint.

Command:

```bash
python -m src.craftly.model_ops.native_serving \
  --checkpoint-manifest artifacts/craftly/train/checkpoint_manifest.json \
  --tokenizer-path artifacts/craftly/tokenizer/tokenizer.json \
  --model-name craftly-scratch \
  --device cuda \
  --host 0.0.0.0 \
  --port 8001
```

Endpoints:

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/models`
- `POST /v1/chat/completions`

The endpoint supports normal JSON responses and SSE streaming. Auth, quota, and
observability middleware are installed by default. Use `--no-auth` and
`--no-quota` only for local validation.

## DeepSpeed Runtime Status

DeepSpeed ZeRO-2/ZeRO-3 is now wired as a real runtime path in the scratch
pretraining loop. When `--distributed-strategy deepspeed_zero2` or
`deepspeed_zero3` is selected, the loop:

- imports DeepSpeed and fails immediately if it is not installed
- initializes distributed state through DeepSpeed
- loads and patches the ZeRO JSON config with real batch sizes
- uses DeepSpeed engine `backward()` and `step()`
- writes DeepSpeed engine checkpoint folders alongside the native checkpoint

Example:

```bash
deepspeed --num_nodes 1 --num_gpus 8 -m src.craftly.model_ops.pretrain_loop \
  --distributed-strategy deepspeed_zero3 \
  --deepspeed-config deploy/gpu/deepspeed_zero3.json \
  --manifest artifacts/craftly/shards/manifest.json \
  --output-dir artifacts/craftly/ds-zero3-run \
  --device cuda \
  --model-profile 7b \
  --dtype bf16 \
  --sequence-length 2048 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --steps 10000 \
  --production
```

This path is built, but still not cluster-proven until an actual multi-GPU
DeepSpeed run saves, reloads, and evaluates a checkpoint. The Megatron launch
adapter now maps the canonical model contract into bounded TP/PP/DP/CP/EP
arguments and a validated multi-node rendezvous. It still requires a pinned
external Megatron-Core checkout and real cluster checkpoint/recovery proof.

## vLLM / TensorRT-LLM / Megatron Adapters

Module:

- `src/craftly/model_ops/production_adapters.py`

Purpose:

Bridge native Craftly scratch checkpoints into production runtime ecosystems
without pretending that external GPU runtimes have been validated locally.

HF/vLLM export:

```bash
python -m src.craftly.model_ops.production_adapters export-hf \
  --checkpoint-manifest artifacts/craftly/train/checkpoint_manifest.json \
  --tokenizer-path artifacts/craftly/tokenizer/tokenizer.json \
  --output-dir artifacts/craftly/exports/hf-llama \
  --torch-dtype bfloat16
```

The HF export path is intentionally dense/GQA-only. It rejects MLA or MoE
checkpoints because mapping those weights into a Llama package would be
incorrect. The 4T profile therefore requires a dedicated MLA/MoE converter and
fixed-prompt logit-parity proof before vLLM or TensorRT-LLM promotion.

This writes:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `craftly_conversion_manifest.json`

Validate vLLM prerequisites:

```bash
python -m src.craftly.model_ops.production_adapters validate-vllm \
  --hf-model-dir artifacts/craftly/exports/hf-llama
```

TensorRT-LLM build plan:

```bash
python -m src.craftly.model_ops.production_adapters plan-tensorrt \
  --hf-model-dir artifacts/craftly/exports/hf-llama \
  --output-dir artifacts/craftly/exports/tensorrt \
  --dtype bfloat16
```

Megatron launch plan:

```bash
python -m src.craftly.model_ops.production_adapters plan-megatron \
  --manifest artifacts/craftly/shards/manifest.json \
  --tokenizer-path artifacts/craftly/tokenizer/tokenizer.json \
  --output-dir artifacts/craftly/megatron \
  --model-profile 7b \
  --tensor-parallel 2 \
  --pipeline-parallel 2 \
  --data-parallel 4 \
  --sequence-length 2048 \
  --micro-batch-size 1 \
  --global-batch-size 16 \
  --train-iters 10000 \
  --num-nodes 2 \
  --gpus-per-node 8 \
  --node-rank 0 \
  --master-addr training-controller.internal \
  --master-port 29500 \
  --megatron-root /opt/Megatron-LM
```

Promotion rule:

- HF export existing is not enough.
- vLLM must load and decode the exported package.
- TensorRT-LLM must build an engine and pass decode parity.
- Megatron must run on a real cluster, save a checkpoint, reload, and evaluate.

## Production Benchmark Pack

The benchmark pack now has production minimum-count checks. A tiny local JSONL
file can still validate adapter shape in dev mode, but cannot pass production
coverage.

Public coding benchmarks can be materialized locally:

```bash
python -m src.craftly.evaluation.benchmark_pack \
  --materialize-public \
  --target-dir data/eval
```

This fetches OpenAI HumanEval and Google Research MBPP from their public
repositories and writes:

- `data/eval/humaneval.jsonl`
- `data/eval/mbpp.jsonl`
- `data/eval/benchmark_materialization_report.json`

SWE-Bench and CyberSecEval remain governed local-file inputs. They are not
silently downloaded into training or evaluation because the real runners,
licenses, and holdout rules must be handled explicitly.

```bash
python -m src.craftly.evaluation.benchmark_pack \
  --production \
  --human-eval data/eval/humaneval.jsonl \
  --mbpp data/eval/mbpp.jsonl \
  --swe-bench data/eval/swe_bench_style.jsonl \
  --cyberseceval data/eval/cyberseceval_style.jsonl \
  --custom-security data/eval/craftly_security.jsonl \
  --output-dir artifacts/craftly/benchmark-pack
```

Default production minimums:

- HumanEval: 164 tasks
- MBPP: 374 tasks
- SWE-style suite: at least 1 local governed task file
- CyberSecEval-style suite: at least 1 local governed task file

### Governed Sprint Day 4-6 Execution

Day 4 materializes the public coding holdouts and verifies their local
contracts without treating adapter validation as model performance:

```powershell
python -m src.craftly.evaluation.benchmark_pack `
  --materialize-public `
  --target-dir data\eval

python -m src.craftly.evaluation.benchmark_pack `
  --human-eval data\eval\humaneval.jsonl `
  --mbpp data\eval\mbpp.jsonl `
  --min-human-eval-tasks 164 `
  --min-mbpp-tasks 374 `
  --non-strict `
  --output-dir artifacts\craftly\benchmark-pack
```

The verified July 2026 local materialization contains 164 HumanEval rows and
974 MBPP rows. `--non-strict` means only these two supplied suites are checked;
it does not waive their minimum counts. The resulting suite status proves
files, schemas, and adapter execution. It is not a checkpoint pass rate.
The materialization report binds each artifact to its immutable upstream
revision, declared license, SHA-256, and `eval_holdout` policy. Protected
benchmark fingerprints remain separately pinned under `data/eval/protected`.

Day 5 exports the current durable Dataset Authority review state:

```powershell
python -m src.craftly.learning.dataset_authority review-report `
  --database artifacts\craftly\dataset-authority.sqlite3 `
  --output artifacts\craftly\review\review_evidence_report.json
```

The report includes a schema version, `empty`, `in_progress`, or `complete`
status, total items, decisions, sources, approved/rejected/pending records, and
paired-review count. An empty authority is valid at this scaffold stage and
must report explicit zero totals. It does not satisfy later human-review
promotion requirements.

Day 6 must use the immutable calibration authority, not the older direct GPU
pipeline command. Training and tokenizer flags do not belong in a governed
crawl-only calibration:

```powershell
python -m src.craftly.learning.calibration_gate run `
  --stage calibration_200 `
  --sources config\data_sources.governed.json `
  --protected-config config\protected_benchmarks.json `
  --protected-manifest data\eval\protected\protected_benchmark_manifest.json `
  --reviewer-roster C:\CraftlyGovernance\data_reviewers.json `
  --reviewer-qualification-report C:\CraftlyGovernance\qualification-result\reviewer_qualification_report.json `
  --legal-evidence-dir C:\CraftlyGovernance\source-approvals `
  --legal-evidence-dir governance\source-approvals `
  --trust-policy config\dataset_trust_policy.json `
  --work-dir artifacts\craftly\calibration-200-v2 `
  --authority-db artifacts\craftly\dataset-authority.sqlite3 `
  --crawl-multiplier 2 `
  --workers 4 `
  --max-depth 1 `
  --delay-seconds 0.5
```

The active trust policy caps each source at 20%, which is stricter than the
old 25% command-line cap. The command creates no crawl output unless all
source approvals, evidence files, reviewer identities, and protected holdouts
validate first.

Current Day 6 status is intentionally `blocked`: the protected pack passes,
but the selected eight-source registry still has zero approved sources and the
reviewer roster has zero identities. A real legal operator must bind all eight
immutable source approvals, and governance must configure two independent
reviewers plus one independent adjudicator. No calibration work directory or
quality result may be interpreted as passed before those dependencies exist.

## Scratch Learning Validation Layer

Purpose:

- detect whether a scratch checkpoint actually learns instead of merely running
- catch tokenizer collapse such as dot/newline/space dominance
- prove the optimizer/model/data path can overfit a controlled high-signal corpus
- replace the 5-prompt smoke comparison with a 50-200 prompt coding/security suite
- provide a non-tiny T4 validation profile before expensive 7B+ cluster runs

Main module:

- `src/craftly/model_ops/learning_validation.py`

What it builds:

- `InstructionRecord` schema with `prompt`, `context`, `answer`, `code_patch`, `tests`, and `verification`
- deterministic controlled instruction corpus for defensive security, agentic coding, debugging, patch generation, and repository reasoning
- tokenizer dominance report with total tokens, top tokens, dot fraction, quote fraction, whitespace fraction, newline fraction, unknown-token rate, single-character token rate, special-token checks, sample efficiency, and code/security pattern coverage
- tokenizer audit Markdown report for quick human inspection
- overfit sanity report with first/final/best loss and required relative loss drop
- expanded checkpoint comparison suite compatible with `run_checkpoint_comparison.py`
- T4 validation command using the real scratch profile `t4_validation`

Run controlled validation:

```bash
python -m src.craftly.model_ops.learning_validation \
  --output-dir artifacts/craftly/learning-validation-v1 \
  --instruction-count 200 \
  --overfit-steps 300 \
  --device cuda \
  --dtype fp16
```

Fast local check:

```bash
python -m src.craftly.model_ops.learning_validation \
  --output-dir artifacts/craftly/learning-validation-smoke \
  --instruction-count 50 \
  --skip-overfit \
  --device cpu
```

Interpretation:

- if tokenizer audit fails, fix tokenizer/data before training longer
- if overfit sanity fails, do not spend serious GPU time yet
- if overfit passes but real-data eval still emits repetitive punctuation, improve instruction-style data mix and generation settings
- if expanded eval remains near zero, the model is still too small or undertrained for reasoning quality
- if checkpoint comparison returns `failed_generation_collapse`, the model is repeating patterns above the allowed threshold and must not be promoted

T4 validation profile:

- model profile: `t4_validation`
- hidden size: 512
- layers: 8
- sequence length target: 256 for Kaggle run, max model context 2048
- gradient checkpointing: enabled
- intended run: 1k-10k steps on Kaggle/Colab T4/L4/A100 after overfit sanity passes

Expanded comparison command:

```bash
python deploy/gpu/run_checkpoint_comparison.py \
  --training-report artifacts/craftly/real-data-validation-v1/reports/real_data_training_report.json \
  --prompt-suite artifacts/craftly/learning-validation-v1/expanded_eval_suite.jsonl \
  --output-dir artifacts/craftly/real-data-validation-v1/reports/checkpoint_compare_expanded \
  --device cuda \
  --repetition-penalty 1.18 \
  --no-repeat-ngram-size 4 \
  --max-repetition-ratio 0.72
```

The deterministic comparison path now supports:

- `repetition_penalty`
- `no_repeat_ngram_size`
- stop tokens
- generation collapse detection
- fail-fast comparison status when repetitive output exceeds the threshold

The real-data pipeline writes tokenizer audit artifacts at:

- `reports/tokenizer_audit_report.json`
- `reports/tokenizer_audit_report.md`

## Curriculum-First Scratch Training

Purpose:

- reduce hallucination and generation collapse by training one capability band
  at a time
- start with stable code/security language before mixing agentic workflows
- keep offensive-security material out of the defensive phase

Supported curriculum modes:

- `fundamentals_only`
- `defensive_security_only`
- `debug_patch_only`
- `agentic_coding_only`
- `balanced`

Implementation:

- `src/craftly/learning/mixer.py`
- `src/craftly/model_ops/learning_validation.py`
- `src/craftly/model_ops/checkpoint_compare.py`

Defensive-only data rules:

- keep only defensive security/coding instruction rows
- reject offensive misuse rows containing patterns such as reverse shells,
  shellcode, credential dumping, exfiltration, C2 callbacks, exploit payload
  instructions, or EDR/AV bypass wording
- convert accepted rows into prompt/context/answer/patch/tests/verification
  training text

Defensive hallucination checks:

- if evidence is missing, output must say it cannot confirm or needs more context
- generated CVE IDs are forbidden unless the prompt already includes that CVE
- claims such as "tests passed" are forbidden unless verification evidence is
  present in the prompt
- exploit steps are forbidden in defensive eval

Generate defensive-only validation assets:

```bash
python -m src.craftly.model_ops.learning_validation \
  --output-dir artifacts/craftly/defensive-learning-validation-v1 \
  --instruction-count 100 \
  --curriculum-mode defensive_security_only \
  --overfit-steps 300 \
  --device cuda \
  --dtype fp16
```

This writes:

- `instruction_corpus.jsonl`
- `expanded_eval_suite.jsonl` with 100 defensive prompts
- `tokenizer_dominance_report.json`
- `tokenizer_dominance_report.md`
- `learning_validation_report.json`
- staged commands for small overfit, 1k Kaggle validation, and 10k Kaggle
  validation

Kaggle defensive 1k validation:

```bash
PYTHONUNBUFFERED=1 python -u deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --work-dir artifacts/craftly/defensive-validation-1k-v1 \
  --kaggle-validation \
  --curriculum-mode defensive_security_only \
  --model-profile t4_validation \
  --checkpoint-compare-prompt-suite artifacts/craftly/defensive-learning-validation-v1/expanded_eval_suite.jsonl \
  --checkpoint-compare-repetition-penalty 1.18 \
  --checkpoint-compare-no-repeat-ngram-size 4 \
  --checkpoint-compare-max-repetition-ratio 0.72 \
  --steps 1000 \
  --sequence-length 128 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --validation-interval 100 \
  --validation-batches 8 \
  --early-stopping-patience 5 \
  --gradient-checkpointing \
  --progress-to-stdout
```

Kaggle defensive 10k validation:

```bash
PYTHONUNBUFFERED=1 python -u deploy/gpu/run_real_data_training_pipeline.py \
  --sources config/data_sources.ultimate.json \
  --work-dir artifacts/craftly/defensive-validation-10k-v1 \
  --kaggle-validation \
  --curriculum-mode defensive_security_only \
  --model-profile t4_validation \
  --checkpoint-compare-prompt-suite artifacts/craftly/defensive-learning-validation-v1/expanded_eval_suite.jsonl \
  --checkpoint-compare-repetition-penalty 1.18 \
  --checkpoint-compare-no-repeat-ngram-size 4 \
  --checkpoint-compare-max-repetition-ratio 0.72 \
  --steps 10000 \
  --sequence-length 256 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --validation-interval 250 \
  --validation-batches 8 \
  --early-stopping-patience 12 \
  --gradient-checkpointing \
  --progress-to-stdout
```

Promotion rule:

- small overfit sanity must pass before spending long GPU time
- 1k defensive validation must pass before 10k
- 10k checkpoint must not regress, collapse, invent CVEs, claim unverified test
  success, or produce exploit steps

## Strict Scanner Bootstrap

Security audit reports now include a scanner install plan. For local Windows
setup:

```powershell
python -m pip install --upgrade bandit semgrep pip-audit
winget install --id GitHub.CodeQL
codeql database create artifacts/craftly/codeql-db --language=python --source-root=.
python -m src.craftly.security.audit --strict-external-tools --output-dir artifacts\\craftly\security-audit
```

Strict mode fails when required scanner tools are missing or scanner findings
fail policy.

## Security Audit Production Behavior

Module:

- `src/craftly/security/audit.py`

Dev behavior:

- missing Bandit/Semgrep/CodeQL/pip-audit is reported as `skipped`
- skipped optional tools do not fail local dev release gates

Production behavior:

```powershell
python -m src.craftly.security.audit --strict-external-tools --output-dir artifacts\\craftly\security-audit
```

In strict mode, missing required scanner CLIs become release blockers. Critical
findings, dependency warnings, failed scanner output, and failed Kubernetes
validation block release.

Every external scanner is bounded so CI cannot hang indefinitely. Defaults are
600 seconds for Bandit and pip-audit, 900 seconds for Semgrep, and 1800 seconds
for each CodeQL phase. Operators can set
`CRAFTLY_BANDIT_TIMEOUT_SECONDS`, `CRAFTLY_SEMGREP_TIMEOUT_SECONDS`, and
`CRAFTLY_PIP_AUDIT_TIMEOUT_SECONDS` to values from 30 through 3600 seconds.
A timeout is a failed audit with explicit evidence, never a skipped or fake
pass.

## Verification Commands

Use these after major changes:

```powershell
python -m compileall -q src\craftly tests deploy\gpu
python -m unittest tests.test_craftly_data_engine tests.test_craftly_production_hardening tests.test_craftly_training_control tests.test_craftly_enterprise_readiness
python -m src.craftly.deployment.k8s_validate --output-dir artifacts\\craftly\k8s-validation
python -m src.craftly.learning.storage --uri local://artifacts/craftly/object-store --work-dir artifacts\\craftly\object-store-lifecycle
python -m src.craftly.security.audit --no-bandit --output-dir artifacts\\craftly\security-audit
python -m src.craftly.evaluation.release_gate
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_craftly_consolidated_smoke.ps1
```

## Training Workspace And Scale Control Plane

### Purpose

The Training Workspace is the operational control plane between a researcher
and scratch pretraining compute. It solves three separate problems without
mixing their responsibilities:

1. The client submits a validated training profile and follows progress.
2. The control plane owns identity, durable state, scheduling, audit, and
   artifact promotion.
3. Trusted workers own data processing, model computation, checkpoint writing,
   and measured evaluation.

Kaggle and Colab are validation clients. They are not represented as 7B-60B
production training infrastructure. Large synchronous pretraining belongs on a
trusted Kubernetes or Slurm cluster with pinned containers and private service
connectivity.

No external model weights or post-training adaptation path is introduced. The
workspace accepts Craftly scratch pretraining, evaluation, and checkpoint
lifecycle jobs only.

### Canonical Components

- `src/craftly/training_workspace.py`: schemas, state machine, Postgres and
  in-memory stores, Redis event bus, scheduler adapters, controller, event
  archiver, checkpoint/evaluation promotion, and readiness.
- `src/craftly/training_client.py`: async SDK, short-lived token refresh, SSE
  replay, and CLI.
- `src/craftly/shared/progress.py`: non-blocking worker event batching,
  heartbeat, coalescing, local WAL, retry, and WAL replay.
- `config/training_profiles.json`: append-only versioned profile contracts.
- `deploy/gpu/run_workspace_validation.py`: direct-kernel notebook validation
  launcher.
- `apps/training-workspace`: React and TypeScript workspace UI.
- `src/craftly/db/migrations/0004_training_workspace.sql`: durable control
  plane schema.
- `deploy/k8s/training-workspace.yaml`: controller identity, RBAC, network
  policy, and deployment.
- `deploy/k8s/training-workspace-ui.yaml`: hardened workspace UI deployment.
- `deploy/k8s/training-monitoring.yaml`: DCGM GPU telemetry.

### Request Flow

```text
SDK / CLI / Web UI
    -> bootstrap exchange
    -> 15-minute scoped access token + <=12-hour refresh session
    -> immutable profile resolution
    -> idempotent job creation
    -> Postgres state transition
    -> controller validation
    -> scheduler submission
    -> trusted worker
    -> Redis live events + local WAL fallback
    -> S3/MinIO durable artifacts
    -> checkpoint reload proof
    -> evaluation gate
    -> audited checkpoint promotion
```

The notebook receives only `CRAFTLY_WORKSPACE_URL` and
`CRAFTLY_BOOTSTRAP_TOKEN`. It never receives Postgres, Redis, Qdrant, or
object-store administration credentials. Cluster workers receive a job-bound
token through a mounted Secret or a mode-0600 Slurm token file. The controller
rotates worker credentials before expiry.

### Immutable Profiles

The registry currently includes:

- `defensive-1k`: direct notebook validation.
- `defensive-10k`: serious single-GPU Kubernetes validation.
- `fundamentals-validation`: coding-fundamentals notebook validation.
- `craftly-1b-single-node`: single-node cluster proof.
- `craftly-7b-fsdp`: multi-node native FSDP.
- `craftly-32b-zero3`: multi-node DeepSpeed ZeRO-3.
- `craftly-60b-hybrid`: Slurm and hybrid/Megatron target.
- `craftly-4t-moe`: immutable 4T-total/128B-active Megatron target with
  TP8/PP12/DP16/CP4/EP16 over 6,144 GPUs and a 30T governed-token exposure
  target.

A profile version is immutable. Postgres uses `(profile_id, version)` as the
identity and stores its SHA-256. Synchronization fails if content changes
without a version bump. A job stores the exact profile hash, spec hash, Git
commit, image digest, tokenizer hash, and dataset manifest hash.

Token accounting uses only independent data-parallel replicas. TP, PP, CP and
EP partition one model execution and do not multiply consumed training tokens.
This prevents large model-parallel profiles from silently overstating their
global batch or token budget.

Allowed overrides have explicit integer ranges. A client cannot submit an
executable, shell fragment, environment secret, mount, image, or scheduler
argument. `build_training_command()` creates argv from validated schema fields
only.

Production mode takes the Git commit and image digest from trusted server
configuration:

```text
CRAFTLY_TRAINING_GIT_COMMIT
CRAFTLY_TRAINING_IMAGE_DIGEST
```

Zero/placeholder values block submission. Images are scheduled by digest, not
mutable tag.

### Job State Machine

```text
validating -> queued -> provisioning -> running
running -> checkpointing -> running
running/checkpointing -> evaluating -> running/succeeded

terminal: succeeded | failed | blocked | cancelled
resumable: infrastructure failed or cancelled -> queued
not resumable: blocked
```

`blocked` is used for data, tokenizer, non-finite-loss, quality, compatibility,
or security failures. Automatic resume is deliberately unavailable for these
failures. `failed` is reserved for infrastructure/runtime failure. Resume
requires a promoted checksum-verified checkpoint.

Every transition uses optimistic version checks. Concurrent writers cannot
silently overwrite state. Attempt state, start time, and finish time follow the
job state. Every create, scheduler submit/failure, transition, cancel, resume,
checkpoint, evaluation, and promotion event is written to the audit table.

### Postgres Schema

- `training_profiles`: immutable versioned profile JSON and hash.
- `training_jobs`: canonical spec, optimistic version, state, active attempt,
  scheduler binding, failure classification, and event cursors.
- `training_attempts`: scheduler attempt and resume checkpoint.
- `training_event_ingress`: `(attempt, rank, source_sequence)` deduplication.
- `training_artifacts`: verified object URI, size, SHA-256, kind, and promotion
  status.
- `checkpoint_versions`: step, immutable hashes, topology, metrics, reload
  proof, and promotion state.
- `evaluation_runs`: checkpoint decision and verified report.
- `service_accounts`: scoped identity and bootstrap token hash.
- `service_account_sessions`: expiring/revocable refresh sessions.
- `training_audit_events`: actor, action, outcome, request, and redacted
  metadata.

Migrations run through the existing migration runner or Alembic chain. The
database is the source of truth. Redis is never used as durable job state.

### Live Event Protocol

Workers emit versioned events with job, attempt, source sequence, rank,
world-size, node, stage, status, step, loss, validation loss, tokens/second,
and GPU memory.

Delivery behavior:

- up to 25 events or one second per batch;
- five-second heartbeat even when the trainer emits nothing;
- bounded in-memory queue;
- metric updates coalesce to their latest value under pressure;
- error, checkpoint, and evaluation events are persisted to WAL rather than
  dropped;
- failed HTTP delivery writes fsync-backed JSONL WAL;
- exponential retry uses jitter;
- restart-safe source sequences and transactional server deduplication;
- WAL replay is safe after an ambiguous network response;
- rank zero emits global metrics; non-zero ranks retain heartbeat, checkpoint,
  and fatal diagnostics.

The SSE endpoint accepts `Last-Event-ID`. SDK and web clients reconnect from
the last acknowledged sequence. The API sends `X-Accel-Buffering: no`; the UI
Nginx proxy disables response buffering. Redis holds the hot stream while the
archiver writes ordered gzip JSONL chunks to object storage.

### Artifact And Checkpoint Lifecycle

Workers request short-lived S3/MinIO presigned PUT URLs. The request binds the
object path, content type, size declaration, and SHA-256 metadata. The notebook
never sees permanent object-store credentials.

Promotion sequence:

```text
write shards and temporary checkpoint prefix
  -> upload manifest through a presigned request
  -> server HEAD size/SHA verification
  -> checkpoint commit
  -> dataset/tokenizer hash comparison with job spec
  -> reload smoke proof
  -> verified evaluation artifact commit
  -> pass/release decision
  -> admin promotion
  -> atomic jobs/{job}/checkpoints/latest.json update
```

Validation-only notebook checkpoints cannot become release checkpoints. A
checkpoint cannot promote without both `reload_verified=true` and a committed
passing evaluation. Promotion is an audited `training:admin` operation.

### Scheduler Safety

`NotebookValidationAdapter` supports one node and no distributed strategy. It
is explicitly validation-only.

`KubernetesSchedulerAdapter` creates a structured `batch/v1 Job`. Before
submission it checks the Kubernetes client, node readiness, GPU count, GPU
memory label, and RDMA label. Nodes must be labeled:

```bash
kubectl label node GPU_NODE craftly.ai/gpu-memory-gib=80
kubectl label node GPU_NODE craftly.ai/rdma=true
```

`KubernetesPyTorchAdapter` additionally requires the Kubeflow Training Operator
and its `PyTorchJob` CRD. Multi-node profiles fail before provisioning when the
operator or sufficient eligible nodes are absent.

Worker pods use:

- pinned image digest;
- non-root UID/GID 10001;
- runtime-default seccomp;
- no privilege escalation;
- all Linux capabilities dropped;
- read-only root filesystem;
- bounded memory-backed `/tmp`;
- node-local `/workspace` cache;
- profile-owned Secret references only;
- job-scoped token mounted read-only;
- explicit CPU, memory, and GPU requests/limits.

`SlurmSchedulerAdapter` uses argv-only `sbatch`, `sacct`, `sinfo`, and
`scancel`. Generated scripts use `set -euo pipefail`; no `eval` exists. The
preflight requires enough nodes with the requested GPU GRES and features
`gpu-memory-<N>g,rdma`. Slurm production workers require an HTTPS workspace
URL.

### Identity And Authorization

Roles are represented by scopes:

- `training:jobs:create`
- `training:jobs:read`
- `training:jobs:cancel`
- `training:events:write`
- `training:artifacts:write`
- `training:artifacts:read`
- `training:admin`

Generic API scope does not grant training scope. Non-admin users see only jobs
they own. Worker tokens are bound to a single `job_id` claim. Event payloads
are limited to 64 KiB, batches to 100, and known secret keys/values are
redacted before Redis, audit, or object storage.

The initial environment bootstrap credential is materialized as a reserved
hashed service account before a refresh session is issued. Rotation updates
the hash and revokes its existing refresh sessions. Explicit logout/revocation
is supported. Production ingress must use HTTPS.

### APIs

```text
POST /v1/training/token/exchange
POST /v1/training/token/refresh
POST /v1/training/token/revoke
GET  /v1/training/profiles
POST /v1/training/jobs
GET  /v1/training/jobs
GET  /v1/training/jobs/{job_id}
POST /v1/training/jobs/{job_id}/claim
POST /v1/training/jobs/{job_id}/cancel
POST /v1/training/jobs/{job_id}/resume
GET  /v1/training/jobs/{job_id}/events
POST /v1/training/jobs/{job_id}/events:batch
GET  /v1/training/jobs/{job_id}/artifacts
POST /v1/training/jobs/{job_id}/artifacts/presign
POST /v1/training/jobs/{job_id}/artifacts/register
GET/POST /v1/training/jobs/{job_id}/checkpoints
GET/POST /v1/training/jobs/{job_id}/evaluations
POST /v1/training/jobs/{job_id}/checkpoints/{checkpoint_id}/promote
GET  /v1/training/jobs/{job_id}/audit
```

### Notebook Operation

Place only these values in Kaggle/Colab Secrets:

```text
CRAFTLY_WORKSPACE_URL=https://workspace.example.com
CRAFTLY_BOOTSTRAP_TOKEN=<bootstrap-value>
```

Clone the repository once, then use direct-kernel execution:

```python
%run -i deploy/gpu/run_workspace_validation.py --profile defensive-1k
```

`%run -i` avoids a nested buffered shell process. The same kernel displays
each progress line immediately. If workspace secrets are absent, the launcher
runs standalone validation and labels that condition in output.

SDK:

```python
from craftly_client import Workspace

workspace = Workspace.from_environment()
run = await workspace.train(profile="defensive-1k", follow=True)
```

CLI:

```bash
python -m pip install -e . --no-deps
craftly train --profile defensive-1k --follow
craftly jobs list
craftly jobs inspect JOB_ID
craftly jobs cancel JOB_ID
craftly jobs resume JOB_ID
```

When the Python Scripts directory is not on `PATH`, use
`python -m src.craftly.training_client` with the same arguments.

### Deployment

Local integration proof:

```bash
cp deploy/prod/.env.example deploy/prod/.env
docker compose --env-file deploy/prod/.env -f deploy/prod/docker-compose.yml --profile training up --build
```

The API is on port 8090 and UI on port 8088. Example/default secrets are
invalid for production and must be replaced.

Kubernetes:

```bash
kubectl apply -f deploy/k8s/postgres-redis.yaml
kubectl apply -f deploy/k8s/minio.yaml
kubectl apply -f deploy/k8s/api.yaml
kubectl apply -f deploy/k8s/training-workspace.yaml
kubectl apply -f deploy/k8s/training-workspace-ui.yaml
kubectl apply -f deploy/k8s/training-monitoring.yaml
```

Secret examples are templates, not deployable credentials. Use External
Secrets, Vault, cloud workload identity, or another approved secret manager in
the real cluster.

### Observability

Control-plane metrics include HTTP rate/latency, job submissions/transitions,
event ingestion and deduplication, training/validation loss, token throughput,
checkpoint commits/promotions, and evaluation decisions. Histogram storage is
constant-memory; the process does not retain every observation.

DCGM Exporter exposes GPU utilization, memory, and temperature. The Kubernetes
Service carries Prometheus scrape annotations. The Grafana production
dashboard includes jobs, loss, throughput, and GPU utilization.

Alerts still require the production Prometheus/Alertmanager policy to be
installed and tested. Missing heartbeat, non-finite loss, OOM, NCCL timeout,
data starvation, checkpoint failure, dependency outage, and evaluation
regression are the required alert classes.

### Honest Readiness

Locally tested code is not cluster proof. Status rules:

- notebook direct-kernel path: `validation_ready`;
- configured Postgres/Redis/S3 workspace: `production_ready_requires_external_service`;
- Kubernetes/PyTorchJob/Slurm profiles before live proof: `built_not_cluster_proven`;
- missing dependency or placeholder image/commit: `blocked_missing_dependency`.

The following cannot be honestly completed on this Windows workstation:

- 24-hour single-GPU soak;
- 10k-step T4/L4/A100 proof;
- two-GPU FSDP save/reload;
- multi-node node-loss recovery;
- one-million-event Redis interruption test;
- Postgres failover;
- MinIO checkpoint interruption;
- 7B, 32B, or 60B scratch pretraining.

The code paths and fail-fast gates exist, but these statuses remain
`built_not_cluster_proven` until the corresponding infrastructure test report
is produced.

### Production Training Policy Contract

`config/training_profiles.json` schema version 3 is the canonical, append-only
training policy registry. JSON `defaults` reduce duplication, but
`TrainingProfileRegistry.from_file()` deep-resolves every profile before
validation and hashing. The immutable hash therefore covers the complete
resolved policy rather than a reference to mutable defaults.

Every profile now binds all of the following:

- AdamW learning rate, beta values, epsilon, weight decay, and gradient clip;
- constant, linear, or cosine schedule with exactly one warmup mode;
- global batch sequences and non-padding target/maximum token budgets;
- checkpoint interval, retained latest/best counts, optimizer/scheduler/RNG
  state, checksum, and reload requirements;
- evaluation interval, suite, strictness, early stopping, and regression limit;
- hot/archive/delete retention and promoted-checkpoint preservation;
- promotion evidence, minimum step, and minimum evaluation count;
- retryable failure classes, exponential backoff, preemption handling, and
  maximum attempts;
- wall time, heartbeat timeout, and graceful termination;
- deterministic dataloader seed, bounded prefetch, workers, checksum policy,
  pinning, and node-local cache budget;
- OCI image repository plus required Python, PyTorch, and CUDA versions;
- namespace/partition, queue, account, storage class, priority, gang
  scheduling, and topology key;
- maximum GPU-hours, projected USD cost, storage, egress, and owner concurrency;
- promoted dataset and tokenizer URI/hash requirements.

Resolution recalculates global batch and target tokens whenever an authorized
step/node override changes topology. Production submissions reject unpromoted
datasets, disallowed URI schemes, mutable image identities, excessive
concurrency, GPU-hour requests above profile quota, and projected GPU cost
above profile quota. Projected cost is the maximum wall time multiplied by the
requested GPU count and the profile's audited per-GPU-hour estimate. These values are passed
to the real pretraining process. The runtime uses configured AdamW and
warmup/decay behavior, reports current learning rate, validates expected
runtime versions, prefetches shard batches, and fails if consumed tokens exceed
the immutable budget.

`steps` is defined as completed optimizer updates on every runtime. The single
canonical contract is `global_batch_sequences = micro_batch_size *
gradient_accumulation_steps * data_parallel_size` and `target_tokens =
optimizer_steps * sequence_length * global_batch_sequences`. Each optimizer
update consumes exactly `gradient_accumulation_steps` microbatches per
data-parallel rank. FSDP and DeepSpeed ranks receive deterministic,
non-overlapping partitions, while distributed validation aggregates loss across
all ranks. Checkpoints are emitted only at optimizer boundaries and bind the
exact epoch/batch cursor, data-parallel topology, scheduler, optimizer, and RNG
state. A checkpoint without the `optimizer_update_v2` contract is rejected by
the production resume path instead of being interpreted ambiguously.

Token-shard schema v2 adds one `<|document_end|>` token per accepted source
document and records train/validation document counts plus the boundary token
ID in the immutable shard manifest. Production training verifies these counts
and rejects legacy unbounded concatenation. This prevents the causal objective
from treating the end of one unrelated repository record and the beginning of
another as a genuine language transition.

### Gated Qualification Staircase

`config/training_qualification_campaigns.json` defines compact ranges rather
than dozens of copied profiles. `defensive-staircase-v1` expands to 37
milestones:

1. 1k, 2k, ..., 20k;
2. 30k, 40k, ..., 100k;
3. 200k, 300k, ..., 1M.

The first ten milestones use notebook validation. Later milestones require the
trusted Kubernetes scheduler. A create request must bind both campaign and
milestone and set the exact milestone step override. For every later
milestone, the service queries durable prior jobs and requires a succeeded
predecessor, reload-verified checkpoint, passing evaluation, <=3% validation
regression, and identical dataset/tokenizer hashes. User-supplied metadata
cannot bypass this gate.

Example:

```powershell
craftly train --profile defensive-staircase-notebook --steps 1000 `
  --campaign defensive-staircase-v1 --milestone steps-0001000 `
  --dataset-manifest-uri file:///data/manifest.json `
  --dataset-manifest-sha256 <sha256> `
  --tokenizer-uri file:///data/tokenizer.json --tokenizer-sha256 <sha256>
```

This validates one model/config progressively. It does not mean that 1M steps
automatically qualifies a 1B or 60B model. Parameter-scale changes begin a new
checkpoint lineage because their tensors are incompatible.

### Measured Infrastructure Proofs

`src/craftly/training_proofs.py` writes a machine-readable report where every
proof is `passed`, `failed`, or `blocked`. `deploy/proof/docker-compose.yml`
provides isolated pinned Postgres, Redis, and MinIO dependencies.

The local proof performs real migrations, stores a job and attempt in
Postgres, checks event sequence deduplication through Redis Streams, executes
MinIO upload/head/download/list/delete with SHA-256, kills a real Docker worker
and verifies node-loss retry/backoff, dumps/restores Postgres, restarts
Postgres/Redis/MinIO, and verifies persistence after restart.

```powershell
docker compose -p craftly-proof -f deploy\proof\docker-compose.yml up -d
python -m src.craftly.training_proofs `
  --output-dir artifacts\craftly\production-proofs\local-docker
```

Full ordered-event and soak commands:

```powershell
python -m src.craftly.training_proofs --event-count 1000000 `
  --skip-disaster-recovery --output-dir artifacts\craftly\production-proofs\million-events
python -m src.craftly.training_proofs --soak-seconds 86400 `
  --output-dir artifacts\craftly\production-proofs\24-hour-soak
```

The local full stress proof executed on 2026-07-16 accepted exactly 1,000,000
ordered events, verified both final and tail sequence as 1,000,000, and measured
1,285.64 events/second with no failed proof. This proves the tested local
Postgres/Redis path; it is not a multi-region or cluster throughput claim. A
30-second soak harness smoke also completed 15 dependency health cycles. The
required 24-hour soak remains unproven until the exact 86,400-second command
finishes without interruption.

Kubernetes/PyTorchJob, Slurm, FSDP, ZeRO-3, Megatron, and 60B resume checks
probe their real dependencies. A missing cluster or GPU allocation is an
explicit blocker. Running a local mock does not promote those proof states.

### Distributed Execution And Checkpoint Proof Contract

`src/craftly/model_ops/distributed_worker.py` is the fixed cluster entrypoint.
It converts Kubernetes replica metadata or Slurm rank variables into canonical
PyTorch rank variables without accepting an arbitrary shell command. A
PyTorchJob replica launches one process per visible GPU. Slurm validates
`SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, master address, and port before
starting either native Craftly pretraining or the pinned Megatron entrypoint.

FSDP checkpoints no longer gather a full model on rank zero. They use PyTorch
Distributed Checkpoint sharded model and optimizer state, per-rank RNG state,
trainer and scheduler state, checksums, an incomplete staging directory, and
an atomic final-directory commit on shared storage. Resume restores the
sharded state and verifies step/token state before training continues.
DeepSpeed uses its native sharded checkpoint client state. Megatron requires a
real `pretrain_gpt.py`, indexed `.bin/.idx` data under the immutable manifest
root, topology divisibility, distributed optimizer state, and `torch_dist`
checkpoint format.

Workers materialize `file://` or `s3://` dataset manifests, tokenizers, and
shards through a bounded local artifact cache. SHA-256 mismatch, path escape,
missing object, or tokenizer/model mismatch blocks the run before optimization.
Native distributed checkpoints are uploaded file by file to workspace object
storage, registered transactionally, committed only after checksum validation,
and marked reload-verified only after an actual reload succeeds.

Real Kubernetes FSDP and ZeRO-3 proof commands:

```powershell
$common = @(
  "--skip-infrastructure", "--skip-disaster-recovery", "--skip-capability-probes",
  "--dataset-manifest-uri", $env:CRAFTLY_PROOF_DATASET_URI,
  "--dataset-manifest-sha256", $env:CRAFTLY_PROOF_DATASET_SHA256,
  "--tokenizer-uri", $env:CRAFTLY_PROOF_TOKENIZER_URI,
  "--tokenizer-sha256", $env:CRAFTLY_PROOF_TOKENIZER_SHA256,
  "--git-commit", $env:CRAFTLY_TRAINING_GIT_COMMIT,
  "--container-digest", $env:CRAFTLY_TRAINING_IMAGE_DIGEST,
  "--live-timeout-seconds", "86400",
  "--inject-worker-loss"
)
python -m src.craftly.training_proofs --live-profile craftly-7b-fsdp @common
python -m src.craftly.training_proofs --live-profile craftly-32b-zero3 @common
```

The proof submits a real immutable job, watches the state machine, requires
heartbeats and a reload-verified checkpoint, deletes one Kubernetes worker only
after that checkpoint exists, and then requires post-injection events and a
successful terminal state. `blocked` means the cluster, CRD, image, data, GPU,
or service dependency was unavailable. It is never rewritten as `passed`.

Real Slurm/Megatron 60B proof:

```powershell
$env:CRAFTLY_MEGATRON_ROOT = "/cluster/software/Megatron-LM"
$env:CRAFTLY_SHARED_CHECKPOINT_ROOT = "/cluster/checkpoints/craftly"
python -m src.craftly.training_proofs --live-profile craftly-60b-hybrid @common
```

For the 60B profile, dataset and tokenizer URIs must resolve to immutable
cluster-mounted files. The manifest binds a `megatron_data_prefix` whose
`.bin/.idx` files remain under the manifest directory. The proof waits for a
multi-file Megatron checkpoint, runs a real `scontrol requeue`, then requires
new events and successful completion. Promotion remains
`built_not_cluster_proven` until this exact run succeeds on the target Slurm
fabric and a subsequent resume reads the same checkpoint.

### Disaster Recovery And Soak Proofs

The 24-hour proof performs real Postgres transactions, Redis write/read/delete,
and MinIO write/read/checksum/delete on every cycle. It records availability,
error-budget consumption, consecutive failures, and p50/p95/p99/max latency.
It passes only after the full 86,400 seconds elapse:

```powershell
python -m src.craftly.training_proofs --soak-seconds 86400 `
  --soak-interval-seconds 30 --skip-capability-probes `
  --output-dir artifacts\craftly\production-proofs\24-hour-soak
```

Local DR proves Postgres dump/restore, Redis AOF recovery, MinIO volume
recovery, and worker-loss retry using isolated Docker services. The live
Kubernetes drill is deliberately disruptive and must be explicitly enabled:

```powershell
python -m src.craftly.training_proofs --skip-infrastructure `
  --skip-disaster-recovery --skip-capability-probes `
  --inject-kubernetes-disaster-recovery `
  --dr-workload default:statefulset:craftly-postgres `
  --dr-workload default:statefulset:craftly-redis `
  --dr-workload default:statefulset:craftly-minio `
  --dr-workload craftly-training:deployment:craftly-training-controller
```

Before disruption it persists a job in Postgres, a sequenced checkpoint event
in Redis Streams, and a checksummed marker in S3/MinIO. Each declared rollout
must recover before the next starts. Fresh clients then verify the exact job,
event, and object checksum. Provider-managed regional failover still requires
that provider's runbook and cannot be truthfully proved by repository code
alone.

### Strict CodeQL Gate

The security audit resolves the installed official CodeQL bundle, builds a
fresh Python database from `src`, runs `python-security-extended.qls`, parses
SARIF, and blocks on every result. Scanner timeout, database creation failure,
analysis failure, or malformed SARIF also blocks. Deployment YAML is evaluated
separately by the Kubernetes policy audit.

```powershell
$env:CRAFTLY_CODEQL_BIN = "$HOME\.craftly\tools\codeql\codeql\codeql.exe"
python -m src.craftly.security.audit `
  --strict-external-tools --output-dir artifacts\craftly\security-audit
```

Do not suppress a CodeQL result merely to make the gate green. Fix the sink or
record an independently reviewed, narrowly scoped exception in the governed
security process.

### Workspace Verification

```powershell
python -m compileall -q src\craftly tests deploy\gpu
python -m unittest tests.test_craftly_training_workspace
python -m src.craftly.training_workspace profiles
python -m src.craftly.training_workspace readiness
python -m src.craftly.evaluation.release_gate
python -m src.craftly.security.audit --output-dir artifacts\craftly\security-audit
python -m src.craftly.deployment.k8s_validate --output-dir artifacts\craftly\k8s-validation
```

Frontend:

```powershell
cd apps\training-workspace
npm ci
npm run build
```

### Governed 50-Task Checkpoint Qualification

`src/craftly/evaluation/qualification_campaign.py` is the single authoritative
control path for the real-repository holdout, current-checkpoint baseline,
defensive curriculum stages, checkpoint comparison, and scale admission. It
does not duplicate repository execution or pretraining. It calls the existing
`agent_scorecard`, `checkpoint_compare`, `checkpoint_eval`,
`audit_tokenizer_dominance`, and `run_pretraining_loop` contracts.

#### Why This Layer Exists

A prompt answer, a historical vulnerability, and a repository-verified patch
are three different facts. Earlier score reports could measure prompt quality,
but a prompt score alone cannot prove compilation, developer tests, or
security-test success. The qualification layer preserves these boundaries:

- historical provenance proves that the task came from a pinned upstream
  benchmark and fixing commit;
- prompt evaluation measures checkpoint output, hallucination, and repetition;
- repository evidence measures official security testcase and developer unit
  tests;
- missing repository evidence is `blocked_missing_evidence`, never zero and
  never a pass;
- benchmark ground truth is a permanent holdout and cannot enter tokenizer,
  sharding, or training inputs.

#### Source Governance

`config/repository_qualification_sources.json` pins SecRepoBench to an exact
commit and declares required source files, HTTPS repository policy, and
evaluation-only use. Pack generation checks Git HEAD and SHA-256 for every
required benchmark artifact.

First generate a local approval template:

```powershell
python -m src.craftly.evaluation.qualification_campaign approval-template `
  --source-root D:\benchmarks\SecRepoBench `
  --source-id secrepobench-318 `
  --registry config\repository_qualification_sources.json `
  --output config\local\secrepobench-approval.json
```

The command deliberately writes `decision: pending`. Repository code cannot
grant legal or license approval. An authorized reviewer must inspect the source
license, intended use, benchmark policy, privacy risk, and redistribution
constraints before setting `decision: approved`. The approval binds the source
URL, pinned commit, intended evaluation-only use, reviewer, timestamp,
rationale, and exact file hashes. A later file change invalidates it.

Build the pack only after approval:

```powershell
python -m src.craftly.evaluation.qualification_campaign build-pack `
  --source-root D:\benchmarks\SecRepoBench `
  --approval config\local\secrepobench-approval.json `
  --output-dir artifacts\craftly\qualification-pack
```

The deterministic selection contains exactly:

| Category | Count |
|---|---:|
| coding | 10 |
| debugging | 10 |
| defensive_security | 10 |
| patch_generation | 10 |
| long_context | 10 |

Only records whose vulnerable testcase crashes, whose developer fix passes the
official security testcase, whose fixing commit is a full hash, and whose
upstream repository uses an allowed HTTPS host are eligible. Prompts contain
pre-fix context and crash class, but never the fixing diff, fixed source, fixing
commit, or target CVE. The oracle is represented only by a SHA-256 digest in the
task catalog.

Artifacts:

```text
qualification_pack_manifest.json
checkpoint_prompts.jsonl
repository_task_catalog.jsonl
qualification_pack_report.md
```

The manifest binds both JSONL files by SHA-256. Production readiness requires
this governed pack and an exact-task strict repository report; an unrelated
50-task report cannot satisfy the gate.

#### Official Repository Evidence

After Craftly completions are executed by the pinned SecRepoBench ARVO/Docker
harness, import the official nested result:

```powershell
python -m src.craftly.evaluation.qualification_campaign import-evidence `
  --pack-manifest artifacts\craftly\qualification-pack\qualification_pack_manifest.json `
  --benchmark-source-root D:\benchmarks\SecRepoBench `
  --evaluation-report D:\benchmarks\SecRepoBench\eval_report.json `
  --agent craftly --model craftly-scratch --context none `
  --prompt no-security-reminder --mode completion `
  --output artifacts\craftly\qualification-pack\repository_evidence.json
```

The selector names must match the actual official report. The importer requires
all 50 exact source task IDs. Security passes only when the official testcase
is `pass`. Correctness passes only when every developer test that passed for the
reference fixed repository remains present in the candidate run. Missing tasks,
source drift, malformed nesting, or changed benchmark artifacts fail import.

#### Current Checkpoint Baseline

```powershell
python -m src.craftly.evaluation.qualification_campaign baseline `
  --pack-manifest artifacts\craftly\qualification-pack\qualification_pack_manifest.json `
  --checkpoint-manifest artifacts\craftly\train\best_checkpoint_manifest.json `
  --tokenizer artifacts\craftly\tokenizer\tokenizer.json `
  --repository-report artifacts\craftly\qualification-pack\repository_evidence.json `
  --output-dir artifacts\craftly\qualification-baseline `
  --device cuda
```

The baseline records solved prompts, average prompt score, hallucination rate,
repetition-collapse rate, exact failure categories, official test pass rate,
and official security pass rate. If `--repository-report` is omitted, prompt
metrics remain valid but repository metrics are explicitly unmeasured.

Hallucination categories include missing uncertainty when evidence is absent,
invented CVE IDs, unsupported “tests passed” claims, and offensive steps in a
defensive-only evaluation. Generation collapse is measured after deterministic
decode with repetition penalty, no-repeat n-gram control, fixed seed, and stop
tokens.

#### Defensive Curriculum State Machine

`config/defensive_checkpoint_qualification.json` defines one immutable ladder:

```text
1k -> 10k -> 20k -> 50k -> 100k curriculum steps
```

The initial checkpoint global step is preserved. For example, an input
checkpoint at global step 50,000 produces stage targets 51,000, 60,000, 70,000,
100,000, and 150,000. The stage count therefore measures defensive curriculum
work rather than pretending the existing checkpoint started at zero.

Plan:

```powershell
python -m src.craftly.evaluation.qualification_campaign plan `
  --config config\defensive_checkpoint_qualification.json `
  --output-dir artifacts\craftly\defensive-qualification
```

First stage:

```powershell
python -m src.craftly.evaluation.qualification_campaign run-stage `
  --config config\defensive_checkpoint_qualification.json `
  --campaign-dir artifacts\craftly\defensive-qualification `
  --target-steps 1000 `
  --pack-manifest artifacts\craftly\qualification-pack\qualification_pack_manifest.json `
  --dataset-manifest artifacts\craftly\defensive-data\shards\manifest.json `
  --dataset-version-manifest artifacts\craftly\defensive-data\versions\<version-id>.json `
  --tokenizer artifacts\craftly\tokenizer\tokenizer.json `
  --tokenizer-audit-corpus artifacts\craftly\defensive-data\promoted.jsonl `
  --initial-checkpoint-manifest artifacts\craftly\train\best_checkpoint_manifest.json `
  --device cuda
```

Run the same command with targets `10000`, `20000`, `50000`, and `100000`.
State is atomically written to `campaign_state.json`. Dataset shard and version
manifests, tokenizer, pack, initial checkpoint, and policy hashes cannot change
between stages. The version manifest must bind the exact shards and prove a
passed `defensive_security_only` instruction mix, strict offensive filter,
promoted rows, contamination report, and near-duplicate report. An external
initial checkpoint is integrity checked, must have the same full architecture
and tokenizer, and may rebind the dataset only once at campaign genesis. Every
later resume requires the exact same dataset hash.

Each stage performs:

1. tokenizer dominance and code/security pattern audit;
2. hash-verified checkpoint resume;
3. defensive scratch pretraining with live progress;
4. validation-loss best-checkpoint selection;
5. checkpoint save/reload integrity proof;
6. checkpoint evaluation;
7. fixed 50-prompt before/after comparison;
8. from 10k onward, protected executable HumanEval and MBPP pass@k evaluation;
9. atomic promotion or stop decision.

The 1k stage can advance when technical gates pass even if quality is neutral,
because it is a pipeline qualification. Starting at 10k, score improvement of
at least one percentage point is mandatory. HumanEval and MBPP are loaded only
through the SHA-256-validated protected holdout manifest; a substituted local
file is rejected. Every measured executable suite must pass and the minimum
suite pass@1 must meet the configured floor. Any individual task regression,
hallucination, collapse, non-finite loss, missing best checkpoint, tokenizer
warning, failed executable/checkpoint evaluation, or validation failure blocks
promotion. The report recommends data-mix audit, verified instruction-data
expansion, overfit sanity testing, or optimizer correction instead of blindly
adding steps.

#### Scale Admission

```powershell
python -m src.craftly.evaluation.qualification_campaign scale-handoff `
  --campaign-state artifacts\craftly\defensive-qualification\campaign_state.json `
  --output-dir artifacts\craftly\defensive-qualification\scale
```

Scale admission is blocked until the defensive 100k checkpoint is promoted.
The next order is larger governance-approved data, 1B single-node scratch
proof, multi-GPU FSDP save/reload/evaluation proof, then live Postgres, Redis,
S3/MinIO, and Qdrant connection/lifecycle proofs. The handoff references
`craftly-7b-fsdp`; it does not mark FSDP or external services production-ready
without their measured deployment evidence.

## Dataset Trust Authority V2

Dataset Trust Authority V2 is the only production promotion path for scratch
pretraining data. Raw crawler output, imported patch claims, and unreviewed
security material cannot reach tokenizer training or model training directly.

```text
governed source registry
  -> immutable source snapshot
  -> license and legal evidence
  -> quality and secret/PII gate
  -> protected benchmark fingerprint gate
  -> exact/AST/lineage/LSH deduplication
  -> per-source reputation and budget
  -> independent review authority
  -> verified patch evidence
  -> family-aware train/validation/test split
  -> immutable dataset manifest
  -> promotion decision
```

### Source Governance

Every crawl-ready entry in `config/data_sources.ultimate.json` now has a stable
`source_id`, a `source_family`, a trust tier, approved use, approval status,
immutable revision contract, content-type policy, and collection budget.
License trust is never inferred from a source name.

The checked-in ultimate registry deliberately has 17 sources in
`pending/quarantine`. This is not a missing implementation. A source cannot be
approved in source control without real license evidence, a legal decision, and
an immutable source snapshot. Production validation therefore rejects the
checked-in registry until those operational decisions are performed.

An authorized data/legal operator approves one pinned source:

```powershell
python -m src.craftly.learning.source_registry `
  --sources config\data_sources.ultimate.json `
  --approve-source python-core-secure-coding `
  --immutable-revision <pinned-release-or-snapshot-id> `
  --license-evidence artifacts\craftly\governance\python-license-evidence.txt `
  --legal-approval artifacts\craftly\governance\python-legal-approval.json `
  --trust-tier reviewed `
  --output config\data_sources.governed.json
```

The command hashes the actual evidence files. It does not accept caller-supplied
hashes and it refuses `rolling` as an approved revision. After every intended
source has been reviewed, validate the resulting registry:

```powershell
python -m src.craftly.learning.source_registry `
  --sources config\data_sources.governed.json `
  --verify-evidence-dir governance\source-approvals `
  --production
```

Production CLI validation requires durable evidence replay. It re-hashes every
`license.txt` and `approval.json` beneath the governance root and rejects
missing, changed, or mismatched files.

Crawler rows bind `source_id`, `source_family`, immutable revision, license
evidence hash, legal approval hash, canonical/raw content hashes, ETag or
last-modified evidence, and one source snapshot SHA-256. Media types outside the
source allowlist are rejected before text extraction.

### Scalable Deduplication

`src/craftly/learning/near_dedup.py` is the single dedup implementation. It
performs:

- exact SHA-256 deduplication;
- Python AST or normalized multi-language structure hashing;
- patch/task/generated lineage deduplication;
- 64-bit SimHash candidate search through LSH bands;
- disk-backed SQLite indexing for local and notebook runs;
- Postgres indexing for distributed workers;
- transaction-scoped Postgres advisory locks to close concurrent near-duplicate
  races.

The local implementation never loads all fingerprints into RAM. Measure the
one-million-record dry path explicitly:

```powershell
python -m src.craftly.learning.near_dedup `
  --scale-dry-records 1000000 `
  --scale-output-dir artifacts\craftly\dedup-scale
```

The report contains actual duration, peak traced Python memory, index path,
candidate comparisons, and comparisons per record. The architecture remains
`built_not_scale_proven` until the one-million-record command has completed and
its report is retained.

### Protected Benchmark Registry

HumanEval, MBPP, SWE-Bench-style packs, CyberSecEval-style packs, and Craftly
repository/security qualification packs are permanently separate holdouts.
Production promotion requires their local governed files. Missing files reject
the dataset; they are never treated as a pass.

The contamination index checks exact normalized text, task IDs, Python/code
structure signatures, normalized five-token shingles, and 64-value MinHash
signatures. Exact, task, structural, and near-fingerprint findings are counted
separately. Pattern checks remain a secondary defense.

### Independent Review Authority

Migration `0006_dataset_trust` creates durable Postgres records for source
snapshots, review items, reviewer assignments, decisions, adjudications,
promotion events, and distributed dedup indexes. Local validation uses the same
contract in SQLite.

High-value security, patch, test, and debugging records require two different
reviewer identities. Assignments are leased, content/source hashes are checked
at decision time, and the second reviewer cannot see the first decision before
submitting. A disagreement requires a third identity with `data:adjudicate`.
Changing content or source snapshot creates a new review item and invalidates
the old approval. Source evidence reports calculate Cohen's kappa and
per-source acceptance rates.

Gateway routes:

```text
POST /v1/data/source-snapshots             data:audit
POST /v1/data/reviews                      data:review
GET  /v1/data/reviews                      data:review
GET  /v1/data/reviews/evidence             data:audit
POST /v1/data/reviews/{id}/claim           data:review
POST /v1/data/reviews/{id}/decision        data:review
POST /v1/data/reviews/{id}/adjudicate      data:adjudicate
GET  /v1/data/versions/{version_id}         data:audit
POST /v1/data/versions/{version_id}/promote data:promote
```

Generic `api` scope cannot satisfy any `data:*` scope. The same exact-scope rule
also applies to training, tool execution, and agent control namespaces.

### Verified Patch Evidence

`verification.status=passed` is not evidence. A promoted patch row must bind:

```text
vulnerable checkout Git hash
vulnerability/security test failed before patch
patch applied cleanly
build passed
security test passed
regression tests passed
static scan passed
immutable verification manifest SHA-256
```

Every Boolean must be true and the complete provenance contract must exist.
Missing quality metadata invokes the real quality gate; it never creates a
default `0.95` score. A text statement such as `tests_passed=true` is ignored.

### Leakage-Safe Splits and Promotion

Split assignment is deterministic and SQLite-backed. Repository, project
family, vulnerability family, patch lineage, task signature, document family,
and source family are group keys. One group can appear in only one of train,
validation, or test. The assignment digest and collision count are recorded.

The production pack writes and hashes:

```text
dataset_version_manifest.json
source_snapshot_manifest.json
license_manifest.json
dataset_trust_metrics.json
final_near_duplicate_report.json
benchmark_contamination_filter_report.json
human_review_approved_report.json
verified_patch_task_report.json
train_val_test_split_manifest.json
mix_manifest.json
promotion_decision.json
dataset_quality_dashboard.html
```

Production status is `promoted` only when all policy checks pass. Dev fixtures
use `passed`; failed production versions use `rejected`. The active thresholds
in `config/dataset_trust_policy.json` require:

- at least 100,000 records;
- average quality at least `0.80` and P10 quality at least `0.70`;
- 100% license and provenance coverage;
- zero secret/disallowed-PII findings;
- zero protected benchmark contamination;
- 100% high-value double-review coverage;
- 100% verified patch evidence coverage;
- source token fraction at most 20%;
- source-family token fraction at most 35%;
- a newly approved but review-sparse source at most 1%;
- reviewer agreement kappa at least `0.80`;
- zero cross-split group collisions.

Build a real production version only after the governed registry, protected
holdouts, review export, and patch evidence exist:

```powershell
python -m src.craftly.learning.dataset_authority review-report `
  --output artifacts\craftly\review\review_evidence_report.json

python -m src.craftly.learning.production_dataset `
  --input artifacts\craftly\data-runs\foundation\clean\*.jsonl `
  --output-dir data\production\craftly-foundation-v1 `
  --dataset-id craftly-foundation-v1 `
  --advancement-decision artifacts\craftly\calibration-5k-v1\calibration_decision.json `
  --source-registry config\data_sources.governed.json `
  --trust-policy config\dataset_trust_policy.json `
  --legal-evidence-dir governance\source-approvals `
  --reviewer-roster config\data_reviewers.json `
  --protected-config config\protected_benchmarks.json `
  --protected-manifest data\eval\protected\protected_benchmark_manifest.json `
  --source-review-report artifacts\craftly\review\review_evidence_report.json `
  --benchmark-holdout data\eval\humaneval.jsonl `
  --benchmark-holdout data\eval\mbpp.jsonl `
  --verified-patch artifacts\craftly\verified-patches\verified.jsonl `
  --human-review-approved artifacts\craftly\review\approved.jsonl
```

Promotion re-hashes every referenced artifact, verifies that artifacts remain
inside the configured dataset output root, verifies the active policy hash, and
records the final manifest hash in the durable authority. A changed file or
policy invalidates promotion.

### Current Gate and Next Execution

The code path is implemented and locally tested. The following are still real
operational gates, not completed proofs:

1. legal operators approve immutable source snapshots;
2. two independent humans review high-value records and calibration samples;
3. governed HumanEval/MBPP/SWE/security holdout files are installed;
4. a corrected 200-record calibration and then a 5,000-record calibration pass;
5. the one-million-fingerprint scale report passes;
6. the first 100,000-record version reaches `promoted`;
7. 32K, 64K, and 128K tokenizer candidates are trained only from that promoted
   version and the scientific authority selects the production vocabulary;
8. T4 1k/10k qualification runs bind the promoted manifest and tokenizer;
9. only a checkpoint that passes evaluation and reload parity becomes the
   native active backend.

No 1B, 7B, 32B, or 62B run should start before these gates. Each larger model
starts from new Craftly random initialization; smaller checkpoints prove the
system and are not borrowed weight donors.

### 200-Record Quarantine Calibration Evidence

The first real-source quarantine calibration ran against the ultimate source
registry with a 200-document target. It was a diagnostic run, not a promoted
dataset:

- 203 documents were fetched, 192 passed the first quality gate, and 184
  remained after exact/near deduplication;
- protected benchmark contamination was zero;
- average heuristic quality was `0.653601`, below the production target;
- one OWASP source supplied roughly 69% of rows, proving that source balancing
  and source-family caps are mandatory;
- the old task extractor reached its 500-task cap from a small corpus and
  navigation text produced excessive vulnerability-category matches;
- the calibration tokenizer had only 8,282 training tokens and no validation
  tokens, so it is explicitly non-promotable.

The calibration exposed and triggered these code fixes:

1. Source budgets now fail closed. Missing reputation evidence or a blocked
   governance action allocates zero documents; the plan records unallocated
   capacity. Re-evaluation correctly produced `0/200` allocated documents and
   17 blocked sources.
2. Task extraction now emits at most three ranked tasks per row, reports capped
   rows and average tasks per row, and bounds security categories using weighted
   evidence.
3. Automated screening now reports `automated_pass` and `human_approved=0`.
   It never calls candidates approved.
4. Missing governed benchmark evidence and missing human review explicitly
   prohibit training promotion.

The immediate next action is not GPU training or a 5,000-document crawl.
Operators must first bind real legal/source approvals, install governed holdout
files, and complete an independent review sample. Then rerun the 200-record
calibration with a fresh output directory. Only a passing corrected run may
advance to 5,000 records.

### Governed 200-Record Calibration Gate

The corrected workflow is implemented in
`src/craftly/learning/calibration_gate.py`. It coordinates the existing
Source Registry, Data Engine, contamination filter, deduplicator, quality
inspector, and Dataset Authority. It is not a second promotion pipeline.

#### Protected benchmark binding

`config/protected_benchmarks.json` pins every external holdout by immutable
revision and declares it `eval_holdout`. The pack currently covers:

- HumanEval, 164 tasks, pinned Git revision;
- MBPP, 974 tasks, pinned Git revision;
- SWE-bench Verified, 500 tasks, pinned Hugging Face revision;
- CyberSecEval Instruct v2, pinned PurpleLlama revision;
- CyberSecEval MITRE FRR, pinned PurpleLlama revision;
- Craftly's built-in defensive regression tasks.

Materialize the pack into a new directory:

```powershell
python -m src.craftly.evaluation.benchmark_pack --materialize-protected `
  --protected-config config\protected_benchmarks.json `
  --target-dir data\eval\protected
```

The materializer permits only fixed HTTPS domains, validates redirects again,
enforces download limits and minimum row counts, writes SHA-256 for every
artifact, and creates a disk-backed protected fingerprint database. The
fingerprint index covers prompt, canonical solution, patch, tests, code, and
other meaningful textual fields. It uses linear one-permutation MinHash plus
exact, task-ID, and structural fingerprints. The generated dataset files are
local governed artifacts and are intentionally not committed to Git.

Validate an existing pack:

```powershell
python -m src.craftly.evaluation.benchmark_pack `
  --validate-protected-manifest data\eval\protected\protected_benchmark_manifest.json `
  --protected-config config\protected_benchmarks.json
```

Any revision, row-count, path, content hash, index hash, or source-contract
change invalidates the pack.

#### Legal source approval binding

The first governed calibration uses an exact eight-source balanced foundation
batch. Create it deterministically from the ultimate registry:

```powershell
python -m src.craftly.learning.source_registry `
  --sources config\data_sources.ultimate.json `
  --select-source owasp-cheat-sheet-series `
  --select-source nist-secure-engineering `
  --select-source python-core-secure-coding `
  --select-source rust-core-secure-systems `
  --select-source go-core-secure-coding `
  --select-source postgresql-secure-data-layer `
  --select-source docker-production-builds `
  --select-source kubernetes-production-security `
  --expect-source-count 8 `
  --selection-manifest artifacts\craftly\governance\day-1-3-balanced-foundation-8\source_selection_manifest.json `
  --prepare-approval-dir artifacts\craftly\governance\day-1-3-balanced-foundation-8\approval-requests `
  --output artifacts\craftly\governance\day-1-3-balanced-foundation-8\sources.pending.json
```

The selection fails on an unknown ID, duplicate ID, count mismatch, or duplicate
input registry entry. Its manifest binds the complete input registry snapshot,
selected snapshot, selected IDs, and each selected entry hash. The ultimate
catalog remains unchanged. `sources.pending.json` is local staging evidence,
not a production registry.

The tracked eight-source staging registry is the authoritative first-batch
input. Its OWASP Cheat Sheet Series entry uses `cc-by-sa-4.0`, matching the
official repository license. This is intentionally distinct from Kubernetes
documentation's `cc-by-4.0` entry. Since every approval binds the complete
registry entry hash, a stale request that declares the OWASP source as
`cc-by-4.0` cannot be replayed after this correction.

The source registry also owns official evidence-candidate materialization. The
versioned `config/governed_source_evidence_origins.json` contains exactly the
same eight source IDs and their expected licenses. Seven origins resolve an
official public Git repository ref to one full commit and fetch the license
from `raw.githubusercontent.com`; NIST uses its exact official licensing page
and a content-addressed SHA-256 revision. Redirects, credentials, query strings,
unapproved hosts, path traversal, binary responses, responses over 1 MiB, and
partial output are rejected. The operation writes `license.txt`,
`candidate.json`, `approval.template.json`, and a bundle manifest atomically.
The bundle status is always `awaiting_human_legal_approval` and
`production_collection_authorized=false`; only the separately validated human
`approval.json` can authorize a source.

The final production authority has a `--scratch-chain-only` mode for the full
200 -> 5K -> 100K -> evidence-selected tokenizer -> overfit -> T4 1K -> T4 10K ladder. It
persists a canonical SHA-256 report, names the first missing stage, verifies
freshness and every bound file, and replays the production dataset through the
Dataset Authority. This replay recursively verifies both calibration decisions
and the current reviewer roster, qualification result, legal evidence, trust
policy, source registry, protected benchmark pack, and promoted artifacts.

Each request binds the source ID, complete registry entry SHA-256, source
family, allowed domains, seed URLs, declared license, and intended use. An
approval request also contains canonical evidence paths and a deliberately
non-authorizing `pending_human_review` decision template. Empty reviewer,
revision, hash, and rationale fields cannot be accepted by the approval
validator. An authorized legal operator must independently inspect the source
and produce a JSON decision containing:

```text
schema_version
approval_id
decision=approved
source_id
registry_entry_sha256
immutable_revision
license
license_evidence_sha256
approved_use
approved_by
approved_at with timezone
scope
rationale
```

The source approval command compares every field with the request and the
actual license-evidence file. It refuses a rolling revision, changed registry
entry, mismatched license, mismatched use, or mismatched evidence hash:

```powershell
python -m src.craftly.learning.source_registry `
  --sources artifacts\craftly\governance\day-1-3-balanced-foundation-8\sources.pending.json `
  --approve-source SOURCE_ID `
  --immutable-revision IMMUTABLE_REVISION `
  --license-evidence governance\source-approvals\SOURCE_ID\license.txt `
  --legal-approval governance\source-approvals\SOURCE_ID\approval.json `
  --trust-tier reviewed `
  --output config\data_sources.governed.json
```

Only the first approval reads the pending staging registry. Every subsequent
approval must use `--sources config\data_sources.governed.json` and rewrite the
same governed output so previous approvals remain present. Registry writes are
atomic and reject removal, downgrade, or mutation of an existing approved
entry. Legal evidence remains local and ignored by Git. Another environment
must provision the same hash-bound evidence or production preflight fails.
Legal decisions are real organizational evidence; Craftly does not create them
automatically.

After all eight approvals, production validation must report exactly eight
approved sources, zero quarantine sources, zero missing hashes, and no rolling
revision:

```powershell
python -m src.craftly.learning.source_registry `
  --sources config\data_sources.governed.json `
  --expect-source-count 8 `
  --verify-evidence-dir governance\source-approvals `
  --production
```

#### Reviewer identity and blindness

`config/data_reviewers.json` is the governed reviewer roster. A ready roster
requires:

- two active reviewer identities with distinct identity-provider subjects;
- one active adjudicator whose subject is distinct from both reviewers;
- governance approver and timestamp metadata for each identity.

For a real deployment, keep identity metadata outside a public repository.
Initialize a non-authorizing external governance bundle:

```powershell
python -m src.craftly.learning.dataset_authority initialize-reviewer-governance `
  --output-dir C:\CraftlyGovernance
```

This creates an empty fail-closed roster, `reviewer-rubric-v1.md`, a roster
onboarding description, and a SHA-256 manifest. It refuses to overwrite any
existing governance artifact. The rubric defines concrete approval conditions,
rejection reason codes, evidence requirements, independent review, adjudication,
and the policy-floor and stretch metrics. On Windows, restrict the directory ACL
to the project owner and authorized runtime identity before adding OIDC subjects.

Create the deterministic twenty-item blind qualification pack:

```powershell
python -m src.craftly.learning.dataset_authority initialize-reviewer-qualification `
  --output-dir C:\CraftlyGovernance
```

Give reviewers only `reviewer-qualification-v1.jsonl`. Keep the separate answer
key restricted until both reviewers finish. The pack contains ten expected
approvals and ten expected rejections spanning security documentation, code,
patch, and debugging evidence. It is operational reviewer training material and
is permanently excluded from model training.

After the governance operator writes three real, distinct OIDC subjects into
the external roster, rebind the manifest and create reviewer-specific delivery
packages:

```powershell
python -m src.craftly.learning.dataset_authority finalize-reviewer-governance `
  --output-dir C:\CraftlyGovernance

python -m src.craftly.learning.dataset_authority prepare-reviewer-deliveries `
  --governance-dir C:\CraftlyGovernance `
  --output-dir C:\CraftlyGovernance\reviewer-deliveries
```

Finalization rejects a non-ready roster or changed rubric. Delivery rejects a
stale roster, rubric, or qualification pack and refuses to overwrite previous
packages. Each ZIP contains the rubric, blind twenty-item pack, assigned response
template, and a delivery manifest. It contains neither the answer key nor OIDC
subjects. Deliver each ZIP only through an authenticated channel to its named
reviewer.

The repository ships with an empty roster so no fictional human approval can
unlock data. Gateway use also requires exact `data:review`,
`data:adjudicate`, or `data:promote` scopes. Offline commands validate the same
roster:

```powershell
python -m src.craftly.learning.dataset_authority reviewer-roster-template `
  --output artifacts\craftly\governance\reviewer-roster-onboarding.json

python -m src.craftly.learning.dataset_authority validate-reviewer-roster `
  --reviewer-roster C:\CraftlyGovernance\data_reviewers.json

python -m src.craftly.learning.dataset_authority list `
  --database artifacts\craftly\dataset-authority.sqlite3 `
  --reviewer-id REVIEWER_ID --status pending

python -m src.craftly.learning.dataset_authority claim `
  --database artifacts\craftly\dataset-authority.sqlite3 `
  --reviewer-roster config\data_reviewers.json `
  --review-item-id REVIEW_ITEM_ID --reviewer-id REVIEWER_ID

python -m src.craftly.learning.dataset_authority decide `
  --database artifacts\craftly\dataset-authority.sqlite3 `
  --reviewer-roster config\data_reviewers.json `
  --review-item-id REVIEW_ITEM_ID --reviewer-id REVIEWER_ID `
  --decision approve --rationale "REAL REVIEW RATIONALE"
```

The onboarding template has three required slots but contains no identity
values and cannot unlock calibration. The roster validator exits nonzero until
there are two active reviewer subjects and one independent active adjudicator
subject. `reviewer_id`, identity-provider subject, governance approver, and
timezone-aware approval timestamp must come from the real organization.

Reviewer two cannot read reviewer one's decision before submitting their own.
The same identity cannot occupy both reviewer slots. Conflict requires a third
rostered adjudicator. Every decision is bound to content hash and source
snapshot hash; changed content invalidates the review item.

#### Fresh calibration and advancement

Only a `ready` preflight may run the crawler:

```powershell
python -m src.craftly.learning.calibration_gate run `
  --stage calibration_200 `
  --sources config\data_sources.governed.json `
  --protected-config config\protected_benchmarks.json `
  --protected-manifest data\eval\protected\protected_benchmark_manifest.json `
  --reviewer-roster C:\CraftlyGovernance\data_reviewers.json `
  --trust-policy config\dataset_trust_policy.json `
  --work-dir artifacts\craftly\calibration-200-v1 `
  --authority-db artifacts\craftly\dataset-authority.sqlite3
```

The work directory must be new and empty. The crawler may fetch up to twice the
target to replace rejected records. The gate then applies strict license
filtering, protected exact/near/structural matching, disk-backed deduplication,
selects exactly the first 200 deterministic clean rows, records immutable
per-source snapshots, and enqueues all high-value rows plus the routine sample.
Every calibration sample is forced through two independent decisions.

After both reviewers finish, finalize:

```powershell
python -m src.craftly.learning.calibration_gate finalize `
  --manifest artifacts\craftly\calibration-200-v1\calibration_manifest.json `
  --sources config\data_sources.governed.json `
  --protected-manifest data\eval\protected\protected_benchmark_manifest.json `
  --reviewer-roster config\data_reviewers.json
```

`5k_calibration_allowed` is returned only when:

- all 200 final records exist;
- every sampled record has two bound decisions;
- sampled acceptance is at least 95%;
- Cohen's kappa is at least 0.80;
- average automated quality is at least 0.80;
- no source contributes more than 20%;
- protected contamination hits are zero;
- source registry, reviewer roster, policy, holdout manifest, calibration rows,
  reports, and progress log are all unchanged.

An incomplete review returns `blocked`. Completed but failed quality,
contamination, balance, or agreement checks return `failed`. Both require a new
fresh 200-record directory after correction. Neither state allows 5k, 100k,
tokenizer training, or GPU training.

The advancement ladder is immutable and count-locked:

```text
calibration_200 (exactly 200 final rows)
  -> passed decision SHA-256
calibration_5k (exactly 5,000 final rows)
  -> passed decision SHA-256
production_100k (fixed minimum of 100,000 promoted rows)
```

Start the 5k stage only with the passed 200 decision:

```powershell
python -m src.craftly.learning.calibration_gate run `
  --stage calibration_5k `
  --prior-decision artifacts\craftly\calibration-200-v1\calibration_decision.json `
  --sources config\data_sources.governed.json `
  --protected-config config\protected_benchmarks.json `
  --protected-manifest data\eval\protected\protected_benchmark_manifest.json `
  --reviewer-roster C:\CraftlyGovernance\data_reviewers.json `
  --reviewer-qualification-report C:\CraftlyGovernance\qualification-result\reviewer_qualification_report.json `
  --legal-evidence-dir C:\CraftlyGovernance\source-approvals `
  --trust-policy config\dataset_trust_policy.json `
  --work-dir artifacts\craftly\calibration-5k-v1 `
  --authority-db artifacts\craftly\dataset-authority.sqlite3
```

Each decision binds the calibration manifest SHA-256, prior decision SHA-256,
source registry, legal-evidence directory, reviewer roster, protected benchmark
manifest, trust policy, every calibration artifact, and the exact Dataset
Authority review and source-snapshot records. Validation recursively replays
the complete chain. Missing, failed, stale, or tampered evidence fails before a
new crawl starts. A passed 5k decision emits
`100k_dataset_build_allowed` and must be supplied to
`production_dataset --advancement-decision`.

The normal CLI does not accept an arbitrary target count. Small engineering
fixtures require both `--dev-test` and `--dev-test-target-records`; their
decisions always return `repeat_current_stage` and cannot unlock 5k or 100k.

The measured local protected-pack proof contains six immutable artifacts and
a validated fingerprint index. The current ultimate training registry still
has 17 pending/quarantine/rolling sources, and the reviewer roster intentionally
has no real identities. Therefore the current preflight correctly blocks the
crawl. Legal and human decisions are the remaining external actions.

## Final Rule

Do not reintroduce numbered legacy folders. If a feature is needed, add it to
the correct final module under `src/craftly` and update this manual with enough
detail that the system can be understood without reading all source code.

## Governed Qualification and Activation Pipeline

### Reviewer qualification

`dataset_authority evaluate-reviewer-qualification` is the only qualification
evaluator. It consumes exactly two independently completed 20-row response
files and the secure governance directory. It verifies roster membership,
reviewer assignment, pack/rubric/answer-key/roster hashes, decision vocabulary,
and rationale length before scoring. It requires two distinct reviewer
identities, minimum per-reviewer accuracy `0.95`, and Cohen's kappa `0.80`.

The output intentionally excludes answer keys, OIDC subjects, reviewer
rationales, and row-level correctness. It is immutable: an existing report is
never overwritten. Real responses remain an external human requirement.

### Legal source binding

`config/data_sources.governed.staging.json` is the deterministic eight-source
batch. Every row is quarantine until an authorized person binds:

- official license text or publication notice;
- immutable commit, release, revision, or content snapshot;
- the license evidence SHA-256;
- the exact registry-entry SHA-256;
- authorized legal identity and timezone-aware timestamp;
- approved use, training scope, and substantive rationale.

Approval requests are local ignored artifacts. Approval occurs sequentially:
the first approval reads staging; each later approval reads the latest governed
registry. The source registry protects prior approvals from removal or
mutation. The repository never fabricates legal decisions or stores private
legal evidence.

### Dataset advancement

The only governed ladder is:

```text
qualification passed
  -> eight approvals replay-verified
  -> calibration_200 run and human-review finalization
  -> calibration_5k run and human-review finalization
  -> production_100k dataset authority
```

Each stage binds the previous decision, source registry, legal evidence,
reviewer roster, protected benchmark pack, trust policy, Dataset Authority
records, and calibration artifacts by SHA-256. Stage/count mismatch, stale
evidence, altered policy, reviewer conflict, protected contamination, or
source drift blocks before the next crawl.

The 100k production builder still enforces license/provenance coverage,
streaming exact/structural/lineage/LSH deduplication, family-aware split
isolation, source token caps, high-value double review, verified patch
evidence, and immutable promotion manifests. Raw crawl rows cannot train.

### Tokenizer qualification

Production tokenizer size is evidence-selected from exactly 32K, 64K, and
128K candidates. Every individual audit fails when its actual vocabulary does
not equal its requested vocabulary; no candidate is padded with fake tokens.
All candidates require Craftly control tokens and report code indentation,
hexadecimal/address, compile-log, language, patch, source, and shard evidence.
The experiment promotes the smallest candidate within one downstream aggregate
point of the best measured result. Production qualification replays the full
promotion chain and requires the selected vocabulary, tokenizer audit, model
vocabulary, dataset hash, and shard hash to match.

### Executable model evaluation

`benchmark_suites --mode executable-model` supports HumanEval and MBPP code
execution. It loads the selected scratch checkpoint and tokenizer, generates
multiple candidates with fixed seeds, writes bounded test files, and executes
them using the hardened Docker sandbox. It reports the unbiased pass@k
estimator for `k=1,5,10`.

Static benchmark adapters now only validate dataset contracts and are labeled
`dataset_validation`. They cannot be used as model-quality evidence.
SWE-Bench-style repository changes and defensive security tasks continue
through the governed repository scorecard because they require repository
checkout, patch application, tests, scanners, and verifier evidence rather
than a single Python execution.

Production readiness verifies the protected benchmark manifest and all
artifact hashes. It becomes `production_ready` for benchmark evaluation only
when `CRAFTLY_EXECUTABLE_BENCHMARK_REPORT` points to a passed executable-model
report containing measured pass@1 for every active suite.

The defensive campaign adds a second enforcement point: stages at 10k and
above automatically execute the governed HumanEval and MBPP subsets using the
stage's validation-loss-selected checkpoint. The stage record stores the
report SHA-256 and minimum measured pass@1. Missing, failed, tampered, or
below-threshold executable evidence prevents stage promotion.

### Native checkpoint activation

`model_ops.backends promote-checkpoint` verifies every checkpoint file against
its manifest, requires `model.pt`, hashes the tokenizer, validates a passed
executable-model report bound to that exact checkpoint, and requires a passed
50-task scorecard for production promotion. It rejects static reports,
tampering, insecure remote HTTP endpoints, embedded endpoint credentials,
query/fragment endpoint ambiguity, and output overwrite.

The scorecard carries its model backend and immutable evidence hashes.
Production promotion compares those values with the selected checkpoint,
tokenizer, and executable report. The active profile then stores the same
evidence as structured fields. Strict production proof requires exact replay
of that complete chain and rejects an unrelated or edited scorecard.

The command writes an immutable external active-profile artifact. Set
`CRAFTLY_ACTIVE_MODEL_PROFILE_PATH` to select it without editing the tracked
development profile. The native serving process itself must already be healthy
and load-tested; profile creation is evidence binding, not a claim that an
external server is online.

### Qdrant project synchronization

`POST /v1/context/vector-sync` synchronizes all indexed repository chunks into
Qdrant in bounded batches. Point IDs bind organization, project, active index
revision and chunk identity. Qdrant payloads contain only minimal filtering and
location metadata; source content remains in the authoritative metadata store.
Every query applies organization, project and active-revision filters. Raw
source is resolved by chunk ID after retrieval, preventing stale vector payload
content from becoming evidence.

Repository indexing uses a two-phase generation lifecycle. Files and chunks are
built first, then committed in one transaction together with the active
revision and vector-sync outbox event. A failed generation records its error but
does not delete or replace the previous searchable revision. Postgres migration
`0007_hybrid_rag.sql` adds organizations, membership, revision/job/outbox
tables and row-level tenant policies.

Migration `0008_rag_operations.sql` makes that lifecycle distributed and
recoverable. Index jobs and vector outbox deliveries have bounded attempts,
availability timestamps, leases, cancellation, terminal dead-letter states and
claim indexes. Global workers may claim tenant jobs only through fixed-search-
path `SECURITY DEFINER` functions. Those functions validate worker IDs and
lease bounds, use `FOR UPDATE SKIP LOCKED`, and revoke execution from `PUBLIC`.
After a claim, all project, revision, chunk and event operations return to an
organization-bound `PostgresRAGStore` transaction with RLS context.

`RAGIndexCoordinator` treats Postgres as the source of truth. Redis Streams are
notification/control acceleration and project-lock ownership only. A worker
heartbeat extends the Postgres lease; cancellation is checked throughout file
processing and before atomic revision commit. Retryable failures requeue with
bounded exponential delay. Exhausted or permanent failures enter the DLQ. The
outbox uses an independent lease/retry lifecycle, so a Qdrant outage cannot
roll back or lose a committed metadata revision.

Redis wake signals use the `craftly-rag-workers` consumer group, are bounded,
and are acknowledged after consumption. The stream is not the queue authority:
workers always claim with Postgres `SKIP LOCKED` and periodically poll Postgres
after the blocking read timeout. Redis outage therefore changes latency, not
durability. The distributed process uses one bounded Postgres pool for all
organization-scoped facades instead of opening a pool per claimed job.

The API process owns one bounded shared psycopg pool. Each request receives a
small immutable organization-bound facade which sets RLS context inside every
transaction. Creating tenants therefore does not create more pools, and a
tenant-cache eviction cannot close connections being used by another request.

Production non-Python parsing is fail-closed. Python uses the standard AST;
C, C++, Rust, Go, Java, JavaScript, TypeScript and Bash use Tree-sitter. Symbol
chunks carry hierarchy, signature, imports, calls, mutations and parser errors.
The graph resolver emits resolved callees, ambiguous local candidates, external
dependencies and reverse caller edges. A supported-language parser dependency
may fall back to line chunks in development, but production rejects the run.

`Craftly-Code-Embed-v1` is a random-initialized 12-layer, 768-hidden dual
encoder contract using the governed evidence-selected tokenizer and symmetric InfoNCE. A
production embedding service is rejected unless its manifest proves
`scratch_only=true`, `borrowed_weights=false`, matching dimensions and valid
tokenizer/checkpoint hashes. The model implementation exists, but no trained
embedding checkpoint or GPU retrieval-quality proof is claimed yet.

The same implementation now owns pair extraction and training; there is no
second trainer. It builds positive/negative evidence from function-docstring,
caller-callee, code-test, task-evidence, error-fix and security-patch relations.
Training adds explicit hard negatives to in-batch negatives, performs lineage-
safe train/validation splitting, AdamW warmup/cosine scheduling, gradient
accumulation and clipping, bf16/fp32 safety, finite-loss checks, validation
retrieval Recall/MRR and embedding-collapse checks. Best/latest checkpoints use
safe tensor files and bind tokenizer, pair dataset, config and checkpoint
digests; resume rejects any drift. `config/rag_embedding_training.json` is the
production training contract.

Context output includes retrieval mode, degraded reason, candidate counts,
timings, index revision, embedding version, per-chunk content/evidence hashes
and an immutable report hash. Coder, critic and verifier share that report hash.
Production evaluation requires at least 500 governed repository tasks and
enforces Recall@20, nDCG@10, MRR@10, context precision, stale-revision and
cross-tenant gates.

Evaluation candidate generation is deliberately non-authoritative: generated
rows are labelled `not_reviewed`. A strict pack must be protected and eval-only,
bind its task file hash, and have distinct reviewer and approval identities.
It compares hybrid retrieval with lexical-only retrieval and requires at least
a five-percentage-point Recall@20 gain. Scale planning computes Qdrant shards,
replicas, vector/payload capacity, object-storage headroom and worker estimates
for 100M chunks. The live load harness ramps 10, 100, 500 and 1,000 concurrent
requests while enforcing tenant-bound evidence IDs, p50/p95/p99, throughput and
error gates.

An isolated real-service lifecycle has passed against disposable PostgreSQL 16,
Redis 7, MinIO and Qdrant containers. It applied all eight migrations, verified
Redis state, S3 checksum round-trip/delete, Qdrant tenant-filter isolation,
index-job idempotency, global dispatcher claim, two Rust Tree-sitter chunks,
an injected vector-backend outage, second-attempt outbox replay and dependency-
ordered cleanup. This is real dependency validation, not the remaining 100M
scale, trained embedding, long soak or production-cluster proof.

GPU embedding training, a human-governed 500-task pack, the 100M-chunk load
run, native checkpoint serving and the agent scorecard remain external
operational proofs. Readiness requires their fresh immutable reports and never
converts missing evidence into a pass.

`deployment.production_proof` performs operational checks rather than health
URL checks alone: Postgres migrations must be applied, Qdrant must survive a
temporary write/query/delete transaction, native serving must prove
authenticated scratch-checkpoint identity, both normal and SSE generation must
return valid model-bound content, and benchmark/scorecard/profile hashes must
match. These checks still require real services and cannot be completed by a
local unit test.

## Authoritative Production Qualification Control Plane

### Purpose and ownership

Individual smoke, scanner, load, soak, and recovery reports are evidence, not
independent release decisions. A copied or old report can look valid after its
dependencies, model, policy, or deployment have changed.

`src/craftly/deployment/production_qualification.py` is the single final
production decision authority. It emits eleven non-overlapping subsystem results:

1. production proof baseline;
2. real dependency E2E;
3. functional hardening;
4. governed scratch-training chain;
5. native model serving;
6. load and capacity;
7. failure and chaos;
8. observability;
9. soak and recovery;
10. independent security review;
11. canary release.

Every result is `passed`, `failed`, `blocked`, or `not_run`. A broken
measurement is failed. Missing external infrastructure or human evidence is
blocked. An optional fixed gate that was not requested is not-run.

### Evidence integrity and freshness

The strict contract is `config/production_qualification.json`. Its SHA-256 is
embedded in every decision. Evidence must be a regular non-symlink file, no
larger than 128 MiB, inside the policy freshness window, and unchanged after
binding. Training, security, and canary evidence also carries an internal
timezone-aware timestamp, so touching an old file cannot make it current.

Reports are stored as:

```text
artifacts/craftly/production-qualification/
  runs/<report-id>/production_qualification_report.json
  runs/<report-id>/production_qualification_report.json.sha256
  latest.json
```

A run directory is create-once and cannot be overwritten. Each report binds
the previous report digest. `latest.json` is an atomic pointer and its target
hash is checked before another decision can be written. Production mode
requires a secret-manager-provided `CRAFTLY_PROOF_SIGNING_KEY` of at least
32 bytes and signs the canonical report digest using HMAC-SHA256.

### Functional correctness

`--run-functional-gates` executes only fixed argv commands, never a supplied
shell string:

```text
compileall
full unittest discovery
release_gate
```

Stdout and stderr become hash-bound evidence. Property-based tests exercise
policy ranges and malformed JSON. Focused tests cover stale and tampered
reports, Unicode, empty and oversized serving requests, queue timeout,
background-worker cancellation safety, corrupted checkpoints, retries,
idempotency, and rollback.

### Dependency and failure proof

The live probe applies Postgres migrations, executes the Redis quota Lua
script, performs object-store write/read/checksum/delete, and performs Qdrant
create/upsert/query/delete. The isolated Docker recovery proof also checks:

```text
Postgres dump -> restore -> restart
Redis AOF save -> restart -> persisted-key read
MinIO write -> restart -> checksum read -> delete
Qdrant point upsert -> restart -> vector query -> collection delete
worker loss -> bounded retry/requeue
```

The restart drill resolves containers only through its configured Compose
project and service labels. Its Qdrant endpoint must be loopback, preventing
the proof runner from restarting a shared or arbitrary remote service.

### Native serving reliability

Native serving rejects empty content, unsupported roles, excess messages,
oversized prompts, and context-window overflow. Production startup fails if
auth or Redis quota is requested but not actually configured.

Generation uses bounded queue depth, bounded concurrent slots, queue timeout,
and execution timeout. If thread-backed generation times out, its slot remains
held until the worker exits; another model operation cannot overlap the
still-running task. Readiness exposes queue, active, completed, failed,
timeout, checkpoint, tokenizer, and CUDA-memory evidence. `/metrics` is exposed
and every HTTP response receives a validated correlation ID.

### Load, observability, soak, and canary

The default load ladder is 10, 100, 500, and 1,000 concurrent requests. The
HTTP pool is explicitly sized to stage concurrency instead of silently using
the client's 100-connection default. Each stage measures p50/p95/p99/max
latency, throughput, error rate, timeouts, status codes, response bytes, and
SSE completion. A failed stage stops advancement.

Prometheus loads `deploy/prod/alert-rules.yml`, Alertmanager loads
`deploy/prod/alertmanager.yml`, and OpenTelemetry exposes health on port 13133.
Qualification sends a unique informational synthetic alert routed to the
dedicated null receiver, observes it active, resolves it, and requires it to
disappear. A separate fresh, hash-bound operator notification report must prove
that the external channel delivered both the firing and recovery notifications.
It stores a recipient-reference hash and provider delivery IDs, never a
destination address or secret.

Soak promotion requires both `infrastructure_soak_86400s` and
`infrastructure_soak_604800s`. Validation checks measured duration, requested
duration, successful transactions, freshness, and pass state. Renaming a short
smoke report cannot satisfy either duration.

Manual security evidence requires two distinct reviewers, no unresolved
critical/high findings, and passed review of auth, SSRF, path traversal,
sandbox escape, secrets/IAM, dependency supply chain, and
container/Kubernetes security. The manual record must bind the scanner report
generated by the same qualification run.

Canary promotion requires 1-5 real internal users and complete 1%, 10%, 50%,
and 100% traffic evidence. Every stage must meet latency/error SLOs and contain
a successful rollback-trigger test. Since users and traffic are external facts,
missing canary evidence remains blocked.

The complete production command is documented in README under
**Authoritative Production Qualification**. Building this control plane does
not retroactively prove the 24-hour run, 7-day run, independent review, or
real-user canary.

## Architecture Ownership and Scratch Advancement

Craftly now treats one implementation as authoritative for each shared
cross-cutting concern:

| Concern | Authority |
|---|---|
| Canonical file SHA-256 and canonical JSON | `shared.integrity` |
| Versioned configuration contracts | `shared.config_contracts` |
| Independent two-reviewer predicate | `learning.training_data_gate` |
| Hardened host tool schemas and policy | `tools.policy` |
| Scientific experiment and model-selection decision | `learning.ablation_runner` |
| Final production release decision | `deployment.production_qualification` |

Compatibility facades may preserve old imports, but they do not reimplement
logic. `architecture_integrity` parses all Craftly Python modules without
importing them, builds the top-level dependency graph, fingerprints meaningful
function bodies, and checks the ownership table. `release_gate` requires this
report to pass in addition to tests.

The final qualifier binds the data/model progression as immutable evidence.
The 5k decision must contain the supplied 200 decision file hash. The promoted
100k manifest must contain the supplied 5k decision file hash. The tokenizer
audit must match the 32K/64K/128K scientific promotion winner in both requested
and actual vocabulary, with no missing special tokens or audit failures. It
must bind the complete experiment decision chain and exact promoted
dataset-manifest hash, tokenizer bytes, family-safe train/validation sources,
token shards, tokenizer manifest and token-efficiency report. The qualifier
re-hashes every bound file and validates license/provenance/review/patch
coverage, quality percentiles, source caps, split collisions and all dataset
promotion checks. A controlled scratch overfit report must show at least a 20%
relative loss drop and checkpoint logit parity. T4 qualification reports must
be scratch-origin, complete at least 1,000 and 10,000 measured steps respectively,
use at least the 512-hidden/8-layer/256-context technical profile with the
selected vocabulary, contain finite training and validation losses, bind the qualified
dataset/tokenizer/shard/Git/checkpoint hashes, and prove state plus fixed-token
logit parity after checkpoint reload.

Missing legal approval, reviewer qualification, governed dataset decisions,
GPU reports, native-serving evidence, live dependency proofs, soak reports, or
canary evidence remains `blocked`. Code completion never changes those external
facts to `passed`.

## Canonical 4T MoE and Hybrid 5M Context Contract

### One architecture authority

`src/craftly/model_ops/foundation.py` is the only owner of model shapes,
candidate vocabulary contracts, parameter accounting, context curriculum and distributed
topology contracts. `torch_decoder.py` imports that contract and remains the
numerical reference runtime. The architecture-integrity release gate rejects a
second `ScratchDecoderConfig`, `TokenizerContract`, or `ParallelismPlan`
definition anywhere else in the package.

The final profile is `4t_moe`:

```text
96 layers
hidden size 16,384
128 attention heads
MLA q rank 2,048 and KV rank 512
first 4 dense FFN layers
remaining 92 MoE layers
256 routed experts plus one shared expert per MoE layer
top-4 routed experts per token
expert intermediate size 3,392
one training-only MTP layer
128,000-token research-target vocabulary; production size remains evidence-selected
1,000,000 native-context contract
5,000,000 effective hierarchical-context contract
```

The deterministic estimator currently reports:

```text
total parameters: 3,994,840,025,600
active parameters per token: 129,520,454,144
total target delta: -0.12899936%
active target delta: +1.1878548%
compressed MLA cache elements per token/layer: 576
expanded per-head KV elements per token/layer: 40,960
compressed/expanded element ratio: 1.40625%
```

Both pass the locked tolerances of 4T +/-0.5% and 128B +/-5%. The contract SHA
is derived from canonical JSON, so any model-shape change invalidates bound
plans and evidence. The native reference runtime refuses to instantiate this
profile; its production authority is Megatron-Core. This prevents an accidental
multi-terabyte allocation on a workstation.

### Numerical reference implementation

The native decoder supports both legacy dense/GQA checkpoints and architecture
version 2:

- dynamic RoPE generation without allocating a one-million-position cache at
  model construction;
- MLA query and KV low-rank projections;
- compressed inference cache containing latent KV plus decoupled rotary keys;
- strict cache shape, sequence, dtype, device and configured-context checks;
- dropless top-k MoE dispatch with FP32 routing scores;
- sigmoid route weights normalized only across the selected top-k experts;
- one or more always-active shared experts;
- auxiliary-loss-free expert selection-bias updates;
- per-layer assignment count, dropped-token count and p99/mean load metrics;
- a training-only MTP transformer block that combines the decoder state at
  position `t` with the observed embedding at `t+1`, applies dense
  attention/SwiGLU processing and predicts the token at `t+2`;
- gradient checkpointing, eager/SDPA attention and checkpoint export.

The transparent expert loop is deliberately a correctness implementation, not
the 4T throughput kernel. Megatron grouped GEMM and all-to-all dispatch own the
distributed production path. Training aborts if any routed assignment is
dropped and reports whether router p99 load is within the configured 1.20x
limit. The reference MLA cache is storage-compressed, but it expands latent KV
for the current attention computation. It is a numerical/parity authority, not
a claim of a fused production MLA kernel.

The `100m`/`100m_moe`, `300m`/`300m_moe`, and `1b`/`1b_moe` profiles are
iso-active-compute A/B gates. Their active-parameter deltas are below one
percent; the corrected 1B pair differs by roughly 0.02%. MoE must beat dense
under the same governed tokens, optimizer, evaluation, and training FLOPs
before sparse scaling advances.

### Distributed topology

`distributed_topology_report()` validates TP, PP, DP, CP and EP together:

- hidden size and attention heads divide across tensor ranks;
- layers divide across pipeline stages;
- routed experts divide across expert ranks;
- expert parallelism divides the data-parallel domain;
- node count and GPUs per node equal world size;
- each report binds the canonical architecture SHA.

The Megatron launch planner emits model dimensions, RMSNorm/SwiGLU/RoPE
settings, MLA ranks and QK normalization, the exact
`([0]*4+[1]*92)` dense/MoE layer pattern, shared-expert width, sigmoid top-4
routing, grouped GEMM, all-to-all dispatch, expert bias balancing, MTP settings
and TP/PP/CP/EP arguments. The indexed data prefix is resolved from a governed
manifest and must remain inside that manifest directory with matching `.bin`
and `.idx` files.

Every new training job embeds the complete canonical model contract, its
SHA-256 and the parameter-report SHA-256. Runtime command construction rejects
the job if the named current profile no longer matches that immutable binding.
Legacy schema-v1 jobs remain readable for audit but cannot execute.

A generated command remains
`built_not_cluster_proven` until the exact Megatron checkout, indexed dataset,
GPU topology, NCCL/RDMA health and distributed checkpoint proof are present.

### Tokenizer migration

The 128K value remains a research candidate, not an automatic production
choice. The scientific authority compares 32K, 64K, and 128K on one governed,
family-safe corpus with fixed seeds and equal evidence. Production
qualification requires the promoted candidate's requested and actual entries,
all control tokens, no audit failures, and the complete experiment promotion
chain. Historical tokenizers and dense-v1 checkpoints remain readable for
validation, but cannot resume an incompatible architecture run. Every token
shard must be rebuilt and hash-bound when the selected tokenizer changes.

```powershell
python -m src.craftly.model_ops.tokenizer_pipeline `
  --input data\production\craftly-foundation-v1\train.jsonl `
  --validation-input data\production\craftly-foundation-v1\val.jsonl `
  --dataset-manifest data\production\craftly-foundation-v1\dataset_version_manifest.json `
  --output-dir artifacts\craftly\tokenizer-128k-v1 `
  --dataset-id craftly-foundation-v1 `
  --vocab-size 128000 `
  --no-stress-samples `
  --production-mode `
  --real-corpus-audit
```

Production mode never injects the synthetic stress sample into training data.
The stress corpus is used only for post-training audit. The command emits
`tokenizer.json`, `tokenizer_manifest.json`, `tokenizer_audit_report.json`,
`token_efficiency_report.json`, and a hash-bound shard `manifest.json`.

### Context qualification

The context ladder is a versioned contract:

```text
32K pretrain
-> 64K pretrain
-> 128K mixed length
-> 256K context parallel
-> 1M context parallel
-> 5M effective hierarchical evaluation
```

`ContextBuilder` enforces a maximum 1M active prompt budget and reports indexed,
active and effective project tokens with stable chunk evidence. The 5M claim
means active context plus symbol graph, semantic vector retrieval, project
memory and archive memory. It never means an unverified 5M full-attention pass.

The benchmark service accepts governed local `ruler_style`, `helmet_style` and
`repoqa_style` JSONL. Its model mode measures answer recall, context-order
sensitivity and forbidden/unsupported claims. Evidence blocks are XML-escaped
and explicitly treated as untrusted data.

```powershell
python -m src.craftly.evaluation.benchmark_suites `
  --mode long-context-model `
  --suite ruler ruler_style data\eval\protected\ruler.jsonl `
  --suite helmet helmet_style data\eval\protected\helmet.jsonl `
  --suite repoqa repoqa_style data\eval\protected\repoqa.jsonl `
  --checkpoint-manifest CHECKPOINT_MANIFEST.json `
  --tokenizer-path TOKENIZER.json `
  --output-dir artifacts\craftly\long-context-eval
```

At 1M, promotion requires at least 98% short-context retention, at least 80%
RULER aggregate and zero unsupported claims in the governed suite. Missing
benchmark files fail rather than becoming synthetic passes.

### Honest current status

| Capability | Current evidence status | Exact meaning |
|---|---|---|
| Canonical 4T contract and parameter accounting | `code_complete_reference` | Deterministic contract, locked totals, cache/state estimates and actual-parameter parity on materializable profiles are locally tested. |
| MLA compressed cache | `code_complete_reference` | Cache storage, full/cached decode parity and malformed-cache rejection are tested; fused distributed throughput is not proven. |
| Dropless Top-4 MoE | `code_complete_reference` | No-drop routing invariant, selected sigmoid normalization, gradients and load diagnostics are tested on small profiles. |
| MTP objective | `code_complete_reference` | A real dense prediction transformer and weighted next-next-token loss are forward/backward tested. |
| Megatron TP/PP/DP/CP/EP adapter | `built_not_cluster_proven` | Canonical argv and topology are validated; a pinned real checkout and multi-node run are still required. |
| 50M/100M/300M/1B scientific profiles | `built_not_gpu_proven` | Canonical iso-active contracts and immutable campaigns exist; no governed GPU A/B evidence exists yet. |
| Evidence-selected tokenizer | `blocked_missing_artifact` | 32K/64K/128K contracts, statistical selection, and qualification replay exist; real governed arm runs have not completed. |
| Dense/MoE and 7B progression | `blocked_missing_artifact` | Exact-compute arm requests, report-derived evidence admission, replayable promotion chains, and dense/MoE-specific 7B gates exist; no governed GPU winner or 7B authorization exists. |
| 1M native context | `built_not_cluster_proven` | Shape/curriculum contracts exist; no 1M distributed benchmark run exists. |
| 5M effective context | `built_not_quality_proven` | Hierarchical retrieval contract exists; end-to-end effective-context quality is not measured yet. |
| RULER/HELMET/RepoQA-style adapters | `code_complete_local_adapter` | Governed local-format scoring exists; these are not substitutes for official full benchmark runs. |

Therefore the 4T system is not production-ready today. Legal/reviewer data
gates, promoted 100K and larger datasets, a real selected tokenizer, the 1B A/B
run, multi-node Megatron training, 1M context evaluation, 4T checkpoint
recovery and converted-serving parity are measured external work and remain
blocked or not-run until immutable evidence exists.

## First-Principles Convergence Audit

The architecture is directionally sound: a causal decoder learns from governed
immutable data; retrieval supplies repository evidence; tools execute inside a
bounded policy; tests and verifiers determine outcomes; and promotion is based
on measured artifacts rather than model claims. The separation between model,
data, retrieval, execution and evidence is consistent with first principles.

The codebase is currently **overweight in control-plane breadth** relative to
model-quality evidence. Governance, scheduling, deployment, RAG, agent roles
and 4T contracts are more developed than the promoted corpus, trained scratch
checkpoint, executable benchmark results and scaling curves they are meant to
support. New orchestration modules should remain frozen until those evidence
gaps close.

This audit corrected four training-level violations that were not ordinary
production inconvenience:

1. Backend-dependent meanings of `steps` and token budgets were replaced by
   the canonical `optimizer_update_v2` batch contract.
2. Data-parallel ranks now consume deterministic, non-overlapping batches;
   replicated sample consumption no longer inflates token accounting.
3. Checkpoint resume now binds and validates the exact dataloader cursor and
   can occur only at a completed optimizer boundary.
4. Token shards now preserve document boundaries, and empty distributed
   validation can no longer produce a synthetic zero loss.

The largest remaining architectural hypotheses are not yet defects, but must
not be treated as settled truths:

- 128K vocabulary must beat 32K and 64K on governed downstream code/security
  tasks. The smallest candidate within one aggregate point of the best result
  is selected to avoid unnecessary embedding and output-layer cost.
- 4T total parameters, 128B active parameters and 5M effective context are
  research targets. Dense/MoE and context scaling evidence must select the
  architecture before capital is committed.
- The former blanket prohibition on all post-pretraining full-weight work has
  been corrected. External weights and adapters remain prohibited, while a
  separately governed full-parameter continuation stage for qualified Craftly-
  owned weights is permitted only after the 1B foundation proof.
- A model-quality claim is blocked while the active backend is a mock, protected
  executable evaluations have not run, and no real checkpoint is serving.

The convergence order is therefore fixed: close governance, build promoted
data, run tokenizer A/B qualification, prove the T4 trainer, perform 1B
dense/MoE scaling experiments, activate a native checkpoint, and only then
deepen infrastructure or scale model size. External approvals, GPUs, live
services, load, chaos and soak tests are ordinary production realities. Silent
objective drift, duplicate rank data, ambiguous resume semantics and missing
document boundaries are fundamental errors and must always block release.



