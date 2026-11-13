# go_paged_attention

A small Go project for studying **PagedAttention**, the KV-cache memory scheme from vLLM
([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)).

## Idea in one paragraph

During decoding, every token's keys and values (the KV cache) are kept so later tokens can attend to them.
Reserving one contiguous buffer per request for the maximum length wastes most of it. PagedAttention splits
the cache into fixed-size **blocks** (like OS pages). Each sequence keeps a **block table** that maps logical
block `i` to a physical block ID, blocks are allocated only when needed, and blocks can be **shared** between
sequences (prefix sharing, beam search) with **copy-on-write**.

## What's here

- **Paged KV cache manager** (`internal/kvcache`, `internal/block`): fixed-size blocks, per-sequence block
  tables, reference counting, and copy-on-write forking.
- **Prefix sharing** (`internal/kvcache/radix.go`): every full block is indexed in a radix tree keyed by token
  IDs, so prompts that start with already-computed tokens reuse those blocks. Blocks no sequence uses stay
  cached and are evicted least-recently-used first when memory runs short.
- **Discrete-event serving simulator** (`internal/sim`): replays seeded synthetic workloads under a fixed KV
  memory budget. Requests queue until memory is available, are batched every step, and are preempted
  (and later recomputed) if blocks run out. Policies compared:
  - `contiguous`: reserves `prompt + max_tokens` per sample in one region (first-fit arena), the pre-vLLM way
  - `paged`: blocks allocated on demand; parallel samples fork and share prompt blocks
  - `paged+prefix`: paged plus the radix-tree prefix cache

  The paged policies drive the real cache manager in metadata-only mode. Step time comes from a simple
  roofline cost model of LLaMA-13B on an A100-40GB.

- **Correctness**: `CheckInvariants` checks refcounts against every block-table and tree reference, the
  allocator's free list, radix-tree structure, and that cached blocks hold the tokens their sequences expect.
  The checks run after every operation in tests, fuzzing, and `-check` mode. Fuzz targets: `FuzzCache`
  (random op sequences against a reference model), `FuzzArena`, and `FuzzSim` (random workloads, every request
  must finish with no leaked memory).
- **Attention kernels** (`internal/attention`): naive (contiguous) vs paged attention, tested to give the
  same output.

## Layout

```
internal/block      physical block pool: allocate / retain / release with refcounts, invariant check
internal/kvcache    block tables, append, fork + copy-on-write, free, radix-tree prefix cache, invariants
internal/attention  Naive (contiguous) vs Paged attention for one decode step
internal/sim        discrete-event simulator: workloads, scheduler, contiguous arena, cost model, metrics
cmd/sim             runs every workload under every policy and prints the comparison
cmd/demo            a few sequences decoding + a fork, prints block tables
```

## Usage

```sh
make test    # unit tests, including invariant-checked simulations
make sim     # go run ./cmd/sim  (flags: -workload -n -rate -seed -block -mem -check)
make fuzz    # fuzz the cache, the arena and the simulator (FUZZTIME=30s each)
make run     # block-table demo
```

## Results (`make sim`, seed 1)

LLaMA-13B on one A100-40GB leaves room for 12,207 tokens of KV cache. Same memory for every policy:

| workload                                   | policy       | req/s       | KV utilization | internal frag | external frag | prefix hit |
| ------------------------------------------ | ------------ | ----------- | -------------- | ------------- | ------------- | ---------- |
| chat (ShareGPT-like lengths)               | contiguous   | 1.37        | 31%            | 50%           | 11%           |            |
|                                            | paged        | 3.02 (2.2x) | 95%            | 2%            | 0%            |            |
| system prompt (4 shared 768-token prompts) | contiguous   | 2.24        | 58%            | 25%           | 8%            |            |
|                                            | paged        | 2.92 (1.3x) | 94%            | 1%            | 0%            |            |
|                                            | paged+prefix | 9.40 (4.2x) | 91%            | 3%            | 0%            | 94%        |
| parallel sampling (n=4)                    | contiguous   | 0.98        | 33%            | 34%           | 20%           |            |
|                                            | paged        | 2.55 (2.6x) | 92%            | 3%            | 0%            |            |

Reserving `max_tokens` up front leaves most of the contiguous baseline's memory empty, which caps its batch
size. Paged allocation wastes at most one partly filled block per sequence, so it fits 3-6x more sequences
in the same memory. Prefix caching then lets requests with a common system prompt share it instead of each
storing a copy.
