# go_paged_attention

A small Go project for studying **PagedAttention**, the KV-cache memory scheme from vLLM
([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)).

## Idea in one paragraph

During decoding, every token's keys and values (the KV cache) are kept so later tokens can attend to them.
Reserving one contiguous buffer per request for the maximum length wastes most of it. PagedAttention splits
the cache into fixed-size **blocks** (like OS pages). Each sequence keeps a **block table** that maps logical
block `i` to a physical block ID, blocks are allocated only when needed, and blocks can be **shared** between
sequences (prefix sharing, beam search) with **copy-on-write**.

## Layout

```
internal/block      physical block pool: allocate / retain / release with refcounts
internal/kvcache    K/V storage [block][slot][head][dim], block tables, append, fork, free
internal/attention  Naive (contiguous) vs Paged attention for one decode step; test checks they match
cmd/demo            simulates several sequences decoding + a fork, prints block tables
```

## Usage

```sh
make test   # go test ./...
make run    # go run ./cmd/demo
```

## Ideas to explore next

- [ ] Benchmark `Naive` vs `Paged` and different block sizes
- [ ] Prefix caching: hash full blocks and reuse them across requests
- [ ] A scheduler that preempts / swaps out sequences when blocks run out
- [ ] Measure internal fragmentation vs a contiguous max-length allocation
- [ ] Parallelize attention across heads with goroutines
