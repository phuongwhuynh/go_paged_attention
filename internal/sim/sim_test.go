package sim

import (
	"testing"
)

// smallConfig is a small memory setup with invariant checks on.
func smallConfig(p Policy) Config {
	return Config{Policy: p, CapacityTokens: 4096, BlockSize: 16, MaxSeqLen: 1024, Cost: LLaMA13BOnA100, Check: true}
}

// small shrinks a preset so tests run quickly.
func small(w Workload, n int) Workload {
	w.Requests = n
	w.PromptMean /= 2
	w.OutputMean /= 2
	w.MaxTokens /= 2
	w.PrefixLen /= 2
	return w
}

// run runs the simulator and fails the test unless every request finishes.
func run(t testing.TB, cfg Config, reqs []Request) Result {
	t.Helper()
	res, err := Run(cfg, reqs)
	if err != nil {
		t.Fatal(err)
	}
	if res.Completed != len(reqs) {
		t.Fatalf("%s: completed %d of %d", cfg.Policy, res.Completed, len(reqs))
	}
	return res
}

// Every preset finishes under every policy with invariant checks on.
func TestAllPresetsWithInvariantChecks(t *testing.T) {
	for _, w := range Presets {
		reqs := small(w, 60).Generate(1, 1024)
		for _, p := range Policies {
			t.Run(w.Name+"/"+p.String(), func(t *testing.T) { run(t, smallConfig(p), reqs) })
		}
	}
}

// The same seed gives exactly the same result.
func TestSeededRunsAreReproducible(t *testing.T) {
	w := small(Presets[1], 80)
	a := run(t, smallConfig(PagedPrefix), w.Generate(7, 1024))
	b := run(t, smallConfig(PagedPrefix), w.Generate(7, 1024))
	if a != b {
		t.Fatalf("same seed, different results:\n%+v\n%+v", a, b)
	}
}

// With the same memory, paged serves clearly more and wastes less.
func TestPagedServesMoreThanContiguous(t *testing.T) {
	reqs := small(Presets[0], 150).Generate(3, 1024)
	c := run(t, smallConfig(Contiguous), reqs)
	p := run(t, smallConfig(Paged), reqs)
	if p.ReqPerSec < 1.5*c.ReqPerSec {
		t.Fatalf("paged %.2f req/s vs contiguous %.2f: expected paged to be clearly faster", p.ReqPerSec, c.ReqPerSec)
	}
	if c.InternalFrag <= p.InternalFrag {
		t.Fatalf("contiguous internal fragmentation %.2f should exceed paged %.2f", c.InternalFrag, p.InternalFrag)
	}
}

// Shared system prompts mostly come from the cache instead of being recomputed.
func TestPrefixCacheHitsSharedSystemPrompts(t *testing.T) {
	reqs := small(Presets[1], 150).Generate(5, 1024)
	p := run(t, smallConfig(Paged), reqs)
	pp := run(t, smallConfig(PagedPrefix), reqs)
	if pp.PrefixHitRate < 0.5 {
		t.Fatalf("prefix hit rate %.2f, want > 0.5 with 4 shared system prompts", pp.PrefixHitRate)
	}
	if pp.PrefillTokens >= p.PrefillTokens {
		t.Fatalf("prefix caching should cut prefill: %d vs %d tokens", pp.PrefillTokens, p.PrefillTokens)
	}
}

// Samples of one prompt share its blocks instead of each storing a copy.
func TestParallelSamplingSharesPromptBlocks(t *testing.T) {
	reqs := small(Presets[2], 60).Generate(9, 1024)
	p := run(t, smallConfig(Paged), reqs)
	if p.Sharing <= 1.2 || p.CoWCopies == 0 {
		t.Fatalf("sharing factor %.2f, CoW copies %d: samples should share prompt blocks", p.Sharing, p.CoWCopies)
	}
}

// Free space split into small holes can't fit a large request; freeing
// everything merges it back into one hole.
func TestArena(t *testing.T) {
	a := newArena(100)
	x, _ := a.alloc(30)
	y, _ := a.alloc(30)
	z, _ := a.alloc(30)
	a.free(x)
	a.free(z)
	if total, largest := a.freeTotal(); total != 70 || largest != 40 {
		t.Fatalf("free %d largest %d, want 70 and 40 (hole of 30 + hole of 40)", total, largest)
	}
	if _, ok := a.alloc(50); ok {
		t.Fatal("50 slots are free in total but no hole is big enough")
	}
	a.free(y)
	if err := a.check(); err != nil {
		t.Fatal(err)
	}
	if total, largest := a.freeTotal(); total != 100 || largest != 100 {
		t.Fatal("freeing everything should coalesce into one hole")
	}
}

// FuzzArena allocates and frees at random; memory must always add up.
func FuzzArena(f *testing.F) {
	f.Add([]byte{10, 20, 200, 5, 201, 30})
	f.Fuzz(func(t *testing.T, ops []byte) {
		a := newArena(256)
		var live []int
		for _, op := range ops {
			if op >= 128 && len(live) > 0 {
				i := int(op) % len(live)
				a.free(live[i])
				live = append(live[:i], live[i+1:]...)
			} else if off, ok := a.alloc(int(op % 64)); ok {
				live = append(live, off)
			}
			if err := a.check(); err != nil {
				t.Fatal(err)
			}
		}
	})
}

// FuzzSim runs random small workloads and memory sizes under every policy.
// Every request must finish and no memory may leak.
func FuzzSim(f *testing.F) {
	f.Add(uint64(1), uint8(0), uint8(4), uint8(1), uint8(2), uint8(0))
	f.Add(uint64(2), uint8(2), uint8(1), uint8(3), uint8(0), uint8(1))
	f.Fuzz(func(t *testing.T, seed uint64, policy, bsLog, samples, prefixes, rate uint8) {
		bs := 1 << (bsLog % 6) // 1..32
		w := Workload{
			Requests: 30, PromptMean: 40, OutputMean: 30, Sigma: 1, MaxTokens: 64,
			Samples: int(samples%4) + 1, NumPrefixes: int(prefixes % 4), PrefixLen: 64,
			Rate: float64(rate % 3 * 20),
		}
		cfg := Config{
			Policy: Policies[int(policy)%len(Policies)], BlockSize: bs, MaxSeqLen: 256,
			CapacityTokens: 4 * (256 + 2*bs) * (int(seed%3) + 1), Cost: LLaMA13BOnA100, Check: true,
		}
		run(t, cfg, w.Generate(seed, cfg.MaxSeqLen))
	})
}
