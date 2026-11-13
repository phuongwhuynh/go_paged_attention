package attention

import (
	"math"
	"math/rand/v2"
	"testing"

	"go_paged_attention/internal/kvcache"
)

// randVec returns n random values in [-1, 1).
func randVec(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

// Paged must match Naive even when a sequence's blocks are scattered.
func TestPagedMatchesNaive(t *testing.T) {
	cfg := kvcache.Config{NumBlocks: 16, BlockSize: 4, NumHeads: 2, HeadDim: 8}
	stride := cfg.NumHeads * cfg.HeadDim
	r := rand.New(rand.NewPCG(1, 2))

	c := kvcache.New(cfg)
	// Interleave two sequences so their blocks are not contiguous in memory.
	c.AddSequence(0)
	c.AddSequence(1)
	var keys, values []float32
	const numTokens = 11
	for range numTokens {
		k, v := randVec(r, stride), randVec(r, stride)
		keys, values = append(keys, k...), append(values, v...)
		if err := c.Append(0, 0, k, v); err != nil {
			t.Fatal(err)
		}
		if err := c.Append(1, 0, randVec(r, stride), randVec(r, stride)); err != nil {
			t.Fatal(err)
		}
	}

	q := randVec(r, stride)
	want := Naive(q, keys, values, numTokens, cfg.NumHeads, cfg.HeadDim)
	got := Paged(q, c, c.Sequence(0))
	for i := range want {
		if math.Abs(float64(want[i]-got[i])) > 1e-5 {
			t.Fatalf("out[%d]: paged=%v naive=%v", i, got[i], want[i])
		}
	}
}
