// Command demo simulates a few sequences decoding into a shared paged KV
// cache and prints how blocks are laid out.
package main

import (
	"fmt"
	"math/rand/v2"

	"go_paged_attention/internal/attention"
	"go_paged_attention/internal/kvcache"
)

func main() {
	cfg := kvcache.Config{NumBlocks: 16, BlockSize: 4, NumHeads: 2, HeadDim: 8}
	stride := cfg.NumHeads * cfg.HeadDim
	c := kvcache.New(cfg)
	r := rand.New(rand.NewPCG(42, 0))
	vec := func() []float32 {
		v := make([]float32, stride)
		for i := range v {
			v[i] = r.Float32()
		}
		return v
	}

	lengths := map[int]int{0: 7, 1: 3, 2: 10}
	for id := range 3 {
		c.AddSequence(id)
	}
	// Decode round-robin, the way a batching server would.
	for step := range 10 {
		for id := range 3 {
			if step < lengths[id] {
				if err := c.Append(id, vec(), vec()); err != nil {
					panic(err)
				}
			}
		}
	}

	// Beam-search style fork: shares blocks until it writes.
	c.Fork(0, 3)
	c.Append(3, vec(), vec())

	for id := range 4 {
		s := c.Sequence(id)
		out := attention.Paged(vec(), c, s)
		fmt.Printf("seq %d: %2d tokens, block table %v, out[0]=%.4f\n", id, s.NumTokens, s.Blocks, out[0])
	}
	fmt.Printf("free blocks: %d / %d\n", c.NumFreeBlocks(), cfg.NumBlocks)
}
