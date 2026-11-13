// Package attention computes attention for one new token, once over a plain
// contiguous K/V array and once through a paged KV cache.
package attention

import (
	"math"

	"go_paged_attention/internal/kvcache"
)

// Naive is the reference version. keys and values are laid out
// [token][head][dim]; q and the result are [head][dim].
func Naive(q, keys, values []float32, numTokens, numHeads, headDim int) []float32 {
	out := make([]float32, numHeads*headDim)
	scale := 1 / float32(math.Sqrt(float64(headDim)))
	scores := make([]float32, numTokens)
	for h := range numHeads {
		qh := q[h*headDim : (h+1)*headDim]
		for t := range numTokens {
			off := (t*numHeads + h) * headDim
			scores[t] = dot(qh, keys[off:off+headDim]) * scale
		}
		softmax(scores)
		oh := out[h*headDim : (h+1)*headDim]
		for t := range numTokens {
			off := (t*numHeads + h) * headDim
			axpy(scores[t], values[off:off+headDim], oh)
		}
	}
	return out
}

// Paged gives the same result as Naive, reading each token's K/V through the
// sequence's block table.
func Paged(q []float32, c *kvcache.Cache, s *kvcache.Sequence) []float32 {
	return PagedPrefix(q, c, s, s.NumTokens)
}

// PagedPrefix is Paged over only the first n tokens, as when computing the
// token at position n-1 during prefill.
func PagedPrefix(q []float32, c *kvcache.Cache, s *kvcache.Sequence, n int) []float32 {
	cfg := c.Cfg
	out := make([]float32, cfg.NumHeads*cfg.HeadDim)
	scale := 1 / float32(math.Sqrt(float64(cfg.HeadDim)))
	scores := make([]float32, n)
	for h := range cfg.NumHeads {
		qh := q[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		for t := range n {
			// Token t is in logical block t/BlockSize, at slot t%BlockSize.
			off := c.Offset(s.Blocks[t/cfg.BlockSize], t%cfg.BlockSize) + h*cfg.HeadDim
			scores[t] = dot(qh, c.K[off:off+cfg.HeadDim]) * scale
		}
		softmax(scores)
		oh := out[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		for t := range n {
			off := c.Offset(s.Blocks[t/cfg.BlockSize], t%cfg.BlockSize) + h*cfg.HeadDim
			axpy(scores[t], c.V[off:off+cfg.HeadDim], oh)
		}
	}
	return out
}

// dot returns the dot product of a and b.
func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// axpy adds a*x to y in place.
func axpy(a float32, x, y []float32) {
	for i := range x {
		y[i] += a * x[i]
	}
}

// softmax turns x into probabilities in place. Subtracting the max first
// keeps exp from overflowing.
func softmax(x []float32) {
	maxV := float32(math.Inf(-1))
	for _, v := range x {
		maxV = max(maxV, v)
	}
	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - maxV)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}
