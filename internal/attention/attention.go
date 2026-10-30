// Package attention implements single-query (decode-step) attention two ways:
// over contiguous K/V (the reference) and over a paged KV cache.
package attention

import (
	"math"

	"go_paged_attention/internal/kvcache"
)

// Naive computes attention for one query token against contiguous keys and
// values laid out as [token][head][dim]. q is [head][dim]; returns [head][dim].
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

// Paged computes the same result as Naive, but reads K/V through the
// sequence's block table: logical token t lives at physical block
// Blocks[t/BlockSize], slot t%BlockSize.
func Paged(q []float32, c *kvcache.Cache, s *kvcache.Sequence) []float32 {
	cfg := c.Cfg
	out := make([]float32, cfg.NumHeads*cfg.HeadDim)
	scale := 1 / float32(math.Sqrt(float64(cfg.HeadDim)))
	scores := make([]float32, s.NumTokens)
	for h := range cfg.NumHeads {
		qh := q[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		for t := range s.NumTokens {
			off := c.Offset(s.Blocks[t/cfg.BlockSize], t%cfg.BlockSize) + h*cfg.HeadDim
			scores[t] = dot(qh, c.K[off:off+cfg.HeadDim]) * scale
		}
		softmax(scores)
		oh := out[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		for t := range s.NumTokens {
			off := c.Offset(s.Blocks[t/cfg.BlockSize], t%cfg.BlockSize) + h*cfg.HeadDim
			axpy(scores[t], c.V[off:off+cfg.HeadDim], oh)
		}
	}
	return out
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// axpy: y += a*x
func axpy(a float32, x, y []float32) {
	for i := range x {
		y[i] += a * x[i]
	}
}

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
