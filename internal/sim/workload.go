package sim

import (
	"math"
	"math/rand/v2"
)

// Request is one client request.
type Request struct {
	ID        int
	Arrival   float64 // seconds
	Prompt    []int32
	MaxTokens int   // client's limit per sample; contiguous reserves this much
	Outputs   []int // real output length per sample; the scheduler can't see these
}

// Workload describes a synthetic request mix. Lengths are lognormal: mostly
// short, with a long tail.
type Workload struct {
	Name        string
	Requests    int
	Rate        float64 // mean arrivals per second (Poisson); 0 = all queued at t=0
	PromptMean  float64 // mean prompt length, excluding any shared prefix
	OutputMean  float64
	Sigma       float64 // lognormal shape
	MaxTokens   int
	Samples     int // outputs per request; 0 or 1 = one
	NumPrefixes int // shared system prompts, picked with Zipf popularity
	PrefixLen   int
}

// Presets are the built-in workloads, each aimed at one feature.
var Presets = []Workload{
	// Lengths match the vLLM paper's ShareGPT averages. Shows the waste
	// from reserving max_tokens.
	{Name: "chat", Requests: 1000, PromptMean: 161, OutputMean: 338, Sigma: 1, MaxTokens: 1024},
	// Requests share 4 long system prompts. Shows the prefix cache.
	{Name: "system-prompt", Requests: 1000, PromptMean: 48, OutputMean: 128, Sigma: 1, MaxTokens: 512,
		NumPrefixes: 4, PrefixLen: 768},
	// 4 outputs per prompt. Shows fork and copy-on-write.
	{Name: "parallel-sampling", Requests: 500, PromptMean: 256, OutputMean: 160, Sigma: 1, MaxTokens: 512,
		Samples: 4},
}

// PresetByName looks up a preset by its name.
func PresetByName(name string) (Workload, bool) {
	for _, w := range Presets {
		if w.Name == name {
			return w, true
		}
	}
	return Workload{}, false
}

const vocab = 32000

// Generate creates the workload's requests. The same seed always gives the
// same requests.
func (w Workload) Generate(seed uint64, maxSeqLen int) []Request {
	r := rand.New(rand.NewPCG(seed, 0x5eed))
	sigma := w.Sigma
	logn := func(mean float64) int {
		mu := math.Log(mean) - sigma*sigma/2
		return max(1, int(math.Round(math.Exp(mu+sigma*r.NormFloat64()))))
	}
	randTokens := func(n int) []int32 {
		t := make([]int32, n)
		for i := range t {
			t[i] = int32(r.IntN(vocab))
		}
		return t
	}

	prefixes := make([][]int32, w.NumPrefixes)
	weights := make([]float64, w.NumPrefixes)
	totalW := 0.0
	for i := range prefixes {
		prefixes[i] = randTokens(w.PrefixLen)
		weights[i] = 1 / float64(i+1) // a few prompts dominate
		totalW += weights[i]
	}

	n := max(1, w.Samples)
	reqs := make([]Request, w.Requests)
	t := 0.0
	for i := range reqs {
		var prompt []int32
		if w.NumPrefixes > 0 {
			// Pick a prefix in proportion to its weight.
			x := r.Float64() * totalW
			k := 0
			for k < len(weights)-1 && x >= weights[k] {
				x -= weights[k]
				k++
			}
			prompt = append(prompt, prefixes[k]...)
		}
		prompt = append(prompt, randTokens(logn(w.PromptMean))...)
		prompt = prompt[:min(len(prompt), maxSeqLen-1)]
		maxTok := min(w.MaxTokens, maxSeqLen-len(prompt))
		outs := make([]int, n)
		for j := range outs {
			outs[j] = min(logn(w.OutputMean), maxTok)
		}
		if w.Rate > 0 {
			t += r.ExpFloat64() / w.Rate
		}
		reqs[i] = Request{ID: i, Arrival: t, Prompt: prompt, MaxTokens: maxTok, Outputs: outs}
	}
	return reqs
}

// outputToken is the k-th token that sample j of request id generates.
// It's a hash, so recomputing after preemption gives the same tokens.
func outputToken(id, j, k int) int32 {
	h := uint64(id)*0x9E3779B97F4A7C15 ^ uint64(j)*0xBF58476D1CE4E5B9 ^ uint64(k)*0x94D049BB133111EB
	h ^= h >> 31
	return int32(h % vocab)
}
