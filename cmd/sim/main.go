// Command sim runs each workload under each memory policy, with the same
// memory, and prints a comparison table.
//
//	go run ./cmd/sim
//	go run ./cmd/sim -workload chat -rate 2 -seed 3
package main

import (
	"flag"
	"fmt"
	"os"
	"text/tabwriter"

	"go_paged_attention/internal/sim"
)

// main parses flags, then runs and prints one table per workload.
func main() {
	workload := flag.String("workload", "all", "preset: all, chat, system-prompt, parallel-sampling")
	n := flag.Int("n", 0, "number of requests (0 = preset default)")
	rate := flag.Float64("rate", 0, "arrival rate in req/s, Poisson (0 = all queued at t=0, measures max throughput)")
	seed := flag.Uint64("seed", 1, "workload seed")
	bs := flag.Int("block", 16, "paged block size in tokens")
	memUtil := flag.Float64("mem", 0.9, "fraction of GPU memory usable (weights + KV)")
	maxSeqLen := flag.Int("max-seq-len", 2048, "model context length")
	check := flag.Bool("check", false, "run invariant checks after every operation (slow)")
	flag.Parse()

	cost := sim.LLaMA13BOnA100
	capacity := cost.KVCapacityTokens(*memUtil)
	fmt.Printf("%s: KV capacity %d tokens (%.0f MB/1k tokens), block size %d, seed %d\n\n",
		cost.Name, capacity, cost.KVBytesPerToken*1000/1e6, *bs, *seed)

	var workloads []sim.Workload
	if *workload == "all" {
		workloads = sim.Presets
	} else if w, ok := sim.PresetByName(*workload); ok {
		workloads = []sim.Workload{w}
	} else {
		fmt.Fprintf(os.Stderr, "unknown workload %q\n", *workload)
		os.Exit(2)
	}

	for _, w := range workloads {
		if *n > 0 {
			w.Requests = *n
		}
		w.Rate = *rate
		reqs := w.Generate(*seed, *maxSeqLen)
		desc := fmt.Sprintf("%d requests, prompt ~%.0f, output ~%.0f, max_tokens %d", w.Requests, w.PromptMean, w.OutputMean, w.MaxTokens)
		if w.NumPrefixes > 0 {
			desc += fmt.Sprintf(", %d shared %d-token system prompts", w.NumPrefixes, w.PrefixLen)
		}
		if w.Samples > 1 {
			desc += fmt.Sprintf(", %d samples each", w.Samples)
		}
		fmt.Printf("== %s: %s ==\n", w.Name, desc)

		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
		fmt.Fprintln(tw, "policy\treq/s\ttok/s\tlatency s\tbatch\t│\tutil\tint-frag\text-frag\tcached\tsharing\t│\tprefix hit\tprefill tok\tpreempt\tevict\tCoW\t")
		var base float64
		for _, p := range sim.Policies {
			r, err := sim.Run(sim.Config{
				Policy: p, CapacityTokens: capacity, BlockSize: *bs, MaxSeqLen: *maxSeqLen,
				MaxBatch: 256, Cost: cost, Check: *check,
			}, reqs)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			if p == sim.Contiguous {
				base = r.ReqPerSec
			}
			fmt.Fprintf(tw, "%s\t%.2f (%.1fx)\t%.0f\t%.1f\t%.1f\t│\t%.0f%%\t%.0f%%\t%.0f%%\t%.0f%%\t%.2fx\t│\t%.0f%%\t%d\t%d\t%d\t%d\t\n",
				p, r.ReqPerSec, r.ReqPerSec/base, r.TokPerSec, r.MeanLatency, r.AvgBatch,
				100*r.Utilization, 100*r.InternalFrag, 100*r.ExternalFrag, 100*r.Cached, r.Sharing,
				100*r.PrefixHitRate, r.PrefillTokens, r.Preemptions, r.Evictions, r.CoWCopies)
		}
		tw.Flush()
		fmt.Println()
	}
	fmt.Println("util = KV slots holding live tokens; int-frag = allocated but empty; ext-frag = free but outside the")
	fmt.Println("largest hole; cached = evictable prefix-cache blocks; sharing = logical / physical tokens. All are")
	fmt.Println("time-averaged fractions of KV capacity.")
}
