// Package sim is a discrete-event simulator of an LLM server with a fixed
// amount of KV-cache memory. It replays a seeded workload under one memory
// policy and reports throughput, latency and how well memory was used.
//
// The paged policies run the real kvcache code, minus the tensors. Step
// times come from CostModel.
package sim

import (
	"container/heap"
	"fmt"

	"go_paged_attention/internal/kvcache"
)

type Policy int

const (
	// Contiguous reserves prompt + max_tokens per sample up front, in one
	// region, the way servers did before vLLM.
	Contiguous Policy = iota
	// Paged allocates blocks as sequences grow. Parallel samples share the
	// prompt's blocks.
	Paged
	// PagedPrefix is Paged plus the prefix cache.
	PagedPrefix
)

var Policies = []Policy{Contiguous, Paged, PagedPrefix}

// String returns the policy's name as shown in results.
func (p Policy) String() string {
	return [...]string{"contiguous", "paged", "paged+prefix"}[p]
}

type Config struct {
	Policy         Policy
	CapacityTokens int // KV memory in tokens, the same for every policy
	BlockSize      int
	MaxSeqLen      int
	MaxBatch       int // max running sequences, 0 = no limit
	Cost           CostModel
	Check          bool // run invariant checks after every operation (slow)
}

// Result summarizes one run. Memory fields are fractions of capacity,
// averaged over time.
type Result struct {
	Policy      Policy
	Completed   int
	Makespan    float64 // seconds from first arrival to last completion
	ReqPerSec   float64
	TokPerSec   float64 // generated tokens per second
	MeanLatency float64 // seconds from arrival to completion
	NormLatency float64 // latency per output token
	AvgBatch    float64 // running sequences per step
	Steps       int

	Utilization  float64 // holding tokens running sequences need
	InternalFrag float64 // given to running sequences but still empty
	ExternalFrag float64 // free, but outside the largest free hole
	Cached       float64 // held only by the prefix cache
	Sharing      float64 // tokens seen by sequences / tokens stored

	PromptTokens  int
	PrefillTokens int // prompt and recompute tokens actually computed
	PrefixHitRate float64
	Preemptions   int
	Evictions     int
	CoWCopies     int
}

// sample is one output stream of a request.
type sample struct {
	seqID     int
	out       int // tokens it will generate
	generated int
	region    int // contiguous: arena offset; -1 once freed
	ctx       int // contiguous: tokens stored
}

// group is a request and its samples, scheduled together.
type group struct {
	req     *Request
	samples []*sample
	freed   int // samples finished and freed
}

// live returns the samples still generating.
func (g *group) live() []*sample {
	var l []*sample
	for _, s := range g.samples {
		if s.generated < s.out {
			l = append(l, s)
		}
	}
	return l
}

// Sequence IDs are request*16 + sample. ID request*16 + 15 is reserved for
// the prompt that samples fork from.
const maxSamples = 15

type event struct {
	t    float64
	seq  int // breaks ties in arrival order, so runs are deterministic
	req  *Request
	done bool // a step finished; otherwise req arrived
}

// eventQueue is a min-heap of events ordered by time.
type eventQueue []event

// Len, Less, Swap, Push and Pop implement heap.Interface.
func (q eventQueue) Len() int { return len(q) }

// Less orders events by time, then by the order they were scheduled.
func (q eventQueue) Less(i, j int) bool {
	if q[i].t != q[j].t {
		return q[i].t < q[j].t
	}
	return q[i].seq < q[j].seq
}
func (q eventQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *eventQueue) Push(x any)   { *q = append(*q, x.(event)) }
func (q *eventQueue) Pop() any {
	old := *q
	e := old[len(old)-1]
	*q = old[:len(old)-1]
	return e
}

type engine struct {
	cfg     Config
	now     float64
	events  eventQueue
	nextSeq int
	busy    bool // a step is in flight
	waiting []*group
	running []*group

	cache *kvcache.Cache // paged policies
	arena *arena         // contiguous

	// Running totals, turned into a Result by finish.
	res       Result
	start     float64
	stepEnd   float64
	batchSum  int
	timeTotal float64
	memSum    struct{ used, internal, external, cached, logical, physical float64 }
	latSum    float64
	normSum   float64
	tokens    int
}

// push schedules an event.
func (e *engine) push(ev event) {
	ev.seq = e.nextSeq
	e.nextSeq++
	heap.Push(&e.events, ev)
}

// Run serves every request to completion and reports what happened. It
// fails if requests can't fit, the engine stalls, or memory leaks.
func Run(cfg Config, reqs []Request) (Result, error) {
	if cfg.BlockSize < 1 || cfg.MaxSeqLen > cfg.CapacityTokens {
		return Result{}, fmt.Errorf("sim: need BlockSize >= 1 and MaxSeqLen <= capacity")
	}
	e := &engine{cfg: cfg, res: Result{Policy: cfg.Policy}}
	if cfg.Policy == Contiguous {
		e.arena = newArena(cfg.CapacityTokens)
	} else {
		e.cache = kvcache.New(kvcache.Config{
			NumBlocks: cfg.CapacityTokens / cfg.BlockSize, BlockSize: cfg.BlockSize,
			PrefixCaching: cfg.Policy == PagedPrefix,
		})
	}
	for i := range reqs {
		r := &reqs[i]
		if len(r.Prompt) == 0 || len(r.Outputs) == 0 || len(r.Outputs) > maxSamples ||
			len(r.Prompt)+r.MaxTokens > cfg.MaxSeqLen {
			return Result{}, fmt.Errorf("sim: request %d does not fit MaxSeqLen %d", r.ID, cfg.MaxSeqLen)
		}
		// A request that can't fit even on its own would wait forever.
		if len(r.Outputs)*(len(r.Prompt)+r.MaxTokens+2*cfg.BlockSize) > cfg.CapacityTokens {
			return Result{}, fmt.Errorf("sim: request %d needs more than the whole capacity", r.ID)
		}
		for _, o := range r.Outputs {
			if o < 1 || o > r.MaxTokens {
				return Result{}, fmt.Errorf("sim: request %d output %d outside [1, %d]", r.ID, o, r.MaxTokens)
			}
		}
		e.push(event{t: r.Arrival, req: r})
	}
	if len(reqs) > 0 {
		e.start = e.events[0].t
	}

	// Main loop: handle the next event, then start a step if the engine is idle.
	for e.events.Len() > 0 {
		ev := heap.Pop(&e.events).(event)
		e.now = ev.t
		if ev.done {
			e.busy = false
			if err := e.finishStep(); err != nil {
				return e.res, err
			}
		} else {
			g := &group{req: ev.req}
			for j, o := range ev.req.Outputs {
				g.samples = append(g.samples, &sample{seqID: ev.req.ID*16 + j, out: o})
			}
			e.waiting = append(e.waiting, g)
		}
		if !e.busy {
			if err := e.startStep(); err != nil {
				return e.res, err
			}
		}
	}
	if e.res.Completed != len(reqs) {
		return e.res, fmt.Errorf("sim: stalled with %d of %d requests done", e.res.Completed, len(reqs))
	}
	// Everything must be handed back by the end.
	if e.cache != nil {
		live := 0
		e.cache.Sequences(func(*kvcache.Sequence) { live++ })
		if free := e.cache.NumAvailableBlocks(); live != 0 || free != e.cache.Cfg.NumBlocks {
			return e.res, fmt.Errorf("sim: leak: %d sequences live, %d of %d blocks available", live, free, e.cache.Cfg.NumBlocks)
		}
	} else if total, _ := e.arena.freeTotal(); total != e.arena.capacity {
		return e.res, fmt.Errorf("sim: leak: %d of %d arena slots free", total, e.arena.capacity)
	}
	return e.finish(), nil
}

// check runs the memory invariant checks when Config.Check is on.
func (e *engine) check() error {
	if !e.cfg.Check {
		return nil
	}
	if e.cache != nil {
		return e.cache.CheckInvariants()
	}
	return e.arena.check()
}

// startStep plans the next step and schedules its completion. With nothing
// to run, the engine goes idle until the next arrival.
func (e *engine) startStep() error {
	computeTokens, kvTokens, err := e.schedule()
	if err != nil {
		return err
	}
	if len(e.running) == 0 {
		e.busy = false
		return nil
	}
	dt := e.cfg.Cost.StepTime(computeTokens, kvTokens)
	e.sampleMemory(dt)
	e.res.Steps++
	e.busy = true
	e.push(event{t: e.now + dt, done: true})
	return nil
}

// schedule builds one step: preempt if memory is short, give every running
// sample one more token, then admit waiting requests while they fit. It
// returns the step's work for the cost model.
func (e *engine) schedule() (computeTokens, kvTokens int, err error) {
	preempted := false
	if e.cache != nil {
		// need counts the new blocks a request's next tokens will take.
		need := func(g *group) int {
			n := 0
			for _, s := range g.live() {
				if e.cache.NeedsBlock(s.seqID) {
					n++
				}
			}
			return n
		}
		total := 0
		for _, g := range e.running {
			total += need(g)
		}
		// Not enough room: evict the newest request. It goes back to the
		// front of the queue and is recomputed when it's admitted again.
		for total > e.cache.NumAvailableBlocks() {
			v := e.running[len(e.running)-1]
			e.running = e.running[:len(e.running)-1]
			total -= need(v)
			for _, s := range v.live() {
				e.cache.Free(s.seqID)
			}
			e.waiting = append([]*group{v}, e.waiting...)
			e.res.Preemptions++
			preempted = true
			if err := e.check(); err != nil {
				return 0, 0, err
			}
		}
	}

	for _, g := range e.running {
		for _, s := range g.live() {
			c, k, err := e.decode(g, s)
			if err != nil {
				return 0, 0, err
			}
			computeTokens += c
			kvTokens += k
		}
	}

	// Admit in arrival order. Skip this right after a preemption, otherwise
	// the new request would just take the memory we freed.
	for !preempted && len(e.waiting) > 0 {
		g := e.waiting[0]
		if e.cfg.MaxBatch > 0 && e.runningSeqs()+len(g.live()) > e.cfg.MaxBatch && len(e.running) > 0 {
			break
		}
		c, k, ok, err := e.admit(g)
		if err != nil {
			return 0, 0, err
		}
		if !ok {
			break
		}
		e.waiting = e.waiting[1:]
		e.running = append(e.running, g)
		computeTokens += c
		kvTokens += k
	}
	return computeTokens, kvTokens, nil
}

// runningSeqs counts the samples currently generating.
func (e *engine) runningSeqs() int {
	n := 0
	for _, g := range e.running {
		n += len(g.live())
	}
	return n
}

// decode generates one token for sample s and stores its KV.
func (e *engine) decode(g *group, s *sample) (computeTokens, kvTokens int, err error) {
	tok := outputToken(g.req.ID, s.seqID%16, s.generated)
	ctx := 0
	if e.cache != nil {
		if err := e.cache.Append(s.seqID, tok, nil, nil); err != nil {
			return 0, 0, fmt.Errorf("sim: decode append for seq %d: %w", s.seqID, err)
		}
		ctx = e.cache.Sequence(s.seqID).NumTokens
	} else {
		s.ctx++
		ctx = s.ctx
	}
	s.generated++
	e.tokens++
	return 1, ctx, e.check()
}

// admit tries to start request g: compute its prompt (plus any tokens it
// had generated before being preempted) and its next token. If memory runs
// out partway, it undoes everything and returns ok = false.
func (e *engine) admit(g *group) (computeTokens, kvTokens int, ok bool, err error) {
	live := g.live()
	prompt := g.req.Prompt
	P := len(prompt)

	if e.arena != nil {
		// Each sample gets one region big enough for its longest possible output.
		size := P + g.req.MaxTokens
		var got []*sample
		for _, s := range live {
			off, ok := e.arena.alloc(size)
			if !ok {
				for _, x := range got {
					e.arena.free(x.region)
				}
				return 0, 0, false, e.check()
			}
			s.region, s.ctx = off, P+s.generated
			got = append(got, s)
		}
		computeTokens = P // computed once, then copied to each sample
		for _, s := range live {
			computeTokens += s.generated
			kvTokens += P + s.generated
		}
		e.res.PromptTokens += P
		e.res.PrefillTokens += computeTokens
		if err := e.check(); err != nil {
			return 0, 0, false, err
		}
		for _, s := range live {
			c, k, err := e.decode(g, s)
			if err != nil {
				return 0, 0, false, err
			}
			computeTokens += c
			kvTokens += k
		}
		return computeTokens, kvTokens, true, nil
	}

	// Rough check before doing any work. Keep 1% of blocks spare, so a new
	// request doesn't force a preemption on the very next step.
	bs := e.cfg.BlockSize
	est := (P - e.cache.CachedPrefixLen(prompt) + bs - 1) / bs
	for _, s := range live {
		est += (s.generated+1)/bs + 2
	}
	reserve := max(1, e.cache.Cfg.NumBlocks/100)
	if len(e.running) == 0 {
		reserve = 0
	}
	if est > e.cache.NumAvailableBlocks()-reserve {
		return 0, 0, false, nil
	}

	base := g.req.ID*16 + maxSamples
	rollback := func() error {
		e.cache.Free(base)
		for _, s := range live {
			e.cache.Free(s.seqID)
		}
		return e.check()
	}
	// Compute the prompt once, reusing whatever the prefix cache has.
	_, hit := e.cache.AddSequenceWithPrefix(base, prompt)
	for _, t := range prompt[hit:] {
		if err := e.cache.Append(base, t, nil, nil); err != nil {
			return 0, 0, false, rollback()
		}
	}
	if err := e.check(); err != nil {
		return 0, 0, false, err
	}
	computeTokens = P - hit
	// Every sample forks the prompt, so they all share its blocks.
	for _, s := range live {
		if _, err := e.cache.Fork(base, s.seqID); err != nil {
			return 0, 0, false, err
		}
		for k := range s.generated { // recompute tokens lost to preemption
			if err := e.cache.Append(s.seqID, outputToken(g.req.ID, s.seqID%16, k), nil, nil); err != nil {
				return 0, 0, false, rollback()
			}
		}
		computeTokens += s.generated
		kvTokens += P + s.generated
	}
	e.cache.Free(base)
	if err := e.check(); err != nil {
		return 0, 0, false, err
	}
	// Make sure every sample's next token fits before generating any.
	needs := 0
	for _, s := range live {
		if e.cache.NeedsBlock(s.seqID) {
			needs++
		}
	}
	if needs > e.cache.NumAvailableBlocks() {
		return 0, 0, false, rollback()
	}
	for _, s := range live {
		c, k, err := e.decode(g, s)
		if err != nil {
			return 0, 0, false, err
		}
		computeTokens += c
		kvTokens += k
	}
	e.res.PromptTokens += P
	e.res.PrefillTokens += computeTokens - len(live)
	return computeTokens, kvTokens, true, nil
}

// finishStep frees samples that just finished and records requests whose
// samples are all done.
func (e *engine) finishStep() error {
	keep := e.running[:0]
	for _, g := range e.running {
		for _, s := range g.samples {
			if s.generated == s.out && s.region >= 0 {
				if e.cache != nil {
					e.cache.Free(s.seqID)
				} else {
					e.arena.free(s.region)
				}
				s.region = -1
				g.freed++
			}
		}
		if err := e.check(); err != nil {
			return err
		}
		if g.freed < len(g.samples) {
			keep = append(keep, g)
			continue
		}
		e.res.Completed++
		lat := e.now - g.req.Arrival
		out := 0
		for _, s := range g.samples {
			out = max(out, s.out)
		}
		e.latSum += lat
		e.normSum += lat / float64(out)
		e.stepEnd = e.now
	}
	e.running = keep
	return nil
}

// sampleMemory records how memory is split up during a step lasting dt.
func (e *engine) sampleMemory(dt float64) {
	capacity := float64(e.cfg.CapacityTokens)
	var used, internal, external, cached, logical, physical float64
	if e.cache != nil {
		// Count each physical block once, even if several sequences share it.
		bs := e.cfg.BlockSize
		fill := map[int]int{} // block -> slots in use
		e.cache.Sequences(func(s *kvcache.Sequence) {
			logical += float64(s.NumTokens)
			for i, b := range s.Blocks {
				fill[b] = max(fill[b], min(bs, s.NumTokens-i*bs))
			}
		})
		for _, f := range fill {
			used += float64(f)
			internal += float64(bs - f)
		}
		physical = used
		cached = float64(e.cache.NumCachedBlocks() * bs)
	} else {
		for _, g := range e.running {
			for _, s := range g.samples {
				if s.region >= 0 {
					used += float64(s.ctx)
					internal += float64(e.arena.used[s.region] - s.ctx)
				}
			}
		}
		logical, physical = used, used
		total, largest := e.arena.freeTotal()
		external = float64(total - largest)
	}
	e.memSum.used += used / capacity * dt
	e.memSum.internal += internal / capacity * dt
	e.memSum.external += external / capacity * dt
	e.memSum.cached += cached / capacity * dt
	e.memSum.logical += logical * dt
	e.memSum.physical += physical * dt
	e.timeTotal += dt
	e.batchSum += e.runningSeqs()
}

// finish turns the running totals into averages and rates.
func (e *engine) finish() Result {
	r := e.res
	r.Makespan = e.stepEnd - e.start
	if r.Makespan > 0 {
		r.ReqPerSec = float64(r.Completed) / r.Makespan
		r.TokPerSec = float64(e.tokens) / r.Makespan
	}
	if r.Completed > 0 {
		r.MeanLatency = e.latSum / float64(r.Completed)
		r.NormLatency = e.normSum / float64(r.Completed)
	}
	if r.Steps > 0 {
		r.AvgBatch = float64(e.batchSum) / float64(r.Steps)
	}
	if e.timeTotal > 0 {
		r.Utilization = e.memSum.used / e.timeTotal
		r.InternalFrag = e.memSum.internal / e.timeTotal
		r.ExternalFrag = e.memSum.external / e.timeTotal
		r.Cached = e.memSum.cached / e.timeTotal
	}
	if e.memSum.physical > 0 {
		r.Sharing = e.memSum.logical / e.memSum.physical
	}
	if e.cache != nil {
		st := e.cache.Stats
		if st.PromptTokens > 0 {
			r.PrefixHitRate = float64(st.HitTokens) / float64(st.PromptTokens)
		}
		r.Evictions, r.CoWCopies = st.Evictions, st.CoWCopies
	}
	return r
}
