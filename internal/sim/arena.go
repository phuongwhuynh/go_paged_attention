package sim

import (
	"fmt"
	"sort"
)

// arena is the contiguous baseline's memory: one range of token slots,
// split first-fit into regions. Unlike blocks, it can end up with free
// space scattered in holes too small to use.
type arena struct {
	capacity int
	holes    []span      // free ranges, sorted by start, never adjacent
	used     map[int]int // start -> size of each allocated region
}

type span struct{ start, size int }

// newArena creates an arena with all capacity slots free.
func newArena(capacity int) *arena {
	return &arena{capacity: capacity, holes: []span{{0, capacity}}, used: map[int]int{}}
}

// alloc carves size slots out of the first hole big enough and returns
// where they start.
func (a *arena) alloc(size int) (int, bool) {
	if size <= 0 {
		return 0, false
	}
	for i, h := range a.holes {
		if h.size < size {
			continue
		}
		if h.size == size {
			a.holes = append(a.holes[:i], a.holes[i+1:]...)
		} else {
			a.holes[i] = span{h.start + size, h.size - size}
		}
		a.used[h.start] = size
		return h.start, true
	}
	return 0, false
}

// free returns the region starting at start, merging it with any free
// neighbours.
func (a *arena) free(start int) {
	size, ok := a.used[start]
	if !ok {
		panic(fmt.Sprintf("arena: free of unallocated region %d", start))
	}
	delete(a.used, start)
	i := sort.Search(len(a.holes), func(i int) bool { return a.holes[i].start > start })
	a.holes = append(a.holes, span{})
	copy(a.holes[i+1:], a.holes[i:])
	a.holes[i] = span{start, size}
	if i+1 < len(a.holes) && a.holes[i].start+a.holes[i].size == a.holes[i+1].start {
		a.holes[i].size += a.holes[i+1].size
		a.holes = append(a.holes[:i+1], a.holes[i+2:]...)
	}
	if i > 0 && a.holes[i-1].start+a.holes[i-1].size == a.holes[i].start {
		a.holes[i-1].size += a.holes[i].size
		a.holes = append(a.holes[:i], a.holes[i+1:]...)
	}
}

// freeTotal returns the total free space and the size of the largest hole.
func (a *arena) freeTotal() (total, largest int) {
	for _, h := range a.holes {
		total += h.size
		largest = max(largest, h.size)
	}
	return
}

// check confirms that holes and regions cover the arena exactly, with no
// gaps or overlaps, and that neighbouring holes were merged.
func (a *arena) check() error {
	type rng struct {
		start, size int
		hole        bool
	}
	var all []rng
	for i, h := range a.holes {
		if h.size <= 0 {
			return fmt.Errorf("arena: empty hole at %d", h.start)
		}
		if i > 0 && a.holes[i-1].start+a.holes[i-1].size >= h.start {
			return fmt.Errorf("arena: holes at %d and %d overlap or are not coalesced", a.holes[i-1].start, h.start)
		}
		all = append(all, rng{h.start, h.size, true})
	}
	for s, n := range a.used {
		if n <= 0 {
			return fmt.Errorf("arena: empty region at %d", s)
		}
		all = append(all, rng{s, n, false})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].start < all[j].start })
	pos := 0
	for _, r := range all {
		if r.start != pos {
			return fmt.Errorf("arena: gap or overlap at %d (next range starts at %d)", pos, r.start)
		}
		pos += r.size
	}
	if pos != a.capacity {
		return fmt.Errorf("arena: ranges cover %d of %d slots", pos, a.capacity)
	}
	return nil
}
