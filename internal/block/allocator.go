// Package block manages a fixed pool of physical KV-cache blocks.
//
// In PagedAttention the KV cache is split into fixed-size blocks (like OS
// memory pages). A sequence owns a list of block IDs (its "block table")
// instead of one large contiguous buffer, so memory is allocated on demand
// and almost nothing is wasted to fragmentation.
package block

import "errors"

var ErrOutOfBlocks = errors.New("block: no free blocks")

// Allocator hands out physical block IDs and tracks reference counts so that
// blocks can be shared between sequences (e.g. a common prompt prefix, or
// beam-search forks) and copied only on write.
type Allocator struct {
	free     []int // stack of free block IDs
	refCount []int
}

func NewAllocator(numBlocks int) *Allocator {
	a := &Allocator{
		free:     make([]int, numBlocks),
		refCount: make([]int, numBlocks),
	}
	for i := range numBlocks {
		a.free[i] = numBlocks - 1 - i // so block 0 is handed out first
	}
	return a
}

// Allocate returns a fresh block with refcount 1.
func (a *Allocator) Allocate() (int, error) {
	if len(a.free) == 0 {
		return 0, ErrOutOfBlocks
	}
	id := a.free[len(a.free)-1]
	a.free = a.free[:len(a.free)-1]
	a.refCount[id] = 1
	return id, nil
}

// Retain adds a reference to an already-allocated block (sharing).
func (a *Allocator) Retain(id int) { a.refCount[id]++ }

// Release drops a reference; the block returns to the pool at zero.
func (a *Allocator) Release(id int) {
	a.refCount[id]--
	if a.refCount[id] == 0 {
		a.free = append(a.free, id)
	}
}

func (a *Allocator) RefCount(id int) int { return a.refCount[id] }
func (a *Allocator) NumFree() int        { return len(a.free) }
func (a *Allocator) NumTotal() int       { return len(a.refCount) }
