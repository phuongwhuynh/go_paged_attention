// Package block manages the pool of physical KV-cache blocks.
// Blocks are fixed-size, like OS memory pages, and reference counted so
// several sequences can share one.
package block

import (
	"errors"
	"fmt"
)

var ErrOutOfBlocks = errors.New("block: no free blocks")

// Allocator hands out block IDs and tracks how many owners each block has.
type Allocator struct {
	free     []int // stack of free block IDs
	refCount []int
}

// NewAllocator creates a pool of numBlocks free blocks.
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

// Allocate takes a block off the free list and gives it one owner.
func (a *Allocator) Allocate() (int, error) {
	if len(a.free) == 0 {
		return 0, ErrOutOfBlocks
	}
	id := a.free[len(a.free)-1]
	a.free = a.free[:len(a.free)-1]
	a.refCount[id] = 1
	return id, nil
}

// Retain adds an owner to a block that is already in use.
func (a *Allocator) Retain(id int) { a.refCount[id]++ }

// Release removes an owner; the last one out returns the block to the pool.
func (a *Allocator) Release(id int) {
	a.refCount[id]--
	if a.refCount[id] == 0 {
		a.free = append(a.free, id)
	}
}

// Check confirms the free list holds exactly the blocks with no owners,
// each once.
func (a *Allocator) Check() error {
	onList := make([]bool, len(a.refCount))
	for _, id := range a.free {
		if id < 0 || id >= len(a.refCount) {
			return fmt.Errorf("block: free list has out-of-range id %d", id)
		}
		if onList[id] {
			return fmt.Errorf("block: %d is on the free list twice", id)
		}
		onList[id] = true
		if a.refCount[id] != 0 {
			return fmt.Errorf("block: free block %d has refcount %d", id, a.refCount[id])
		}
	}
	for id, rc := range a.refCount {
		if rc < 0 {
			return fmt.Errorf("block: %d has negative refcount %d", id, rc)
		}
		if rc == 0 && !onList[id] {
			return fmt.Errorf("block: %d has refcount 0 but is not free (leaked)", id)
		}
	}
	return nil
}

// RefCount returns how many owners a block has.
func (a *Allocator) RefCount(id int) int { return a.refCount[id] }

// NumFree returns how many blocks are on the free list.
func (a *Allocator) NumFree() int { return len(a.free) }

// NumTotal returns the size of the pool.
func (a *Allocator) NumTotal() int { return len(a.refCount) }
