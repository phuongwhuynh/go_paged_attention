// Package kvcache stores keys/values in fixed-size physical blocks and maps
// each sequence's logical token positions onto them via a block table.
package kvcache

import (
	"fmt"

	"go_paged_attention/internal/block"
)

type Config struct {
	NumBlocks int // size of the physical pool
	BlockSize int // tokens per block
	NumHeads  int
	HeadDim   int
}

// Sequence is the per-request view: logical block i lives in physical
// block Blocks[i].
type Sequence struct {
	ID        int
	Blocks    []int // the block table (logical->physical mapping)
	NumTokens int
}

// Cache owns the physical K/V memory, laid out as
// [block][slot][head][dim] in one flat slice each.
type Cache struct {
	Cfg   Config
	K, V  []float32         // physical memory for keys and values (flat slices of [block][slot][head][dim])
	alloc *block.Allocator  // which physical blocks are free and how many references each has (shared between sequences)
	seqs  map[int]*Sequence // map of sequence IDs to sequences
}

func New(cfg Config) *Cache {
	n := cfg.NumBlocks * cfg.BlockSize * cfg.NumHeads * cfg.HeadDim
	return &Cache{
		Cfg:   cfg,
		K:     make([]float32, n),
		V:     make([]float32, n),
		alloc: block.NewAllocator(cfg.NumBlocks),
		seqs:  map[int]*Sequence{},
	}
}

func (c *Cache) tokenStride() int { return c.Cfg.NumHeads * c.Cfg.HeadDim }

// Offset returns the index into K/V for (physical block, slot in block).
func (c *Cache) Offset(blockID, slot int) int {
	return (blockID*c.Cfg.BlockSize + slot) * c.tokenStride()
}

func (c *Cache) AddSequence(id int) *Sequence {
	s := &Sequence{ID: id}
	c.seqs[id] = s
	return s
}

func (c *Cache) Sequence(id int) *Sequence { return c.seqs[id] }

// Append writes one token's K and V (each NumHeads*HeadDim floats) to the
// end of the sequence, allocating a new block when the last one is full and
// copying the last block first if it is shared (copy-on-write).
func (c *Cache) Append(seqID int, k, v []float32) error {
	s, ok := c.seqs[seqID]
	if !ok {
		return fmt.Errorf("kvcache: unknown sequence %d", seqID)
	}
	if len(k) != c.tokenStride() || len(v) != c.tokenStride() {
		return fmt.Errorf("kvcache: k/v must have %d floats", c.tokenStride())
	}

	slot := s.NumTokens % c.Cfg.BlockSize
	if slot == 0 {
		id, err := c.alloc.Allocate()
		if err != nil {
			return err
		}
		s.Blocks = append(s.Blocks, id)
	} else if err := c.copyOnWrite(s); err != nil {
		return err
	}

	last := s.Blocks[len(s.Blocks)-1]
	off := c.Offset(last, slot)
	copy(c.K[off:], k)
	copy(c.V[off:], v)
	s.NumTokens++
	return nil
}

// copyOnWrite gives s a private copy of its last block if it is shared.
func (c *Cache) copyOnWrite(s *Sequence) error {
	i := len(s.Blocks) - 1
	old := s.Blocks[i]
	if c.alloc.RefCount(old) == 1 {
		return nil
	}
	fresh, err := c.alloc.Allocate()
	if err != nil {
		return err
	}
	src, dst := c.Offset(old, 0), c.Offset(fresh, 0)
	n := c.Cfg.BlockSize * c.tokenStride()
	copy(c.K[dst:dst+n], c.K[src:src+n])
	copy(c.V[dst:dst+n], c.V[src:src+n])
	c.alloc.Release(old)
	s.Blocks[i] = fresh
	return nil
}

// Fork creates a new sequence sharing all of parent's blocks
// shallow copy of physical blocks (bump reference counts), but deep copy of the logical block table.
func (c *Cache) Fork(parentID, childID int) (*Sequence, error) {
	p, ok := c.seqs[parentID]
	if !ok {
		return nil, fmt.Errorf("kvcache: unknown sequence %d", parentID)
	}
	child := &Sequence{ID: childID, Blocks: append([]int(nil), p.Blocks...), NumTokens: p.NumTokens}
	for _, b := range child.Blocks {
		c.alloc.Retain(b)
	}
	c.seqs[childID] = child
	return child, nil
}

// Free releases all blocks held by a sequence.
func (c *Cache) Free(seqID int) {
	s, ok := c.seqs[seqID]
	if !ok {
		return
	}
	for _, b := range s.Blocks {
		c.alloc.Release(b)
	}
	delete(c.seqs, seqID)
}

func (c *Cache) NumFreeBlocks() int { return c.alloc.NumFree() }
