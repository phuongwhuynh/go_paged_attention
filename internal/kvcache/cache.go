// Package kvcache is a paged KV cache: each sequence's tokens live in
// fixed-size blocks, found through the sequence's block table.
//
// Blocks can be shared. Fork shares a whole sequence and copies a block only
// when one side writes to it. With prefix caching on, full blocks are also
// indexed by their tokens so new prompts can reuse them.
package kvcache

import (
	"errors"
	"fmt"

	"go_paged_attention/internal/block"
)

var ErrOutOfBlocks = block.ErrOutOfBlocks

type Config struct {
	NumBlocks int // size of the physical pool
	BlockSize int // tokens per block
	// Floats stored per token is NumHeads*HeadDim. Use 0 to track blocks
	// only, without storing any K/V (the simulator does this).
	NumHeads      int
	HeadDim       int
	PrefixCaching bool
}

// Sequence is one request's view of the cache.
type Sequence struct {
	ID        int
	Blocks    []int   // block table: logical block i is physical block Blocks[i]
	Tokens    []int32 // token IDs stored so far
	NumTokens int
}

type Stats struct {
	PromptTokens int // prompt tokens looked up in the prefix cache
	HitTokens    int // of those, tokens found
	Evictions    int // cached blocks evicted to make room
	CoWCopies    int // shared blocks copied on write
}

// Cache owns the K/V memory, laid out as [block][slot][head][dim].
type Cache struct {
	Cfg   Config
	K, V  []float32
	alloc *block.Allocator
	seqs  map[int]*Sequence
	tree  *radixTree // prefix cache, nil when disabled
	Stats Stats
}

// New creates an empty cache with all blocks free.
func New(cfg Config) *Cache {
	n := cfg.NumBlocks * cfg.BlockSize * cfg.NumHeads * cfg.HeadDim
	c := &Cache{
		Cfg:   cfg,
		K:     make([]float32, n),
		V:     make([]float32, n),
		alloc: block.NewAllocator(cfg.NumBlocks),
		seqs:  map[int]*Sequence{},
	}
	if cfg.PrefixCaching {
		c.tree = newRadixTree(cfg.BlockSize, cfg.NumBlocks)
	}
	return c
}

// tokenStride is the number of floats one token takes in K (or V).
func (c *Cache) tokenStride() int { return c.Cfg.NumHeads * c.Cfg.HeadDim }

// Offset returns where a slot of a physical block starts in K and V.
func (c *Cache) Offset(blockID, slot int) int {
	return (blockID*c.Cfg.BlockSize + slot) * c.tokenStride()
}

// AddSequence registers an empty sequence.
func (c *Cache) AddSequence(id int) *Sequence {
	s := &Sequence{ID: id}
	c.seqs[id] = s
	return s
}

// AddSequenceWithPrefix registers a sequence and reuses as much of prompt as
// is already cached. It returns how many tokens were reused; the caller
// appends the rest. The last prompt token is never reused, because the
// engine must compute it to produce the first output.
func (c *Cache) AddSequenceWithPrefix(id int, prompt []int32) (*Sequence, int) {
	s := c.AddSequence(id)
	if c.tree == nil || len(prompt) == 0 {
		return s, 0
	}
	blocks := c.tree.match(prompt, (len(prompt)-1)/c.Cfg.BlockSize, true)
	for _, b := range blocks {
		c.alloc.Retain(b)
	}
	hit := len(blocks) * c.Cfg.BlockSize
	s.Blocks = blocks
	s.Tokens = append([]int32(nil), prompt[:hit]...)
	s.NumTokens = hit
	c.Stats.PromptTokens += len(prompt)
	c.Stats.HitTokens += hit
	return s, hit
}

// CachedPrefixLen is a dry run of AddSequenceWithPrefix: how many tokens
// would be reused, without changing anything.
func (c *Cache) CachedPrefixLen(prompt []int32) int {
	if c.tree == nil || len(prompt) == 0 {
		return 0
	}
	return len(c.tree.match(prompt, (len(prompt)-1)/c.Cfg.BlockSize, false)) * c.Cfg.BlockSize
}

// Sequence returns the sequence with the given ID, or nil.
func (c *Cache) Sequence(id int) *Sequence { return c.seqs[id] }

// allocate gets a free block, evicting an idle cached block if none are free.
func (c *Cache) allocate() (int, error) {
	id, err := c.alloc.Allocate()
	if !errors.Is(err, block.ErrOutOfBlocks) || c.tree == nil {
		return id, err
	}
	b, ok := c.tree.evict(c.alloc.RefCount)
	if !ok {
		return 0, err
	}
	c.alloc.Release(b) // the tree was the only owner, so this frees it
	c.Stats.Evictions++
	return c.alloc.Allocate()
}

// NeedsBlock reports whether the next Append to seqID will use up a block:
// either the last block is full, or it is shared and must be copied.
func (c *Cache) NeedsBlock(seqID int) bool {
	s := c.seqs[seqID]
	if s.NumTokens%c.Cfg.BlockSize == 0 {
		return true
	}
	return c.alloc.RefCount(s.Blocks[len(s.Blocks)-1]) > 1
}

// Append adds one token, with its K and V, to the end of a sequence.
func (c *Cache) Append(seqID int, token int32, k, v []float32) error {
	s, ok := c.seqs[seqID]
	if !ok {
		return fmt.Errorf("kvcache: unknown sequence %d", seqID)
	}
	if len(k) != c.tokenStride() || len(v) != c.tokenStride() {
		return fmt.Errorf("kvcache: k/v must have %d floats", c.tokenStride())
	}

	slot := s.NumTokens % c.Cfg.BlockSize
	if slot == 0 {
		id, err := c.allocate()
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
	s.Tokens = append(s.Tokens, token)
	s.NumTokens++

	// The last block just filled up, so it can now be shared with later prompts.
	if c.tree != nil && s.NumTokens%c.Cfg.BlockSize == 0 {
		nb := s.NumTokens / c.Cfg.BlockSize
		for _, b := range c.tree.insert(s.Tokens[:nb*c.Cfg.BlockSize], s.Blocks[:nb]) {
			c.alloc.Retain(b)
		}
	}
	return nil
}

// copyOnWrite gives s its own copy of its last block if anyone else uses it.
func (c *Cache) copyOnWrite(s *Sequence) error {
	i := len(s.Blocks) - 1
	old := s.Blocks[i]
	if c.alloc.RefCount(old) == 1 {
		return nil
	}
	fresh, err := c.allocate()
	if err != nil {
		return err
	}
	src, dst := c.Offset(old, 0), c.Offset(fresh, 0)
	n := c.Cfg.BlockSize * c.tokenStride()
	copy(c.K[dst:dst+n], c.K[src:src+n])
	copy(c.V[dst:dst+n], c.V[src:src+n])
	c.alloc.Release(old)
	s.Blocks[i] = fresh
	c.Stats.CoWCopies++
	return nil
}

// Fork creates childID as a copy of parentID. The block table is copied but
// the blocks are shared, so forking costs no memory until someone writes.
func (c *Cache) Fork(parentID, childID int) (*Sequence, error) {
	p, ok := c.seqs[parentID]
	if !ok {
		return nil, fmt.Errorf("kvcache: unknown sequence %d", parentID)
	}
	if _, exists := c.seqs[childID]; exists {
		return nil, fmt.Errorf("kvcache: sequence %d already exists", childID)
	}
	child := &Sequence{
		ID: childID, NumTokens: p.NumTokens,
		Blocks: append([]int(nil), p.Blocks...),
		Tokens: append([]int32(nil), p.Tokens...),
	}
	for _, b := range child.Blocks {
		c.alloc.Retain(b)
	}
	c.seqs[childID] = child
	return child, nil
}

// Free drops a sequence and its references to its blocks. Blocks the prefix
// cache also holds stay in memory for later reuse.
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

// NumFreeBlocks returns how many blocks are on the free list.
func (c *Cache) NumFreeBlocks() int { return c.alloc.NumFree() }

// NumCachedBlocks returns how many blocks only the prefix cache holds. No
// sequence is using them, so they can be evicted when memory runs out.
func (c *Cache) NumCachedBlocks() int {
	if c.tree == nil {
		return 0
	}
	n := 0
	for b, in := range c.tree.inTree {
		if in && c.alloc.RefCount(b) == 1 {
			n++
		}
	}
	return n
}

// NumAvailableBlocks returns how many blocks could be allocated right now,
// counting cached blocks that would be evicted.
func (c *Cache) NumAvailableBlocks() int { return c.NumFreeBlocks() + c.NumCachedBlocks() }

// RefCount returns how many owners a block has.
func (c *Cache) RefCount(blockID int) int { return c.alloc.RefCount(blockID) }

// Sequences calls f for every live sequence, in no particular order.
func (c *Cache) Sequences(f func(*Sequence)) {
	for _, s := range c.seqs {
		f(s)
	}
}
