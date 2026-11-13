package kvcache

import (
	"fmt"
	"slices"
)

// CheckInvariants returns an error if the cache's bookkeeping is
// inconsistent. Tests, fuzzing and the simulator's -check mode call it after
// every operation. It checks that:
//
//   - each block's refcount matches the references to it, and none leak;
//   - each sequence has the right number of blocks for its tokens;
//   - the radix tree is well formed;
//   - each cached block a sequence uses really holds that sequence's tokens.
func (c *Cache) CheckInvariants() error {
	if err := c.alloc.Check(); err != nil {
		return err
	}
	bs := c.Cfg.BlockSize
	want := make([]int, c.Cfg.NumBlocks) // expected refcount per block

	for id, s := range c.seqs {
		if s.ID != id {
			return fmt.Errorf("kvcache: sequence stored under %d has ID %d", id, s.ID)
		}
		if len(s.Tokens) != s.NumTokens {
			return fmt.Errorf("kvcache: seq %d has %d token IDs for %d tokens", id, len(s.Tokens), s.NumTokens)
		}
		if nb := (s.NumTokens + bs - 1) / bs; len(s.Blocks) != nb {
			return fmt.Errorf("kvcache: seq %d has %d blocks for %d tokens", id, len(s.Blocks), s.NumTokens)
		}
		for _, b := range s.Blocks {
			if b < 0 || b >= c.Cfg.NumBlocks {
				return fmt.Errorf("kvcache: seq %d references block %d out of range", id, b)
			}
			want[b]++
		}
	}

	// Walk the tree, counting its references and recording where[b]: the
	// path of tokens and blocks from the root down to block b.
	type pathEnd struct {
		tokens []int32
		blocks []int
	}
	where := map[int]pathEnd{}
	if t := c.tree; t != nil {
		inTree := make([]bool, c.Cfg.NumBlocks)
		var walk func(n *radixNode, tokens []int32, blocks []int) error
		walk = func(n *radixNode, tokens []int32, blocks []int) error {
			_, isLeaf := t.leaves[n]
			if n != t.root {
				if len(n.blocks) == 0 || len(n.tokens) != len(n.blocks)*bs {
					return fmt.Errorf("kvcache: radix node has %d tokens for %d blocks", len(n.tokens), len(n.blocks))
				}
				if isLeaf != (len(n.children) == 0) {
					return fmt.Errorf("kvcache: radix leaf set wrong for node with %d children", len(n.children))
				}
			} else if isLeaf {
				return fmt.Errorf("kvcache: radix root is in the leaf set")
			}
			for j, b := range n.blocks {
				if inTree[b] {
					return fmt.Errorf("kvcache: block %d indexed twice in radix tree", b)
				}
				inTree[b] = true
				want[b]++
				tokens = append(slices.Clip(tokens), t.blockTokens(n.tokens, j)...)
				blocks = append(slices.Clip(blocks), b)
				where[b] = pathEnd{tokens, blocks}
			}
			for k, ch := range n.children {
				if ch.parent != n {
					return fmt.Errorf("kvcache: radix child has wrong parent")
				}
				if k != key(t.blockTokens(ch.tokens, 0)) {
					return fmt.Errorf("kvcache: radix child key does not match its first block")
				}
				if err := walk(ch, tokens, blocks); err != nil {
					return err
				}
			}
			return nil
		}
		if err := walk(t.root, nil, nil); err != nil {
			return err
		}
		for l := range t.leaves {
			if len(l.blocks) == 0 || !inTree[l.blocks[0]] || where[l.blocks[0]].blocks == nil {
				return fmt.Errorf("kvcache: radix leaf set holds a detached or empty node")
			}
		}
		if !slices.Equal(inTree, t.inTree) {
			return fmt.Errorf("kvcache: radix inTree flags out of sync with the tree")
		}
	}

	for b, n := range want {
		if rc := c.alloc.RefCount(b); rc != n {
			return fmt.Errorf("kvcache: block %d has refcount %d, but %d references", b, rc, n)
		}
	}

	// A cached block at position i must sit at depth i on the same path.
	for id, s := range c.seqs {
		for i, b := range s.Blocks {
			p, ok := where[b]
			if !ok {
				continue
			}
			if len(p.blocks) != i+1 || !slices.Equal(p.blocks, s.Blocks[:i+1]) {
				return fmt.Errorf("kvcache: seq %d uses cached block %d at position %d off its tree path", id, b, i)
			}
			if !slices.Equal(p.tokens, s.Tokens[:(i+1)*bs]) {
				return fmt.Errorf("kvcache: seq %d tokens differ from cached block %d", id, b)
			}
		}
	}
	return nil
}
