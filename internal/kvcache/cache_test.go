package kvcache

import (
	"errors"
	"slices"
	"testing"
)

// tok returns a K or V vector of stride copies of val.
func tok(stride int, val float32) []float32 {
	v := make([]float32, stride)
	for i := range v {
		v[i] = val
	}
	return v
}

// mustCheck fails the test if any cache invariant is broken.
func mustCheck(t testing.TB, c *Cache) {
	t.Helper()
	if err := c.CheckInvariants(); err != nil {
		t.Fatal(err)
	}
}

// appendTokens appends tokens, using each token ID as its K and V value so
// tests can read tokens back from memory. Needs NumHeads*HeadDim == 1.
func appendTokens(t testing.TB, c *Cache, id int, tokens ...int32) {
	t.Helper()
	for _, x := range tokens {
		if err := c.Append(id, x, tok(1, float32(x)), tok(1, float32(x))); err != nil {
			t.Fatal(err)
		}
		mustCheck(t, c)
	}
}

// storedTokens reads a sequence's tokens back from K via its block table.
func storedTokens(c *Cache, s *Sequence) []int32 {
	out := make([]int32, s.NumTokens)
	for i := range out {
		out[i] = int32(c.K[c.Offset(s.Blocks[i/c.Cfg.BlockSize], i%c.Cfg.BlockSize)])
	}
	return out
}

// Blocks are taken only as tokens arrive, and all come back on Free.
func TestAllocatesBlocksOnDemand(t *testing.T) {
	c := New(Config{NumBlocks: 4, BlockSize: 2, NumHeads: 1, HeadDim: 2})
	c.AddSequence(0)
	for i := range 3 {
		if err := c.Append(0, int32(i), tok(2, float32(i)), tok(2, float32(i))); err != nil {
			t.Fatal(err)
		}
		mustCheck(t, c)
	}
	if got := len(c.Sequence(0).Blocks); got != 2 {
		t.Fatalf("3 tokens with block size 2 should use 2 blocks, got %d", got)
	}
	c.Free(0)
	mustCheck(t, c)
	if c.NumFreeBlocks() != 4 {
		t.Fatalf("expected all blocks free after Free, got %d", c.NumFreeBlocks())
	}
}

// A fork shares blocks until the child writes, then gets its own copy.
func TestForkSharesThenCopiesOnWrite(t *testing.T) {
	c := New(Config{NumBlocks: 4, BlockSize: 4, NumHeads: 1, HeadDim: 1})
	c.AddSequence(0)
	appendTokens(t, c, 0, 1, 2)

	child, _ := c.Fork(0, 1)
	mustCheck(t, c)
	if c.NumFreeBlocks() != 3 {
		t.Fatalf("fork should not allocate, free=%d", c.NumFreeBlocks())
	}

	appendTokens(t, c, 1, 9)
	parent := c.Sequence(0)
	if parent.Blocks[0] == child.Blocks[0] {
		t.Fatal("child should get its own block after writing to a shared one")
	}
	if got := storedTokens(c, child); !slices.Equal(got, []int32{1, 2, 9}) {
		t.Fatalf("child holds %v, want [1 2 9]", got)
	}
	if got := storedTokens(c, parent); !slices.Equal(got, []int32{1, 2}) {
		t.Fatalf("parent holds %v, want [1 2]: it must not see the child's write", got)
	}
	if c.Stats.CoWCopies != 1 {
		t.Fatalf("CoWCopies = %d, want 1", c.Stats.CoWCopies)
	}
}

// A new prompt reuses the full blocks it shares with an earlier one.
func TestPrefixCacheReusesFullBlocks(t *testing.T) {
	c := New(Config{NumBlocks: 8, BlockSize: 4, NumHeads: 1, HeadDim: 1, PrefixCaching: true})
	a := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	_, hit := c.AddSequenceWithPrefix(0, a)
	if hit != 0 {
		t.Fatalf("empty cache hit %d tokens", hit)
	}
	appendTokens(t, c, 0, a...)

	// Shares the first two blocks (8 tokens), then diverges.
	b := []int32{1, 2, 3, 4, 5, 6, 7, 8, 42, 43}
	if got := c.CachedPrefixLen(b); got != 8 {
		t.Fatalf("CachedPrefixLen = %d, want 8", got)
	}
	sb, hit := c.AddSequenceWithPrefix(1, b)
	mustCheck(t, c)
	if hit != 8 {
		t.Fatalf("hit %d tokens, want 8", hit)
	}
	appendTokens(t, c, 1, b[hit:]...)
	sa := c.Sequence(0)
	if !slices.Equal(sa.Blocks[:2], sb.Blocks[:2]) || sa.Blocks[2] == sb.Blocks[2] {
		t.Fatalf("blocks a=%v b=%v: want first two shared, third private", sa.Blocks, sb.Blocks)
	}
	if got := storedTokens(c, sb); !slices.Equal(got, b) {
		t.Fatalf("b holds %v, want %v", got, b)
	}

	// A prompt that is exactly cached still leaves its last token to compute.
	if got := c.CachedPrefixLen(a[:8]); got != 4 {
		t.Fatalf("CachedPrefixLen(8 cached tokens) = %d, want 4", got)
	}
}

// The radix tree keeps runs compressed and splits a node where prompts diverge.
func TestPrefixCacheSplitsOnDivergence(t *testing.T) {
	c := New(Config{NumBlocks: 16, BlockSize: 2, NumHeads: 1, HeadDim: 1, PrefixCaching: true})
	c.AddSequenceWithPrefix(0, nil)
	appendTokens(t, c, 0, 1, 1, 2, 2, 3, 3, 4, 4)
	node := c.tree.root.children[key([]int32{1, 1})]
	if len(c.tree.root.children) != 1 || len(node.blocks) != 4 {
		t.Fatal("4 blocks inserted one at a time should form one compressed node")
	}

	// Computed the same first blocks itself (no prefix lookup), so its
	// physical blocks differ: the tree keeps seq 0's copy, seq 1's stay private.
	c.AddSequence(1)
	appendTokens(t, c, 1, 1, 1, 2, 2, 9, 9)
	if len(c.tree.root.children) != 1 || len(node.children) != 0 {
		t.Fatal("a duplicate prefix computed elsewhere must not be inserted")
	}

	// Matches seq 0's first two blocks, then diverges: splits the node.
	_, hit := c.AddSequenceWithPrefix(2, []int32{1, 1, 2, 2, 7, 7, 7})
	if hit != 4 {
		t.Fatalf("hit %d, want 4", hit)
	}
	appendTokens(t, c, 2, 7, 7, 7)
	upper := c.tree.root.children[key([]int32{1, 1})]
	if len(upper.blocks) != 2 || len(upper.children) != 2 {
		t.Fatalf("after divergence: upper node has %d blocks and %d children, want 2 and 2",
			len(upper.blocks), len(upper.children))
	}
	for _, id := range []int{0, 1, 2} {
		c.Free(id)
		mustCheck(t, c)
	}
	if c.NumCachedBlocks() == 0 {
		t.Fatal("cached blocks should survive their sequences")
	}
}

// When memory runs out, the least recently used idle cached blocks go first.
func TestLRUEviction(t *testing.T) {
	c := New(Config{NumBlocks: 4, BlockSize: 2, NumHeads: 1, HeadDim: 1, PrefixCaching: true})
	old := []int32{1, 1, 2, 2}
	recent := []int32{5, 5, 6, 6}
	for id, p := range [][]int32{old, recent} {
		c.AddSequenceWithPrefix(id, p)
		appendTokens(t, c, id, p...)
		c.Free(id)
		mustCheck(t, c)
	}
	if c.NumFreeBlocks() != 0 || c.NumCachedBlocks() != 4 {
		t.Fatalf("free=%d cached=%d, want 0 and 4", c.NumFreeBlocks(), c.NumCachedBlocks())
	}
	// Touch `old` so it becomes the most recently used.
	c.AddSequenceWithPrefix(9, append(old, 0))
	c.Free(9)

	// Needs two blocks: both must come from `recent`, the LRU entry.
	c.AddSequence(2)
	appendTokens(t, c, 2, 8, 8, 8)
	if c.Stats.Evictions != 2 {
		t.Fatalf("evictions = %d, want 2", c.Stats.Evictions)
	}
	if c.CachedPrefixLen(append(old, 0)) != 4 {
		t.Fatal("recently used prefix was evicted")
	}
	if c.CachedPrefixLen(append(recent, 0)) != 0 {
		t.Fatal("least recently used prefix should have been evicted")
	}

	// Cached blocks in use by a sequence are never evicted.
	c.AddSequenceWithPrefix(3, append(old, 0))
	c.AddSequence(4)
	if err := c.Append(4, 1, tok(1, 1), tok(1, 1)); !errors.Is(err, ErrOutOfBlocks) {
		t.Fatalf("expected ErrOutOfBlocks, got %v", err)
	}
	mustCheck(t, c)
}

// FuzzCache applies random operations to a small cache. After each one it
// checks the invariants and compares every sequence with a plain token list.
func FuzzCache(f *testing.F) {
	f.Add([]byte{0, 5, 1, 2, 1, 2, 1, 0, 1, 2, 0, 6, 1, 2, 1, 2, 2, 2, 3, 0, 1, 1})
	f.Add([]byte{1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 1, 1, 3, 1, 0, 7, 0, 0, 0, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}
		c := New(Config{NumBlocks: 8, BlockSize: 2, NumHeads: 1, HeadDim: 1, PrefixCaching: data[0]%2 == 1})
		model := map[int][]int32{}
		var live []int
		nextID := 0
		pos := 1
		next := func() int {
			if pos >= len(data) {
				return 0
			}
			pos++
			return int(data[pos-1])
		}
		pick := func() int { return live[next()%len(live)] }
		appendTok := func(id int, x int32) bool {
			err := c.Append(id, x, tok(1, float32(x)), tok(1, float32(x)))
			if errors.Is(err, ErrOutOfBlocks) {
				return false
			}
			if err != nil {
				t.Fatal(err)
			}
			model[id] = append(model[id], x)
			return true
		}

		for pos < len(data) {
			switch op := next() % 4; {
			case op == 0 || len(live) == 0: // new sequence with a prompt
				prompt := make([]int32, next()%7+1)
				for i := range prompt {
					prompt[i] = int32(next() % 3) // tiny alphabet: lots of shared prefixes
				}
				id := nextID
				nextID++
				s, hit := c.AddSequenceWithPrefix(id, prompt)
				if !slices.Equal(s.Tokens, prompt[:hit]) {
					t.Fatalf("prefix hit %d tokens %v for prompt %v", hit, s.Tokens, prompt)
				}
				model[id] = append([]int32(nil), prompt[:hit]...)
				live = append(live, id)
				for _, x := range prompt[hit:] {
					if !appendTok(id, x) {
						break
					}
				}
			case op == 1:
				appendTok(pick(), int32(next()%3))
			case op == 2:
				parent := pick()
				id := nextID
				nextID++
				if _, err := c.Fork(parent, id); err != nil {
					t.Fatal(err)
				}
				model[id] = slices.Clone(model[parent])
				live = append(live, id)
			case op == 3:
				i := next() % len(live)
				c.Free(live[i])
				delete(model, live[i])
				live = slices.Delete(live, i, i+1)
			}

			if err := c.CheckInvariants(); err != nil {
				t.Fatal(err)
			}
			for _, id := range live {
				s := c.Sequence(id)
				if !slices.Equal(s.Tokens, model[id]) {
					t.Fatalf("seq %d token IDs %v, model %v", id, s.Tokens, model[id])
				}
				if got := storedTokens(c, s); !slices.Equal(got, model[id]) {
					t.Fatalf("seq %d stored K %v, model %v", id, got, model[id])
				}
			}
		}
	})
}
