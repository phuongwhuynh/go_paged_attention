package kvcache

import "testing"

func tok(stride int, val float32) []float32 {
	v := make([]float32, stride)
	for i := range v {
		v[i] = val
	}
	return v
}

func TestAllocatesBlocksOnDemand(t *testing.T) {
	c := New(Config{NumBlocks: 4, BlockSize: 2, NumHeads: 1, HeadDim: 2})
	c.AddSequence(0)
	for i := range 3 {
		if err := c.Append(0, tok(2, float32(i)), tok(2, float32(i))); err != nil {
			t.Fatal(err)
		}
	}
	if got := len(c.Sequence(0).Blocks); got != 2 {
		t.Fatalf("3 tokens with block size 2 should use 2 blocks, got %d", got)
	}
	c.Free(0)
	if c.NumFreeBlocks() != 4 {
		t.Fatalf("expected all blocks free after Free, got %d", c.NumFreeBlocks())
	}
}

func TestForkSharesThenCopiesOnWrite(t *testing.T) {
	c := New(Config{NumBlocks: 4, BlockSize: 4, NumHeads: 1, HeadDim: 1})
	c.AddSequence(0)
	c.Append(0, tok(1, 1), tok(1, 1))
	c.Append(0, tok(1, 2), tok(1, 2))

	child, _ := c.Fork(0, 1)
	if c.NumFreeBlocks() != 3 {
		t.Fatalf("fork should not allocate, free=%d", c.NumFreeBlocks())
	}

	c.Append(1, tok(1, 9), tok(1, 9))
	parent := c.Sequence(0)
	if parent.Blocks[0] == child.Blocks[0] {
		t.Fatal("child should get its own block after writing to a shared one")
	}
	if c.K[c.Offset(child.Blocks[0], 1)] != 2 {
		t.Fatal("copy-on-write should preserve earlier tokens")
	}
	if c.K[c.Offset(parent.Blocks[0], 2)] != 0 {
		t.Fatal("parent block must not see the child's write")
	}
}
