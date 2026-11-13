package kvcache

import "encoding/binary"

// radixTree is the prefix cache. It maps token sequences to the blocks that
// hold them, one full block per step down the tree.
//
// Each node holds a run of one or more blocks, and its children are keyed
// by the tokens of their first block. The tree owns a reference to every
// block in it, so cached blocks outlive the sequences that wrote them.
type radixTree struct {
	bs     int
	root   *radixNode
	leaves map[*radixNode]struct{} // eviction candidates
	inTree []bool                  // per physical block
	clock  uint64                  // logical time, for LRU
}

type radixNode struct {
	tokens   []int32 // len(blocks) * BlockSize tokens
	blocks   []int
	parent   *radixNode
	children map[string]*radixNode
	lastUse  uint64
}

// newRadixTree creates an empty tree for a pool of numBlocks blocks.
func newRadixTree(bs, numBlocks int) *radixTree {
	return &radixTree{
		bs:     bs,
		root:   &radixNode{children: map[string]*radixNode{}},
		leaves: map[*radixNode]struct{}{},
		inTree: make([]bool, numBlocks),
	}
}

// key turns one block's tokens into a map key.
func key(tokens []int32) string {
	b := make([]byte, 4*len(tokens))
	for i, tok := range tokens {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(tok))
	}
	return string(b)
}

// equalTokens reports whether two token slices are identical.
func equalTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// tick advances the LRU clock and returns the new time.
func (t *radixTree) tick() uint64 { t.clock++; return t.clock }

// blockTokens returns the tokens of block i within tokens.
func (t *radixTree) blockTokens(tokens []int32, i int) []int32 {
	return tokens[i*t.bs : (i+1)*t.bs]
}

// match walks down the tree as far as tokens agree, up to maxBlocks, and
// returns the blocks it passed. With touch set, those nodes count as used.
func (t *radixTree) match(tokens []int32, maxBlocks int, touch bool) []int {
	var out []int
	n, i := t.root, 0
	for i < maxBlocks {
		child := n.children[key(t.blockTokens(tokens, i))]
		if child == nil {
			break
		}
		if touch {
			child.lastUse = t.tick()
		}
		j := 0
		for j < len(child.blocks) && i < maxBlocks && equalTokens(t.blockTokens(child.tokens, j), t.blockTokens(tokens, i)) {
			out = append(out, child.blocks[j])
			i++
			j++
		}
		if j < len(child.blocks) {
			break // stopped partway through this node
		}
		n = child
	}
	return out
}

// insert adds a sequence's full blocks to the tree and returns the ones that
// were new, which the caller must Retain.
//
// If the tree already holds the same tokens in a different block (two
// sequences computed the same prefix at once), insert stops and keeps the
// tree's copy. This guarantees a sequence's cached blocks always form one
// path from the root.
func (t *radixTree) insert(tokens []int32, blocks []int) []int {
	n, i := t.root, 0
	for i < len(blocks) {
		k := key(t.blockTokens(tokens, i))
		child := n.children[k]
		if child == nil && n != t.root && len(n.children) == 0 {
			// Sequences insert a block at a time as they grow, so extend the
			// leaf in place rather than chaining one-block nodes.
			n.tokens = append(n.tokens, tokens[i*t.bs:]...)
			n.blocks = append(n.blocks, blocks[i:]...)
			n.lastUse = t.tick()
			for _, b := range blocks[i:] {
				t.inTree[b] = true
			}
			return blocks[i:]
		}
		if child == nil {
			leaf := &radixNode{
				tokens:   append([]int32(nil), tokens[i*t.bs:]...),
				blocks:   append([]int(nil), blocks[i:]...),
				parent:   n,
				children: map[string]*radixNode{},
				lastUse:  t.tick(),
			}
			n.children[k] = leaf
			delete(t.leaves, n)
			t.leaves[leaf] = struct{}{}
			for _, b := range leaf.blocks {
				t.inTree[b] = true
			}
			return leaf.blocks
		}
		child.lastUse = t.tick()
		j := 0
		for j < len(child.blocks) && i < len(blocks) && equalTokens(t.blockTokens(child.tokens, j), t.blockTokens(tokens, i)) {
			if child.blocks[j] != blocks[i] {
				return nil // same tokens, different block: keep the tree's copy
			}
			i++
			j++
		}
		if j < len(child.blocks) && i < len(blocks) {
			n = t.split(child, j) // diverged inside child
		} else {
			n = child
		}
	}
	return nil
}

// split breaks node after its first j blocks and returns the new top half.
func (t *radixTree) split(node *radixNode, j int) *radixNode {
	k := key(t.blockTokens(node.tokens, 0))
	upper := &radixNode{
		tokens:   node.tokens[: j*t.bs : j*t.bs],
		blocks:   node.blocks[:j:j],
		parent:   node.parent,
		children: map[string]*radixNode{},
		lastUse:  node.lastUse,
	}
	node.tokens = node.tokens[j*t.bs:]
	node.blocks = node.blocks[j:]
	node.parent = upper
	upper.children[key(t.blockTokens(node.tokens, 0))] = node
	upper.parent.children[k] = upper
	return upper
}

// evict removes one block from the tree and returns it; the caller releases
// the tree's reference. It takes the last block of the least recently used
// leaf that no sequence is using. Inner blocks are never taken, since their
// descendants would become unreachable.
func (t *radixTree) evict(refCount func(int) int) (int, bool) {
	var best *radixNode
	for l := range t.leaves {
		if refCount(l.blocks[len(l.blocks)-1]) == 1 && (best == nil || l.lastUse < best.lastUse) {
			best = l
		}
	}
	if best == nil {
		return 0, false
	}
	last := len(best.blocks) - 1
	b := best.blocks[last]
	t.inTree[b] = false
	if last == 0 {
		p := best.parent
		delete(p.children, key(t.blockTokens(best.tokens, 0)))
		delete(t.leaves, best)
		if p != t.root && len(p.children) == 0 {
			t.leaves[p] = struct{}{}
		}
	} else {
		best.blocks = best.blocks[:last]
		best.tokens = best.tokens[:last*t.bs]
	}
	return b, true
}
