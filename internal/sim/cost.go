package sim

// CostModel estimates how long one step takes on a GPU. A step reads all
// the weights once whatever the batch size, which is why bigger batches pay
// off. It's an estimate, not a measurement:
//
//	step = max((weights + KV read) / bandwidth, 2 * params * tokens / FLOPs)
type CostModel struct {
	Name            string
	WeightBytes     float64
	KVBytesPerToken float64
	Params          float64
	MemBW           float64 // bytes/s
	FLOPs           float64 // effective FLOP/s
	MemBytes        float64 // total device memory
}

// LLaMA13BOnA100 is the vLLM paper's main setup. KV per token is
// 40 layers x 5120 x (K and V) x 2 bytes = 800 KB. FLOPs assumes half of
// the A100's 312 TFLOP/s peak.
var LLaMA13BOnA100 = CostModel{
	Name:            "LLaMA-13B on A100-40GB",
	WeightBytes:     26e9,
	KVBytesPerToken: 2 * 40 * 5120 * 2,
	Params:          13e9,
	MemBW:           1.555e12,
	FLOPs:           156e12,
	MemBytes:        40e9,
}

// KVCapacityTokens returns how many tokens of KV fit next to the weights
// when using a memUtil fraction of GPU memory.
func (c CostModel) KVCapacityTokens(memUtil float64) int {
	return int((c.MemBytes*memUtil - c.WeightBytes) / c.KVBytesPerToken)
}

// StepTime returns a step's duration in seconds. computeTokens is how many
// tokens the step processes; kvTokens is how much KV it reads or writes.
func (c CostModel) StepTime(computeTokens, kvTokens int) float64 {
	mem := (c.WeightBytes + float64(kvTokens)*c.KVBytesPerToken) / c.MemBW
	compute := 2 * c.Params * float64(computeTokens) / c.FLOPs
	return max(mem, compute)
}
