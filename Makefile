.PHONY: test run sim vet fuzz

test:
	go test ./...

run:
	go run ./cmd/demo

sim:
	go run ./cmd/sim

vet:
	go vet ./...

FUZZTIME ?= 30s
fuzz:
	go test -run '^$$' -fuzz=FuzzCache -fuzztime=$(FUZZTIME) ./internal/kvcache/
	go test -run '^$$' -fuzz=FuzzArena -fuzztime=$(FUZZTIME) ./internal/sim/
	go test -run '^$$' -fuzz=FuzzSim -fuzztime=$(FUZZTIME) ./internal/sim/
