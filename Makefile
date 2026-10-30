.PHONY: test run bench vet

test:
	go test ./...

run:
	go run ./cmd/demo

bench:
	go test -bench=. ./...

vet:
	go vet ./...
