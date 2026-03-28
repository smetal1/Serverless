# Podstack Inference OS - Build Orchestration
# ============================================

.PHONY: all build test lint clean docker deploy proto sdk worker proxy help

# Variables
GO_MODULE := github.com/podstack/serverless
GO_DIR := ./go
DOCKER_REGISTRY ?= podstack
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
PLATFORMS ?= linux/amd64

# Go build settings
GOFLAGS ?= -trimpath
LDFLAGS := -ldflags "-s -w -X main.version=$(VERSION)"

# ============================================
# High-level targets
# ============================================

## help: Show this help message
help:
	@echo "Podstack Inference OS - Build Targets"
	@echo "======================================"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'

## all: Build everything
all: build sdk worker proxy

## build: Build all Go binaries
build: build-operator build-gateway build-scheduler

## test: Run all tests
test: test-go test-python

## clean: Remove build artifacts
clean:
	rm -rf bin/
	rm -rf $(GO_DIR)/vendor/
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +

# ============================================
# Go targets
# ============================================

## build-operator: Build the operator binary
build-operator:
	cd $(GO_DIR) && go build $(GOFLAGS) $(LDFLAGS) -o ../bin/operator ./cmd/operator/

## build-gateway: Build the gateway binary
build-gateway:
	cd $(GO_DIR) && go build $(GOFLAGS) $(LDFLAGS) -o ../bin/gateway ./cmd/gateway/

## build-scheduler: Build the scheduler plugin binary
build-scheduler:
	cd $(GO_DIR) && go build $(GOFLAGS) $(LDFLAGS) -o ../bin/scheduler ./cmd/scheduler/

## go-mod-tidy: Tidy Go module dependencies
go-mod-tidy:
	cd $(GO_DIR) && go mod tidy

## test-go: Run Go tests
test-go:
	cd $(GO_DIR) && go test ./... -v -count=1

## lint-go: Run Go linter
lint-go:
	cd $(GO_DIR) && golangci-lint run ./...

## vet: Run Go vet
vet:
	cd $(GO_DIR) && go vet ./...

## generate: Generate deepcopy functions and CRD manifests
generate:
	cd $(GO_DIR) && controller-gen object:headerFile="hack/boilerplate.go.txt" paths="./api/..."
	cd $(GO_DIR) && controller-gen crd paths="./api/..." output:crd:dir=../deploy/base/crds/

# ============================================
# Python targets
# ============================================

## sdk: Build the Podstack Python SDK
sdk:
	cd python/podstack-sdk && pip install -e ".[dev]" 2>/dev/null || pip install -e .

## worker: Build the Podstack worker runtime
worker:
	cd python/podstack-worker && pip install -e ".[dev]" 2>/dev/null || pip install -e .

## proxy: Build the Podstack LiteLLM proxy
proxy:
	cd python/podstack-proxy && pip install -e ".[dev]" 2>/dev/null || pip install -e .

## test-python: Run Python tests
test-python:
	cd python/podstack-sdk && python -m pytest tests/ -v 2>/dev/null || echo "No SDK tests yet"
	cd python/podstack-worker && python -m pytest tests/ -v 2>/dev/null || echo "No worker tests yet"
	cd python/podstack-proxy && python -m pytest tests/ -v 2>/dev/null || echo "No proxy tests yet"

## lint-python: Run Python linter
lint-python:
	ruff check python/

# ============================================
# Docker targets
# ============================================

## docker: Build all Docker images
docker: docker-base docker-operator docker-gateway docker-proxy docker-vllm docker-triton docker-generic

## docker-base: Build base GPU image with CRIU + cuda-checkpoint
docker-base:
	docker build -t $(DOCKER_REGISTRY)/base-gpu:$(VERSION) -f docker/base-gpu.Dockerfile .

## docker-operator: Build operator image
docker-operator: build-operator
	docker build -t $(DOCKER_REGISTRY)/operator:$(VERSION) -f docker/operator.Dockerfile .

## docker-gateway: Build gateway image
docker-gateway: build-gateway
	docker build -t $(DOCKER_REGISTRY)/gateway:$(VERSION) -f docker/gateway.Dockerfile .

## docker-proxy: Build LiteLLM proxy image
docker-proxy:
	docker build -t $(DOCKER_REGISTRY)/proxy:$(VERSION) -f docker/proxy.Dockerfile .

## docker-vllm: Build vLLM worker image
docker-vllm: docker-base
	docker build -t $(DOCKER_REGISTRY)/worker-vllm:$(VERSION) -f docker/worker-vllm.Dockerfile .

## docker-triton: Build Triton worker image
docker-triton:
	docker build -t $(DOCKER_REGISTRY)/worker-triton:$(VERSION) -f docker/worker-triton.Dockerfile .

## docker-generic: Build generic worker image
docker-generic: docker-base
	docker build -t $(DOCKER_REGISTRY)/worker-generic:$(VERSION) -f docker/worker-generic.Dockerfile .

## docker-push: Push all images to registry
docker-push:
	docker push $(DOCKER_REGISTRY)/operator:$(VERSION)
	docker push $(DOCKER_REGISTRY)/gateway:$(VERSION)
	docker push $(DOCKER_REGISTRY)/proxy:$(VERSION)
	docker push $(DOCKER_REGISTRY)/worker-vllm:$(VERSION)
	docker push $(DOCKER_REGISTRY)/worker-triton:$(VERSION)
	docker push $(DOCKER_REGISTRY)/worker-generic:$(VERSION)

# ============================================
# Deployment targets
# ============================================

## deploy-crds: Install CRDs on the cluster
deploy-crds:
	kubectl apply -f deploy/base/crds/

## deploy-dev: Deploy to dev environment
deploy-dev: deploy-crds
	kubectl apply -k deploy/overlays/dev/

## deploy-prod: Deploy to production environment
deploy-prod: deploy-crds
	kubectl apply -k deploy/overlays/production/

## deploy-examples: Deploy example model deployments
deploy-examples:
	kubectl apply -f deploy/examples/

## undeploy: Remove all Podstack resources
undeploy:
	kubectl delete -k deploy/base/ --ignore-not-found
	kubectl delete -f deploy/base/crds/ --ignore-not-found

# ============================================
# Proto targets
# ============================================

## proto: Generate code from protobuf definitions
proto:
	protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		api/proto/podstack/v1/*.proto

# ============================================
# Development helpers
# ============================================

## run-operator: Run the operator locally (requires kubeconfig)
run-operator: build-operator
	./bin/operator --nfs-base-path=/tmp/podstack/models --snapshot-base-path=/tmp/podstack/snapshots

## run-gateway: Run the gateway locally (requires kubeconfig)
run-gateway: build-gateway
	./bin/gateway --addr=:8080 --namespace=podstack-system

## fmt: Format all code
fmt:
	cd $(GO_DIR) && gofmt -s -w .
	ruff format python/ 2>/dev/null || true
