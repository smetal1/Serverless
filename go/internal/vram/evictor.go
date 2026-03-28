package vram

import (
	"context"
	"time"

	"github.com/go-logr/logr"
)

// Evictor implements LRU eviction policy for GPU VRAM.
// When VRAM is needed for a new model, evicts the least recently used idle model.
type Evictor struct {
	log logr.Logger
}

// EvictionCandidate represents a model that can be evicted.
type EvictionCandidate struct {
	ModelName     string
	Namespace     string
	PodName       string
	GPUUUID       string
	VRAMUsedMB    int64
	LastRequestAt time.Time
	IdleDuration  time.Duration
}

// NewEvictor creates a new LRU evictor.
func NewEvictor(log logr.Logger) *Evictor {
	return &Evictor{log: log}
}

// FindCandidates returns models eligible for eviction, sorted by LRU (oldest first).
// Phase 2: Will query K8s for idle ModelDeployments and sort by lastRequestAt.
func (e *Evictor) FindCandidates(ctx context.Context, gpuUUID string) ([]EvictionCandidate, error) {
	// TODO(Phase 2): Implement LRU candidate selection
	// 1. List all ModelDeployments in Idle or Active phase on this GPU
	// 2. Sort by lastRequestAt (oldest first)
	// 3. Filter out models with minReplicas > 0 (pinned models)
	e.log.V(1).Info("FindCandidates called (stub)", "gpu", gpuUUID)
	return nil, nil
}

// Evict removes a model's GPU allocation and transitions it to Standby.
// Phase 2: Will trigger GPU state checkpoint and release GPU resources.
func (e *Evictor) Evict(ctx context.Context, candidate EvictionCandidate) error {
	// TODO(Phase 2): Implement eviction
	// 1. Trigger CUDA checkpoint (save GPU state if not already snapshotted)
	// 2. Update ModelDeployment phase to Idle → Standby
	// 3. Release GPU resources from pod
	// 4. Update VRAM tracking
	e.log.Info("Evict called (stub)", "model", candidate.ModelName, "gpu", candidate.GPUUUID)
	return nil
}

// EvictForSpace evicts enough models to free the requested VRAM on a GPU.
// Phase 2: Will iteratively evict LRU models until enough space is freed.
func (e *Evictor) EvictForSpace(ctx context.Context, gpuUUID string, neededMB int64) (int64, error) {
	// TODO(Phase 2): Implement space-based eviction
	// 1. Get sorted candidates
	// 2. Evict one at a time until enough space is freed
	// 3. Return total freed VRAM
	e.log.Info("EvictForSpace called (stub)", "gpu", gpuUUID, "neededMB", neededMB)
	return 0, nil
}
