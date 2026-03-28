// Package vram manages GPU VRAM as a first-class resource.
// Phase 2: Tracks per-GPU VRAM budgets and enforces limits.
package vram

import (
	"context"
	"sync"

	"github.com/go-logr/logr"
)

// Manager tracks VRAM budget per GPU across the cluster.
// It maintains a real-time view of VRAM allocation and enforces limits.
type Manager struct {
	mu         sync.RWMutex
	gpuBudgets map[string]*GPUBudget // gpuUUID → budget
	evictor    *Evictor
	log        logr.Logger
}

// GPUBudget tracks VRAM allocation for a single GPU.
type GPUBudget struct {
	GPUUUID    string
	NodeName   string
	TotalMB    int64
	AllocatedMB int64
	Allocations map[string]int64 // modelName → VRAM MB
}

// NewManager creates a new VRAM manager.
func NewManager(log logr.Logger) *Manager {
	return &Manager{
		gpuBudgets: make(map[string]*GPUBudget),
		evictor:    NewEvictor(log),
		log:        log,
	}
}

// Allocate reserves VRAM on a GPU for a model.
// Phase 2: Will enforce budgets and trigger eviction if needed.
func (m *Manager) Allocate(ctx context.Context, gpuUUID string, modelName string, vramMB int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	budget, ok := m.gpuBudgets[gpuUUID]
	if !ok {
		// TODO(Phase 2): Auto-discover GPU and its total VRAM
		m.log.V(1).Info("GPU not tracked, allowing allocation (stub)", "gpu", gpuUUID)
		return nil
	}

	if budget.AllocatedMB+vramMB > budget.TotalMB {
		// TODO(Phase 2): Trigger eviction
		m.log.Info("Insufficient VRAM, eviction needed (stub)", "gpu", gpuUUID,
			"requested", vramMB, "available", budget.TotalMB-budget.AllocatedMB)
		return nil
	}

	budget.AllocatedMB += vramMB
	budget.Allocations[modelName] = vramMB
	return nil
}

// Release frees VRAM on a GPU when a model is unloaded.
func (m *Manager) Release(ctx context.Context, gpuUUID string, modelName string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	budget, ok := m.gpuBudgets[gpuUUID]
	if !ok {
		return nil
	}

	if vram, exists := budget.Allocations[modelName]; exists {
		budget.AllocatedMB -= vram
		delete(budget.Allocations, modelName)
	}
	return nil
}

// GetAvailable returns available VRAM in MB for a GPU.
func (m *Manager) GetAvailable(gpuUUID string) int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	budget, ok := m.gpuBudgets[gpuUUID]
	if !ok {
		return 0
	}
	return budget.TotalMB - budget.AllocatedMB
}

// RegisterGPU adds a GPU to the manager's tracking.
func (m *Manager) RegisterGPU(gpuUUID string, nodeName string, totalMB int64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.gpuBudgets[gpuUUID] = &GPUBudget{
		GPUUUID:     gpuUUID,
		NodeName:    nodeName,
		TotalMB:     totalMB,
		AllocatedMB: 0,
		Allocations: make(map[string]int64),
	}
}

// SyncFromMetrics updates VRAM tracking from DCGM metrics.
// Phase 2: Will reconcile tracked state with actual GPU usage.
func (m *Manager) SyncFromMetrics(ctx context.Context) error {
	// TODO(Phase 2): Query DCGM for actual VRAM usage and reconcile
	m.log.V(1).Info("SyncFromMetrics called (stub)")
	return nil
}
