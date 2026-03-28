package vram

import (
	"context"

	"github.com/go-logr/logr"
)

// Pager implements layer-wise VRAM paging for transformer models.
// Phase 3: Only active transformer layers are kept in VRAM;
// inactive layers are paged to CPU memory or NFS.
type Pager struct {
	log logr.Logger
}

// PageConfig configures the paging behavior for a model.
type PageConfig struct {
	ModelName       string
	TotalLayers     int
	ActiveLayers    int   // Number of layers to keep in VRAM
	PageSizeMB      int64 // Size of each layer in VRAM
	PrefetchCount   int   // Number of layers to prefetch ahead
	NFSBackingPath  string
}

// NewPager creates a new VRAM pager.
func NewPager(log logr.Logger) *Pager {
	return &Pager{log: log}
}

// Setup initializes paging for a model.
// Phase 3: Will configure layer-wise memory management.
func (p *Pager) Setup(ctx context.Context, config PageConfig) error {
	// TODO(Phase 3): Implement layer-wise paging setup
	// 1. Analyze model architecture to determine layer boundaries
	// 2. Allocate VRAM for active window (activeLayers * pageSizeMB)
	// 3. Set up CPU memory backing store
	// 4. Configure NFS backing for cold layers
	// 5. Initialize prefetch pipeline
	p.log.Info("Pager Setup called (stub)", "model", config.ModelName, "layers", config.TotalLayers)
	return nil
}

// PageIn brings a layer into VRAM from CPU memory or NFS.
// Phase 3: Will implement async DMA transfer.
func (p *Pager) PageIn(ctx context.Context, modelName string, layerIndex int) error {
	// TODO(Phase 3): Implement page-in via PCIe DMA
	p.log.V(2).Info("PageIn called (stub)", "model", modelName, "layer", layerIndex)
	return nil
}

// PageOut evicts a layer from VRAM to CPU memory.
// Phase 3: Will implement async DMA transfer.
func (p *Pager) PageOut(ctx context.Context, modelName string, layerIndex int) error {
	// TODO(Phase 3): Implement page-out via PCIe DMA
	p.log.V(2).Info("PageOut called (stub)", "model", modelName, "layer", layerIndex)
	return nil
}

// Teardown cleans up paging resources for a model.
func (p *Pager) Teardown(ctx context.Context, modelName string) error {
	// TODO(Phase 3): Cleanup
	p.log.Info("Pager Teardown called (stub)", "model", modelName)
	return nil
}
