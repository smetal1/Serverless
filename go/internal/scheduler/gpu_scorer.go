package scheduler

import (
	"context"

	"github.com/go-logr/logr"
)

// GPUScorer scores nodes based on GPU utilization and snapshot locality.
// Phase 2: Will consume DCGM Prometheus metrics for real-time GPU data.
type GPUScorer struct {
	prometheusAddr string
	log            logr.Logger
}

// NewGPUScorer creates a new GPU scorer.
func NewGPUScorer(prometheusAddr string, log logr.Logger) *GPUScorer {
	return &GPUScorer{
		prometheusAddr: prometheusAddr,
		log:            log,
	}
}

// NodeGPUInfo contains GPU information for a node.
type NodeGPUInfo struct {
	NodeName    string
	GPUs        []GPUInfo
	TotalVRAMMB int64
	UsedVRAMMB  int64
	FreeVRAMMB  int64
}

// GPUInfo contains information about a single GPU.
type GPUInfo struct {
	UUID           string
	Model          string
	VRAMTotalMB    int64
	VRAMUsedMB     int64
	UtilizationPct float64
	TemperatureC   float64
}

// ScoreNode returns a score (0-100) for scheduling a pod on a given node.
// Higher score = better fit.
// Phase 2: Will query DCGM metrics via Prometheus.
func (g *GPUScorer) ScoreNode(ctx context.Context, nodeName string, requiredVRAMMB int64) (int64, error) {
	// TODO(Phase 2): Implement real scoring based on:
	// 1. Available VRAM (must meet minimum)
	// 2. GPU utilization (prefer partially utilized for binpacking)
	// 3. Temperature (avoid hot GPUs)
	// 4. Snapshot locality (bonus for local NVMe cache)
	g.log.V(1).Info("ScoreNode called (stub)", "node", nodeName, "requiredVRAM", requiredVRAMMB)
	return 50, nil
}

// GetNodeGPUInfo retrieves GPU information for a node from Prometheus.
// Phase 2: Will query DCGM exporter metrics.
func (g *GPUScorer) GetNodeGPUInfo(ctx context.Context, nodeName string) (*NodeGPUInfo, error) {
	// TODO(Phase 2): Query Prometheus for DCGM metrics
	g.log.V(1).Info("GetNodeGPUInfo called (stub)", "node", nodeName)
	return &NodeGPUInfo{
		NodeName:    nodeName,
		TotalVRAMMB: 48000,
		UsedVRAMMB:  0,
		FreeVRAMMB:  48000,
	}, nil
}
