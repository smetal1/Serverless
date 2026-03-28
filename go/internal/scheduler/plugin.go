// Package scheduler implements a Kubernetes scheduler framework plugin for GPU-aware scheduling.
// Phase 2: Currently scaffolded with stubs.
package scheduler

import (
	"context"
	"fmt"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
)

const (
	// PluginName is the name of the scheduler plugin.
	PluginName = "PodstackGPUScheduler"
)

// Plugin implements the K8s scheduler framework interfaces:
// Filter, Score, PostFilter for GPU-aware pod placement.
type Plugin struct {
	gpuScorer   *GPUScorer
	prewarmer   *Prewarmer
	log         logr.Logger
}

// NewPlugin creates a new scheduler plugin instance.
func NewPlugin(prometheusAddr string, log logr.Logger) *Plugin {
	return &Plugin{
		gpuScorer: NewGPUScorer(prometheusAddr, log),
		prewarmer: NewPrewarmer(prometheusAddr, log),
		log:       log,
	}
}

// Name returns the plugin name.
func (p *Plugin) Name() string {
	return PluginName
}

// Filter checks if a node has sufficient GPU resources for the pod.
// Phase 2: Will check per-GPU VRAM availability via DCGM metrics.
func (p *Plugin) Filter(ctx context.Context, pod *corev1.Pod, nodeName string) (bool, error) {
	// TODO(Phase 2): Implement GPU-aware filtering
	// 1. Read GPU resource requests from pod spec
	// 2. Query DCGM metrics for available VRAM on each GPU on this node
	// 3. Check if any GPU has enough free VRAM
	// 4. Check if snapshot exists locally on this node (prefer local)
	p.log.V(1).Info("Filter called (stub)", "pod", pod.Name, "node", nodeName)
	return true, nil
}

// Score assigns a scheduling score to a node based on GPU optimization criteria.
// Phase 2: Will prefer nodes with cached snapshots and binpack GPUs.
func (p *Plugin) Score(ctx context.Context, pod *corev1.Pod, nodeName string) (int64, error) {
	// TODO(Phase 2): Implement GPU-aware scoring
	// 1. Prefer nodes where snapshot is already cached (NVMe local)
	// 2. Prefer nodes with partially utilized GPUs (binpacking)
	// 3. Avoid nodes at thermal limits
	// 4. Consider network proximity to NFS
	p.log.V(1).Info("Score called (stub)", "pod", pod.Name, "node", nodeName)
	return 50, nil
}

// PostFilter handles cases where no node passes the Filter.
// Phase 2: Will preempt idle models to free GPU resources.
func (p *Plugin) PostFilter(ctx context.Context, pod *corev1.Pod) (string, error) {
	// TODO(Phase 2): Implement preemption of idle models
	// 1. Find idle model pods on nodes with matching GPU type
	// 2. Evict the least-recently-used idle model
	// 3. Return the freed node name
	p.log.V(1).Info("PostFilter called (stub)", "pod", pod.Name)
	return "", fmt.Errorf("no suitable node found and preemption not yet implemented")
}
