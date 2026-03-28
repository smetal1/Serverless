package scheduler

import (
	"context"
	"time"

	"github.com/go-logr/logr"
)

// MetricsCollector aggregates metrics from Prometheus for the scheduler.
// Phase 2: Will provide real-time GPU and model usage data to scheduling decisions.
type MetricsCollector struct {
	prometheusAddr string
	log            logr.Logger
}

// NewMetricsCollector creates a new metrics collector.
func NewMetricsCollector(prometheusAddr string, log logr.Logger) *MetricsCollector {
	return &MetricsCollector{
		prometheusAddr: prometheusAddr,
		log:            log,
	}
}

// ClusterGPUMetrics represents cluster-wide GPU metrics.
type ClusterGPUMetrics struct {
	TotalGPUs       int
	ActiveGPUs      int
	TotalVRAMMB     int64
	UsedVRAMMB      int64
	AvgUtilization  float64
	CollectedAt     time.Time
}

// ModelMetrics represents usage metrics for a specific model.
type ModelMetrics struct {
	ModelName       string
	RequestsPerMin  float64
	AvgLatencyMs    float64
	P99LatencyMs    float64
	ActiveReplicas  int
	VRAMUsedMB      int64
	LastRequestAt   time.Time
}

// CollectClusterMetrics gathers cluster-wide GPU metrics from Prometheus.
// Phase 2: Will query DCGM exporter metrics.
func (m *MetricsCollector) CollectClusterMetrics(ctx context.Context) (*ClusterGPUMetrics, error) {
	// TODO(Phase 2): Query Prometheus for cluster-wide GPU metrics
	m.log.V(1).Info("CollectClusterMetrics called (stub)")
	return &ClusterGPUMetrics{
		CollectedAt: time.Now(),
	}, nil
}

// CollectModelMetrics gathers usage metrics for a specific model.
// Phase 2: Will query request rate, latency, and resource usage.
func (m *MetricsCollector) CollectModelMetrics(ctx context.Context, modelName string) (*ModelMetrics, error) {
	// TODO(Phase 2): Query Prometheus for model-specific metrics
	m.log.V(1).Info("CollectModelMetrics called (stub)", "model", modelName)
	return &ModelMetrics{
		ModelName: modelName,
	}, nil
}

// RunCollectionLoop periodically collects and caches metrics.
// Phase 2: Will maintain a local cache refreshed every scrapeInterval.
func (m *MetricsCollector) RunCollectionLoop(ctx context.Context, scrapeInterval time.Duration) error {
	// TODO(Phase 2): Implement periodic collection loop
	m.log.Info("MetricsCollector loop started (stub)", "interval", scrapeInterval)
	<-ctx.Done()
	return ctx.Err()
}
