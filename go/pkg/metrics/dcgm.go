// Package metrics provides Prometheus-based metric collection for the Podstack
// Inference OS, including GPU metrics from DCGM and per-tenant billing counters.
package metrics

import (
	"context"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/common/model"

	promapi "github.com/prometheus/client_golang/api"
	promv1 "github.com/prometheus/client_golang/api/prometheus/v1"
)

// GPUMetrics contains GPU-level telemetry scraped from DCGM exporter via
// Prometheus. Fields map to the standard DCGM-exporter metric names.
type GPUMetrics struct {
	// GPUUUID is the unique identifier of the GPU (e.g. "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").
	GPUUUID string

	// NodeName is the Kubernetes node hosting this GPU.
	NodeName string

	// UtilizationPct is the GPU compute utilization percentage (0-100).
	UtilizationPct float64

	// VRAMUsedMB is the current GPU memory usage in megabytes.
	VRAMUsedMB int64

	// VRAMTotalMB is the total GPU memory capacity in megabytes.
	VRAMTotalMB int64

	// TemperatureC is the GPU core temperature in degrees Celsius.
	TemperatureC float64

	// PowerWatts is the current GPU power draw in watts.
	PowerWatts float64
}

// DCGMCollector reads GPU metrics from DCGM exporter via the Prometheus
// query API. It connects to a Prometheus instance that scrapes the
// dcgm-exporter DaemonSet running on GPU nodes.
type DCGMCollector struct {
	prometheusAddr string
	log            logr.Logger
	client         promapi.Client
	api            promv1.API
}

// NewDCGMCollector creates a new collector that queries Prometheus at the
// given address (e.g. "http://prometheus.monitoring:9090") for DCGM metrics.
func NewDCGMCollector(prometheusAddr string, log logr.Logger) *DCGMCollector {
	client, err := promapi.NewClient(promapi.Config{
		Address: prometheusAddr,
	})
	if err != nil {
		log.Error(err, "failed to create Prometheus client", "address", prometheusAddr)
	}

	var api promv1.API
	if client != nil {
		api = promv1.NewAPI(client)
	}

	return &DCGMCollector{
		prometheusAddr: prometheusAddr,
		log:            log.WithName("dcgm-collector"),
		client:         client,
		api:            api,
	}
}

// CollectAll queries Prometheus for GPU metrics from all GPUs across all nodes.
// It returns a slice of GPUMetrics, one per GPU device discovered.
func (d *DCGMCollector) CollectAll(ctx context.Context) ([]GPUMetrics, error) {
	if d.api == nil {
		return nil, fmt.Errorf("dcgm collector: prometheus client not initialized")
	}

	// Query GPU utilization to discover all GPU UUIDs.
	utilResult, _, err := d.api.Query(ctx, "DCGM_FI_DEV_GPU_UTIL", time.Now())
	if err != nil {
		return nil, fmt.Errorf("dcgm collector: failed to query GPU utilization: %w", err)
	}

	vector, ok := utilResult.(model.Vector)
	if !ok {
		return nil, fmt.Errorf("dcgm collector: unexpected result type for GPU utilization query")
	}

	// Collect metrics for each GPU discovered.
	var allMetrics []GPUMetrics
	for _, sample := range vector {
		gpuUUID := string(sample.Metric["gpu"])
		nodeName := string(sample.Metric["Hostname"])
		if nodeName == "" {
			nodeName = string(sample.Metric["instance"])
		}

		gm := GPUMetrics{
			GPUUUID:        gpuUUID,
			NodeName:       nodeName,
			UtilizationPct: float64(sample.Value),
		}

		// Fill in memory, temperature, and power from additional queries.
		if err := d.fillMemoryMetrics(ctx, gpuUUID, &gm); err != nil {
			d.log.V(1).Info("failed to fetch memory metrics", "gpu", gpuUUID, "error", err)
		}
		if err := d.fillThermalMetrics(ctx, gpuUUID, &gm); err != nil {
			d.log.V(1).Info("failed to fetch thermal metrics", "gpu", gpuUUID, "error", err)
		}

		allMetrics = append(allMetrics, gm)
	}

	d.log.V(1).Info("collected GPU metrics", "count", len(allMetrics))
	return allMetrics, nil
}

// GetGPUMetrics returns metrics for a single GPU identified by its UUID.
func (d *DCGMCollector) GetGPUMetrics(ctx context.Context, gpuUUID string) (*GPUMetrics, error) {
	if d.api == nil {
		return nil, fmt.Errorf("dcgm collector: prometheus client not initialized")
	}

	gm := &GPUMetrics{GPUUUID: gpuUUID}

	// GPU utilization.
	query := fmt.Sprintf(`DCGM_FI_DEV_GPU_UTIL{gpu=%q}`, gpuUUID)
	result, _, err := d.api.Query(ctx, query, time.Now())
	if err != nil {
		return nil, fmt.Errorf("dcgm collector: failed to query utilization for GPU %s: %w", gpuUUID, err)
	}
	if vec, ok := result.(model.Vector); ok && len(vec) > 0 {
		gm.UtilizationPct = float64(vec[0].Value)
		gm.NodeName = string(vec[0].Metric["Hostname"])
		if gm.NodeName == "" {
			gm.NodeName = string(vec[0].Metric["instance"])
		}
	}

	// Memory and thermal metrics.
	if err := d.fillMemoryMetrics(ctx, gpuUUID, gm); err != nil {
		d.log.V(1).Info("failed to fetch memory metrics", "gpu", gpuUUID, "error", err)
	}
	if err := d.fillThermalMetrics(ctx, gpuUUID, gm); err != nil {
		d.log.V(1).Info("failed to fetch thermal metrics", "gpu", gpuUUID, "error", err)
	}

	return gm, nil
}

// GetAvailableVRAM returns the available (unused) VRAM in megabytes for the
// specified GPU. This is calculated as total VRAM minus used VRAM.
func (d *DCGMCollector) GetAvailableVRAM(ctx context.Context, gpuUUID string) (int64, error) {
	gm, err := d.GetGPUMetrics(ctx, gpuUUID)
	if err != nil {
		return 0, err
	}
	available := gm.VRAMTotalMB - gm.VRAMUsedMB
	if available < 0 {
		available = 0
	}
	return available, nil
}

// fillMemoryMetrics populates VRAM used and total fields from Prometheus.
func (d *DCGMCollector) fillMemoryMetrics(ctx context.Context, gpuUUID string, gm *GPUMetrics) error {
	// DCGM reports memory in MiB by default.
	usedQuery := fmt.Sprintf(`DCGM_FI_DEV_FB_USED{gpu=%q}`, gpuUUID)
	usedResult, _, err := d.api.Query(ctx, usedQuery, time.Now())
	if err != nil {
		return fmt.Errorf("failed to query VRAM used: %w", err)
	}
	if vec, ok := usedResult.(model.Vector); ok && len(vec) > 0 {
		gm.VRAMUsedMB = int64(vec[0].Value)
	}

	totalQuery := fmt.Sprintf(`DCGM_FI_DEV_FB_FREE{gpu=%q} + DCGM_FI_DEV_FB_USED{gpu=%q}`, gpuUUID, gpuUUID)
	totalResult, _, err := d.api.Query(ctx, totalQuery, time.Now())
	if err != nil {
		return fmt.Errorf("failed to query VRAM total: %w", err)
	}
	if vec, ok := totalResult.(model.Vector); ok && len(vec) > 0 {
		gm.VRAMTotalMB = int64(vec[0].Value)
	}

	return nil
}

// fillThermalMetrics populates temperature and power fields from Prometheus.
func (d *DCGMCollector) fillThermalMetrics(ctx context.Context, gpuUUID string, gm *GPUMetrics) error {
	// Temperature in degrees Celsius.
	tempQuery := fmt.Sprintf(`DCGM_FI_DEV_GPU_TEMP{gpu=%q}`, gpuUUID)
	tempResult, _, err := d.api.Query(ctx, tempQuery, time.Now())
	if err != nil {
		return fmt.Errorf("failed to query temperature: %w", err)
	}
	if vec, ok := tempResult.(model.Vector); ok && len(vec) > 0 {
		gm.TemperatureC = float64(vec[0].Value)
	}

	// Power in watts.
	powerQuery := fmt.Sprintf(`DCGM_FI_DEV_POWER_USAGE{gpu=%q}`, gpuUUID)
	powerResult, _, err := d.api.Query(ctx, powerQuery, time.Now())
	if err != nil {
		return fmt.Errorf("failed to query power: %w", err)
	}
	if vec, ok := powerResult.(model.Vector); ok && len(vec) > 0 {
		gm.PowerWatts = float64(vec[0].Value)
	}

	return nil
}
