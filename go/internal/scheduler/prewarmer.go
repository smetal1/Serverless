package scheduler

import (
	"context"
	"time"

	"github.com/go-logr/logr"
)

// Prewarmer predicts which models will be needed and pre-warms them to standby state.
// Phase 2: Will use ML-based intent prediction from Prometheus metrics.
type Prewarmer struct {
	prometheusAddr string
	log            logr.Logger
}

// NewPrewarmer creates a new predictive pre-warmer.
func NewPrewarmer(prometheusAddr string, log logr.Logger) *Prewarmer {
	return &Prewarmer{
		prometheusAddr: prometheusAddr,
		log:            log,
	}
}

// ModelPrediction represents a predicted model usage.
type ModelPrediction struct {
	ModelName   string
	Probability float64
	PredictedAt time.Time
	Reason      string // e.g., "time_pattern", "request_trend", "correlated_model"
}

// PredictNextModels analyzes recent request patterns and returns models
// likely to be needed in the near future.
// Phase 2: Will implement time-series analysis on Prometheus metrics.
func (p *Prewarmer) PredictNextModels(ctx context.Context) ([]ModelPrediction, error) {
	// TODO(Phase 2): Implement prediction based on:
	// 1. Time-of-day patterns (e.g., TTS models used during business hours)
	// 2. Request frequency trends (recently popular models)
	// 3. Correlated model access (model A often followed by model B)
	// 4. Tenant-specific patterns
	p.log.V(1).Info("PredictNextModels called (stub)")
	return nil, nil
}

// ShouldPrewarm decides whether a model should be pre-warmed to standby.
// Phase 2: Will consider VRAM budget, prediction confidence, and model size.
func (p *Prewarmer) ShouldPrewarm(ctx context.Context, prediction ModelPrediction, availableVRAMMB int64) bool {
	// TODO(Phase 2): Implement decision logic
	// Only pre-warm if prediction probability > threshold (e.g., 0.7)
	// and enough VRAM budget exists for potential GPU restore
	p.log.V(1).Info("ShouldPrewarm called (stub)", "model", prediction.ModelName)
	return false
}

// RunLoop runs the pre-warming loop at regular intervals.
// Phase 2: Will periodically check predictions and trigger pre-warming.
func (p *Prewarmer) RunLoop(ctx context.Context, interval time.Duration) error {
	// TODO(Phase 2): Implement pre-warming loop
	p.log.Info("Prewarmer loop started (stub)", "interval", interval)
	<-ctx.Done()
	return ctx.Err()
}
