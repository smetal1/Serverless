package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

// TenantUsage holds aggregated usage counters for a single tenant.
type TenantUsage struct {
	// TotalRequests is the cumulative number of inference requests.
	TotalRequests int64

	// TotalInputTokens is the cumulative count of input (prompt) tokens.
	TotalInputTokens int64

	// TotalOutputTokens is the cumulative count of output (completion) tokens.
	TotalOutputTokens int64

	// TotalDurationMs is the cumulative inference latency in milliseconds.
	TotalDurationMs int64
}

// BillingCollector tracks per-tenant token and request usage using Prometheus
// counters and histograms. All metrics are labelled by tenant and model to
// enable fine-grained cost attribution and billing.
type BillingCollector struct {
	mu sync.RWMutex

	// In-memory per-tenant aggregates for fast lookup.
	tenantUsage map[string]*TenantUsage

	// Prometheus metrics.
	requestsTotal     *prometheus.CounterVec
	inputTokensTotal  *prometheus.CounterVec
	outputTokensTotal *prometheus.CounterVec
	durationMs        *prometheus.HistogramVec
}

// NewBillingCollector creates a new BillingCollector and registers its
// Prometheus metrics. The metrics are registered with the default Prometheus
// registerer; callers should ensure this is called only once.
func NewBillingCollector() *BillingCollector {
	b := &BillingCollector{
		tenantUsage: make(map[string]*TenantUsage),

		requestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "podstack",
				Subsystem: "billing",
				Name:      "requests_total",
				Help:      "Total number of inference requests per tenant and model.",
			},
			[]string{"tenant", "model"},
		),
		inputTokensTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "podstack",
				Subsystem: "billing",
				Name:      "input_tokens_total",
				Help:      "Total number of input (prompt) tokens consumed per tenant and model.",
			},
			[]string{"tenant", "model"},
		),
		outputTokensTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "podstack",
				Subsystem: "billing",
				Name:      "output_tokens_total",
				Help:      "Total number of output (completion) tokens produced per tenant and model.",
			},
			[]string{"tenant", "model"},
		),
		durationMs: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "podstack",
				Subsystem: "billing",
				Name:      "request_duration_milliseconds",
				Help:      "Inference request duration in milliseconds per tenant and model.",
				Buckets:   []float64{10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000},
			},
			[]string{"tenant", "model"},
		),
	}

	// Register all metrics with the default Prometheus registerer.
	prometheus.MustRegister(b.requestsTotal)
	prometheus.MustRegister(b.inputTokensTotal)
	prometheus.MustRegister(b.outputTokensTotal)
	prometheus.MustRegister(b.durationMs)

	return b
}

// RecordRequest records a single inference request's metrics for the given
// tenant and model. Token counts and duration are accumulated both in
// Prometheus counters and in the in-memory per-tenant aggregates.
func (b *BillingCollector) RecordRequest(tenant, model string, inputTokens, outputTokens int64, durationMs int64) {
	// Update Prometheus counters.
	b.requestsTotal.WithLabelValues(tenant, model).Inc()
	b.inputTokensTotal.WithLabelValues(tenant, model).Add(float64(inputTokens))
	b.outputTokensTotal.WithLabelValues(tenant, model).Add(float64(outputTokens))
	b.durationMs.WithLabelValues(tenant, model).Observe(float64(durationMs))

	// Update in-memory aggregates.
	b.mu.Lock()
	defer b.mu.Unlock()

	usage, ok := b.tenantUsage[tenant]
	if !ok {
		usage = &TenantUsage{}
		b.tenantUsage[tenant] = usage
	}

	usage.TotalRequests++
	usage.TotalInputTokens += inputTokens
	usage.TotalOutputTokens += outputTokens
	usage.TotalDurationMs += durationMs
}

// GetTenantUsage returns the aggregated usage for the specified tenant.
// Returns nil if the tenant has not yet recorded any requests.
func (b *BillingCollector) GetTenantUsage(tenant string) *TenantUsage {
	b.mu.RLock()
	defer b.mu.RUnlock()

	usage, ok := b.tenantUsage[tenant]
	if !ok {
		return nil
	}

	// Return a copy to prevent data races.
	result := *usage
	return &result
}
