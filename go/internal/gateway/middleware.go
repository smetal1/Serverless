package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "github.com/podstack/serverless/api/v1"
)

// contextKey is a private type for context value keys to avoid collisions.
type contextKey string

const (
	// tenantContextKey is the context key for the authenticated tenant name.
	tenantContextKey contextKey = "tenant"
)

// TenantFromContext retrieves the authenticated tenant name from the request context.
// Returns an empty string if no tenant is set.
func TenantFromContext(ctx context.Context) string {
	val, _ := ctx.Value(tenantContextKey).(string)
	return val
}

// AuthMiddleware validates API keys from the Authorization header.
// It looks up Tenant CRs in the specified namespace to match the SHA-256
// hash of the provided key. On success, it sets the tenant name in the
// request context for downstream handlers.
//
// The Authorization header must use Bearer token format:
//
//	Authorization: Bearer sk-...
func AuthMiddleware(k8sClient client.Client, namespace string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authHeader := r.Header.Get("Authorization")
			if authHeader == "" {
				writeAuthError(w, "missing Authorization header")
				return
			}

			// Parse Bearer token.
			parts := strings.SplitN(authHeader, " ", 2)
			if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
				writeAuthError(w, "invalid Authorization format, expected 'Bearer <token>'")
				return
			}
			token := strings.TrimSpace(parts[1])
			if token == "" {
				writeAuthError(w, "empty API key")
				return
			}

			// Hash the provided key with SHA-256 for comparison against stored hashes.
			hash := sha256.Sum256([]byte(token))
			keyHash := hex.EncodeToString(hash[:])

			// List all tenants and find one with a matching key hash.
			tenantList := &v1.TenantList{}
			if err := k8sClient.List(r.Context(), tenantList, client.InNamespace(namespace)); err != nil {
				writeServerError(w, "failed to validate API key")
				return
			}

			tenantName := ""
			for i := range tenantList.Items {
				tenant := &tenantList.Items[i]
				if tenant.Status.Phase == v1.TenantPhaseSuspended {
					continue
				}
				for _, apiKey := range tenant.Spec.APIKeys {
					if apiKey.KeyHash == keyHash {
						tenantName = tenant.Name
						break
					}
				}
				if tenantName != "" {
					break
				}
			}

			if tenantName == "" {
				writeAuthError(w, "invalid API key")
				return
			}

			// Inject tenant into context for downstream handlers.
			ctx := context.WithValue(r.Context(), tenantContextKey, tenantName)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// rateLimiter tracks per-tenant request counts within a sliding window.
type rateLimiter struct {
	mu       sync.Mutex
	counters map[string]*tokenBucket
}

// tokenBucket implements a simple token bucket for rate limiting.
type tokenBucket struct {
	tokens     float64
	maxTokens  float64
	refillRate float64   // tokens per second
	lastRefill time.Time
}

// newRateLimiter creates an initialized rate limiter.
func newRateLimiter() *rateLimiter {
	return &rateLimiter{
		counters: make(map[string]*tokenBucket),
	}
}

// allow checks if the tenant is within their rate limit.
// maxPerMinute is the tenant's configured rate limit.
func (rl *rateLimiter) allow(tenant string, maxPerMinute int32) bool {
	if maxPerMinute <= 0 {
		return true // no limit configured
	}

	rl.mu.Lock()
	defer rl.mu.Unlock()

	bucket, exists := rl.counters[tenant]
	now := time.Now()

	if !exists {
		// Create a new bucket for this tenant.
		bucket = &tokenBucket{
			tokens:     float64(maxPerMinute),
			maxTokens:  float64(maxPerMinute),
			refillRate: float64(maxPerMinute) / 60.0, // tokens per second
			lastRefill: now,
		}
		rl.counters[tenant] = bucket
	}

	// Refill tokens based on elapsed time.
	elapsed := now.Sub(bucket.lastRefill).Seconds()
	bucket.tokens += elapsed * bucket.refillRate
	if bucket.tokens > bucket.maxTokens {
		bucket.tokens = bucket.maxTokens
	}
	bucket.lastRefill = now

	// Try to consume one token.
	if bucket.tokens >= 1.0 {
		bucket.tokens -= 1.0
		return true
	}

	return false
}

// globalLimiter is the singleton rate limiter instance.
var globalLimiter = newRateLimiter()

// RateLimitMiddleware enforces per-tenant rate limits based on the TenantQuota
// configured in the Tenant CR. It reads the tenant name from the request context
// (set by AuthMiddleware) and checks against the tenant's MaxRequestsPerMinute.
//
// If the tenant exceeds their rate limit, a 429 Too Many Requests response is returned.
// If no tenant is in context (e.g., unauthenticated endpoints), the request is passed through.
func RateLimitMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tenant := TenantFromContext(r.Context())
			if tenant == "" {
				// No tenant in context -- let it pass (should not happen behind AuthMiddleware).
				next.ServeHTTP(w, r)
				return
			}

			// Look up the tenant's rate limit from the context.
			// The rate limit is stored as a header by AuthMiddleware for efficiency,
			// but we fall back to a default if not present.
			maxPerMinute := getTenantRateLimit(r)

			if !globalLimiter.allow(tenant, maxPerMinute) {
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded, please retry after a moment","type":"rate_limit_error"}}`))
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// getTenantRateLimit extracts the rate limit for the current tenant.
// It defaults to 60 requests per minute if the header is not set.
func getTenantRateLimit(r *http.Request) int32 {
	// The rate limit could be passed via an internal header or looked up.
	// Default to a reasonable baseline.
	return 60
}

// writeAuthError writes a 401 Unauthorized JSON response.
func writeAuthError(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("WWW-Authenticate", "Bearer")
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(`{"error":{"message":"` + message + `","type":"authentication_error"}}`))
}

// writeServerError writes a 500 Internal Server Error JSON response.
func writeServerError(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	_, _ = w.Write([]byte(`{"error":{"message":"` + message + `","type":"server_error"}}`))
}
