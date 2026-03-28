// Package gateway implements the API Gateway for the Podstack Inference OS.
// It provides an OpenAI-compatible HTTP API that dispatches inference requests
// to warm, standby, or cold model pods with intelligent cold-start orchestration.
package gateway

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Server is the HTTP gateway that fronts all inference traffic.
type Server struct {
	router     *Router
	httpServer *http.Server
	log        logr.Logger
}

// Config holds the configuration for the gateway server.
type Config struct {
	// Addr is the listen address (e.g. ":8080").
	Addr string

	// K8sClient is the controller-runtime client for accessing CRDs.
	K8sClient client.Client

	// Namespace is the Kubernetes namespace where model resources live.
	Namespace string

	// ReadTimeout is the maximum duration for reading the entire request.
	ReadTimeout time.Duration

	// WriteTimeout is the maximum duration before timing out writes of the response.
	WriteTimeout time.Duration
}

// NewServer creates a new gateway Server with the given configuration.
func NewServer(cfg Config, log logr.Logger) *Server {
	if cfg.ReadTimeout == 0 {
		cfg.ReadTimeout = 30 * time.Second
	}
	if cfg.WriteTimeout == 0 {
		cfg.WriteTimeout = 120 * time.Second
	}

	r := NewRouter(cfg.K8sClient, cfg.Namespace, log.WithName("router"))

	mux := chi.NewRouter()

	// Standard middleware stack.
	mux.Use(middleware.RequestID)
	mux.Use(middleware.RealIP)
	mux.Use(middleware.Logger)
	mux.Use(middleware.Recoverer)

	// Health and readiness probes are unauthenticated.
	mux.Get("/health", handleHealth)
	mux.Get("/ready", handleReady(r))

	// Authenticated routes.
	mux.Group(func(mux chi.Router) {
		mux.Use(AuthMiddleware(cfg.K8sClient, cfg.Namespace))
		mux.Use(RateLimitMiddleware())

		// OpenAI-compatible inference endpoints.
		mux.Post("/v1/chat/completions", r.HandleInference)
		mux.Post("/v1/completions", r.HandleInference)
		mux.Post("/v1/embeddings", r.HandleInference)
		mux.Post("/v1/images/generations", r.HandleInference)
		mux.Post("/v1/audio/transcriptions", r.HandleInference)
		mux.Post("/v1/audio/speech", r.HandleInference)
		mux.Post("/v1/inference", r.HandleInference)

		// Model listing.
		mux.Get("/v1/models", HandleListModels(cfg.K8sClient, cfg.Namespace))
	})

	s := &Server{
		router: r,
		log:    log.WithName("gateway"),
		httpServer: &http.Server{
			Addr:         cfg.Addr,
			Handler:      mux,
			ReadTimeout:  cfg.ReadTimeout,
			WriteTimeout: cfg.WriteTimeout,
		},
	}

	return s
}

// Start begins serving HTTP traffic and starts the background pool watcher.
// It blocks until the context is cancelled or the server encounters a fatal error.
func (s *Server) Start(ctx context.Context) error {
	// Start the background watcher that keeps warm/standby pools current.
	go func() {
		if err := s.router.StartWatcher(ctx); err != nil {
			s.log.Error(err, "pool watcher exited with error")
		}
	}()

	// Perform an initial pool refresh before accepting traffic.
	if err := s.router.RefreshPools(ctx); err != nil {
		s.log.Error(err, "initial pool refresh failed, starting with empty pools")
	}

	s.log.Info("starting gateway server", "addr", s.httpServer.Addr)

	errCh := make(chan error, 1)
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- fmt.Errorf("http server error: %w", err)
		}
		close(errCh)
	}()

	select {
	case <-ctx.Done():
		s.log.Info("context cancelled, shutting down gateway server")
		return s.Stop(context.Background())
	case err := <-errCh:
		return err
	}
}

// Stop gracefully shuts down the HTTP server with a 15-second deadline.
func (s *Server) Stop(ctx context.Context) error {
	s.log.Info("stopping gateway server")
	shutdownCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	return s.httpServer.Shutdown(shutdownCtx)
}

// handleHealth returns 200 OK if the server process is alive.
func handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

// handleReady returns 200 OK if the router has at least performed one pool refresh.
func handleReady(r *Router) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		r.mu.RLock()
		poolsLoaded := r.warmPool != nil
		r.mu.RUnlock()

		w.Header().Set("Content-Type", "application/json")
		if poolsLoaded {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status":"ready"}`))
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"status":"not_ready"}`))
		}
	}
}
