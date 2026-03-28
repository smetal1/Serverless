package gateway

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "github.com/podstack/serverless/api/v1"
)

const (
	// coldStartTimeout is the maximum time to wait for a cold-started model to become ready.
	coldStartTimeout = 30 * time.Second

	// poolRefreshInterval is how often the background watcher refreshes pools.
	poolRefreshInterval = 5 * time.Second

	// labelModel is the pod label key that identifies which model a pod serves.
	labelModel = "podstack.io/model"

	// labelRuntime is the pod label key that identifies the runtime type.
	labelRuntime = "podstack.io/runtime"

	// annotationBootTrigger is the annotation set on standby pods to trigger boot.
	annotationBootTrigger = "podstack.io/boot-trigger"

	// defaultInferencePort is the default port that inference runtimes listen on.
	defaultInferencePort = 8000
)

// ModelEndpoint represents a running model pod that can serve inference requests.
type ModelEndpoint struct {
	// Name is the ModelDeployment name.
	Name string

	// Address is the pod IP:port for direct proxying.
	Address string

	// Ready indicates the pod is passing readiness checks.
	Ready bool

	// LastUsed tracks the most recent request time for idle detection.
	LastUsed time.Time
}

// StandbyEntry represents a pre-warmed standby pod with model weights loaded
// on CPU but no GPU allocated. Can boot to full inference in <2s.
type StandbyEntry struct {
	// ModelName is the model identifier matching ModelDeployment.spec.modelName.
	ModelName string

	// PodName is the Kubernetes pod name.
	PodName string

	// Namespace is the Kubernetes namespace.
	Namespace string

	// SnapshotRef is the name of the Snapshot CR to restore from.
	SnapshotRef string
}

// ColdStartQueue tracks requests waiting for a model to become ready.
type ColdStartQueue struct {
	mu      sync.Mutex
	waiters map[string][]chan *ModelEndpoint // model name -> waiting request channels
}

// newColdStartQueue creates an initialized ColdStartQueue.
func newColdStartQueue() *ColdStartQueue {
	return &ColdStartQueue{
		waiters: make(map[string][]chan *ModelEndpoint),
	}
}

// enqueue adds a waiter for the given model and returns the channel to wait on.
func (q *ColdStartQueue) enqueue(model string) chan *ModelEndpoint {
	q.mu.Lock()
	defer q.mu.Unlock()

	ch := make(chan *ModelEndpoint, 1)
	q.waiters[model] = append(q.waiters[model], ch)
	return ch
}

// notify sends the endpoint to all waiters for the given model and clears the queue.
func (q *ColdStartQueue) notify(model string, ep *ModelEndpoint) {
	q.mu.Lock()
	defer q.mu.Unlock()

	for _, ch := range q.waiters[model] {
		select {
		case ch <- ep:
		default:
		}
	}
	delete(q.waiters, model)
}

// hasPending returns true if there are requests waiting for the given model.
func (q *ColdStartQueue) hasPending(model string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.waiters[model]) > 0
}

// Router is the core request dispatcher that manages warm, standby, and cold model pools.
// It implements the three-tier cold-start orchestration:
//   - Warm pool:    model is running with GPU, proxy immediately (~10ms)
//   - Standby pool: model weights on CPU, needs GPU attach, queue and wait (<2s)
//   - Cold start:   no pod exists, restore from snapshot or pull fresh (<5s with snapshot)
type Router struct {
	mu          sync.RWMutex
	warmPool    map[string]*ModelEndpoint  // model name -> ready endpoint
	standbyPool map[string]*StandbyEntry   // model name -> standby pod
	coldQueue   *ColdStartQueue
	k8sClient   client.Client
	namespace   string
	log         logr.Logger
}

// NewRouter creates a new Router with empty pools.
func NewRouter(k8sClient client.Client, namespace string, log logr.Logger) *Router {
	return &Router{
		warmPool:    make(map[string]*ModelEndpoint),
		standbyPool: make(map[string]*StandbyEntry),
		coldQueue:   newColdStartQueue(),
		k8sClient:   k8sClient,
		namespace:   namespace,
		log:         log,
	}
}

// HandleInference is the main dispatch handler for all inference requests.
// It implements the three-tier routing strategy:
//  1. Check warm pool -> proxy immediately (~10ms overhead)
//  2. Check standby pool -> trigger boot, queue request, wait for ready (<2s)
//  3. Neither exists -> trigger full restore from snapshot, queue request (<5s)
//
// The request body is buffered so it can be re-read when proxying to the backend.
func (r *Router) HandleInference(w http.ResponseWriter, req *http.Request) {
	// Buffer the request body so it can be re-read for proxying.
	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, `{"error":{"message":"failed to read request body","type":"server_error"}}`, http.StatusBadRequest)
		return
	}
	defer req.Body.Close()

	// Extract the model name from the request payload.
	model := extractModel(body)
	if model == "" {
		http.Error(w, `{"error":{"message":"missing or empty 'model' field in request","type":"invalid_request_error"}}`, http.StatusBadRequest)
		return
	}

	r.log.V(1).Info("inference request", "model", model, "path", req.URL.Path, "method", req.Method)

	// Tier 1: Check warm pool for an immediately available endpoint.
	r.mu.RLock()
	ep, warm := r.warmPool[model]
	r.mu.RUnlock()

	if warm && ep.Ready {
		r.log.V(1).Info("warm pool hit", "model", model, "address", ep.Address)
		ep.LastUsed = time.Now()
		req.Body = io.NopCloser(bytes.NewReader(body))
		r.proxyToEndpoint(w, req, ep)
		return
	}

	// Tier 2: Check standby pool for a pre-warmed pod that needs GPU boot.
	r.mu.RLock()
	sb, standby := r.standbyPool[model]
	r.mu.RUnlock()

	if standby {
		r.log.Info("standby pool hit, triggering boot", "model", model, "pod", sb.PodName)
		if err := r.triggerBoot(req.Context(), sb); err != nil {
			r.log.Error(err, "failed to trigger boot for standby pod", "pod", sb.PodName)
			// Fall through to cold start path.
		} else {
			// Queue this request and wait for the model to become ready.
			r.waitAndProxy(w, req, body, model)
			return
		}
	}

	// Tier 3: Cold start -- trigger full restore from NFS snapshot or fresh pull.
	r.log.Info("cold start triggered", "model", model)

	// Only trigger a restore if there is not already one in progress for this model.
	if !r.coldQueue.hasPending(model) {
		if err := r.triggerFullRestore(req.Context(), model); err != nil {
			r.log.Error(err, "failed to trigger full restore", "model", model)
			http.Error(w, `{"error":{"message":"failed to start model, please try again","type":"server_error"}}`, http.StatusServiceUnavailable)
			return
		}
	}

	r.waitAndProxy(w, req, body, model)
}

// waitAndProxy enqueues a request in the cold start queue and waits for the model
// to become ready, then proxies the request to the endpoint.
func (r *Router) waitAndProxy(w http.ResponseWriter, req *http.Request, body []byte, model string) {
	ctx, cancel := context.WithTimeout(req.Context(), coldStartTimeout)
	defer cancel()

	waiter := r.coldQueue.enqueue(model)

	select {
	case ep := <-waiter:
		if ep == nil {
			http.Error(w, `{"error":{"message":"model startup failed","type":"server_error"}}`, http.StatusServiceUnavailable)
			return
		}
		r.log.Info("cold start complete, proxying request", "model", model, "address", ep.Address)
		ep.LastUsed = time.Now()
		req.Body = io.NopCloser(bytes.NewReader(body))
		r.proxyToEndpoint(w, req, ep)

	case <-ctx.Done():
		r.log.Error(ctx.Err(), "cold start timed out", "model", model)
		http.Error(w, `{"error":{"message":"model cold start timed out, please retry","type":"timeout_error"}}`, http.StatusServiceUnavailable)
	}
}

// StartWatcher runs a polling loop that refreshes warm and standby pools by
// watching pod state in Kubernetes. It blocks until the context is cancelled.
func (r *Router) StartWatcher(ctx context.Context) error {
	r.log.Info("starting pool watcher", "interval", poolRefreshInterval)
	ticker := time.NewTicker(poolRefreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			r.log.Info("pool watcher stopped")
			return ctx.Err()
		case <-ticker.C:
			if err := r.RefreshPools(ctx); err != nil {
				r.log.Error(err, "pool refresh failed")
			}
		}
	}
}

// RefreshPools scans Kubernetes for current model pods and updates the warm
// and standby pools accordingly. It also notifies any queued cold-start waiters
// when their model becomes ready.
func (r *Router) RefreshPools(ctx context.Context) error {
	// List all pods with the podstack model label in the configured namespace.
	podList := &corev1.PodList{}
	if err := r.k8sClient.List(ctx, podList,
		client.InNamespace(r.namespace),
		client.HasLabels{labelModel},
	); err != nil {
		return fmt.Errorf("listing model pods: %w", err)
	}

	// List all ModelDeployments to get snapshot references and standby info.
	mdList := &v1.ModelDeploymentList{}
	if err := r.k8sClient.List(ctx, mdList, client.InNamespace(r.namespace)); err != nil {
		return fmt.Errorf("listing model deployments: %w", err)
	}

	// Build a map of model name -> ModelDeployment for quick lookup.
	mdByModel := make(map[string]*v1.ModelDeployment, len(mdList.Items))
	for i := range mdList.Items {
		md := &mdList.Items[i]
		mdByModel[md.Spec.ModelName] = md
	}

	newWarm := make(map[string]*ModelEndpoint)
	newStandby := make(map[string]*StandbyEntry)

	for i := range podList.Items {
		pod := &podList.Items[i]
		modelName := pod.Labels[labelModel]
		if modelName == "" {
			continue
		}

		// Reverse the model path encoding to get the original model name.
		modelName = strings.ReplaceAll(modelName, "--", "/")

		switch pod.Status.Phase {
		case corev1.PodRunning:
			// Check if the pod is ready (all containers passing readiness probes).
			ready := isPodReady(pod)
			if ready && pod.Status.PodIP != "" {
				addr := fmt.Sprintf("%s:%d", pod.Status.PodIP, inferencePort(pod))
				ep := &ModelEndpoint{
					Name:     modelName,
					Address:  addr,
					Ready:    true,
					LastUsed: time.Now(),
				}
				newWarm[modelName] = ep

				// Notify any cold-start waiters that this model is now available.
				r.coldQueue.notify(modelName, ep)
			}

		case corev1.PodPending:
			// Pods in Pending with model weights but no GPU are standby candidates.
			md, exists := mdByModel[modelName]
			if exists && md.Status.Phase == v1.PhaseStandby {
				entry := &StandbyEntry{
					ModelName:   modelName,
					PodName:     pod.Name,
					Namespace:   pod.Namespace,
					SnapshotRef: md.Status.SnapshotRef,
				}
				newStandby[modelName] = entry
			}
		}
	}

	// Also check ModelDeployments that are in Standby phase but might not have
	// pods yet (operator creates them lazily).
	for modelName, md := range mdByModel {
		if md.Status.Phase == v1.PhaseStandby {
			if _, already := newWarm[modelName]; !already {
				if _, already := newStandby[modelName]; !already {
					newStandby[modelName] = &StandbyEntry{
						ModelName:   modelName,
						Namespace:   md.Namespace,
						SnapshotRef: md.Status.SnapshotRef,
					}
				}
			}
		}
	}

	// Atomically swap the pools.
	r.mu.Lock()
	r.warmPool = newWarm
	r.standbyPool = newStandby
	r.mu.Unlock()

	r.log.V(1).Info("pools refreshed", "warm", len(newWarm), "standby", len(newStandby))
	return nil
}

// triggerBoot annotates the standby pod with a boot trigger so that the
// Podstack operator picks it up and attaches a GPU, transitioning it from
// Standby to Booting to Active.
func (r *Router) triggerBoot(ctx context.Context, sb *StandbyEntry) error {
	if sb.PodName == "" {
		// No specific pod -- update the ModelDeployment to signal the operator.
		return r.triggerModelTransition(ctx, sb.ModelName, v1.PhaseBooting)
	}

	pod := &corev1.Pod{}
	key := types.NamespacedName{Name: sb.PodName, Namespace: sb.Namespace}
	if err := r.k8sClient.Get(ctx, key, pod); err != nil {
		return fmt.Errorf("getting standby pod %s: %w", sb.PodName, err)
	}

	// Add the boot trigger annotation with a timestamp.
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	pod.Annotations[annotationBootTrigger] = time.Now().UTC().Format(time.RFC3339)

	if err := r.k8sClient.Update(ctx, pod); err != nil {
		return fmt.Errorf("annotating standby pod %s: %w", sb.PodName, err)
	}

	r.log.Info("boot trigger set on standby pod", "pod", sb.PodName, "model", sb.ModelName)

	// Remove from standby pool since it is now booting.
	r.mu.Lock()
	delete(r.standbyPool, sb.ModelName)
	r.mu.Unlock()

	return nil
}

// triggerModelTransition updates a ModelDeployment's status phase to signal
// the operator to transition the model.
func (r *Router) triggerModelTransition(ctx context.Context, model string, targetPhase string) error {
	mdList := &v1.ModelDeploymentList{}
	if err := r.k8sClient.List(ctx, mdList, client.InNamespace(r.namespace)); err != nil {
		return fmt.Errorf("listing model deployments: %w", err)
	}

	for i := range mdList.Items {
		md := &mdList.Items[i]
		if md.Spec.ModelName == model {
			// Add annotation to signal desired transition.
			if md.Annotations == nil {
				md.Annotations = make(map[string]string)
			}
			md.Annotations["podstack.io/desired-phase"] = targetPhase
			md.Annotations["podstack.io/transition-trigger"] = time.Now().UTC().Format(time.RFC3339)

			if err := r.k8sClient.Update(ctx, md); err != nil {
				return fmt.Errorf("updating ModelDeployment %s: %w", md.Name, err)
			}
			r.log.Info("triggered model transition", "model", model, "targetPhase", targetPhase)
			return nil
		}
	}

	return fmt.Errorf("ModelDeployment not found for model %q", model)
}

// triggerFullRestore creates or updates the ModelDeployment to trigger a full
// restore from an NFS snapshot. If no ModelDeployment exists for the model,
// one must be created by the tenant ahead of time -- this method signals the
// existing CR to begin the restore sequence.
func (r *Router) triggerFullRestore(ctx context.Context, model string) error {
	mdList := &v1.ModelDeploymentList{}
	if err := r.k8sClient.List(ctx, mdList, client.InNamespace(r.namespace)); err != nil {
		return fmt.Errorf("listing model deployments: %w", err)
	}

	for i := range mdList.Items {
		md := &mdList.Items[i]
		if md.Spec.ModelName == model {
			// Annotate to trigger the operator's restore flow.
			if md.Annotations == nil {
				md.Annotations = make(map[string]string)
			}
			md.Annotations["podstack.io/restore-trigger"] = time.Now().UTC().Format(time.RFC3339)
			md.Annotations["podstack.io/desired-phase"] = v1.PhaseBooting

			if err := r.k8sClient.Update(ctx, md); err != nil {
				return fmt.Errorf("updating ModelDeployment %s for restore: %w", md.Name, err)
			}

			r.log.Info("triggered full restore", "model", model, "deployment", md.Name)
			return nil
		}
	}

	return fmt.Errorf("no ModelDeployment found for model %q; create one first", model)
}

// notifyWaiters signals all queued requests that the model is ready.
func (r *Router) notifyWaiters(model string, ep *ModelEndpoint) {
	r.coldQueue.notify(model, ep)
}

// proxyToEndpoint reverse-proxies the HTTP request to the given model endpoint.
// For streaming requests it uses the SSE proxy; for standard requests it uses
// httputil.ReverseProxy.
func (r *Router) proxyToEndpoint(w http.ResponseWriter, req *http.Request, ep *ModelEndpoint) {
	target, err := url.Parse(fmt.Sprintf("http://%s", ep.Address))
	if err != nil {
		r.log.Error(err, "invalid endpoint address", "address", ep.Address)
		http.Error(w, `{"error":{"message":"internal routing error","type":"server_error"}}`, http.StatusInternalServerError)
		return
	}

	// Read the body to check for streaming before proxying.
	var bodyBytes []byte
	if req.Body != nil {
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			http.Error(w, `{"error":{"message":"failed to read request body","type":"server_error"}}`, http.StatusInternalServerError)
			return
		}
	}

	streaming := isStreamingRequest(bodyBytes)

	// Restore the body for the proxy.
	req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	req.ContentLength = int64(len(bodyBytes))

	proxy := &httputil.ReverseProxy{
		Director: func(outReq *http.Request) {
			outReq.URL.Scheme = target.Scheme
			outReq.URL.Host = target.Host
			outReq.Host = target.Host

			// Preserve the original path -- the backend expects the same
			// OpenAI-compatible paths.
			outReq.Header.Set("X-Podstack-Model", ep.Name)
			outReq.Header.Set("X-Forwarded-For", req.RemoteAddr)

			// Remove hop-by-hop headers.
			outReq.Header.Del("Connection")
		},
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			r.log.Error(err, "proxy error", "model", ep.Name, "address", ep.Address)
			http.Error(w, `{"error":{"message":"upstream model error","type":"upstream_error"}}`, http.StatusBadGateway)
		},
		ModifyResponse: func(resp *http.Response) error {
			// For streaming responses, let the SSE proxy handle flushing.
			if streaming && resp.StatusCode == http.StatusOK {
				resp.Header.Set("Content-Type", "text/event-stream")
				resp.Header.Set("Cache-Control", "no-cache")
				resp.Header.Set("Connection", "keep-alive")
				resp.Header.Set("X-Accel-Buffering", "no")
			}
			return nil
		},
		// Disable buffering for streaming responses.
		FlushInterval: -1,
	}

	proxy.ServeHTTP(w, req)

	// Update last used time after successful proxy.
	r.mu.Lock()
	if existing, ok := r.warmPool[ep.Name]; ok {
		existing.LastUsed = time.Now()
	}
	r.mu.Unlock()
}

// isPodReady checks if all containers in a pod are ready.
func isPodReady(pod *corev1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

// inferencePort extracts the inference port from pod annotations or returns the default.
func inferencePort(pod *corev1.Pod) int32 {
	// Check for explicit port annotation.
	if portStr, ok := pod.Annotations["podstack.io/inference-port"]; ok {
		var port int32
		if _, err := fmt.Sscanf(portStr, "%d", &port); err == nil && port > 0 {
			return port
		}
	}

	// Try to find the port from container spec.
	for _, container := range pod.Spec.Containers {
		for _, p := range container.Ports {
			if p.Name == "http" || p.Name == "inference" {
				return p.ContainerPort
			}
		}
	}

	return defaultInferencePort
}

// updateModelLastRequest updates the ModelDeployment's lastRequestAt timestamp.
// This is a fire-and-forget operation used for idle tracking.
func (r *Router) updateModelLastRequest(model string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	mdList := &v1.ModelDeploymentList{}
	if err := r.k8sClient.List(ctx, mdList, client.InNamespace(r.namespace)); err != nil {
		r.log.V(1).Error(err, "failed to list deployments for last-request update")
		return
	}

	now := metav1.Now()
	for i := range mdList.Items {
		md := &mdList.Items[i]
		if md.Spec.ModelName == model {
			md.Status.LastRequestAt = &now
			if err := r.k8sClient.Status().Update(ctx, md); err != nil {
				r.log.V(1).Error(err, "failed to update lastRequestAt", "model", model)
			}
			return
		}
	}
}
