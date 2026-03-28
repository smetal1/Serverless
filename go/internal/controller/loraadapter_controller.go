package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/go-logr/logr"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "github.com/podstack/serverless/api/v1"
	"github.com/podstack/serverless/pkg/nfs"
)

const (
	loraFinalizerName = "podstack.io/lora-adapter-finalizer"

	// loraRequeueDownloading is the requeue interval while downloading adapter weights.
	loraRequeueDownloading = 10 * time.Second

	// loraRequeueCached is the requeue interval while waiting for the base model to be active.
	loraRequeueCached = 15 * time.Second
)

// LoRAAdapterReconciler reconciles LoRAAdapter CRs and manages their lifecycle
// through the phases: Downloading -> Cached -> Loaded -> Failed. It downloads
// adapter weights to NFS and hot-swaps them onto active vLLM model instances.
type LoRAAdapterReconciler struct {
	client.Client
	Scheme     *runtime.Scheme
	ModelCache *nfs.ModelCache
	Log        logr.Logger
}

// +kubebuilder:rbac:groups=podstack.io,resources=loraadapters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=podstack.io,resources=loraadapters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=podstack.io,resources=loraadapters/finalizers,verbs=update
// +kubebuilder:rbac:groups=podstack.io,resources=modeldeployments,verbs=get;list;watch

// Reconcile manages the LoRA adapter lifecycle:
// Downloading -> Cached -> Loaded -> Failed
func (r *LoRAAdapterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("loraadapter", req.NamespacedName)
	logger.V(1).Info("reconciling LoRAAdapter")

	// Fetch the LoRAAdapter CR.
	adapter := &v1.LoRAAdapter{}
	if err := r.Get(ctx, req.NamespacedName, adapter); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("LoRAAdapter not found, must have been deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "unable to fetch LoRAAdapter")
		return ctrl.Result{}, err
	}

	// Handle deletion via finalizer.
	if !adapter.DeletionTimestamp.IsZero() {
		return r.handleDeletion(ctx, logger, adapter)
	}

	// Ensure finalizer is present.
	if !controllerutil.ContainsFinalizer(adapter, loraFinalizerName) {
		controllerutil.AddFinalizer(adapter, loraFinalizerName)
		if err := r.Update(ctx, adapter); err != nil {
			logger.Error(err, "failed to add finalizer to LoRAAdapter")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Phase dispatch.
	switch adapter.Status.Phase {
	case "", v1.LoRAPhaseDownloading:
		return r.handleDownloading(ctx, logger, adapter)
	case v1.LoRAPhaseCached:
		return r.handleCached(ctx, logger, adapter)
	case v1.LoRAPhaseLoaded:
		return r.handleLoaded(ctx, logger, adapter)
	case v1.LoRAPhaseFailed:
		return r.handleFailed(ctx, logger, adapter)
	default:
		logger.Info("unknown LoRA adapter phase, resetting to Downloading", "phase", adapter.Status.Phase)
		adapter.Status.Phase = v1.LoRAPhaseDownloading
		if err := r.Status().Update(ctx, adapter); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}
}

// handleDownloading manages the adapter weight download process. It checks
// if the weights are already cached on NFS and transitions to Cached when
// the download is complete.
func (r *LoRAAdapterReconciler) handleDownloading(ctx context.Context, logger logr.Logger, adapter *v1.LoRAAdapter) (ctrl.Result, error) {
	logger.Info("handling Downloading phase",
		"adapter", adapter.Spec.AdapterName,
		"source", adapter.Spec.Source,
		"sourcePath", adapter.Spec.SourcePath,
	)

	r.setLoRACondition(adapter, "Downloading", metav1.ConditionTrue, "InProgress",
		fmt.Sprintf("Downloading adapter weights from %s:%s", adapter.Spec.Source, adapter.Spec.SourcePath))

	// Ensure the LoRA directory exists on NFS.
	if r.ModelCache != nil {
		loraDir := r.ModelCache.LoRAPath(adapter.Spec.TenantRef, adapter.Spec.AdapterName)
		if err := os.MkdirAll(loraDir, 0o755); err != nil {
			logger.Error(err, "failed to ensure LoRA directory on NFS")
			adapter.Status.Phase = v1.LoRAPhaseFailed
			adapter.Status.Message = fmt.Sprintf("Failed to create NFS directory: %s", err.Error())
			_ = r.Status().Update(ctx, adapter)
			return ctrl.Result{RequeueAfter: requeueMedium}, err
		}
	}

	// Check if the adapter weights already exist on NFS.
	if r.ModelCache != nil && r.ModelCache.LoRAExists(adapter.Spec.TenantRef, adapter.Spec.AdapterName) {
		logger.Info("adapter weights found on NFS, transitioning to Cached")
		adapter.Status.Phase = v1.LoRAPhaseCached
		adapter.Status.CachePath = r.ModelCache.LoRAPath(adapter.Spec.TenantRef, adapter.Spec.AdapterName)
		adapter.Status.Message = "Adapter weights cached on NFS"

		r.setLoRACondition(adapter, "Cached", metav1.ConditionTrue, "Downloaded",
			"Adapter weights available on NFS")

		if err := r.Status().Update(ctx, adapter); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// In a real implementation, this would trigger a download job (e.g., a
	// Kubernetes Job that pulls from HuggingFace, S3, etc.). For now, we
	// set the cache path and requeue to check for completion.
	if r.ModelCache != nil {
		adapter.Status.CachePath = r.ModelCache.LoRAPath(adapter.Spec.TenantRef, adapter.Spec.AdapterName)
	}
	adapter.Status.Message = fmt.Sprintf("Downloading weights from %s", adapter.Spec.SourcePath)

	if err := r.Status().Update(ctx, adapter); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: loraRequeueDownloading}, nil
}

// handleCached manages an adapter whose weights are cached on NFS. If the
// base model is Active and AutoLoad is enabled, it triggers loading the
// adapter into the running vLLM instance.
func (r *LoRAAdapterReconciler) handleCached(ctx context.Context, logger logr.Logger, adapter *v1.LoRAAdapter) (ctrl.Result, error) {
	logger.Info("handling Cached phase", "adapter", adapter.Spec.AdapterName, "baseModel", adapter.Spec.BaseModelRef)

	// If AutoLoad is not enabled, stay in Cached state.
	if !adapter.Spec.AutoLoad {
		logger.V(1).Info("autoLoad not enabled, remaining in Cached state")
		return ctrl.Result{}, nil
	}

	// Look up the base model deployment to check if it is Active.
	baseMD := &v1.ModelDeployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      adapter.Spec.BaseModelRef,
		Namespace: adapter.Namespace,
	}, baseMD)

	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("base model deployment not found, waiting", "baseModelRef", adapter.Spec.BaseModelRef)
			return ctrl.Result{RequeueAfter: loraRequeueCached}, nil
		}
		logger.Error(err, "failed to look up base model deployment")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	// Wait for the base model to be Active.
	if baseMD.Status.Phase != v1.PhaseActive {
		logger.Info("base model not Active yet, waiting",
			"baseModelPhase", baseMD.Status.Phase,
			"baseModel", adapter.Spec.BaseModelRef,
		)
		return ctrl.Result{RequeueAfter: loraRequeueCached}, nil
	}

	// Base model is Active. Attempt to load the adapter via the vLLM API.
	if baseMD.Status.Endpoint == "" {
		logger.Info("base model endpoint not available yet, waiting")
		return ctrl.Result{RequeueAfter: loraRequeueCached}, nil
	}

	logger.Info("base model is Active, loading LoRA adapter",
		"endpoint", baseMD.Status.Endpoint,
		"adapterName", adapter.Spec.AdapterName,
		"cachePath", adapter.Status.CachePath,
	)

	if err := r.loadLoRAAdapter(ctx, logger, baseMD.Status.Endpoint, adapter); err != nil {
		logger.Error(err, "failed to load LoRA adapter via vLLM API")
		adapter.Status.Message = fmt.Sprintf("Failed to load adapter: %s", err.Error())
		r.setLoRACondition(adapter, "LoadFailed", metav1.ConditionTrue, "APIError", err.Error())
		_ = r.Status().Update(ctx, adapter)
		return ctrl.Result{RequeueAfter: requeueMedium}, nil
	}

	// Successfully loaded.
	adapter.Status.Phase = v1.LoRAPhaseLoaded
	adapter.Status.Message = "Adapter loaded on base model"

	r.setLoRACondition(adapter, "Loaded", metav1.ConditionTrue, "Active",
		fmt.Sprintf("Adapter loaded on %s", adapter.Spec.BaseModelRef))

	if err := r.Status().Update(ctx, adapter); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// handleLoaded monitors a loaded LoRA adapter. It periodically verifies that
// the base model is still Active and the adapter is still loaded.
func (r *LoRAAdapterReconciler) handleLoaded(ctx context.Context, logger logr.Logger, adapter *v1.LoRAAdapter) (ctrl.Result, error) {
	logger.V(1).Info("handling Loaded phase", "adapter", adapter.Spec.AdapterName)

	// Verify the base model is still Active.
	baseMD := &v1.ModelDeployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      adapter.Spec.BaseModelRef,
		Namespace: adapter.Namespace,
	}, baseMD)

	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("base model deleted, transitioning back to Cached")
			adapter.Status.Phase = v1.LoRAPhaseCached
			adapter.Status.LoadedOnPods = nil
			adapter.Status.Message = "Base model no longer exists"
			_ = r.Status().Update(ctx, adapter)
			return ctrl.Result{Requeue: true}, nil
		}
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	// If the base model is no longer Active, transition back to Cached.
	if baseMD.Status.Phase != v1.PhaseActive {
		logger.Info("base model no longer Active, transitioning back to Cached",
			"baseModelPhase", baseMD.Status.Phase)
		adapter.Status.Phase = v1.LoRAPhaseCached
		adapter.Status.LoadedOnPods = nil
		adapter.Status.Message = "Base model is no longer active"

		r.setLoRACondition(adapter, "Loaded", metav1.ConditionFalse, "BaseModelInactive",
			"Base model transitioned away from Active phase")

		if err := r.Status().Update(ctx, adapter); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Adapter is loaded and base model is Active. Steady state.
	return ctrl.Result{RequeueAfter: requeueIdle}, nil
}

// handleFailed processes an adapter in the Failed state.
func (r *LoRAAdapterReconciler) handleFailed(ctx context.Context, logger logr.Logger, adapter *v1.LoRAAdapter) (ctrl.Result, error) {
	logger.Info("handling Failed phase", "adapter", adapter.Spec.AdapterName, "message", adapter.Status.Message)

	r.setLoRACondition(adapter, "Failed", metav1.ConditionTrue, "Error", adapter.Status.Message)

	if err := r.Status().Update(ctx, adapter); err != nil {
		return ctrl.Result{}, err
	}

	// Do not auto-retry on failure. Manual intervention or re-creation is needed.
	return ctrl.Result{}, nil
}

// handleDeletion processes the finalizer for LoRAAdapter deletion.
func (r *LoRAAdapterReconciler) handleDeletion(ctx context.Context, logger logr.Logger, adapter *v1.LoRAAdapter) (ctrl.Result, error) {
	logger.Info("handling LoRAAdapter deletion", "adapter", adapter.Spec.AdapterName)

	if controllerutil.ContainsFinalizer(adapter, loraFinalizerName) {
		// If the adapter is currently loaded, attempt to unload it from the
		// base model. Best-effort -- if the base model is gone, skip.
		if adapter.Status.Phase == v1.LoRAPhaseLoaded {
			baseMD := &v1.ModelDeployment{}
			err := r.Get(ctx, types.NamespacedName{
				Name:      adapter.Spec.BaseModelRef,
				Namespace: adapter.Namespace,
			}, baseMD)

			if err == nil && baseMD.Status.Phase == v1.PhaseActive && baseMD.Status.Endpoint != "" {
				if unloadErr := r.unloadLoRAAdapter(ctx, logger, baseMD.Status.Endpoint, adapter); unloadErr != nil {
					logger.Error(unloadErr, "failed to unload LoRA adapter during deletion (best-effort)")
				} else {
					logger.Info("unloaded LoRA adapter from base model")
				}
			}
		}

		// Remove the finalizer.
		controllerutil.RemoveFinalizer(adapter, loraFinalizerName)
		if err := r.Update(ctx, adapter); err != nil {
			logger.Error(err, "failed to remove finalizer from LoRAAdapter")
			return ctrl.Result{}, err
		}
		logger.Info("finalizer removed from LoRAAdapter")
	}

	return ctrl.Result{}, nil
}

// loadLoRAAdapter calls the vLLM hot-swap API to load a LoRA adapter onto the
// running inference server.
func (r *LoRAAdapterReconciler) loadLoRAAdapter(ctx context.Context, logger logr.Logger, endpoint string, adapter *v1.LoRAAdapter) error {
	payload := map[string]string{
		"lora_name": adapter.Spec.AdapterName,
		"lora_path": adapter.Status.CachePath,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal LoRA load request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/load_lora_adapter", endpoint)
	logger.Info("calling vLLM LoRA load API", "url", url, "adapterName", adapter.Spec.AdapterName)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("HTTP request to load LoRA adapter failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("vLLM LoRA load API returned status %d", resp.StatusCode)
	}

	logger.Info("LoRA adapter loaded successfully", "adapterName", adapter.Spec.AdapterName)
	return nil
}

// unloadLoRAAdapter calls the vLLM API to unload a LoRA adapter from the
// running inference server.
func (r *LoRAAdapterReconciler) unloadLoRAAdapter(ctx context.Context, logger logr.Logger, endpoint string, adapter *v1.LoRAAdapter) error {
	payload := map[string]string{
		"lora_name": adapter.Spec.AdapterName,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal LoRA unload request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/unload_lora_adapter", endpoint)
	logger.Info("calling vLLM LoRA unload API", "url", url, "adapterName", adapter.Spec.AdapterName)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("HTTP request to unload LoRA adapter failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("vLLM LoRA unload API returned status %d", resp.StatusCode)
	}

	logger.Info("LoRA adapter unloaded successfully", "adapterName", adapter.Spec.AdapterName)
	return nil
}

// setLoRACondition sets or updates a condition on the LoRAAdapter status.
func (r *LoRAAdapterReconciler) setLoRACondition(adapter *v1.LoRAAdapter, condType string, status metav1.ConditionStatus, reason, message string) {
	now := metav1.NewTime(time.Now())
	condition := metav1.Condition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	for i, existing := range adapter.Status.Conditions {
		if existing.Type == condType {
			if existing.Status != status {
				adapter.Status.Conditions[i] = condition
			} else {
				adapter.Status.Conditions[i].Reason = reason
				adapter.Status.Conditions[i].Message = message
			}
			return
		}
	}
	adapter.Status.Conditions = append(adapter.Status.Conditions, condition)
}

// SetupWithManager sets up the controller with the Manager.
func (r *LoRAAdapterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1.LoRAAdapter{}).
		Complete(r)
}
