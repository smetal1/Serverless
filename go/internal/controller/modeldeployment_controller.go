package controller

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "github.com/podstack/serverless/api/v1"
	"github.com/podstack/serverless/internal/snapshot"
	"github.com/podstack/serverless/pkg/nfs"
	pkgruntime "github.com/podstack/serverless/pkg/runtime"
)

const (
	finalizerName = "podstack.io/model-deployment-finalizer"

	// Label used to associate child resources with a ModelDeployment.
	labelModelDeployment = "podstack.io/model-deployment"

	// Annotation that triggers a standby pod to boot with GPU.
	annotationBoot = "podstack.io/boot"

	// Annotation containing VRAM usage reported by the inference pod.
	annotationVRAMUsed = "podstack.io/vram-used-mb"

	// Default requeue intervals for various phases.
	requeueShort  = 5 * time.Second
	requeueMedium = 15 * time.Second
	requeueLong   = 30 * time.Second
	requeueIdle   = 60 * time.Second
)

// ModelDeploymentReconciler reconciles a ModelDeployment object using a state
// machine pattern. Each lifecycle phase is handled by a dedicated method that
// manages Kubernetes resources, interacts with the SnapshotManager and
// ModelCache, and transitions to the next phase when conditions are met.
type ModelDeploymentReconciler struct {
	client.Client
	Scheme          *runtime.Scheme
	SnapshotManager *snapshot.Manager
	ModelCache      *nfs.ModelCache
	Log             logr.Logger
}

// +kubebuilder:rbac:groups=podstack.io,resources=modeldeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=podstack.io,resources=modeldeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=podstack.io,resources=modeldeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=podstack.io,resources=snapshots,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete

// Reconcile implements the state machine for ModelDeployment lifecycle management.
func (r *ModelDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("modeldeployment", req.NamespacedName)
	logger.V(1).Info("reconciling ModelDeployment")

	// Fetch the ModelDeployment CR.
	md := &v1.ModelDeployment{}
	if err := r.Get(ctx, req.NamespacedName, md); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("ModelDeployment not found, must have been deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "unable to fetch ModelDeployment")
		return ctrl.Result{}, err
	}

	// Handle deletion via finalizer.
	if !md.DeletionTimestamp.IsZero() {
		return r.handleDeletion(ctx, logger, md)
	}

	// Ensure finalizer is present.
	if !controllerutil.ContainsFinalizer(md, finalizerName) {
		controllerutil.AddFinalizer(md, finalizerName)
		if err := r.Update(ctx, md); err != nil {
			logger.Error(err, "failed to add finalizer")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// State machine dispatch based on current phase.
	switch md.Status.Phase {
	case "", v1.PhasePending:
		return r.handlePending(ctx, logger, md)
	case v1.PhaseSnapshotting:
		return r.handleSnapshotting(ctx, logger, md)
	case v1.PhaseStandby:
		return r.handleStandby(ctx, logger, md)
	case v1.PhaseBooting:
		return r.handleBooting(ctx, logger, md)
	case v1.PhaseActive:
		return r.handleActive(ctx, logger, md)
	case v1.PhaseIdle:
		return r.handleIdle(ctx, logger, md)
	case v1.PhaseEvicted:
		return r.handleEvicted(ctx, logger, md)
	default:
		logger.Info("unknown phase, resetting to Pending", "phase", md.Status.Phase)
		return r.updatePhase(ctx, md, v1.PhasePending)
	}
}

// ---------------------------------------------------------------------------
// Phase handlers
// ---------------------------------------------------------------------------

// handlePending validates the model, checks the NFS cache, and determines
// whether to proceed with snapshotting or direct activation.
func (r *ModelDeploymentReconciler) handlePending(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) (ctrl.Result, error) {
	logger.Info("handling Pending phase", "model", md.Spec.ModelName, "runtime", md.Spec.Runtime)

	// Set initial condition.
	r.setCondition(md, "Reconciling", metav1.ConditionTrue, "Pending", "Validating model and checking cache")

	// Validate model existence in the NFS cache.
	if r.ModelCache != nil {
		modelPath := r.ModelCache.ModelPath(md.Spec.ModelName)
		if !r.ModelCache.ModelExists(md.Spec.ModelName) {
			logger.Info("model not found in NFS cache, will be downloaded at runtime",
				"model", md.Spec.ModelName, "expectedPath", modelPath)
		} else {
			logger.V(1).Info("model found in NFS cache", "model", md.Spec.ModelName, "path", modelPath)
		}
	}

	// Check if a Snapshot CR already exists for this deployment.
	if md.Status.SnapshotRef != "" {
		snap := &v1.Snapshot{}
		err := r.Get(ctx, types.NamespacedName{Name: md.Status.SnapshotRef, Namespace: md.Namespace}, snap)
		if err == nil && snap.Status.Phase == v1.SnapshotPhaseReady {
			logger.Info("existing snapshot found, transitioning to Standby", "snapshot", snap.Name)
			return r.updatePhase(ctx, md, v1.PhaseStandby)
		}
	}

	// Use the SnapshotManager to check for an existing snapshot in the
	// blobstore (verified against both the Snapshot CR and disk).
	if r.SnapshotManager != nil {
		existingSnap, exists, err := r.SnapshotManager.SnapshotExists(ctx, md.Spec.ModelName, md.Spec.GPU.Type)
		if err != nil {
			logger.Error(err, "failed to check for existing snapshot via SnapshotManager")
		} else if exists && existingSnap != nil {
			logger.Info("found verified snapshot via SnapshotManager, transitioning to Standby",
				"snapshot", existingSnap.Name)
			md.Status.SnapshotRef = existingSnap.Name
			if err := r.Status().Update(ctx, md); err != nil {
				return ctrl.Result{}, err
			}
			return r.updatePhase(ctx, md, v1.PhaseStandby)
		}
	}

	// Fall back to listing Snapshot CRs by label (covers cases where
	// SnapshotManager is not configured or the labels differ).
	snapList := &v1.SnapshotList{}
	if err := r.List(ctx, snapList, client.InNamespace(md.Namespace), client.MatchingLabels{
		labelModelDeployment: md.Name,
	}); err != nil {
		logger.Error(err, "failed to list snapshots")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	for i := range snapList.Items {
		snap := &snapList.Items[i]
		if snap.Status.Phase == v1.SnapshotPhaseReady {
			logger.Info("found ready snapshot via label search, transitioning to Standby", "snapshot", snap.Name)
			md.Status.SnapshotRef = snap.Name
			if err := r.Status().Update(ctx, md); err != nil {
				return ctrl.Result{}, err
			}
			return r.updatePhase(ctx, md, v1.PhaseStandby)
		}
	}

	// No snapshot available. Decide path based on snapshot configuration.
	if md.Spec.Snapshot.Enabled {
		logger.Info("snapshot enabled but none found, transitioning to Snapshotting")
		return r.updatePhase(ctx, md, v1.PhaseSnapshotting)
	}

	// Snapshot not enabled: create full model pod directly and go Active.
	logger.Info("snapshot not enabled, creating model pod and transitioning to Active")
	if err := r.createModelPod(ctx, logger, md); err != nil {
		logger.Error(err, "failed to create model pod")
		r.setCondition(md, "PodCreationFailed", metav1.ConditionTrue, "Error", err.Error())
		_ = r.Status().Update(ctx, md)
		return ctrl.Result{RequeueAfter: requeueMedium}, err
	}

	if err := r.createService(ctx, logger, md); err != nil {
		logger.Error(err, "failed to create service")
		return ctrl.Result{RequeueAfter: requeueMedium}, err
	}

	return r.updatePhase(ctx, md, v1.PhaseActive)
}

// handleSnapshotting manages the snapshot creation process. It creates a full
// model pod with GPU, waits for it to become ready, and then triggers snapshot
// creation via the SnapshotManager.
func (r *ModelDeploymentReconciler) handleSnapshotting(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) (ctrl.Result, error) {
	logger.Info("handling Snapshotting phase")

	r.setCondition(md, "Reconciling", metav1.ConditionTrue, "Snapshotting", "Creating CUDA snapshot")

	// Check if a snapshot already exists (another reconciliation may have created it).
	if r.SnapshotManager != nil {
		existingSnap, exists, err := r.SnapshotManager.SnapshotExists(ctx, md.Spec.ModelName, md.Spec.GPU.Type)
		if err != nil {
			logger.Error(err, "failed to check for existing snapshot")
		} else if exists && existingSnap != nil && existingSnap.Status.Phase == v1.SnapshotPhaseReady {
			logger.Info("snapshot already exists and is ready, transitioning to Standby", "snapshot", existingSnap.Name)
			md.Status.SnapshotRef = existingSnap.Name
			return r.updatePhase(ctx, md, v1.PhaseStandby)
		}
	}

	// Ensure the model pod exists for snapshotting.
	pod, err := r.getPodForDeployment(ctx, md, "inference")
	if err != nil {
		logger.Error(err, "failed to look up model pod")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	if pod == nil {
		logger.Info("creating model pod for snapshot capture")
		if err := r.createModelPod(ctx, logger, md); err != nil {
			logger.Error(err, "failed to create model pod for snapshotting")
			return ctrl.Result{RequeueAfter: requeueMedium}, err
		}
		return ctrl.Result{RequeueAfter: requeueMedium}, nil
	}

	// Wait for the pod to be ready.
	if !isPodReady(pod) {
		logger.Info("model pod not ready yet, waiting", "podPhase", pod.Status.Phase)
		return ctrl.Result{RequeueAfter: requeueMedium}, nil
	}

	logger.Info("model pod is ready, triggering snapshot creation via SnapshotManager")

	// Trigger snapshot creation via the SnapshotManager. The Manager handles
	// CUDA context locking, CRIU/cuda-checkpoint, blobstore archival, and
	// Snapshot CR creation internally.
	if r.SnapshotManager != nil {
		snap, err := r.SnapshotManager.CreateSnapshot(ctx, pod, md)
		if err != nil {
			logger.Error(err, "snapshot creation failed")
			r.setCondition(md, "SnapshotFailed", metav1.ConditionTrue, "Failed", err.Error())
			_ = r.Status().Update(ctx, md)
			// Retry by resetting to Pending after a delay.
			return ctrl.Result{RequeueAfter: requeueLong}, nil
		}

		if snap != nil && snap.Status.Phase == v1.SnapshotPhaseReady {
			logger.Info("snapshot created successfully, transitioning to Standby", "snapshot", snap.Name)
			md.Status.SnapshotRef = snap.Name

			// Delete the full model pod since we now have a snapshot.
			if err := r.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
				logger.Error(err, "failed to delete model pod after snapshot")
				return ctrl.Result{RequeueAfter: requeueShort}, err
			}
			logger.Info("deleted model pod after snapshot capture")

			return r.updatePhase(ctx, md, v1.PhaseStandby)
		}

		// Snapshot CR created but not yet Ready (should not happen normally
		// since CreateSnapshot is synchronous, but handle gracefully).
		if snap != nil {
			md.Status.SnapshotRef = snap.Name
		}
	}

	// Snapshot still in progress or SnapshotManager not configured.
	return ctrl.Result{RequeueAfter: requeueMedium}, r.Status().Update(ctx, md)
}

// handleStandby ensures a lightweight standby pod is running and watches for
// the boot annotation to trigger GPU activation.
func (r *ModelDeploymentReconciler) handleStandby(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) (ctrl.Result, error) {
	logger.Info("handling Standby phase")

	r.setCondition(md, "Reconciling", metav1.ConditionTrue, "Standby", "Standby pod running, waiting for boot signal")

	// Ensure the standby pod exists.
	standbyPod, err := r.getPodForDeployment(ctx, md, "standby")
	if err != nil {
		logger.Error(err, "failed to look up standby pod")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	if standbyPod == nil {
		logger.Info("creating standby pod")
		if err := r.createStandbyPod(ctx, logger, md); err != nil {
			logger.Error(err, "failed to create standby pod")
			return ctrl.Result{RequeueAfter: requeueMedium}, err
		}
		md.Status.StandbyReplicas = 1
		md.Status.ReadyReplicas = 0
		if err := r.Status().Update(ctx, md); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: requeueShort}, nil
	}

	// Check for boot annotation on the ModelDeployment itself.
	if md.Annotations != nil {
		if bootVal, ok := md.Annotations[annotationBoot]; ok && bootVal == "true" {
			logger.Info("boot annotation detected, transitioning to Booting")

			// Remove the boot annotation to prevent re-triggering.
			delete(md.Annotations, annotationBoot)
			if err := r.Update(ctx, md); err != nil {
				logger.Error(err, "failed to remove boot annotation")
				return ctrl.Result{}, err
			}

			return r.updatePhase(ctx, md, v1.PhaseBooting)
		}
	}

	// Update standby replica count.
	md.Status.StandbyReplicas = 1
	md.Status.ReadyReplicas = 0
	if err := r.Status().Update(ctx, md); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: requeueLong}, nil
}

// handleBooting restores GPU state from the snapshot and transitions to Active.
func (r *ModelDeploymentReconciler) handleBooting(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) (ctrl.Result, error) {
	logger.Info("handling Booting phase", "snapshotRef", md.Status.SnapshotRef)
	bootStart := time.Now()

	r.setCondition(md, "Reconciling", metav1.ConditionTrue, "Booting", "Restoring GPU state from snapshot")

	// Delete the standby pod if it exists.
	standbyPod, err := r.getPodForDeployment(ctx, md, "standby")
	if err != nil {
		logger.Error(err, "failed to look up standby pod")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}
	if standbyPod != nil {
		if err := r.Delete(ctx, standbyPod); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete standby pod")
			return ctrl.Result{RequeueAfter: requeueShort}, err
		}
		logger.Info("deleted standby pod for boot")
	}

	// Create the full inference pod with GPU resources.
	inferencePod, err := r.getPodForDeployment(ctx, md, "inference")
	if err != nil {
		logger.Error(err, "failed to look up inference pod")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	if inferencePod == nil {
		logger.Info("creating inference pod for boot")
		if err := r.createModelPod(ctx, logger, md); err != nil {
			logger.Error(err, "failed to create inference pod during boot")
			r.setCondition(md, "BootFailed", metav1.ConditionTrue, "Error", err.Error())
			_ = r.Status().Update(ctx, md)
			return ctrl.Result{RequeueAfter: requeueMedium}, err
		}
		return ctrl.Result{RequeueAfter: requeueShort}, nil
	}

	// Wait for the pod to be ready.
	if !isPodReady(inferencePod) {
		logger.Info("inference pod not ready yet during boot", "podPhase", inferencePod.Status.Phase)
		return ctrl.Result{RequeueAfter: requeueShort}, nil
	}

	// Ensure service exists.
	if err := r.createService(ctx, logger, md); err != nil {
		logger.Error(err, "failed to create service during boot")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	// Calculate cold start time.
	coldStartMs := time.Since(bootStart).Milliseconds()
	md.Status.ColdStartMs = coldStartMs
	md.Status.Endpoint = fmt.Sprintf("http://%s.%s.svc.cluster.local", md.Name, md.Namespace)
	md.Status.ReadyReplicas = 1
	md.Status.StandbyReplicas = 0

	now := metav1.NewTime(time.Now())
	md.Status.LastRequestAt = &now

	logger.Info("boot complete", "coldStartMs", coldStartMs, "endpoint", md.Status.Endpoint)

	return r.updatePhase(ctx, md, v1.PhaseActive)
}

// handleActive ensures the service is pointing to the active model pod,
// monitors VRAM usage, and checks idle timeout for scale-down.
func (r *ModelDeploymentReconciler) handleActive(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) (ctrl.Result, error) {
	logger.V(1).Info("handling Active phase")

	r.setCondition(md, "Ready", metav1.ConditionTrue, "Active", "Model is actively serving inference requests")

	// Ensure the inference pod exists.
	pod, err := r.getPodForDeployment(ctx, md, "inference")
	if err != nil {
		logger.Error(err, "failed to look up inference pod")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	if pod == nil {
		logger.Info("inference pod missing while Active, transitioning back to Pending")
		md.Status.ReadyReplicas = 0
		_ = r.Status().Update(ctx, md)
		return r.updatePhase(ctx, md, v1.PhasePending)
	}

	if !isPodReady(pod) {
		logger.Info("inference pod not ready, waiting", "podPhase", pod.Status.Phase)
		md.Status.ReadyReplicas = 0
		_ = r.Status().Update(ctx, md)
		return ctrl.Result{RequeueAfter: requeueShort}, nil
	}

	// Ensure the service exists.
	if err := r.createService(ctx, logger, md); err != nil {
		logger.Error(err, "failed to ensure service in Active phase")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	// Update VRAM usage from pod annotation.
	if pod.Annotations != nil {
		if vramStr, ok := pod.Annotations[annotationVRAMUsed]; ok {
			vram, parseErr := strconv.ParseInt(vramStr, 10, 32)
			if parseErr == nil {
				md.Status.VRAMUsedMB = int32(vram)
			}
		}
	}

	// Update endpoint.
	md.Status.Endpoint = fmt.Sprintf("http://%s.%s.svc.cluster.local", md.Name, md.Namespace)
	md.Status.ReadyReplicas = 1
	md.Status.StandbyReplicas = 0

	// Check idle timeout.
	if md.Spec.Scaling.IdleTimeoutSeconds > 0 && md.Status.LastRequestAt != nil {
		idleTimeout := time.Duration(md.Spec.Scaling.IdleTimeoutSeconds) * time.Second
		elapsed := time.Since(md.Status.LastRequestAt.Time)

		if elapsed > idleTimeout {
			logger.Info("model is idle, transitioning to Idle",
				"elapsed", elapsed.String(),
				"timeout", idleTimeout.String(),
			)
			if err := r.Status().Update(ctx, md); err != nil {
				return ctrl.Result{}, err
			}
			return r.updatePhase(ctx, md, v1.PhaseIdle)
		}

		// Requeue to check idle timeout again.
		remainingTimeout := idleTimeout - elapsed
		if err := r.Status().Update(ctx, md); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: remainingTimeout}, nil
	}

	if err := r.Status().Update(ctx, md); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: requeueIdle}, nil
}

// handleIdle releases GPU resources and transitions back to Standby.
func (r *ModelDeploymentReconciler) handleIdle(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) (ctrl.Result, error) {
	logger.Info("handling Idle phase, releasing GPU resources")

	r.setCondition(md, "Reconciling", metav1.ConditionTrue, "Idle", "Releasing GPU resources due to inactivity")

	// Delete the inference pod to release GPU.
	pod, err := r.getPodForDeployment(ctx, md, "inference")
	if err != nil {
		logger.Error(err, "failed to look up inference pod for idle cleanup")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}

	if pod != nil {
		if err := r.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete inference pod during idle")
			return ctrl.Result{RequeueAfter: requeueShort}, err
		}
		logger.Info("deleted inference pod to release GPU")
	}

	// Delete the service since no active pod is serving.
	svc, err := r.getServiceForDeployment(ctx, md)
	if err != nil {
		logger.Error(err, "failed to look up service for idle cleanup")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}
	if svc != nil {
		if err := r.Delete(ctx, svc); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete service during idle")
			return ctrl.Result{RequeueAfter: requeueShort}, err
		}
		logger.Info("deleted service during idle")
	}

	// Clear active status.
	md.Status.ReadyReplicas = 0
	md.Status.VRAMUsedMB = 0
	md.Status.Endpoint = ""

	if err := r.Status().Update(ctx, md); err != nil {
		return ctrl.Result{}, err
	}

	// Transition to Standby if snapshot exists for fast re-boot,
	// or to Pending to re-create from scratch.
	if md.Status.SnapshotRef != "" || md.Spec.Snapshot.Enabled {
		logger.Info("snapshot available, transitioning to Standby")
		return r.updatePhase(ctx, md, v1.PhaseStandby)
	}

	logger.Info("no snapshot available, transitioning to Pending")
	return r.updatePhase(ctx, md, v1.PhasePending)
}

// handleEvicted cleans up all pods and services while preserving the snapshot
// on NFS for future restore.
func (r *ModelDeploymentReconciler) handleEvicted(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) (ctrl.Result, error) {
	logger.Info("handling Evicted phase, cleaning up resources")

	r.setCondition(md, "Evicted", metav1.ConditionTrue, "Evicted", "Model evicted, snapshot persisted on NFS")

	// Delete inference pod if it exists.
	inferencePod, err := r.getPodForDeployment(ctx, md, "inference")
	if err != nil {
		logger.Error(err, "failed to look up inference pod for eviction")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}
	if inferencePod != nil {
		if err := r.Delete(ctx, inferencePod); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete inference pod during eviction")
			return ctrl.Result{RequeueAfter: requeueShort}, err
		}
	}

	// Delete standby pod if it exists.
	standbyPod, err := r.getPodForDeployment(ctx, md, "standby")
	if err != nil {
		logger.Error(err, "failed to look up standby pod for eviction")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}
	if standbyPod != nil {
		if err := r.Delete(ctx, standbyPod); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete standby pod during eviction")
			return ctrl.Result{RequeueAfter: requeueShort}, err
		}
	}

	// Delete service if it exists.
	svc, err := r.getServiceForDeployment(ctx, md)
	if err != nil {
		logger.Error(err, "failed to look up service for eviction")
		return ctrl.Result{RequeueAfter: requeueShort}, err
	}
	if svc != nil {
		if err := r.Delete(ctx, svc); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete service during eviction")
			return ctrl.Result{RequeueAfter: requeueShort}, err
		}
	}

	// Clear status fields.
	md.Status.ReadyReplicas = 0
	md.Status.StandbyReplicas = 0
	md.Status.VRAMUsedMB = 0
	md.Status.Endpoint = ""

	if err := r.Status().Update(ctx, md); err != nil {
		return ctrl.Result{}, err
	}

	// Snapshot remains on NFS; no further action needed until un-evicted.
	logger.Info("eviction complete, snapshot persisted", "snapshotRef", md.Status.SnapshotRef)
	return ctrl.Result{}, nil
}

// ---------------------------------------------------------------------------
// Deletion handler
// ---------------------------------------------------------------------------

// handleDeletion processes the finalizer logic when a ModelDeployment is being
// deleted. It cleans up all child resources and removes the finalizer.
func (r *ModelDeploymentReconciler) handleDeletion(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) (ctrl.Result, error) {
	logger.Info("handling deletion")

	if controllerutil.ContainsFinalizer(md, finalizerName) {
		// Clean up inference pod.
		inferencePod, err := r.getPodForDeployment(ctx, md, "inference")
		if err == nil && inferencePod != nil {
			_ = r.Delete(ctx, inferencePod)
		}

		// Clean up standby pod.
		standbyPod, err := r.getPodForDeployment(ctx, md, "standby")
		if err == nil && standbyPod != nil {
			_ = r.Delete(ctx, standbyPod)
		}

		// Clean up service.
		svc, err := r.getServiceForDeployment(ctx, md)
		if err == nil && svc != nil {
			_ = r.Delete(ctx, svc)
		}

		// Optionally clean up snapshot from blobstore.
		if md.Status.SnapshotRef != "" && r.SnapshotManager != nil {
			if err := r.SnapshotManager.DeleteSnapshotByModel(md.Spec.ModelName, md.Spec.GPU.Type); err != nil {
				logger.Error(err, "failed to delete snapshot from blobstore",
					"model", md.Spec.ModelName, "gpuType", md.Spec.GPU.Type)
			}
		}

		// Remove finalizer.
		controllerutil.RemoveFinalizer(md, finalizerName)
		if err := r.Update(ctx, md); err != nil {
			logger.Error(err, "failed to remove finalizer")
			return ctrl.Result{}, err
		}
		logger.Info("finalizer removed, deletion complete")
	}

	return ctrl.Result{}, nil
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

// createModelPod creates the full inference pod with GPU resources based on
// the configured runtime (vLLM, Triton, or generic).
func (r *ModelDeploymentReconciler) createModelPod(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) error {
	// Check if the pod already exists.
	existing, err := r.getPodForDeployment(ctx, md, "inference")
	if err != nil {
		return err
	}
	if existing != nil {
		logger.V(1).Info("inference pod already exists", "pod", existing.Name)
		return nil
	}

	// Generate the pod template based on runtime type.
	var podTemplate *corev1.PodTemplateSpec
	switch md.Spec.Runtime {
	case v1.RuntimeVLLM:
		podTemplate = pkgruntime.VLLMPodTemplate(md)
	case v1.RuntimeTriton:
		podTemplate = pkgruntime.TritonPodTemplate(md)
	case v1.RuntimeGeneric:
		podTemplate = pkgruntime.GenericPodTemplate(md)
	default:
		return fmt.Errorf("unsupported runtime: %s", md.Spec.Runtime)
	}

	// Create the Pod from the template.
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("%s-%s", md.Name, md.Spec.Runtime),
			Namespace:   md.Namespace,
			Labels:      podTemplate.Labels,
			Annotations: podTemplate.Annotations,
		},
		Spec: podTemplate.Spec,
	}

	// Ensure the model-deployment label is set.
	if pod.Labels == nil {
		pod.Labels = make(map[string]string)
	}
	pod.Labels[labelModelDeployment] = md.Name
	pod.Labels["podstack.io/role"] = "inference"

	// Set the controller reference so the pod is owned by this ModelDeployment.
	if err := controllerutil.SetControllerReference(md, pod, r.Scheme); err != nil {
		return fmt.Errorf("failed to set controller reference on pod: %w", err)
	}

	if err := r.Create(ctx, pod); err != nil {
		if apierrors.IsAlreadyExists(err) {
			logger.V(1).Info("inference pod already exists (race)", "pod", pod.Name)
			return nil
		}
		return fmt.Errorf("failed to create inference pod: %w", err)
	}

	logger.Info("created inference pod", "pod", pod.Name, "runtime", md.Spec.Runtime)
	return nil
}

// createStandbyPod creates a lightweight standby pod with no GPU resources.
// It runs a minimal process to keep the container alive for fast boot.
func (r *ModelDeploymentReconciler) createStandbyPod(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) error {
	// Check if a standby pod already exists.
	existing, err := r.getPodForDeployment(ctx, md, "standby")
	if err != nil {
		return err
	}
	if existing != nil {
		logger.V(1).Info("standby pod already exists", "pod", existing.Name)
		return nil
	}

	labels := map[string]string{
		"app.kubernetes.io/name":       "podstack-inference",
		"app.kubernetes.io/instance":   md.Name,
		"app.kubernetes.io/managed-by": "podstack-controller",
		labelModelDeployment:           md.Name,
		"podstack.io/role":             "standby",
		"podstack.io/runtime":          md.Spec.Runtime,
	}

	annotations := map[string]string{
		"podstack.io/standby":     "true",
		"podstack.io/snapshotRef": md.Status.SnapshotRef,
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("%s-standby", md.Name),
			Namespace:   md.Namespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "standby",
					Image: "busybox:1.36",
					Command: []string{
						"sh", "-c",
						"echo 'standby pod ready' && sleep infinity",
					},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("50m"),
							corev1.ResourceMemory: resource.MustParse("64Mi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("200Mi"),
						},
					},
				},
			},
			RestartPolicy: corev1.RestartPolicyAlways,
		},
	}

	if err := controllerutil.SetControllerReference(md, pod, r.Scheme); err != nil {
		return fmt.Errorf("failed to set controller reference on standby pod: %w", err)
	}

	if err := r.Create(ctx, pod); err != nil {
		if apierrors.IsAlreadyExists(err) {
			logger.V(1).Info("standby pod already exists (race)", "pod", pod.Name)
			return nil
		}
		return fmt.Errorf("failed to create standby pod: %w", err)
	}

	logger.Info("created standby pod", "pod", pod.Name)
	return nil
}

// createService creates a ClusterIP service for the ModelDeployment if one
// does not already exist.
func (r *ModelDeploymentReconciler) createService(ctx context.Context, logger logr.Logger, md *v1.ModelDeployment) error {
	existing, err := r.getServiceForDeployment(ctx, md)
	if err != nil {
		return err
	}
	if existing != nil {
		logger.V(1).Info("service already exists", "service", existing.Name)
		return nil
	}

	// Generate the service based on runtime type.
	var svc *corev1.Service
	switch md.Spec.Runtime {
	case v1.RuntimeVLLM:
		svc = pkgruntime.VLLMServiceForModel(md)
	case v1.RuntimeTriton:
		svc = pkgruntime.TritonServiceForModel(md)
	case v1.RuntimeGeneric:
		svc = pkgruntime.GenericServiceForModel(md)
	default:
		return fmt.Errorf("unsupported runtime for service creation: %s", md.Spec.Runtime)
	}

	// Ensure required labels are set.
	if svc.Labels == nil {
		svc.Labels = make(map[string]string)
	}
	svc.Labels[labelModelDeployment] = md.Name

	// Ensure the selector includes the model-deployment label.
	if svc.Spec.Selector == nil {
		svc.Spec.Selector = make(map[string]string)
	}
	svc.Spec.Selector[labelModelDeployment] = md.Name
	svc.Spec.Selector["podstack.io/role"] = "inference"

	// Set the controller reference.
	if err := controllerutil.SetControllerReference(md, svc, r.Scheme); err != nil {
		return fmt.Errorf("failed to set controller reference on service: %w", err)
	}

	if err := r.Create(ctx, svc); err != nil {
		if apierrors.IsAlreadyExists(err) {
			logger.V(1).Info("service already exists (race)", "service", svc.Name)
			return nil
		}
		return fmt.Errorf("failed to create service: %w", err)
	}

	logger.Info("created service", "service", svc.Name)
	return nil
}

// updatePhase updates the ModelDeployment status phase and persists the change.
func (r *ModelDeploymentReconciler) updatePhase(ctx context.Context, md *v1.ModelDeployment, phase string) (ctrl.Result, error) {
	md.Status.Phase = phase
	if err := r.Status().Update(ctx, md); err != nil {
		return ctrl.Result{RequeueAfter: requeueShort}, fmt.Errorf("failed to update phase to %s: %w", phase, err)
	}
	return ctrl.Result{Requeue: true}, nil
}

// getPodForDeployment finds a pod owned by the given ModelDeployment with the
// specified role (inference or standby).
func (r *ModelDeploymentReconciler) getPodForDeployment(ctx context.Context, md *v1.ModelDeployment, role string) (*corev1.Pod, error) {
	podList := &corev1.PodList{}
	if err := r.List(ctx, podList,
		client.InNamespace(md.Namespace),
		client.MatchingLabels{
			labelModelDeployment: md.Name,
			"podstack.io/role":   role,
		},
	); err != nil {
		return nil, fmt.Errorf("failed to list pods for deployment %s role %s: %w", md.Name, role, err)
	}

	if len(podList.Items) == 0 {
		return nil, nil
	}

	// Return the first non-deleted pod.
	for i := range podList.Items {
		pod := &podList.Items[i]
		if pod.DeletionTimestamp.IsZero() {
			return pod, nil
		}
	}

	return nil, nil
}

// getServiceForDeployment finds the service owned by the given ModelDeployment.
func (r *ModelDeploymentReconciler) getServiceForDeployment(ctx context.Context, md *v1.ModelDeployment) (*corev1.Service, error) {
	svc := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: md.Name, Namespace: md.Namespace}, svc)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get service for deployment %s: %w", md.Name, err)
	}
	return svc, nil
}

// setCondition sets or updates a condition on the ModelDeployment status.
func (r *ModelDeploymentReconciler) setCondition(md *v1.ModelDeployment, condType string, status metav1.ConditionStatus, reason, message string) {
	now := metav1.NewTime(time.Now())
	condition := metav1.Condition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	// Update existing condition or append new one.
	for i, existing := range md.Status.Conditions {
		if existing.Type == condType {
			if existing.Status != status {
				md.Status.Conditions[i] = condition
			} else {
				md.Status.Conditions[i].Reason = reason
				md.Status.Conditions[i].Message = message
			}
			return
		}
	}
	md.Status.Conditions = append(md.Status.Conditions, condition)
}

// isPodReady returns true if the pod is in the Running phase and all containers
// have their Ready condition set to true.
func isPodReady(pod *corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

// SetupWithManager sets up the controller with the Manager.
func (r *ModelDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1.ModelDeployment{}).
		Owns(&corev1.Pod{}).
		Owns(&corev1.Service{}).
		Complete(r)
}
