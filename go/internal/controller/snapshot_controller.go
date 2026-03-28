package controller

import (
	"context"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "github.com/podstack/serverless/api/v1"
	"github.com/podstack/serverless/internal/snapshot"
)

const (
	snapshotFinalizerName = "podstack.io/snapshot-finalizer"
)

// SnapshotReconciler reconciles Snapshot CRs and manages their lifecycle
// through the phases: Creating -> Ready, or Creating -> Failed. On deletion,
// it cleans up the snapshot data from the blobstore.
type SnapshotReconciler struct {
	client.Client
	Scheme          *runtime.Scheme
	SnapshotManager *snapshot.Manager
	Log             logr.Logger
}

// +kubebuilder:rbac:groups=podstack.io,resources=snapshots,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=podstack.io,resources=snapshots/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=podstack.io,resources=snapshots/finalizers,verbs=update

// Reconcile watches Snapshot CRs and manages their lifecycle.
func (r *SnapshotReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("snapshot", req.NamespacedName)
	logger.V(1).Info("reconciling Snapshot")

	// Fetch the Snapshot CR.
	snap := &v1.Snapshot{}
	if err := r.Get(ctx, req.NamespacedName, snap); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("Snapshot not found, must have been deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "unable to fetch Snapshot")
		return ctrl.Result{}, err
	}

	// Handle deletion via finalizer.
	if !snap.DeletionTimestamp.IsZero() {
		return r.handleDeletion(ctx, logger, snap)
	}

	// Ensure finalizer is present.
	if !controllerutil.ContainsFinalizer(snap, snapshotFinalizerName) {
		controllerutil.AddFinalizer(snap, snapshotFinalizerName)
		if err := r.Update(ctx, snap); err != nil {
			logger.Error(err, "failed to add finalizer to Snapshot")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Phase dispatch.
	switch snap.Status.Phase {
	case "", v1.SnapshotPhaseCreating:
		return r.handleCreating(ctx, logger, snap)
	case v1.SnapshotPhaseReady:
		return r.handleReady(ctx, logger, snap)
	case v1.SnapshotPhaseRestoring:
		return r.handleRestoring(ctx, logger, snap)
	case v1.SnapshotPhaseFailed:
		return r.handleFailed(ctx, logger, snap)
	default:
		logger.Info("unknown snapshot phase, resetting to Creating", "phase", snap.Status.Phase)
		snap.Status.Phase = v1.SnapshotPhaseCreating
		if err := r.Status().Update(ctx, snap); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}
}

// handleCreating processes a snapshot that is being created. It verifies
// that the snapshot data exists in the blobstore and transitions to Ready
// when complete.
func (r *SnapshotReconciler) handleCreating(ctx context.Context, logger logr.Logger, snap *v1.Snapshot) (ctrl.Result, error) {
	logger.Info("handling Creating phase", "modelDeploymentRef", snap.Spec.ModelDeploymentRef)

	r.setSnapshotCondition(snap, "Creating", metav1.ConditionTrue, "InProgress", "Snapshot creation in progress")

	// Check if the snapshot data already exists in the blobstore using the
	// SnapshotExists method which searches by model name and GPU type.
	if r.SnapshotManager != nil {
		existingSnap, exists, err := r.SnapshotManager.SnapshotExists(ctx, snap.Spec.ModelDeploymentRef, snap.Spec.GPUType)
		if err != nil {
			logger.Error(err, "failed to check snapshot existence")
			return ctrl.Result{RequeueAfter: requeueMedium}, nil
		}

		if exists && existingSnap != nil {
			logger.Info("snapshot data found in blobstore, transitioning to Ready")

			snap.Spec.SizeBytes = existingSnap.Spec.SizeBytes

			now := metav1.NewTime(time.Now())
			snap.Status.Phase = v1.SnapshotPhaseReady
			snap.Status.CreatedAt = &now
			snap.Status.Verified = false
			snap.Status.Message = "Snapshot created successfully"

			r.setSnapshotCondition(snap, "Ready", metav1.ConditionTrue, "Created", "Snapshot data available in blobstore")

			if err := r.Status().Update(ctx, snap); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, nil
		}
	}

	// Snapshot data not yet available. The ModelDeployment controller is
	// responsible for triggering the actual snapshot capture via the
	// SnapshotManager. We just wait for the data to appear.
	logger.V(1).Info("snapshot data not yet available, waiting")
	snap.Status.Phase = v1.SnapshotPhaseCreating
	snap.Status.Message = "Waiting for snapshot data to be captured"

	if err := r.Status().Update(ctx, snap); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: requeueMedium}, nil
}

// handleReady processes a snapshot that is ready for use. This is a steady
// state -- no action needed unless the snapshot data disappears.
func (r *SnapshotReconciler) handleReady(ctx context.Context, logger logr.Logger, snap *v1.Snapshot) (ctrl.Result, error) {
	logger.V(1).Info("handling Ready phase")

	// Verify the snapshot data still exists by checking the blobstore.
	if r.SnapshotManager != nil {
		_, exists, err := r.SnapshotManager.SnapshotExists(ctx, snap.Spec.ModelDeploymentRef, snap.Spec.GPUType)
		if err != nil {
			logger.Error(err, "failed to verify snapshot existence")
			return ctrl.Result{RequeueAfter: requeueLong}, nil
		}
		if !exists {
			logger.Info("snapshot data missing from blobstore, marking as Failed")
			snap.Status.Phase = v1.SnapshotPhaseFailed
			snap.Status.Message = "Snapshot data no longer available in blobstore"
			snap.Status.Verified = false

			r.setSnapshotCondition(snap, "Ready", metav1.ConditionFalse, "DataMissing", "Snapshot file not found in blobstore")

			if err := r.Status().Update(ctx, snap); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{Requeue: true}, nil
		}
	}

	// Snapshot is healthy. No periodic requeue needed for Ready state.
	return ctrl.Result{}, nil
}

// handleRestoring processes a snapshot that is currently being restored. The
// actual restore is handled by the ModelDeployment controller; this handler
// monitors the progress.
func (r *SnapshotReconciler) handleRestoring(ctx context.Context, logger logr.Logger, snap *v1.Snapshot) (ctrl.Result, error) {
	logger.Info("handling Restoring phase")

	r.setSnapshotCondition(snap, "Restoring", metav1.ConditionTrue, "InProgress", "Snapshot restore in progress")

	if err := r.Status().Update(ctx, snap); err != nil {
		return ctrl.Result{}, err
	}

	// The restore is driven by the ModelDeployment controller. Once restore
	// completes, the MD controller will update the Snapshot back to Ready.
	return ctrl.Result{RequeueAfter: requeueShort}, nil
}

// handleFailed processes a snapshot in the Failed state. It logs the failure
// and allows manual intervention.
func (r *SnapshotReconciler) handleFailed(ctx context.Context, logger logr.Logger, snap *v1.Snapshot) (ctrl.Result, error) {
	logger.Info("handling Failed phase", "message", snap.Status.Message)

	r.setSnapshotCondition(snap, "Failed", metav1.ConditionTrue, "Error", snap.Status.Message)

	if err := r.Status().Update(ctx, snap); err != nil {
		return ctrl.Result{}, err
	}

	// Do not requeue automatically. The user or ModelDeployment controller
	// must take corrective action (delete and recreate the Snapshot CR).
	return ctrl.Result{}, nil
}

// handleDeletion processes the finalizer logic for snapshot deletion.
func (r *SnapshotReconciler) handleDeletion(ctx context.Context, logger logr.Logger, snap *v1.Snapshot) (ctrl.Result, error) {
	logger.Info("handling snapshot deletion")

	if controllerutil.ContainsFinalizer(snap, snapshotFinalizerName) {
		// Clean up blobstore files using model name and GPU type.
		if r.SnapshotManager != nil {
			if err := r.SnapshotManager.DeleteSnapshotByModel(snap.Spec.ModelDeploymentRef, snap.Spec.GPUType); err != nil {
				logger.Error(err, "failed to delete snapshot from blobstore", "snapshot", snap.Name)
				// Continue with finalizer removal even if blobstore cleanup fails.
			} else {
				logger.Info("deleted snapshot data from blobstore", "snapshot", snap.Name)
			}
		}

		// Remove the finalizer.
		controllerutil.RemoveFinalizer(snap, snapshotFinalizerName)
		if err := r.Update(ctx, snap); err != nil {
			logger.Error(err, "failed to remove finalizer from Snapshot")
			return ctrl.Result{}, err
		}
		logger.Info("finalizer removed from Snapshot")
	}

	return ctrl.Result{}, nil
}

// setSnapshotCondition sets or updates a condition on the Snapshot status.
func (r *SnapshotReconciler) setSnapshotCondition(snap *v1.Snapshot, condType string, status metav1.ConditionStatus, reason, message string) {
	now := metav1.NewTime(time.Now())
	condition := metav1.Condition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	for i, existing := range snap.Status.Conditions {
		if existing.Type == condType {
			if existing.Status != status {
				snap.Status.Conditions[i] = condition
			} else {
				snap.Status.Conditions[i].Reason = reason
				snap.Status.Conditions[i].Message = message
			}
			return
		}
	}
	snap.Status.Conditions = append(snap.Status.Conditions, condition)
}

// SetupWithManager sets up the controller with the Manager.
func (r *SnapshotReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1.Snapshot{}).
		Complete(r)
}

// Ensure interface compliance.
var _ fmt.Stringer = (*SnapshotReconciler)(nil)

// String implements fmt.Stringer for logging purposes.
func (r *SnapshotReconciler) String() string {
	return "SnapshotReconciler"
}
