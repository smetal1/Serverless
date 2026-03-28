package controller

import (
	"context"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
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
)

const (
	tenantFinalizerName = "podstack.io/tenant-finalizer"

	// tenantNamespacePrefix is the prefix for auto-generated tenant namespaces.
	tenantNamespacePrefix = "podstack-tenant-"

	// Label applied to tenant namespaces and resources.
	labelTenant = "podstack.io/tenant"
)

// TenantReconciler reconciles Tenant CRs and manages their lifecycle through
// the phases: Provisioning -> Active -> Suspended. It provisions namespaces,
// sets up RBAC for tenant isolation, and monitors resource quota usage.
type TenantReconciler struct {
	client.Client
	Scheme *runtime.Scheme
	Log    logr.Logger
}

// +kubebuilder:rbac:groups=podstack.io,resources=tenants,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=podstack.io,resources=tenants/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=podstack.io,resources=tenants/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=namespaces,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=resourcequotas,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles;rolebindings,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=podstack.io,resources=modeldeployments,verbs=get;list;watch

// Reconcile manages the tenant lifecycle: Provisioning -> Active -> Suspended.
func (r *TenantReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("tenant", req.NamespacedName)
	logger.V(1).Info("reconciling Tenant")

	// Fetch the Tenant CR.
	tenant := &v1.Tenant{}
	if err := r.Get(ctx, req.NamespacedName, tenant); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("Tenant not found, must have been deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "unable to fetch Tenant")
		return ctrl.Result{}, err
	}

	// Handle deletion via finalizer.
	if !tenant.DeletionTimestamp.IsZero() {
		return r.handleDeletion(ctx, logger, tenant)
	}

	// Ensure finalizer is present.
	if !controllerutil.ContainsFinalizer(tenant, tenantFinalizerName) {
		controllerutil.AddFinalizer(tenant, tenantFinalizerName)
		if err := r.Update(ctx, tenant); err != nil {
			logger.Error(err, "failed to add finalizer to Tenant")
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Phase dispatch.
	switch tenant.Status.Phase {
	case "", v1.TenantPhaseProvisioning:
		return r.handleProvisioning(ctx, logger, tenant)
	case v1.TenantPhaseActive:
		return r.handleActive(ctx, logger, tenant)
	case v1.TenantPhaseSuspended:
		return r.handleSuspended(ctx, logger, tenant)
	default:
		logger.Info("unknown tenant phase, resetting to Provisioning", "phase", tenant.Status.Phase)
		tenant.Status.Phase = v1.TenantPhaseProvisioning
		if err := r.Status().Update(ctx, tenant); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}
}

// handleProvisioning sets up the tenant namespace, RBAC, and resource quotas.
func (r *TenantReconciler) handleProvisioning(ctx context.Context, logger logr.Logger, tenant *v1.Tenant) (ctrl.Result, error) {
	logger.Info("handling Provisioning phase", "displayName", tenant.Spec.DisplayName)

	r.setTenantCondition(tenant, "Provisioning", metav1.ConditionTrue, "InProgress", "Setting up tenant resources")

	// Determine the tenant namespace.
	nsName := tenant.Spec.Namespace
	if nsName == "" {
		nsName = tenantNamespacePrefix + tenant.Name
	}

	// Create the tenant namespace.
	if err := r.ensureNamespace(ctx, logger, tenant, nsName); err != nil {
		logger.Error(err, "failed to create tenant namespace")
		r.setTenantCondition(tenant, "NamespaceReady", metav1.ConditionFalse, "Error", err.Error())
		_ = r.Status().Update(ctx, tenant)
		return ctrl.Result{RequeueAfter: requeueMedium}, err
	}

	// Set up RBAC for tenant isolation.
	if err := r.ensureRBAC(ctx, logger, tenant, nsName); err != nil {
		logger.Error(err, "failed to set up RBAC")
		r.setTenantCondition(tenant, "RBACReady", metav1.ConditionFalse, "Error", err.Error())
		_ = r.Status().Update(ctx, tenant)
		return ctrl.Result{RequeueAfter: requeueMedium}, err
	}

	// Set up ResourceQuota.
	if err := r.ensureResourceQuota(ctx, logger, tenant, nsName); err != nil {
		logger.Error(err, "failed to set up resource quota")
		r.setTenantCondition(tenant, "QuotaReady", metav1.ConditionFalse, "Error", err.Error())
		_ = r.Status().Update(ctx, tenant)
		return ctrl.Result{RequeueAfter: requeueMedium}, err
	}

	// Provisioning complete. Transition to Active.
	logger.Info("tenant provisioning complete, transitioning to Active", "namespace", nsName)

	tenant.Status.Phase = v1.TenantPhaseActive
	r.setTenantCondition(tenant, "Ready", metav1.ConditionTrue, "Provisioned", "Tenant namespace and RBAC configured")

	if err := r.Status().Update(ctx, tenant); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{Requeue: true}, nil
}

// handleActive monitors an active tenant: counts models, GPU usage, VRAM
// consumption, and checks whether the tenant has exceeded its budget.
func (r *TenantReconciler) handleActive(ctx context.Context, logger logr.Logger, tenant *v1.Tenant) (ctrl.Result, error) {
	logger.V(1).Info("handling Active phase")

	// Determine the tenant namespace.
	nsName := tenant.Spec.Namespace
	if nsName == "" {
		nsName = tenantNamespacePrefix + tenant.Name
	}

	// List all ModelDeployments belonging to this tenant.
	mdList := &v1.ModelDeploymentList{}
	if err := r.List(ctx, mdList, client.InNamespace(nsName)); err != nil {
		logger.Error(err, "failed to list ModelDeployments for tenant")
		return ctrl.Result{RequeueAfter: requeueMedium}, err
	}

	// Count active models, GPU usage, and VRAM.
	var activeModels int32
	var gpusInUse int32
	var vramUsedMB int32

	for i := range mdList.Items {
		md := &mdList.Items[i]
		if md.Status.Phase == v1.PhaseActive || md.Status.Phase == v1.PhaseBooting {
			activeModels++
			gpusInUse += md.Spec.GPU.Count
			vramUsedMB += md.Status.VRAMUsedMB
		}
	}

	// Update status with current usage.
	tenant.Status.ActiveModels = activeModels
	tenant.Status.GPUsInUse = gpusInUse
	tenant.Status.VRAMUsedMB = vramUsedMB

	// Check quota violations.
	if tenant.Spec.Quota.MaxModels > 0 && activeModels > tenant.Spec.Quota.MaxModels {
		logger.Info("tenant exceeding max models quota",
			"active", activeModels, "max", tenant.Spec.Quota.MaxModels)
		r.setTenantCondition(tenant, "QuotaExceeded", metav1.ConditionTrue, "MaxModelsExceeded",
			fmt.Sprintf("Active models (%d) exceeds quota (%d)", activeModels, tenant.Spec.Quota.MaxModels))
	} else {
		r.setTenantCondition(tenant, "QuotaExceeded", metav1.ConditionFalse, "WithinQuota", "Resource usage within limits")
	}

	if tenant.Spec.Quota.MaxGPUs > 0 && gpusInUse > tenant.Spec.Quota.MaxGPUs {
		logger.Info("tenant exceeding max GPUs quota",
			"used", gpusInUse, "max", tenant.Spec.Quota.MaxGPUs)
		r.setTenantCondition(tenant, "GPUQuotaExceeded", metav1.ConditionTrue, "MaxGPUsExceeded",
			fmt.Sprintf("GPUs in use (%d) exceeds quota (%d)", gpusInUse, tenant.Spec.Quota.MaxGPUs))
	}

	if tenant.Spec.Quota.MaxVRAMMB > 0 && vramUsedMB > tenant.Spec.Quota.MaxVRAMMB {
		logger.Info("tenant exceeding max VRAM quota",
			"used", vramUsedMB, "max", tenant.Spec.Quota.MaxVRAMMB)
		r.setTenantCondition(tenant, "VRAMQuotaExceeded", metav1.ConditionTrue, "MaxVRAMExceeded",
			fmt.Sprintf("VRAM used (%dMB) exceeds quota (%dMB)", vramUsedMB, tenant.Spec.Quota.MaxVRAMMB))
	}

	// Check budget limits.
	if tenant.Spec.Quota.BudgetCentsPerMonth > 0 && tenant.Status.CurrentSpendCents > tenant.Spec.Quota.BudgetCentsPerMonth {
		logger.Info("tenant exceeding budget, suspending",
			"spend", tenant.Status.CurrentSpendCents,
			"budget", tenant.Spec.Quota.BudgetCentsPerMonth,
		)
		r.setTenantCondition(tenant, "BudgetExceeded", metav1.ConditionTrue, "OverBudget",
			fmt.Sprintf("Current spend (%d cents) exceeds monthly budget (%d cents)",
				tenant.Status.CurrentSpendCents, tenant.Spec.Quota.BudgetCentsPerMonth))

		tenant.Status.Phase = v1.TenantPhaseSuspended
		if err := r.Status().Update(ctx, tenant); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	r.setTenantCondition(tenant, "Ready", metav1.ConditionTrue, "Active", "Tenant is active and within quota")

	if err := r.Status().Update(ctx, tenant); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{RequeueAfter: requeueIdle}, nil
}

// handleSuspended manages a suspended tenant. Suspended tenants cannot create
// new model deployments and existing deployments may be evicted.
func (r *TenantReconciler) handleSuspended(ctx context.Context, logger logr.Logger, tenant *v1.Tenant) (ctrl.Result, error) {
	logger.Info("handling Suspended phase", "displayName", tenant.Spec.DisplayName)

	r.setTenantCondition(tenant, "Suspended", metav1.ConditionTrue, "Suspended",
		"Tenant is suspended due to quota or budget violation")

	// Check if the suspension condition has been resolved (e.g., budget reset).
	if tenant.Spec.Quota.BudgetCentsPerMonth > 0 &&
		tenant.Status.CurrentSpendCents <= tenant.Spec.Quota.BudgetCentsPerMonth {
		logger.Info("budget is now within limits, reactivating tenant")
		tenant.Status.Phase = v1.TenantPhaseActive
		r.setTenantCondition(tenant, "Suspended", metav1.ConditionFalse, "Reactivated",
			"Budget is within limits, tenant reactivated")

		if err := r.Status().Update(ctx, tenant); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	if err := r.Status().Update(ctx, tenant); err != nil {
		return ctrl.Result{}, err
	}

	// Check periodically for reactivation conditions.
	return ctrl.Result{RequeueAfter: requeueIdle}, nil
}

// handleDeletion processes the finalizer for Tenant deletion.
func (r *TenantReconciler) handleDeletion(ctx context.Context, logger logr.Logger, tenant *v1.Tenant) (ctrl.Result, error) {
	logger.Info("handling Tenant deletion", "displayName", tenant.Spec.DisplayName)

	if controllerutil.ContainsFinalizer(tenant, tenantFinalizerName) {
		// Determine the tenant namespace.
		nsName := tenant.Spec.Namespace
		if nsName == "" {
			nsName = tenantNamespacePrefix + tenant.Name
		}

		// Delete all ModelDeployments in the tenant namespace.
		mdList := &v1.ModelDeploymentList{}
		if err := r.List(ctx, mdList, client.InNamespace(nsName)); err == nil {
			for i := range mdList.Items {
				if err := r.Delete(ctx, &mdList.Items[i]); err != nil && !apierrors.IsNotFound(err) {
					logger.Error(err, "failed to delete ModelDeployment during tenant cleanup",
						"modelDeployment", mdList.Items[i].Name)
				}
			}
		}

		// Delete all LoRAAdapters in the tenant namespace.
		loraList := &v1.LoRAAdapterList{}
		if err := r.List(ctx, loraList, client.InNamespace(nsName)); err == nil {
			for i := range loraList.Items {
				if err := r.Delete(ctx, &loraList.Items[i]); err != nil && !apierrors.IsNotFound(err) {
					logger.Error(err, "failed to delete LoRAAdapter during tenant cleanup",
						"loraAdapter", loraList.Items[i].Name)
				}
			}
		}

		// Note: We do not delete the namespace itself to avoid data loss.
		// The namespace can be cleaned up manually or by a separate garbage
		// collection controller.
		logger.Info("tenant resources cleaned up, namespace preserved", "namespace", nsName)

		// Remove the finalizer.
		controllerutil.RemoveFinalizer(tenant, tenantFinalizerName)
		if err := r.Update(ctx, tenant); err != nil {
			logger.Error(err, "failed to remove finalizer from Tenant")
			return ctrl.Result{}, err
		}
		logger.Info("finalizer removed from Tenant")
	}

	return ctrl.Result{}, nil
}

// ---------------------------------------------------------------------------
// Provisioning helpers
// ---------------------------------------------------------------------------

// ensureNamespace creates the tenant namespace if it does not exist.
func (r *TenantReconciler) ensureNamespace(ctx context.Context, logger logr.Logger, tenant *v1.Tenant, nsName string) error {
	ns := &corev1.Namespace{}
	err := r.Get(ctx, types.NamespacedName{Name: nsName}, ns)
	if err == nil {
		logger.V(1).Info("tenant namespace already exists", "namespace", nsName)
		return nil
	}
	if !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to check namespace %s: %w", nsName, err)
	}

	ns = &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: nsName,
			Labels: map[string]string{
				labelTenant:                    tenant.Name,
				"app.kubernetes.io/managed-by": "podstack-controller",
			},
			Annotations: map[string]string{
				"podstack.io/tenant-display-name": tenant.Spec.DisplayName,
			},
		},
	}

	if err := r.Create(ctx, ns); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return fmt.Errorf("failed to create namespace %s: %w", nsName, err)
	}

	logger.Info("created tenant namespace", "namespace", nsName)
	return nil
}

// ensureRBAC creates a Role and RoleBinding in the tenant namespace for
// tenant isolation. The role allows managing ModelDeployments, Snapshots,
// and LoRAAdapters within the tenant's namespace.
func (r *TenantReconciler) ensureRBAC(ctx context.Context, logger logr.Logger, tenant *v1.Tenant, nsName string) error {
	// Create the tenant Role.
	roleName := fmt.Sprintf("%s-tenant-role", tenant.Name)
	role := &rbacv1.Role{}
	err := r.Get(ctx, types.NamespacedName{Name: roleName, Namespace: nsName}, role)
	if apierrors.IsNotFound(err) {
		role = &rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{
				Name:      roleName,
				Namespace: nsName,
				Labels: map[string]string{
					labelTenant: tenant.Name,
				},
			},
			Rules: []rbacv1.PolicyRule{
				{
					APIGroups: []string{"podstack.io"},
					Resources: []string{"modeldeployments", "snapshots", "loraadapters"},
					Verbs:     []string{"get", "list", "watch", "create", "update", "patch", "delete"},
				},
				{
					APIGroups: []string{""},
					Resources: []string{"pods", "services"},
					Verbs:     []string{"get", "list", "watch"},
				},
				{
					APIGroups: []string{""},
					Resources: []string{"pods/log"},
					Verbs:     []string{"get"},
				},
			},
		}

		if err := r.Create(ctx, role); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create Role %s: %w", roleName, err)
			}
		} else {
			logger.Info("created tenant Role", "role", roleName, "namespace", nsName)
		}
	} else if err != nil {
		return fmt.Errorf("failed to check Role %s: %w", roleName, err)
	}

	// Create the tenant RoleBinding.
	rbName := fmt.Sprintf("%s-tenant-binding", tenant.Name)
	rb := &rbacv1.RoleBinding{}
	err = r.Get(ctx, types.NamespacedName{Name: rbName, Namespace: nsName}, rb)
	if apierrors.IsNotFound(err) {
		rb = &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:      rbName,
				Namespace: nsName,
				Labels: map[string]string{
					labelTenant: tenant.Name,
				},
			},
			RoleRef: rbacv1.RoleRef{
				APIGroup: rbacv1.GroupName,
				Kind:     "Role",
				Name:     roleName,
			},
			Subjects: []rbacv1.Subject{
				{
					Kind:      rbacv1.GroupKind,
					Name:      fmt.Sprintf("tenant:%s", tenant.Name),
					Namespace: nsName,
				},
			},
		}

		if err := r.Create(ctx, rb); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create RoleBinding %s: %w", rbName, err)
			}
		} else {
			logger.Info("created tenant RoleBinding", "roleBinding", rbName, "namespace", nsName)
		}
	} else if err != nil {
		return fmt.Errorf("failed to check RoleBinding %s: %w", rbName, err)
	}

	return nil
}

// ensureResourceQuota creates a ResourceQuota in the tenant namespace that
// enforces the tenant's GPU and model limits.
func (r *TenantReconciler) ensureResourceQuota(ctx context.Context, logger logr.Logger, tenant *v1.Tenant, nsName string) error {
	quotaName := fmt.Sprintf("%s-quota", tenant.Name)
	quota := &corev1.ResourceQuota{}
	err := r.Get(ctx, types.NamespacedName{Name: quotaName, Namespace: nsName}, quota)
	if apierrors.IsNotFound(err) {
		hard := corev1.ResourceList{}

		// Set GPU quota if specified.
		if tenant.Spec.Quota.MaxGPUs > 0 {
			hard["requests.nvidia.com/gpu"] = *resource.NewQuantity(int64(tenant.Spec.Quota.MaxGPUs), resource.DecimalSI)
			hard["limits.nvidia.com/gpu"] = *resource.NewQuantity(int64(tenant.Spec.Quota.MaxGPUs), resource.DecimalSI)
		}

		// Set pod count limit based on max models (models + standby + buffer).
		if tenant.Spec.Quota.MaxModels > 0 {
			maxPods := tenant.Spec.Quota.MaxModels * 3 // inference + standby + buffer
			hard[corev1.ResourcePods] = *resource.NewQuantity(int64(maxPods), resource.DecimalSI)
		}

		quota = &corev1.ResourceQuota{
			ObjectMeta: metav1.ObjectMeta{
				Name:      quotaName,
				Namespace: nsName,
				Labels: map[string]string{
					labelTenant: tenant.Name,
				},
			},
			Spec: corev1.ResourceQuotaSpec{
				Hard: hard,
			},
		}

		if err := r.Create(ctx, quota); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("failed to create ResourceQuota %s: %w", quotaName, err)
			}
		} else {
			logger.Info("created tenant ResourceQuota", "quota", quotaName, "namespace", nsName)
		}
	} else if err != nil {
		return fmt.Errorf("failed to check ResourceQuota %s: %w", quotaName, err)
	}

	return nil
}

// setTenantCondition sets or updates a condition on the Tenant status.
func (r *TenantReconciler) setTenantCondition(tenant *v1.Tenant, condType string, status metav1.ConditionStatus, reason, message string) {
	now := metav1.NewTime(time.Now())
	condition := metav1.Condition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: now,
		Reason:             reason,
		Message:            message,
	}

	for i, existing := range tenant.Status.Conditions {
		if existing.Type == condType {
			if existing.Status != status {
				tenant.Status.Conditions[i] = condition
			} else {
				tenant.Status.Conditions[i].Reason = reason
				tenant.Status.Conditions[i].Message = message
			}
			return
		}
	}
	tenant.Status.Conditions = append(tenant.Status.Conditions, condition)
}

// SetupWithManager sets up the controller with the Manager.
func (r *TenantReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1.Tenant{}).
		Owns(&corev1.Namespace{}).
		Complete(r)
}
