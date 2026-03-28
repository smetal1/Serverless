package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Tenant lifecycle phases.
const (
	TenantPhaseProvisioning = "Provisioning"
	TenantPhaseActive       = "Active"
	TenantPhaseSuspended    = "Suspended"
)

// TenantSpec defines the desired state of a Tenant.
type TenantSpec struct {
	// DisplayName is a human-readable name for the tenant.
	DisplayName string `json:"displayName"`

	// Namespace is the Kubernetes namespace for this tenant's resources.
	// If empty, defaults to "podstack-tenant-{name}".
	// +optional
	Namespace string `json:"namespace,omitempty"`

	// APIKeys is a list of API key hashes for authentication.
	// Keys are SHA-256 hashed; plaintext is never stored.
	APIKeys []APIKeySpec `json:"apiKeys"`

	// Quota defines resource limits for the tenant.
	Quota TenantQuota `json:"quota"`
}

// APIKeySpec represents a hashed API key.
type APIKeySpec struct {
	// Name is a human-readable identifier for the key.
	Name string `json:"name"`

	// KeyHash is the SHA-256 hash of the API key.
	KeyHash string `json:"keyHash"`

	// CreatedAt is when the key was created.
	// +optional
	CreatedAt *metav1.Time `json:"createdAt,omitempty"`
}

// TenantQuota defines resource limits for a tenant.
type TenantQuota struct {
	// MaxModels is the maximum number of ModelDeployments.
	MaxModels int32 `json:"maxModels"`

	// MaxGPUs is the maximum number of concurrent GPUs.
	MaxGPUs int32 `json:"maxGPUs"`

	// MaxVRAMMB is the maximum total VRAM in megabytes.
	MaxVRAMMB int32 `json:"maxVRAMMB"`

	// MaxRequestsPerMinute is the rate limit.
	MaxRequestsPerMinute int32 `json:"maxRequestsPerMinute"`

	// BudgetCentsPerMonth is the monthly spending cap in cents.
	// +optional
	BudgetCentsPerMonth int64 `json:"budgetCentsPerMonth,omitempty"`
}

// TenantStatus defines the observed state of a Tenant.
type TenantStatus struct {
	// Phase is the tenant lifecycle: Provisioning | Active | Suspended.
	Phase string `json:"phase"`

	// ActiveModels is the number of active ModelDeployments.
	ActiveModels int32 `json:"activeModels"`

	// GPUsInUse is the current GPU count.
	GPUsInUse int32 `json:"gpusInUse"`

	// VRAMUsedMB is the current total VRAM consumption.
	VRAMUsedMB int32 `json:"vramUsedMB"`

	// CurrentSpendCents is the current month's spend in cents.
	CurrentSpendCents int64 `json:"currentSpendCents"`

	// Conditions represent the latest available observations.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Display",type=string,JSONPath=`.spec.displayName`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Models",type=integer,JSONPath=`.status.activeModels`
// +kubebuilder:printcolumn:name="GPUs",type=integer,JSONPath=`.status.gpusInUse`
// +kubebuilder:printcolumn:name="VRAM(MB)",type=integer,JSONPath=`.status.vramUsedMB`

// Tenant is the Schema for the tenants API.
// It represents a tenant in the multi-tenant Podstack Inference OS.
type Tenant struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TenantSpec   `json:"spec,omitempty"`
	Status TenantStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TenantList contains a list of Tenant.
type TenantList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Tenant `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Tenant{}, &TenantList{})
}
