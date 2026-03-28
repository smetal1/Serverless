package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// LoRA adapter lifecycle phases.
const (
	LoRAPhaseDownloading = "Downloading"
	LoRAPhaseCached      = "Cached"
	LoRAPhaseLoaded      = "Loaded"
	LoRAPhaseFailed      = "Failed"
)

// LoRAAdapterSpec defines the desired state of a LoRA adapter.
type LoRAAdapterSpec struct {
	// AdapterName is the human-readable name of the adapter.
	AdapterName string `json:"adapterName"`

	// BaseModelRef is the name of the ModelDeployment this adapter targets.
	BaseModelRef string `json:"baseModelRef"`

	// Source is where to download the adapter from: huggingface | nfs | s3.
	Source string `json:"source"`

	// SourcePath is the HuggingFace ID, NFS path, or S3 URI.
	SourcePath string `json:"sourcePath"`

	// TenantRef is the owning tenant.
	// +optional
	TenantRef string `json:"tenantRef,omitempty"`

	// AutoLoad automatically loads the adapter when the base model is active.
	// +optional
	AutoLoad bool `json:"autoLoad,omitempty"`
}

// LoRAAdapterStatus defines the observed state of a LoRA adapter.
type LoRAAdapterStatus struct {
	// Phase is the adapter lifecycle: Downloading | Cached | Loaded | Failed.
	Phase string `json:"phase"`

	// CachePath is the NFS path where the adapter weights are stored.
	// +optional
	CachePath string `json:"cachePath,omitempty"`

	// LoadedOnPods lists the pod names where this adapter is currently loaded.
	// +optional
	LoadedOnPods []string `json:"loadedOnPods,omitempty"`

	// Message provides additional status information.
	// +optional
	Message string `json:"message,omitempty"`

	// Conditions represent the latest available observations.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Adapter",type=string,JSONPath=`.spec.adapterName`
// +kubebuilder:printcolumn:name="BaseModel",type=string,JSONPath=`.spec.baseModelRef`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Tenant",type=string,JSONPath=`.spec.tenantRef`

// LoRAAdapter is the Schema for the loraadapters API.
// It represents a LoRA adapter that can be hot-swapped onto a base model at runtime.
type LoRAAdapter struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   LoRAAdapterSpec   `json:"spec,omitempty"`
	Status LoRAAdapterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// LoRAAdapterList contains a list of LoRAAdapter.
type LoRAAdapterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []LoRAAdapter `json:"items"`
}

func init() {
	SchemeBuilder.Register(&LoRAAdapter{}, &LoRAAdapterList{})
}
