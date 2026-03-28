package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Snapshot lifecycle phases.
const (
	SnapshotPhaseCreating  = "Creating"
	SnapshotPhaseReady     = "Ready"
	SnapshotPhaseRestoring = "Restoring"
	SnapshotPhaseFailed    = "Failed"
)

// SnapshotSpec defines the desired state of a CUDA snapshot.
type SnapshotSpec struct {
	// ModelDeploymentRef is the name of the ModelDeployment this snapshot belongs to.
	ModelDeploymentRef string `json:"modelDeploymentRef"`

	// StoragePath is the NFS/NVMe path to the snapshot archive.
	StoragePath string `json:"storagePath"`

	// GPUType is the GPU model this snapshot was captured on.
	// Snapshots must be restored on the same GPU type.
	GPUType string `json:"gpuType"`

	// CUDAVersion is the CUDA toolkit version used during capture.
	CUDAVersion string `json:"cudaVersion"`

	// DriverVersion is the NVIDIA driver version used during capture.
	DriverVersion string `json:"driverVersion"`

	// SizeBytes is the total size of the snapshot archive.
	SizeBytes int64 `json:"sizeBytes"`
}

// SnapshotStatus defines the observed state of a Snapshot.
type SnapshotStatus struct {
	// Phase is the current snapshot lifecycle: Creating | Ready | Restoring | Failed.
	Phase string `json:"phase"`

	// CreatedAt is the timestamp when the snapshot was created.
	// +optional
	CreatedAt *metav1.Time `json:"createdAt,omitempty"`

	// RestoreTimeMs is the last measured restore duration in milliseconds.
	RestoreTimeMs int64 `json:"restoreTimeMs"`

	// Verified indicates whether an inference test passed after restore.
	Verified bool `json:"verified"`

	// Message provides additional status information.
	// +optional
	Message string `json:"message,omitempty"`

	// Conditions represent the latest available observations.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.modelDeploymentRef`
// +kubebuilder:printcolumn:name="GPU",type=string,JSONPath=`.spec.gpuType`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Size",type=integer,JSONPath=`.spec.sizeBytes`
// +kubebuilder:printcolumn:name="RestoreTime(ms)",type=integer,JSONPath=`.status.restoreTimeMs`
// +kubebuilder:printcolumn:name="Verified",type=boolean,JSONPath=`.status.verified`

// Snapshot is the Schema for the snapshots API.
// It represents a CUDA checkpoint of a model's GPU state for sub-second restore.
type Snapshot struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SnapshotSpec   `json:"spec,omitempty"`
	Status SnapshotStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// SnapshotList contains a list of Snapshot.
type SnapshotList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Snapshot `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Snapshot{}, &SnapshotList{})
}
