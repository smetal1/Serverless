package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ModelDeployment lifecycle phases.
const (
	PhasePending      = "Pending"
	PhaseSnapshotting = "Snapshotting"
	PhaseStandby      = "Standby"
	PhaseBooting      = "Booting"
	PhaseActive       = "Active"
	PhaseIdle         = "Idle"
	PhaseEvicted      = "Evicted"
)

// Model source types.
const (
	SourceHuggingFace = "huggingface"
	SourceNFS         = "nfs"
	SourceS3          = "s3"
	SourceOCI         = "oci"
)

// Model types.
const (
	ModelTypeLLM       = "llm"
	ModelTypeDiffusion = "diffusion"
	ModelTypeTTS       = "tts"
	ModelTypeASR       = "asr"
	ModelTypeVision    = "vision"
	ModelTypeCustom    = "custom"
)

// Runtime types.
const (
	RuntimeVLLM    = "vllm"
	RuntimeTriton  = "triton"
	RuntimeGeneric = "generic"
)

// ModelDeploymentSpec defines the desired state of a ModelDeployment.
type ModelDeploymentSpec struct {
	// ModelName is the model identifier: HuggingFace ID, NFS path, or OCI image.
	ModelName string `json:"modelName"`

	// ModelSource specifies where to fetch the model from: huggingface | nfs | s3 | oci.
	ModelSource string `json:"modelSource"`

	// ModelType classifies the model: llm | diffusion | tts | asr | vision | custom.
	ModelType string `json:"modelType"`

	// Image is the container image for generic runtime deployments.
	// +optional
	Image string `json:"image,omitempty"`

	// Runtime selects the inference backend: vllm | triton | generic.
	Runtime string `json:"runtime"`

	// RuntimeArgs are additional arguments passed to the runtime.
	// +optional
	RuntimeArgs []string `json:"runtimeArgs,omitempty"`

	// GPU specifies GPU resource requirements.
	GPU GPUSpec `json:"gpu"`

	// Scaling configures autoscaling and cold start behavior.
	Scaling ScalingSpec `json:"scaling"`

	// Snapshot configures CUDA checkpoint/restore for sub-second boot.
	Snapshot ModelSnapshotSpec `json:"snapshot"`

	// LoRA configures LoRA adapter multiplexing (LLMs only).
	// +optional
	LoRA *LoRAMuxSpec `json:"lora,omitempty"`

	// TenantRef references the owning Tenant CR.
	// +optional
	TenantRef string `json:"tenantRef,omitempty"`
}

// GPUSpec defines GPU resource requirements.
type GPUSpec struct {
	// Count is the number of GPUs required.
	Count int32 `json:"count"`

	// MemoryMB is the VRAM limit in megabytes (for fractional GPU via HAMi).
	// +optional
	MemoryMB int32 `json:"memoryMB,omitempty"`

	// CoresPercent is the percentage of GPU cores to allocate.
	// +optional
	CoresPercent int32 `json:"coresPercent,omitempty"`

	// Type specifies the GPU model: l40s, a100, h100, etc.
	// +optional
	Type string `json:"type,omitempty"`
}

// ScalingSpec defines autoscaling behavior.
type ScalingSpec struct {
	// MinReplicas is the minimum number of replicas. 0 enables scale-to-zero.
	MinReplicas int32 `json:"minReplicas"`

	// MaxReplicas is the maximum number of replicas.
	MaxReplicas int32 `json:"maxReplicas"`

	// IdleTimeoutSeconds is how long to wait with no requests before transitioning to Idle.
	IdleTimeoutSeconds int32 `json:"idleTimeoutSeconds"`

	// StandbyPool is the number of pre-warmed standby containers (CPU only, 0 GPU).
	StandbyPool int32 `json:"standbyPool"`
}

// ModelSnapshotSpec configures CUDA snapshot behavior.
type ModelSnapshotSpec struct {
	// Enabled turns on CUDA snapshot-based cold start optimization.
	Enabled bool `json:"enabled"`

	// AutoSnapshot automatically creates a snapshot after the first model load.
	// +optional
	AutoSnapshot bool `json:"autoSnapshot,omitempty"`

	// WarmupRequests is the number of inference requests to run before snapshotting
	// (allows caches to warm up for better restored performance).
	// +optional
	WarmupRequests int32 `json:"warmupRequests,omitempty"`
}

// LoRAMuxSpec configures LoRA adapter multiplexing on a base model.
type LoRAMuxSpec struct {
	// MaxAdapters is the maximum number of LoRA adapters to keep loaded.
	MaxAdapters int32 `json:"maxAdapters,omitempty"`

	// AdapterRefs lists LoRAAdapter CRs to load on this base model.
	// +optional
	AdapterRefs []string `json:"adapterRefs,omitempty"`
}

// ModelDeploymentStatus defines the observed state of a ModelDeployment.
type ModelDeploymentStatus struct {
	// Phase is the current lifecycle phase:
	// Pending → Snapshotting → Standby → Booting → Active → Idle → Evicted
	Phase string `json:"phase"`

	// ReadyReplicas is the number of replicas actively serving with GPU.
	ReadyReplicas int32 `json:"readyReplicas"`

	// StandbyReplicas is the number of pre-warmed replicas (CPU only).
	StandbyReplicas int32 `json:"standbyReplicas"`

	// Endpoint is the internal service URL for this model.
	// +optional
	Endpoint string `json:"endpoint,omitempty"`

	// SnapshotRef references the Snapshot CR for this model.
	// +optional
	SnapshotRef string `json:"snapshotRef,omitempty"`

	// VRAMUsedMB is the current VRAM consumption in megabytes.
	VRAMUsedMB int32 `json:"vramUsedMB"`

	// LastRequestAt is the timestamp of the last inference request.
	// +optional
	LastRequestAt *metav1.Time `json:"lastRequestAt,omitempty"`

	// ColdStartMs is the last measured cold start duration in milliseconds.
	ColdStartMs int64 `json:"coldStartMs"`

	// Conditions represent the latest available observations of the deployment's state.
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.modelName`
// +kubebuilder:printcolumn:name="Runtime",type=string,JSONPath=`.spec.runtime`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
// +kubebuilder:printcolumn:name="Standby",type=integer,JSONPath=`.status.standbyReplicas`
// +kubebuilder:printcolumn:name="VRAM(MB)",type=integer,JSONPath=`.status.vramUsedMB`
// +kubebuilder:printcolumn:name="ColdStart(ms)",type=integer,JSONPath=`.status.coldStartMs`

// ModelDeployment is the Schema for the modeldeployments API.
// It represents a model inference endpoint managed by the Podstack Inference OS.
type ModelDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ModelDeploymentSpec   `json:"spec,omitempty"`
	Status ModelDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ModelDeploymentList contains a list of ModelDeployment.
type ModelDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ModelDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ModelDeployment{}, &ModelDeploymentList{})
}
