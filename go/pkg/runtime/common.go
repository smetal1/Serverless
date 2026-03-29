// Package runtime provides Kubernetes pod template generators for the
// inference runtimes supported by the Podstack Inference OS: vLLM, Triton,
// and generic CUDA containers.
package runtime

import (
	"fmt"
	"strings"

	v1 "github.com/podstack/serverless/api/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	// nfsBaseMountPath is where model weights are mounted from NFS.
	nfsBaseMountPath = "/models/base"

	// nfsLoraMountPath is where LoRA adapters are mounted.
	nfsLoraMountPath = "/models/lora"

	// snapshotMountPath is where CUDA snapshots are mounted.
	snapshotMountPath = "/models/snapshots"

	// nfsVolumeName is the name of the NFS volume for model weights.
	nfsVolumeName = "model-store"

	// shmVolumeName is the name of the shared memory emptyDir volume.
	shmVolumeName = "dshm"

	// snapshotVolumeName is the name of the snapshot volume.
	snapshotVolumeName = "snapshot-store"

	// nfsServer is the default NFS server address.
	nfsServer = "192.168.29.5"

	// nfsSharePath is the NFS export path.
	nfsSharePath = "/mnt/tank/podstack/serverless-models"

	// snapshotNFSPath is the NFS export for snapshot storage (separate from models).
	snapshotNFSPath = "/mnt/tank/podstack/serverless-snapshots"

	// snapshotAgentImage is the sidecar container image for CUDA snapshot management.
	snapshotAgentImage = "saurav7055/podstack-snapshot-agent:latest"
)

// modelPath converts a HuggingFace model ID (e.g. "meta-llama/Llama-3-8B")
// into an NFS-safe directory name (e.g. "meta-llama--Llama-3-8B").
func modelPath(modelName string) string {
	return strings.ReplaceAll(modelName, "/", "--")
}

// commonLabels returns the standard labels applied to all runtime pods.
func commonLabels(md *v1.ModelDeployment) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":       "podstack-inference",
		"app.kubernetes.io/instance":   md.Name,
		"app.kubernetes.io/component":  md.Spec.Runtime,
		"app.kubernetes.io/managed-by": "podstack-controller",
		"podstack.io/model":            modelPath(md.Spec.ModelName),
		"podstack.io/runtime":          md.Spec.Runtime,
	}
}

// hamiAnnotations returns HAMi annotations for fractional GPU scheduling.
func hamiAnnotations(md *v1.ModelDeployment) map[string]string {
	annotations := map[string]string{}
	if md.Spec.GPU.MemoryMB > 0 {
		annotations["hami.io/vgpu-memory"] = fmt.Sprintf("%d", md.Spec.GPU.MemoryMB)
	}
	if md.Spec.GPU.CoresPercent > 0 {
		annotations["hami.io/vgpu-cores"] = fmt.Sprintf("%d", md.Spec.GPU.CoresPercent)
	}
	if md.Spec.GPU.Type != "" {
		annotations["hami.io/gpu-type"] = md.Spec.GPU.Type
	}
	return annotations
}

// commonEnvVars returns environment variables injected into every inference container.
func commonEnvVars(md *v1.ModelDeployment) []corev1.EnvVar {
	modelDir := fmt.Sprintf("%s/%s", nfsBaseMountPath, modelPath(md.Spec.ModelName))
	envVars := []corev1.EnvVar{
		{Name: "PODSTACK_MODEL_PATH", Value: modelDir},
		{Name: "PODSTACK_MODEL_NAME", Value: md.Spec.ModelName},
		{Name: "PODSTACK_RUNTIME", Value: md.Spec.Runtime},
		{Name: "PODSTACK_AUTO_SNAPSHOT", Value: fmt.Sprintf("%t", md.Spec.Snapshot.AutoSnapshot)},
		{Name: "PODSTACK_SNAPSHOT_PATH", Value: snapshotMountPath},
		{Name: "NVIDIA_VISIBLE_DEVICES", Value: "all"},
		{Name: "NVIDIA_DRIVER_CAPABILITIES", Value: "compute,utility"},
	}
	if md.Spec.TenantRef != "" {
		envVars = append(envVars, corev1.EnvVar{Name: "PODSTACK_TENANT", Value: md.Spec.TenantRef})
	}
	return envVars
}

// gpuResources returns the resource requirements for the GPU container.
func gpuResources(md *v1.ModelDeployment) corev1.ResourceRequirements {
	gpuCount := md.Spec.GPU.Count
	if gpuCount < 1 {
		gpuCount = 1
	}
	return corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			"nvidia.com/gpu": *resource.NewQuantity(int64(gpuCount), resource.DecimalSI),
		},
		Requests: corev1.ResourceList{
			"nvidia.com/gpu": *resource.NewQuantity(int64(gpuCount), resource.DecimalSI),
		},
	}
}

// nfsModelVolume returns the NFS volume and volume mount for the model store.
// The entire NFS share is mounted at /models so that all sub-paths
// (base/, triton/, lora/, snapshots/) are accessible.
func nfsModelVolume() (corev1.Volume, corev1.VolumeMount) {
	vol := corev1.Volume{
		Name: nfsVolumeName,
		VolumeSource: corev1.VolumeSource{
			NFS: &corev1.NFSVolumeSource{
				Server:   nfsServer,
				Path:     nfsSharePath,
				ReadOnly: true,
			},
		},
	}
	mount := corev1.VolumeMount{
		Name:      nfsVolumeName,
		MountPath: "/models",
		ReadOnly:  true,
	}
	return vol, mount
}

// shmVolume returns an emptyDir volume backed by tmpfs for shared memory.
func shmVolume() (corev1.Volume, corev1.VolumeMount) {
	medium := corev1.StorageMediumMemory
	vol := corev1.Volume{
		Name: shmVolumeName,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{
				Medium: medium,
			},
		},
	}
	mount := corev1.VolumeMount{
		Name:      shmVolumeName,
		MountPath: "/dev/shm",
	}
	return vol, mount
}

// snapshotVolume returns the NFS volume and mount for snapshot storage.
func snapshotVolume() (corev1.Volume, corev1.VolumeMount) {
	vol := corev1.Volume{
		Name: snapshotVolumeName,
		VolumeSource: corev1.VolumeSource{
			NFS: &corev1.NFSVolumeSource{
				Server:   nfsServer,
				Path:     snapshotNFSPath,
				ReadOnly: false,
			},
		},
	}
	mount := corev1.VolumeMount{
		Name:      snapshotVolumeName,
		MountPath: snapshotMountPath,
	}
	return vol, mount
}

// snapshotAgentSidecar returns the snapshot agent sidecar container spec.
func snapshotAgentSidecar(md *v1.ModelDeployment) corev1.Container {
	return corev1.Container{
		Name:  "snapshot-agent",
		Image: snapshotAgentImage,
		Env: []corev1.EnvVar{
			{Name: "PODSTACK_MODEL_NAME", Value: md.Spec.ModelName},
			{Name: "PODSTACK_SNAPSHOT_PATH", Value: snapshotMountPath},
			{Name: "PODSTACK_AUTO_SNAPSHOT", Value: fmt.Sprintf("%t", md.Spec.Snapshot.AutoSnapshot)},
			{Name: "PODSTACK_WARMUP_REQUESTS", Value: fmt.Sprintf("%d", md.Spec.Snapshot.WarmupRequests)},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: snapshotVolumeName, MountPath: snapshotMountPath},
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("128Mi"),
			},
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("500m"),
				corev1.ResourceMemory: resource.MustParse("256Mi"),
			},
		},
	}
}

// newService creates a ClusterIP service targeting pods of the given ModelDeployment
// on the specified port.
func newService(md *v1.ModelDeployment, port int32, portName string) *corev1.Service {
	labels := commonLabels(md)
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      md.Name,
			Namespace: md.Namespace,
			Labels:    labels,
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: v1.GroupVersion.String(),
					Kind:       "ModelDeployment",
					Name:       md.Name,
					UID:        md.UID,
					Controller: boolPtr(true),
				},
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: labels,
			Type:     corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{
				{
					Name:       portName,
					Port:       port,
					TargetPort: intstr.FromInt32(port),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}
}

func boolPtr(b bool) *bool {
	return &b
}

func int32Ptr(i int32) *int32 {
	return &i
}

func int64Ptr(i int64) *int64 {
	return &i
}
