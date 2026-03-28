package runtime

import (
	"fmt"
	"strconv"

	v1 "github.com/podstack/serverless/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	// vllmImage is the default vLLM inference server image.
	vllmImage = "vllm/vllm-openai:latest"

	// vllmPort is the HTTP port vLLM serves on.
	vllmPort int32 = 8000

	// vllmHealthPath is the health check endpoint.
	vllmHealthPath = "/health"
)

// VLLMPodTemplate creates a complete Kubernetes PodTemplateSpec for running
// vLLM-based large language model inference. The generated template includes:
//   - Model weights mounted from NFS at /models/base/{model}
//   - Shared memory via emptyDir with Medium: Memory for PyTorch IPC
//   - GPU resources from the ModelDeployment spec
//   - HAMi annotations for fractional GPU scheduling
//   - vLLM CLI arguments: --model, --enable-lora, --max-model-len, etc.
//   - HTTP health and readiness probes on /health
//   - Snapshot volume mount and agent sidecar when snapshots are enabled
//   - Standard Podstack environment variables
func VLLMPodTemplate(md *v1.ModelDeployment) *corev1.PodTemplateSpec {
	labels := commonLabels(md)
	annotations := hamiAnnotations(md)

	modelDir := fmt.Sprintf("%s/%s", nfsBaseMountPath, modelPath(md.Spec.ModelName))

	// Build vLLM command-line arguments.
	args := []string{
		"--model", modelDir,
		"--host", "0.0.0.0",
		"--port", strconv.Itoa(int(vllmPort)),
	}

	// Enable LoRA multiplexing when configured.
	if md.Spec.LoRA != nil {
		args = append(args, "--enable-lora")
		if md.Spec.LoRA.MaxAdapters > 0 {
			args = append(args, "--max-loras", strconv.Itoa(int(md.Spec.LoRA.MaxAdapters)))
		}
		args = append(args, "--lora-modules")
		for _, ref := range md.Spec.LoRA.AdapterRefs {
			loraPath := fmt.Sprintf("%s/%s", nfsLoraMountPath, ref)
			args = append(args, fmt.Sprintf("%s=%s", ref, loraPath))
		}
	}

	// GPU tensor parallelism for multi-GPU deployments.
	if md.Spec.GPU.Count > 1 {
		args = append(args, "--tensor-parallel-size", strconv.Itoa(int(md.Spec.GPU.Count)))
	}

	// Append user-provided runtime arguments.
	args = append(args, md.Spec.RuntimeArgs...)

	// Assemble volumes and mounts.
	nfsVol, nfsMount := nfsModelVolume(nfsBaseMountPath)
	shmVol, shmMount := shmVolume()

	volumes := []corev1.Volume{nfsVol, shmVol}
	mounts := []corev1.VolumeMount{nfsMount, shmMount}

	// LoRA volume mount (separate NFS sub-path).
	if md.Spec.LoRA != nil && len(md.Spec.LoRA.AdapterRefs) > 0 {
		loraVol := corev1.Volume{
			Name: "lora-store",
			VolumeSource: corev1.VolumeSource{
				NFS: &corev1.NFSVolumeSource{
					Server:   nfsServer,
					Path:     nfsSharePath + "/lora",
					ReadOnly: true,
				},
			},
		}
		loraMount := corev1.VolumeMount{
			Name:      "lora-store",
			MountPath: nfsLoraMountPath,
			ReadOnly:  true,
		}
		volumes = append(volumes, loraVol)
		mounts = append(mounts, loraMount)
	}

	// Snapshot volume and sidecar.
	var initContainers []corev1.Container
	var sidecarContainers []corev1.Container
	if md.Spec.Snapshot.Enabled {
		snapVol, snapMount := snapshotVolume()
		volumes = append(volumes, snapVol)
		mounts = append(mounts, snapMount)
		sidecarContainers = append(sidecarContainers, snapshotAgentSidecar(md))
	}

	// Main vLLM container.
	mainContainer := corev1.Container{
		Name:         "vllm",
		Image:        vllmImage,
		Args:         args,
		Ports:        []corev1.ContainerPort{{Name: "http", ContainerPort: vllmPort, Protocol: corev1.ProtocolTCP}},
		Env:          commonEnvVars(md),
		Resources:    gpuResources(md),
		VolumeMounts: mounts,
		LivenessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: vllmHealthPath,
					Port: intstr.FromInt32(vllmPort),
				},
			},
			InitialDelaySeconds: 120,
			PeriodSeconds:       30,
			TimeoutSeconds:      5,
			FailureThreshold:    3,
		},
		ReadinessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: vllmHealthPath,
					Port: intstr.FromInt32(vllmPort),
				},
			},
			InitialDelaySeconds: 30,
			PeriodSeconds:       10,
			TimeoutSeconds:      5,
			FailureThreshold:    3,
			SuccessThreshold:    1,
		},
		StartupProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: vllmHealthPath,
					Port: intstr.FromInt32(vllmPort),
				},
			},
			InitialDelaySeconds: 10,
			PeriodSeconds:       10,
			TimeoutSeconds:      5,
			FailureThreshold:    30,
		},
	}

	containers := []corev1.Container{mainContainer}
	containers = append(containers, sidecarContainers...)

	return &corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: corev1.PodSpec{
			InitContainers:                initContainers,
			Containers:                    containers,
			Volumes:                       volumes,
			RestartPolicy:                 corev1.RestartPolicyAlways,
			TerminationGracePeriodSeconds: int64Ptr(30),
			SecurityContext: &corev1.PodSecurityContext{
				RunAsNonRoot: boolPtr(true),
				RunAsUser:    int64Ptr(1000),
				FSGroup:      int64Ptr(1000),
			},
			Tolerations: []corev1.Toleration{
				{
					Key:      "nvidia.com/gpu",
					Operator: corev1.TolerationOpExists,
					Effect:   corev1.TaintEffectNoSchedule,
				},
			},
			// Prefer scheduling on nodes with the specified GPU type.
			Affinity: gpuAffinity(md),
		},
	}
}

// VLLMServiceForModel creates a ClusterIP Service that routes traffic to vLLM
// pods for the given ModelDeployment.
func VLLMServiceForModel(md *v1.ModelDeployment) *corev1.Service {
	svc := newService(md, vllmPort, "http")
	svc.Annotations = map[string]string{
		"podstack.io/runtime": v1.RuntimeVLLM,
	}
	return svc
}

// gpuAffinity returns node affinity rules to prefer nodes with the requested GPU type.
func gpuAffinity(md *v1.ModelDeployment) *corev1.Affinity {
	if md.Spec.GPU.Type == "" {
		return nil
	}
	return &corev1.Affinity{
		NodeAffinity: &corev1.NodeAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
				{
					Weight: 100,
					Preference: corev1.NodeSelectorTerm{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      "nvidia.com/gpu.product",
								Operator: corev1.NodeSelectorOpIn,
								Values:   []string{md.Spec.GPU.Type},
							},
						},
					},
				},
			},
		},
	}
}

