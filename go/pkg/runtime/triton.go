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
	// tritonImage is the default NVIDIA Triton Inference Server image.
	tritonImage = "nvcr.io/nvidia/tritonserver:24.05-py3"

	// tritonHTTPPort is the HTTP endpoint port.
	tritonHTTPPort int32 = 8000

	// tritonGRPCPort is the gRPC endpoint port.
	tritonGRPCPort int32 = 8001

	// tritonMetricsPort is the Prometheus metrics port.
	tritonMetricsPort int32 = 8002

	// tritonHealthPath is the HTTP health check endpoint.
	tritonHealthPath = "/v2/health/ready"

	// tritonModelRepoBase is the NFS sub-path for Triton model repositories.
	tritonModelRepoBase = "/models/triton"
)

// TritonPodTemplate creates a complete Kubernetes PodTemplateSpec for running
// NVIDIA Triton Inference Server. The generated template includes:
//   - Model repository mounted from NFS at /models/triton/{model}
//   - Shared memory via emptyDir with Medium: Memory
//   - GPU resources from the ModelDeployment spec
//   - HAMi annotations for fractional GPU scheduling
//   - Triton CLI arguments: --model-repository, --allow-gpu-metrics, etc.
//   - HTTP health and readiness probes on /v2/health/ready
//   - Snapshot volume mount and agent sidecar when snapshots are enabled
//   - Support for ONNX, TensorRT, PyTorch, and TensorFlow backends
//   - Standard Podstack environment variables
func TritonPodTemplate(md *v1.ModelDeployment) *corev1.PodTemplateSpec {
	labels := commonLabels(md)
	annotations := hamiAnnotations(md)

	modelRepoPath := fmt.Sprintf("%s/%s", tritonModelRepoBase, modelPath(md.Spec.ModelName))

	// Build Triton command-line arguments.
	args := []string{
		"tritonserver",
		"--model-repository=" + modelRepoPath,
		"--allow-gpu-metrics=true",
		"--allow-metrics=true",
		"--metrics-port=" + strconv.Itoa(int(tritonMetricsPort)),
		"--http-port=" + strconv.Itoa(int(tritonHTTPPort)),
		"--grpc-port=" + strconv.Itoa(int(tritonGRPCPort)),
		"--strict-model-config=false",
	}

	// Append user-provided runtime arguments.
	args = append(args, md.Spec.RuntimeArgs...)

	// Assemble volumes and mounts.
	nfsVol, nfsMount := nfsModelVolume()
	shmVol, shmMount := shmVolume()

	volumes := []corev1.Volume{nfsVol, shmVol}
	mounts := []corev1.VolumeMount{nfsMount, shmMount}

	// Snapshot volume and sidecar.
	var sidecarContainers []corev1.Container
	if md.Spec.Snapshot.Enabled {
		snapVol, snapMount := snapshotVolume()
		volumes = append(volumes, snapVol)
		mounts = append(mounts, snapMount)
		sidecarContainers = append(sidecarContainers, snapshotAgentSidecar(md))
	}

	// Main Triton container.
	mainContainer := corev1.Container{
		Name:    "triton",
		Image:   tritonImage,
		Command: []string{"/bin/bash", "-c"},
		Args:    []string{joinArgs(args)},
		Ports: []corev1.ContainerPort{
			{Name: "http", ContainerPort: tritonHTTPPort, Protocol: corev1.ProtocolTCP},
			{Name: "grpc", ContainerPort: tritonGRPCPort, Protocol: corev1.ProtocolTCP},
			{Name: "metrics", ContainerPort: tritonMetricsPort, Protocol: corev1.ProtocolTCP},
		},
		Env:          commonEnvVars(md),
		Resources:    gpuResources(md),
		VolumeMounts: mounts,
		LivenessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: tritonHealthPath,
					Port: intstr.FromInt32(tritonHTTPPort),
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
					Path: tritonHealthPath,
					Port: intstr.FromInt32(tritonHTTPPort),
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
					Path: tritonHealthPath,
					Port: intstr.FromInt32(tritonHTTPPort),
				},
			},
			InitialDelaySeconds: 10,
			PeriodSeconds:       10,
			TimeoutSeconds:      5,
			FailureThreshold:    60,
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
			Affinity: gpuAffinity(md),
		},
	}
}

// TritonServiceForModel creates a ClusterIP Service that routes traffic to
// Triton pods for the given ModelDeployment. The service exposes HTTP, gRPC,
// and Prometheus metrics ports.
func TritonServiceForModel(md *v1.ModelDeployment) *corev1.Service {
	labels := commonLabels(md)
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      md.Name,
			Namespace: md.Namespace,
			Labels:    labels,
			Annotations: map[string]string{
				"podstack.io/runtime": v1.RuntimeTriton,
			},
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
					Name:       "http",
					Port:       tritonHTTPPort,
					TargetPort: intstr.FromInt32(tritonHTTPPort),
					Protocol:   corev1.ProtocolTCP,
				},
				{
					Name:       "grpc",
					Port:       tritonGRPCPort,
					TargetPort: intstr.FromInt32(tritonGRPCPort),
					Protocol:   corev1.ProtocolTCP,
				},
				{
					Name:       "metrics",
					Port:       tritonMetricsPort,
					TargetPort: intstr.FromInt32(tritonMetricsPort),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}
}

// joinArgs joins a slice of strings with spaces, suitable for passing to
// bash -c as a single command string.
func joinArgs(args []string) string {
	result := ""
	for i, a := range args {
		if i > 0 {
			result += " "
		}
		result += a
	}
	return result
}
