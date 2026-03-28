package runtime

import (
	"fmt"

	v1 "github.com/podstack/serverless/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	// genericDefaultPort is the default port for generic containers.
	genericDefaultPort int32 = 8080

	// genericHealthPath is the default health check endpoint.
	genericHealthPath = "/health"
)

// GenericPodTemplate creates a complete Kubernetes PodTemplateSpec for running
// any custom Docker container with CUDA support. The generated template uses
// the user's specified Image from ModelDeploymentSpec and includes:
//   - NFS model mount at /models/base/{model}
//   - Shared memory via emptyDir with Medium: Memory
//   - GPU resources from the ModelDeployment spec
//   - HAMi annotations for fractional GPU scheduling
//   - HTTP health and readiness probes on /health
//   - Snapshot agent sidecar when snapshots are enabled
//   - User-provided RuntimeArgs passed as container arguments
//   - Standard Podstack environment variables
func GenericPodTemplate(md *v1.ModelDeployment) *corev1.PodTemplateSpec {
	labels := commonLabels(md)
	annotations := hamiAnnotations(md)

	// Determine the container image. For generic runtime, the user must
	// specify the image in the spec; fall back to a placeholder if missing.
	image := md.Spec.Image
	if image == "" {
		image = "nvidia/cuda:12.4.1-runtime-ubuntu22.04"
	}

	// Assemble volumes and mounts.
	nfsVol, nfsMount := nfsModelVolume(nfsBaseMountPath)
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

	// Build environment variables with the model path injected.
	envVars := commonEnvVars(md)
	envVars = append(envVars, corev1.EnvVar{
		Name:  "MODEL_PATH",
		Value: fmt.Sprintf("%s/%s", nfsBaseMountPath, modelPath(md.Spec.ModelName)),
	})

	// Main inference container using the user-specified image.
	mainContainer := corev1.Container{
		Name:         "inference",
		Image:        image,
		Args:         md.Spec.RuntimeArgs,
		Ports:        []corev1.ContainerPort{{Name: "http", ContainerPort: genericDefaultPort, Protocol: corev1.ProtocolTCP}},
		Env:          envVars,
		Resources:    gpuResources(md),
		VolumeMounts: mounts,
		LivenessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: genericHealthPath,
					Port: intstr.FromInt32(genericDefaultPort),
				},
			},
			InitialDelaySeconds: 60,
			PeriodSeconds:       30,
			TimeoutSeconds:      5,
			FailureThreshold:    3,
		},
		ReadinessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: genericHealthPath,
					Port: intstr.FromInt32(genericDefaultPort),
				},
			},
			InitialDelaySeconds: 15,
			PeriodSeconds:       10,
			TimeoutSeconds:      5,
			FailureThreshold:    3,
			SuccessThreshold:    1,
		},
		StartupProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: genericHealthPath,
					Port: intstr.FromInt32(genericDefaultPort),
				},
			},
			InitialDelaySeconds: 5,
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

// GenericServiceForModel creates a ClusterIP Service that routes traffic to
// the generic inference container pods for the given ModelDeployment.
func GenericServiceForModel(md *v1.ModelDeployment) *corev1.Service {
	svc := newService(md, genericDefaultPort, "http")
	svc.Annotations = map[string]string{
		"podstack.io/runtime": v1.RuntimeGeneric,
	}
	return svc
}
