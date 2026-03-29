package snapshot

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "github.com/podstack/serverless/api/v1"
)

const (
	// NFS configuration for snapshot storage.
	snapshotNFSServer = "192.168.29.5"
	snapshotNFSPath   = "/mnt/tank/podstack/serverless-snapshots"

	// copyPodTimeout is how long to wait for the checkpoint copy pod to complete.
	copyPodTimeout = 10 * time.Minute
)

// Manager orchestrates CUDA snapshot creation and restoration. It coordinates
// between the kubelet checkpoint API (CPU/memory state via CRIU), cuda-checkpoint
// (GPU state), the blobstore (persistent NFS storage), and the Kubernetes API
// (Snapshot CRs).
type Manager struct {
	k8sClient client.Client
	clientset kubernetes.Interface // for kubelet checkpoint API calls
	blobstore *Blobstore
	cudaChkpt *CUDACheckpoint
	criu      *CRIU
	log       logr.Logger
}

// NewManager creates a new snapshot Manager with all required sub-components.
// The restConfig is used to create a kubernetes.Clientset for kubelet API calls.
func NewManager(k8sClient client.Client, restConfig *rest.Config, blobstoreBasePath string, log logr.Logger) *Manager {
	managerLog := log.WithName("snapshot-manager")

	var clientset kubernetes.Interface
	if restConfig != nil {
		cs, err := kubernetes.NewForConfig(restConfig)
		if err != nil {
			managerLog.Error(err, "failed to create kubernetes clientset; kubelet checkpoint API unavailable")
		} else {
			clientset = cs
		}
	}

	return &Manager{
		k8sClient: k8sClient,
		clientset: clientset,
		blobstore: NewBlobstore(blobstoreBasePath, managerLog),
		cudaChkpt: NewCUDACheckpoint(managerLog),
		criu:      NewCRIU(managerLog),
		log:       managerLog,
	}
}

// snapshotID builds a deterministic snapshot identifier from a model name and
// GPU type. The result is safe for use as a filesystem path component.
func snapshotID(modelName, gpuType string) string {
	sanitized := strings.NewReplacer("/", "--", ":", "-", " ", "_").Replace(modelName)
	return fmt.Sprintf("%s_%s", sanitized, gpuType)
}

// snapshotCRName returns a Kubernetes-safe name for a Snapshot custom resource.
func snapshotCRName(modelName, gpuType string) string {
	sanitized := strings.NewReplacer("/", "-", ":", "-", " ", "-", "_", "-").Replace(
		strings.ToLower(modelName),
	)
	return fmt.Sprintf("snap-%s-%s", sanitized, strings.ToLower(gpuType))
}

// getPIDFromPod extracts the main inference process PID from the pod. It checks
// for a well-known annotation first, then falls back to PID 1 inside the
// container namespace.
func getPIDFromPod(pod *corev1.Pod) (int, error) {
	if pod.Annotations != nil {
		if pidStr, ok := pod.Annotations["podstack.io/main-pid"]; ok {
			pid, err := strconv.Atoi(pidStr)
			if err != nil {
				return 0, fmt.Errorf("invalid pid annotation value %q: %w", pidStr, err)
			}
			return pid, nil
		}
	}
	// Default: assume the inference process is PID 1 inside the container.
	return 1, nil
}

// getContainerID extracts the container ID for the first container in the pod.
// The container runtime prefix (e.g. "containerd://") is stripped.
func getContainerID(pod *corev1.Pod) (string, error) {
	if len(pod.Status.ContainerStatuses) == 0 {
		return "", fmt.Errorf("pod %s/%s has no container statuses", pod.Namespace, pod.Name)
	}
	fullID := pod.Status.ContainerStatuses[0].ContainerID
	if fullID == "" {
		return "", fmt.Errorf("pod %s/%s container has empty container ID", pod.Namespace, pod.Name)
	}
	// Strip the runtime prefix (e.g. "containerd://abc123" -> "abc123").
	parts := strings.SplitN(fullID, "://", 2)
	if len(parts) == 2 {
		return parts[1], nil
	}
	return fullID, nil
}

// CreateSnapshot orchestrates the full snapshot creation flow for a model.
//
// It uses direct CRIU dump from a privileged helper pod (bypassing
// nvidia-container-runtime which doesn't support CRIU checkpointing of GPU
// containers). cuda-checkpoint locks/unlocks GPU state before/after CRIU.
//
// Flow:
//  1. Create Snapshot CR in Creating phase
//  2. Get container's host PID via containerd state
//  3. cuda-checkpoint --action lock (pause CUDA, release GPU)
//  4. Direct CRIU dump via privileged helper pod with nsenter
//  5. cuda-checkpoint --action unlock (resume CUDA)
//  6. Copy CRIU dump from node to NFS
//  7. Update the Snapshot CR to Ready
//
// Falls back to a placeholder snapshot if any step fails.
func (m *Manager) CreateSnapshot(ctx context.Context, pod *corev1.Pod, md *v1.ModelDeployment) (*v1.Snapshot, error) {
	modelName := md.Spec.ModelName
	gpuType := md.Spec.GPU.Type
	snapID := snapshotID(modelName, gpuType)
	crName := snapshotCRName(modelName, gpuType)

	m.log.Info("starting snapshot creation",
		"model", modelName,
		"gpuType", gpuType,
		"pod", pod.Name,
		"namespace", pod.Namespace,
		"snapshotID", snapID,
	)

	// Initialise the Snapshot CR in Creating phase.
	now := metav1.Now()
	var cudaVersion, driverVersion string
	if pod.Annotations != nil {
		cudaVersion = pod.Annotations["podstack.io/cuda-version"]
		driverVersion = pod.Annotations["podstack.io/driver-version"]
	}
	if cudaVersion == "" {
		cudaVersion = "unknown"
	}
	if driverVersion == "" {
		driverVersion = "unknown"
	}

	sanitizedModel := strings.ReplaceAll(modelName, "/", "--")
	storagePath := fmt.Sprintf("%s/%s", snapshotNFSPath, snapID)

	snapshot := &v1.Snapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:      crName,
			Namespace: pod.Namespace,
			Labels: map[string]string{
				"podstack.io/model":    sanitizedModel,
				"podstack.io/gpu-type": gpuType,
			},
		},
		Spec: v1.SnapshotSpec{
			ModelDeploymentRef: md.Name,
			StoragePath:        storagePath,
			GPUType:            gpuType,
			CUDAVersion:        cudaVersion,
			DriverVersion:      driverVersion,
		},
		Status: v1.SnapshotStatus{
			Phase:     v1.SnapshotPhaseCreating,
			CreatedAt: &now,
		},
	}

	if err := m.createOrUpdateSnapshotCR(ctx, snapshot); err != nil {
		return nil, fmt.Errorf("failed to create Snapshot CR: %w", err)
	}

	// Try direct CRIU checkpoint (bypasses nvidia-container-runtime).
	// Flow: CUDA lock → direct CRIU dump → CUDA unlock → copy to NFS.
	if m.clientset != nil {
		containerID, cidErr := getContainerIDByName(pod, "vllm")
		if cidErr != nil {
			m.log.Error(cidErr, "failed to get vllm container ID")
		} else if err := m.runCUDAAction(ctx, pod, containerID, "lock"); err != nil {
			m.log.Error(err, "failed to lock CUDA context, falling back to placeholder")
		} else {
			m.log.Info("CUDA context locked, running direct CRIU dump")

			dumpDir := fmt.Sprintf("/tmp/criu-dump-%s", snapID)
			dumpErr := m.runCRIUDump(ctx, pod, containerID, dumpDir)

			// Always resume CUDA, even if dump failed.
			if unlockErr := m.runCUDAAction(ctx, pod, containerID, "unlock"); unlockErr != nil {
				m.log.Error(unlockErr, "failed to unlock CUDA context after dump")
			} else {
				m.log.Info("CUDA context unlocked")
			}

			if dumpErr != nil {
				m.log.Error(dumpErr, "direct CRIU dump failed after CUDA lock")
			} else {
				m.log.Info("CRIU dump completed, copying to NFS",
					"dumpDir", dumpDir,
					"node", pod.Spec.NodeName,
				)

				// Copy the CRIU dump dir from the node to NFS.
				if cpErr := m.copyCRIUDumpToNFS(ctx, pod.Spec.NodeName, pod.Namespace, dumpDir, snapID); cpErr != nil {
					m.log.Error(cpErr, "failed to copy CRIU dump to NFS, falling back to placeholder")
				} else {
					snapshot.Status.Phase = v1.SnapshotPhaseReady
					snapshot.Status.Message = fmt.Sprintf("CRIU checkpoint created (node: %s)", pod.Spec.NodeName)

					if err := m.createOrUpdateSnapshotCR(ctx, snapshot); err != nil {
						return nil, fmt.Errorf("failed to update Snapshot CR to Ready: %w", err)
					}
					if err := m.updateSnapshotStatus(ctx, snapshot); err != nil {
						return nil, fmt.Errorf("failed to update Snapshot status to Ready: %w", err)
					}

					m.log.Info("snapshot creation complete (direct CRIU dump)",
						"snapshotID", snapID,
						"crName", crName,
						"storagePath", storagePath,
					)
					return snapshot, nil
				}
			}
		}
	}

	// Fallback: create a placeholder snapshot so the pipeline can progress.
	// The standby pod pattern still provides value without a real checkpoint.
	m.log.Info("creating placeholder snapshot (CRIU checkpoint unavailable or failed)",
		"snapshotID", snapID,
		"crName", crName,
	)

	snapshot.Status.Phase = v1.SnapshotPhaseReady
	snapshot.Status.Message = "Placeholder snapshot (real checkpoint unavailable)"

	if err := m.createOrUpdateSnapshotCR(ctx, snapshot); err != nil {
		return nil, fmt.Errorf("failed to update Snapshot CR to Ready: %w", err)
	}
	if err := m.updateSnapshotStatus(ctx, snapshot); err != nil {
		return nil, fmt.Errorf("failed to update Snapshot status to Ready: %w", err)
	}

	return snapshot, nil
}

// kubeletCheckpoint calls the Kubernetes kubelet checkpoint API via the API
// server proxy. This triggers CRIU on the node (via containerd) to checkpoint
// the specified container.
//
// API: POST /api/v1/nodes/{nodeName}/proxy/checkpoint/{namespace}/{pod}/{container}
// Response: {"items": ["/var/lib/kubelet/checkpoints/checkpoint-<pod>_<ns>-<container>-<timestamp>.tar"]}
func (m *Manager) kubeletCheckpoint(ctx context.Context, pod *corev1.Pod, containerName string) (string, error) {
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		return "", fmt.Errorf("pod %s/%s has no nodeName assigned", pod.Namespace, pod.Name)
	}

	m.log.Info("calling kubelet checkpoint API",
		"node", nodeName,
		"namespace", pod.Namespace,
		"pod", pod.Name,
		"container", containerName,
	)

	result := m.clientset.CoreV1().RESTClient().Post().
		Resource("nodes").
		Name(nodeName).
		SubResource("proxy", "checkpoint", pod.Namespace, pod.Name, containerName).
		Do(ctx)

	raw, err := result.Raw()
	if err != nil {
		return "", fmt.Errorf("kubelet checkpoint API call failed for %s/%s/%s on node %s: %w",
			pod.Namespace, pod.Name, containerName, nodeName, err)
	}

	var response struct {
		Items []string `json:"items"`
	}
	if err := json.Unmarshal(raw, &response); err != nil {
		return "", fmt.Errorf("failed to parse kubelet checkpoint response: %w (raw: %s)", err, string(raw))
	}

	if len(response.Items) == 0 {
		return "", fmt.Errorf("kubelet checkpoint API returned no checkpoint paths (raw: %s)", string(raw))
	}

	checkpointPath := response.Items[0]
	m.log.Info("kubelet checkpoint created",
		"path", checkpointPath,
		"node", nodeName,
	)
	return checkpointPath, nil
}

// copyCheckpointToNFS creates a short-lived Pod on the same node as the
// checkpoint to copy the tar from /var/lib/kubelet/checkpoints/ to NFS.
func (m *Manager) copyCheckpointToNFS(ctx context.Context, nodeName, namespace, checkpointPath, snapID string) error {
	checkpointFile := filepath.Base(checkpointPath)
	// Truncate snapID for pod name (must be <=63 chars, DNS-safe).
	safeName := strings.ToLower(strings.NewReplacer("/", "-", "_", "-").Replace(snapID))
	if len(safeName) > 40 {
		safeName = safeName[:40]
	}
	podName := fmt.Sprintf("chkpt-copy-%s", safeName)

	m.log.Info("creating checkpoint copy pod",
		"podName", podName,
		"node", nodeName,
		"source", checkpointFile,
		"destDir", snapID,
	)

	copyPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels: map[string]string{
				"podstack.io/role": "checkpoint-copy",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			Containers: []corev1.Container{
				{
					Name:  "copy",
					Image: "busybox:latest",
					Command: []string{
						"/bin/sh", "-c",
						fmt.Sprintf("mkdir -p /nfs/%s && cp /checkpoints/%s /nfs/%s/checkpoint.tar && echo done",
							snapID, checkpointFile, snapID),
					},
					VolumeMounts: []corev1.VolumeMount{
						{Name: "kubelet-checkpoints", MountPath: "/checkpoints", ReadOnly: true},
						{Name: "nfs-snapshots", MountPath: "/nfs"},
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "kubelet-checkpoints",
					VolumeSource: corev1.VolumeSource{
						HostPath: &corev1.HostPathVolumeSource{
							Path: "/var/lib/kubelet/checkpoints",
						},
					},
				},
				{
					Name: "nfs-snapshots",
					VolumeSource: corev1.VolumeSource{
						NFS: &corev1.NFSVolumeSource{
							Server: snapshotNFSServer,
							Path:   snapshotNFSPath,
						},
					},
				},
			},
			RestartPolicy: corev1.RestartPolicyNever,
		},
	}

	// Delete any leftover copy pod from a previous attempt.
	existing := &corev1.Pod{}
	existingKey := types.NamespacedName{Name: podName, Namespace: namespace}
	if err := m.k8sClient.Get(ctx, existingKey, existing); err == nil {
		_ = m.k8sClient.Delete(ctx, existing)
		// Wait briefly for deletion.
		time.Sleep(2 * time.Second)
	}

	if err := m.k8sClient.Create(ctx, copyPod); err != nil {
		return fmt.Errorf("failed to create checkpoint copy pod: %w", err)
	}

	// Wait for the copy pod to complete.
	deadline := time.Now().Add(copyPodTimeout)
	for time.Now().Before(deadline) {
		fetchedPod := &corev1.Pod{}
		if err := m.k8sClient.Get(ctx, existingKey, fetchedPod); err != nil {
			return fmt.Errorf("failed to get copy pod status: %w", err)
		}

		switch fetchedPod.Status.Phase {
		case corev1.PodSucceeded:
			m.log.Info("checkpoint copy pod completed successfully", "podName", podName)
			_ = m.k8sClient.Delete(ctx, fetchedPod)
			return nil
		case corev1.PodFailed:
			_ = m.k8sClient.Delete(ctx, fetchedPod)
			return fmt.Errorf("checkpoint copy pod failed (phase: %s)", fetchedPod.Status.Phase)
		}

		time.Sleep(3 * time.Second)
	}

	// Timeout — clean up and return error.
	_ = m.k8sClient.Delete(ctx, copyPod)
	return fmt.Errorf("checkpoint copy pod timed out after %v", copyPodTimeout)
}

// getContainerIDByName extracts the container ID for a named container in the pod.
func getContainerIDByName(pod *corev1.Pod, containerName string) (string, error) {
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name == containerName {
			fullID := cs.ContainerID
			if fullID == "" {
				return "", fmt.Errorf("container %s in pod %s/%s has empty container ID", containerName, pod.Namespace, pod.Name)
			}
			parts := strings.SplitN(fullID, "://", 2)
			if len(parts) == 2 {
				return parts[1], nil
			}
			return fullID, nil
		}
	}
	return "", fmt.Errorf("container %s not found in pod %s/%s", containerName, pod.Namespace, pod.Name)
}

// runCUDAAction creates a privileged helper Pod on the same node as the target
// pod to run cuda-checkpoint with the specified action (lock/unlock) against
// the vLLM container's host PID.
//
// The helper pod uses hostPID to access host processes and reads the container's
// init PID from containerd's state file.
func (m *Manager) runCUDAAction(ctx context.Context, pod *corev1.Pod, containerID, action string) error {
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		return fmt.Errorf("pod %s/%s has no nodeName", pod.Namespace, pod.Name)
	}

	safePodName := strings.ToLower(strings.NewReplacer("/", "-", "_", "-").Replace(pod.Name))
	if len(safePodName) > 40 {
		safePodName = safePodName[:40]
	}
	helperName := fmt.Sprintf("cuda-%s-%s", action, safePodName)

	m.log.Info("creating CUDA helper pod",
		"action", action,
		"helperPod", helperName,
		"node", nodeName,
		"containerID", containerID[:12],
	)

	// The script uses nsenter to run cuda-checkpoint in the host's mount
	// namespace, giving access to all host libraries (glibc, NVIDIA drivers).
	// This avoids the musl/glibc incompatibility that occurs with busybox.
	// Note: --timeout is only valid for --action lock.
	timeoutFlag := ""
	if action == "lock" {
		timeoutFlag = "--timeout 30000"
	}
	script := fmt.Sprintf(`set -e
echo "Entering host mount namespace via nsenter..."
nsenter --target 1 --mount -- bash -c '
  set -e
  INIT_PID_FILE="/run/k3s/containerd/io.containerd.runtime.v2.task/k8s.io/%s/init.pid"
  if [ ! -f "$INIT_PID_FILE" ]; then
      echo "ERROR: init.pid not found at $INIT_PID_FILE"
      exit 1
  fi
  HOST_PID=$(cat "$INIT_PID_FILE")
  echo "Found host PID: $HOST_PID for container %s"
  echo "Running: cuda-checkpoint --action %s --pid $HOST_PID %s"
  /usr/local/bin/cuda-checkpoint --action %s --pid $HOST_PID %s
  echo "cuda-checkpoint %s completed successfully"
'
`, containerID, containerID[:12], action, timeoutFlag, action, timeoutFlag, action)

	privileged := true
	helperPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      helperName,
			Namespace: pod.Namespace,
			Labels: map[string]string{
				"podstack.io/role": "cuda-helper",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			HostPID:  true,
			Containers: []corev1.Container{
				{
					Name:    "cuda-action",
					Image:   "ubuntu:22.04",
					Command: []string{"/bin/bash", "-c", script},
					SecurityContext: &corev1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			RestartPolicy: corev1.RestartPolicyNever,
		},
	}

	// Clean up any leftover helper pod.
	helperKey := types.NamespacedName{Name: helperName, Namespace: pod.Namespace}
	existing := &corev1.Pod{}
	if err := m.k8sClient.Get(ctx, helperKey, existing); err == nil {
		_ = m.k8sClient.Delete(ctx, existing)
		time.Sleep(3 * time.Second)
	}

	if err := m.k8sClient.Create(ctx, helperPod); err != nil {
		return fmt.Errorf("failed to create CUDA %s helper pod: %w", action, err)
	}

	// Wait for the helper pod to complete.
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		fetched := &corev1.Pod{}
		if err := m.k8sClient.Get(ctx, helperKey, fetched); err != nil {
			return fmt.Errorf("failed to get CUDA helper pod: %w", err)
		}

		switch fetched.Status.Phase {
		case corev1.PodSucceeded:
			m.log.Info("CUDA helper pod completed", "action", action, "pod", helperName)
			_ = m.k8sClient.Delete(ctx, fetched)
			return nil
		case corev1.PodFailed:
			// Try to get logs for debugging.
			_ = m.k8sClient.Delete(ctx, fetched)
			return fmt.Errorf("CUDA %s helper pod failed", action)
		}

		time.Sleep(2 * time.Second)
	}

	_ = m.k8sClient.Delete(ctx, helperPod)
	return fmt.Errorf("CUDA %s helper pod timed out", action)
}

// runCRIUDump creates a privileged helper Pod on the same node as the target
// pod to run CRIU dump directly against the vLLM process's host PID. This
// bypasses nvidia-container-runtime (which doesn't support CRIU for GPU
// containers) by running criu directly via nsenter in the host namespace.
//
// The dump output is written to dumpDir on the host filesystem.
func (m *Manager) runCRIUDump(ctx context.Context, pod *corev1.Pod, containerID, dumpDir string) error {
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		return fmt.Errorf("pod %s/%s has no nodeName", pod.Namespace, pod.Name)
	}

	safePodName := strings.ToLower(strings.NewReplacer("/", "-", "_", "-").Replace(pod.Name))
	if len(safePodName) > 40 {
		safePodName = safePodName[:40]
	}
	helperName := fmt.Sprintf("criu-dump-%s", safePodName)

	m.log.Info("creating CRIU dump helper pod",
		"helperPod", helperName,
		"node", nodeName,
		"containerID", containerID[:12],
		"dumpDir", dumpDir,
	)

	// The script runs CRIU dump directly using nsenter to access the host's
	// filesystem and process namespace.
	//
	// Key: nvidia-container-runtime injects bind mounts (NVIDIA libs, configs)
	// into the container. CRIU must be told about the container root and these
	// external mounts via --root and --external flags.
	//
	// The script auto-discovers external bind mounts from /proc/<pid>/mountinfo
	// and passes them to CRIU so it doesn't try to serialize the backing store.
	script := fmt.Sprintf(`set -e
echo "Starting CRIU dump via nsenter..."
nsenter --target 1 --mount -- bash -c '
  set -e
  INIT_PID_FILE="/run/k3s/containerd/io.containerd.runtime.v2.task/k8s.io/%s/init.pid"
  if [ ! -f "$INIT_PID_FILE" ]; then
      echo "ERROR: init.pid not found at $INIT_PID_FILE"
      exit 1
  fi
  HOST_PID=$(cat "$INIT_PID_FILE")
  echo "Host PID: $HOST_PID for container %s"

  DUMP_DIR="%s"
  mkdir -p "$DUMP_DIR"

  # Discover external bind mounts from mountinfo.
  # These are nvidia-injected mounts that CRIU cannot serialize.
  # Format: --external mnt[<mount_id>]:<mount_point>
  EXT_MOUNTS=""
  while IFS=" " read -r mnt_id parent_id major_minor root mount_point rest; do
    # Skip the root mount (mount_id is typically 0 or the container root)
    if [ "$mount_point" = "/" ]; then
      continue
    fi
    EXT_MOUNTS="$EXT_MOUNTS --external mnt[$mnt_id]:$mount_point"
  done < /proc/$HOST_PID/mountinfo
  echo "Discovered external mounts for CRIU"

  echo "Running: criu dump -t $HOST_PID -D $DUMP_DIR --root /proc/$HOST_PID/root ..."
  criu dump \
    -t $HOST_PID \
    -D "$DUMP_DIR" \
    --root /proc/$HOST_PID/root \
    --manage-cgroups \
    --leave-running \
    --shell-job \
    --tcp-established \
    --file-locks \
    --ext-unix-sk \
    $EXT_MOUNTS \
    -v4 \
    --log-file dump.log \
    2>&1 || {
      echo "CRIU dump failed. Dump log (last 50 lines):"
      tail -50 "$DUMP_DIR/dump.log" 2>/dev/null || echo "(no dump log)"
      exit 1
    }

  echo "CRIU dump completed successfully"
  ls -la "$DUMP_DIR" | head -20
  echo "Total size: $(du -sh "$DUMP_DIR" | cut -f1)"
'
`, containerID, containerID[:12], dumpDir)

	privileged := true
	helperPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      helperName,
			Namespace: pod.Namespace,
			Labels: map[string]string{
				"podstack.io/role": "criu-dump",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			HostPID:  true,
			Containers: []corev1.Container{
				{
					Name:    "criu-dump",
					Image:   "ubuntu:22.04",
					Command: []string{"/bin/bash", "-c", script},
					SecurityContext: &corev1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			RestartPolicy: corev1.RestartPolicyNever,
		},
	}

	// Clean up any leftover helper pod.
	helperKey := types.NamespacedName{Name: helperName, Namespace: pod.Namespace}
	existing := &corev1.Pod{}
	if err := m.k8sClient.Get(ctx, helperKey, existing); err == nil {
		_ = m.k8sClient.Delete(ctx, existing)
		time.Sleep(3 * time.Second)
	}

	if err := m.k8sClient.Create(ctx, helperPod); err != nil {
		return fmt.Errorf("failed to create CRIU dump helper pod: %w", err)
	}

	// Wait for the helper pod to complete (CRIU dump can take a while for large processes).
	deadline := time.Now().Add(5 * time.Minute)
	for time.Now().Before(deadline) {
		fetched := &corev1.Pod{}
		if err := m.k8sClient.Get(ctx, helperKey, fetched); err != nil {
			return fmt.Errorf("failed to get CRIU dump helper pod: %w", err)
		}

		switch fetched.Status.Phase {
		case corev1.PodSucceeded:
			m.log.Info("CRIU dump helper pod completed", "pod", helperName)
			_ = m.k8sClient.Delete(ctx, fetched)
			return nil
		case corev1.PodFailed:
			_ = m.k8sClient.Delete(ctx, fetched)
			return fmt.Errorf("CRIU dump helper pod failed")
		}

		time.Sleep(3 * time.Second)
	}

	_ = m.k8sClient.Delete(ctx, helperPod)
	return fmt.Errorf("CRIU dump helper pod timed out after 5m")
}

// copyCRIUDumpToNFS creates a short-lived Pod on the same node to tar and
// copy the CRIU dump directory from the host to NFS storage.
func (m *Manager) copyCRIUDumpToNFS(ctx context.Context, nodeName, namespace, dumpDir, snapID string) error {
	safeName := strings.ToLower(strings.NewReplacer("/", "-", "_", "-").Replace(snapID))
	if len(safeName) > 40 {
		safeName = safeName[:40]
	}
	podName := fmt.Sprintf("chkpt-copy-%s", safeName)

	m.log.Info("creating CRIU dump copy pod",
		"podName", podName,
		"node", nodeName,
		"source", dumpDir,
		"destDir", snapID,
	)

	copyPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels: map[string]string{
				"podstack.io/role": "checkpoint-copy",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			Containers: []corev1.Container{
				{
					Name:  "copy",
					Image: "ubuntu:22.04",
					Command: []string{
						"/bin/bash", "-c",
						fmt.Sprintf(`set -e
mkdir -p /nfs/%s
echo "Tarring CRIU dump from /host-dump to /nfs/%s/checkpoint.tar"
tar -cf /nfs/%s/checkpoint.tar -C /host-dump .
echo "Copy done. Size: $(du -sh /nfs/%s/checkpoint.tar | cut -f1)"
`, snapID, snapID, snapID, snapID),
					},
					VolumeMounts: []corev1.VolumeMount{
						{Name: "host-dump", MountPath: "/host-dump", ReadOnly: true},
						{Name: "nfs-snapshots", MountPath: "/nfs"},
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "host-dump",
					VolumeSource: corev1.VolumeSource{
						HostPath: &corev1.HostPathVolumeSource{
							Path: dumpDir,
						},
					},
				},
				{
					Name: "nfs-snapshots",
					VolumeSource: corev1.VolumeSource{
						NFS: &corev1.NFSVolumeSource{
							Server: snapshotNFSServer,
							Path:   snapshotNFSPath,
						},
					},
				},
			},
			RestartPolicy: corev1.RestartPolicyNever,
		},
	}

	// Delete any leftover copy pod from a previous attempt.
	existingKey := types.NamespacedName{Name: podName, Namespace: namespace}
	existingPod := &corev1.Pod{}
	if err := m.k8sClient.Get(ctx, existingKey, existingPod); err == nil {
		_ = m.k8sClient.Delete(ctx, existingPod)
		time.Sleep(2 * time.Second)
	}

	if err := m.k8sClient.Create(ctx, copyPod); err != nil {
		return fmt.Errorf("failed to create CRIU dump copy pod: %w", err)
	}

	// Wait for the copy pod to complete.
	deadline := time.Now().Add(copyPodTimeout)
	for time.Now().Before(deadline) {
		fetchedPod := &corev1.Pod{}
		if err := m.k8sClient.Get(ctx, existingKey, fetchedPod); err != nil {
			return fmt.Errorf("failed to get copy pod status: %w", err)
		}

		switch fetchedPod.Status.Phase {
		case corev1.PodSucceeded:
			m.log.Info("CRIU dump copy pod completed successfully", "podName", podName)
			_ = m.k8sClient.Delete(ctx, fetchedPod)
			return nil
		case corev1.PodFailed:
			_ = m.k8sClient.Delete(ctx, fetchedPod)
			return fmt.Errorf("CRIU dump copy pod failed (phase: %s)", fetchedPod.Status.Phase)
		}

		time.Sleep(3 * time.Second)
	}

	_ = m.k8sClient.Delete(ctx, copyPod)
	return fmt.Errorf("CRIU dump copy pod timed out after %v", copyPodTimeout)
}

// RestoreToGPU restores a model from a snapshot:
//  1. Load snapshot archive from blobstore
//  2. Extract to a staging directory
//  3. CRIU restore process (CPU state)
//  4. cuda-checkpoint restore GPU state
//
// It returns the total time taken for the restore operation.
func (m *Manager) RestoreToGPU(ctx context.Context, snapshot *v1.Snapshot, pod *corev1.Pod) (time.Duration, error) {
	start := time.Now()
	modelRef := snapshot.Spec.ModelDeploymentRef
	gpuType := snapshot.Spec.GPUType

	// Derive the snapshot ID from the model label (which preserves the
	// original model name) rather than from the deployment ref.
	modelName := modelRef
	if snapshot.Labels != nil {
		if labelModel, ok := snapshot.Labels["podstack.io/model"]; ok {
			modelName = labelModel
		}
	}
	snapID := snapshotID(modelName, gpuType)

	m.log.Info("starting snapshot restore",
		"snapshotID", snapID,
		"model", modelName,
		"gpuType", gpuType,
		"pod", pod.Name,
	)

	// Update Snapshot CR to Restoring phase.
	snapshot.Status.Phase = v1.SnapshotPhaseRestoring
	snapshot.Status.Message = "Restore in progress"
	if err := m.updateSnapshotStatus(ctx, snapshot); err != nil {
		m.log.Error(err, "failed to update Snapshot CR to Restoring phase")
	}

	// Step 1: Load the snapshot archive from the blobstore.
	if !m.blobstore.Exists(snapID) {
		return 0, fmt.Errorf("snapshot archive not found in blobstore: %s", snapID)
	}

	reader, err := m.blobstore.Load(snapID)
	if err != nil {
		return 0, fmt.Errorf("failed to load snapshot from blobstore: %w", err)
	}
	defer reader.Close()

	// Step 2: Extract to a staging directory.
	stagingDir := fmt.Sprintf("/tmp/podstack-restore-%s", snapID)
	if err := os.MkdirAll(stagingDir, 0o755); err != nil {
		return 0, fmt.Errorf("failed to create restore staging dir: %w", err)
	}
	defer func() {
		if removeErr := os.RemoveAll(stagingDir); removeErr != nil {
			m.log.Error(removeErr, "failed to clean up restore staging directory", "path", stagingDir)
		}
	}()

	if err := extractTarArchive(reader, stagingDir); err != nil {
		return 0, fmt.Errorf("failed to extract snapshot archive: %w", err)
	}

	cpuDir := stagingDir + "/cpu"
	gpuDir := stagingDir + "/gpu"

	// Step 3: CRIU restore (CPU/memory state).
	m.log.Info("restoring CPU state via CRIU", "snapshotPath", cpuDir)
	restoredPID, err := m.criu.Restore(cpuDir)
	if err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("CRIU restore failed: %v", err))
		return 0, fmt.Errorf("CRIU restore failed: %w", err)
	}

	pid, err := strconv.Atoi(restoredPID)
	if err != nil {
		return 0, fmt.Errorf("invalid restored PID %q: %w", restoredPID, err)
	}

	// Step 4: cuda-checkpoint restore (GPU state).
	m.log.Info("restoring GPU state via cuda-checkpoint", "pid", pid, "inputDir", gpuDir)
	if err := m.cudaChkpt.Restore(pid, gpuDir); err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("cuda-checkpoint restore failed: %v", err))
		return 0, fmt.Errorf("cuda-checkpoint restore failed: %w", err)
	}

	elapsed := time.Since(start)

	// Update the Snapshot CR with restore metrics.
	snapshot.Status.Phase = v1.SnapshotPhaseReady
	snapshot.Status.RestoreTimeMs = elapsed.Milliseconds()
	snapshot.Status.Message = fmt.Sprintf("Restored in %dms", elapsed.Milliseconds())
	if err := m.updateSnapshotStatus(ctx, snapshot); err != nil {
		m.log.Error(err, "failed to update Snapshot CR after restore")
	}

	m.log.Info("snapshot restore complete",
		"snapshotID", snapID,
		"restoreTimeMs", elapsed.Milliseconds(),
		"restoredPID", pid,
	)

	return elapsed, nil
}

// PrewarmStandby creates a standby pod that has the snapshot metadata loaded
// into CPU memory but does not attach a GPU. This enables near-instant GPU
// attach when a request arrives (hot standby pattern).
func (m *Manager) PrewarmStandby(ctx context.Context, snapshot *v1.Snapshot, md *v1.ModelDeployment, namespace string) (*corev1.Pod, error) {
	m.log.Info("creating standby pod",
		"model", md.Spec.ModelName,
		"namespace", namespace,
		"snapshotRef", snapshot.Name,
	)

	podName := fmt.Sprintf("%s-standby-%d", md.Name, time.Now().UnixMilli()%10000)

	// Build a lightweight pod that loads snapshot metadata into memory but
	// does not request GPU resources.
	standbyPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels: map[string]string{
				"podstack.io/model":       md.Spec.ModelName,
				"podstack.io/role":        "standby",
				"podstack.io/snapshot-ref": snapshot.Name,
				"podstack.io/deployment":  md.Name,
			},
			Annotations: map[string]string{
				"podstack.io/snapshot-path":  snapshot.Spec.StoragePath,
				"podstack.io/gpu-type":       snapshot.Spec.GPUType,
				"podstack.io/cuda-version":   snapshot.Spec.CUDAVersion,
				"podstack.io/driver-version": snapshot.Spec.DriverVersion,
				"podstack.io/standby":        "true",
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "standby",
					Image: md.Spec.Image,
					Command: []string{
						"/bin/sh", "-c",
						fmt.Sprintf(
							"echo 'Loading snapshot metadata from %s into memory...' && "+
								"cat %s > /dev/null && "+
								"echo 'Standby ready. Waiting for GPU attach.' && "+
								"sleep infinity",
							snapshot.Spec.StoragePath,
							snapshot.Spec.StoragePath,
						),
					},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("256Mi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("500m"),
							corev1.ResourceMemory: resource.MustParse("2Gi"),
						},
					},
					VolumeMounts: []corev1.VolumeMount{
						{
							Name:      "snapshot-storage",
							MountPath: "/snapshots",
							ReadOnly:  true,
						},
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "snapshot-storage",
					VolumeSource: corev1.VolumeSource{
						HostPath: &corev1.HostPathVolumeSource{
							Path: m.blobstore.basePath,
						},
					},
				},
			},
			// No GPU resources -- CPU-only standby.
			RestartPolicy: corev1.RestartPolicyAlways,
			// Schedule on a node that has the matching GPU type available for
			// when the pod is promoted from standby to active.
			Affinity: &corev1.Affinity{
				NodeAffinity: &corev1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
						NodeSelectorTerms: []corev1.NodeSelectorTerm{
							{
								MatchExpressions: []corev1.NodeSelectorRequirement{
									{
										Key:      "nvidia.com/gpu.product",
										Operator: corev1.NodeSelectorOpIn,
										Values:   []string{snapshot.Spec.GPUType},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	if err := m.k8sClient.Create(ctx, standbyPod); err != nil {
		return nil, fmt.Errorf("failed to create standby pod %s/%s: %w", namespace, podName, err)
	}

	m.log.Info("standby pod created",
		"pod", podName,
		"namespace", namespace,
		"model", md.Spec.ModelName,
		"snapshotRef", snapshot.Name,
	)

	return standbyPod, nil
}

// SnapshotExists checks whether a valid, Ready-phase snapshot exists for the
// given model and GPU type combination. It searches for Snapshot CRs with
// matching labels and verifies the underlying blobstore file is present.
func (m *Manager) SnapshotExists(ctx context.Context, modelName string, gpuType string) (*v1.Snapshot, bool, error) {
	m.log.V(1).Info("checking for existing snapshot", "model", modelName, "gpuType", gpuType)

	// Sanitize model name to match the label format used during creation.
	sanitizedModel := strings.ReplaceAll(modelName, "/", "--")

	snapshotList := &v1.SnapshotList{}
	listOpts := []client.ListOption{
		client.MatchingLabels{
			"podstack.io/model":    sanitizedModel,
			"podstack.io/gpu-type": gpuType,
		},
	}

	if err := m.k8sClient.List(ctx, snapshotList, listOpts...); err != nil {
		return nil, false, fmt.Errorf("failed to list snapshots for model=%s gpu=%s: %w", modelName, gpuType, err)
	}

	// Find the most recent Ready snapshot whose blobstore file actually exists.
	var bestSnapshot *v1.Snapshot
	for i := range snapshotList.Items {
		snap := &snapshotList.Items[i]
		if snap.Status.Phase != v1.SnapshotPhaseReady {
			continue
		}
		snapFileID := snapshotID(modelName, gpuType)
		if !m.blobstore.Exists(snapFileID) {
			m.log.Info("snapshot CR exists but blobstore file missing; skipping",
				"snapshot", snap.Name,
				"snapshotID", snapFileID,
			)
			continue
		}
		if bestSnapshot == nil || snap.CreationTimestamp.After(bestSnapshot.CreationTimestamp.Time) {
			bestSnapshot = snap
		}
	}

	if bestSnapshot != nil {
		m.log.V(1).Info("found existing snapshot",
			"snapshot", bestSnapshot.Name,
			"model", modelName,
			"gpuType", gpuType,
		)
		return bestSnapshot, true, nil
	}

	m.log.V(1).Info("no existing snapshot found", "model", modelName, "gpuType", gpuType)
	return nil, false, nil
}

// DeleteSnapshot removes a snapshot from the blobstore by its snapshot ID.
func (m *Manager) DeleteSnapshot(snapshotID string) error {
	m.log.Info("deleting snapshot from blobstore", "snapshotID", snapshotID)
	return m.blobstore.Delete(snapshotID)
}

// DeleteSnapshotByModel removes a snapshot for the given model and GPU type
// from the blobstore.
func (m *Manager) DeleteSnapshotByModel(modelName, gpuType string) error {
	snapID := snapshotID(modelName, gpuType)
	return m.DeleteSnapshot(snapID)
}

// Blobstore returns a reference to the underlying blobstore.
func (m *Manager) Blobstore() *Blobstore {
	return m.blobstore
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// createOrUpdateSnapshotCR creates a new Snapshot CR or updates an existing one.
func (m *Manager) createOrUpdateSnapshotCR(ctx context.Context, snapshot *v1.Snapshot) error {
	existing := &v1.Snapshot{}
	key := types.NamespacedName{Name: snapshot.Name, Namespace: snapshot.Namespace}

	err := m.k8sClient.Get(ctx, key, existing)
	if errors.IsNotFound(err) {
		if createErr := m.k8sClient.Create(ctx, snapshot); createErr != nil {
			return fmt.Errorf("failed to create Snapshot CR %s: %w", snapshot.Name, createErr)
		}
		m.log.Info("created Snapshot CR", "name", snapshot.Name, "namespace", snapshot.Namespace)
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to get Snapshot CR %s: %w", snapshot.Name, err)
	}

	// Update the existing resource.
	existing.Spec = snapshot.Spec
	existing.Status = snapshot.Status
	existing.Labels = snapshot.Labels

	if err := m.k8sClient.Update(ctx, existing); err != nil {
		return fmt.Errorf("failed to update Snapshot CR %s: %w", snapshot.Name, err)
	}

	// Propagate the latest resource version and UID back to the caller.
	snapshot.ResourceVersion = existing.ResourceVersion
	snapshot.UID = existing.UID

	m.log.Info("updated Snapshot CR", "name", snapshot.Name, "namespace", snapshot.Namespace)
	return nil
}

// updateSnapshotStatus updates only the status subresource of a Snapshot CR.
func (m *Manager) updateSnapshotStatus(ctx context.Context, snapshot *v1.Snapshot) error {
	if err := m.k8sClient.Status().Update(ctx, snapshot); err != nil {
		return fmt.Errorf("failed to update Snapshot status %s: %w", snapshot.Name, err)
	}
	return nil
}

// setSnapshotFailed transitions a Snapshot CR to the Failed phase with a
// human-readable message.
func (m *Manager) setSnapshotFailed(ctx context.Context, snapshot *v1.Snapshot, message string) {
	snapshot.Status.Phase = v1.SnapshotPhaseFailed
	snapshot.Status.Message = message
	if err := m.updateSnapshotStatus(ctx, snapshot); err != nil {
		m.log.Error(err, "failed to update Snapshot CR to Failed phase",
			"snapshot", snapshot.Name,
			"message", message,
		)
	}
}

// createTarArchive creates a tar archive of sourceDir at destPath.
func createTarArchive(sourceDir, destPath string) error {
	cmd := exec.Command("tar", "-cf", destPath, "-C", sourceDir, ".")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("tar create failed: %w (output: %s)", err, string(output))
	}
	return nil
}

// extractTarArchive extracts a tar archive read from reader into destDir.
func extractTarArchive(reader interface{ Read([]byte) (int, error) }, destDir string) error {
	cmd := exec.Command("tar", "-xf", "-", "-C", destDir)
	cmd.Stdin = reader
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("tar extract failed: %w (output: %s)", err, string(output))
	}
	return nil
}
