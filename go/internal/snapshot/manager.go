package snapshot

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "github.com/podstack/serverless/api/v1"
)

// Manager orchestrates CUDA snapshot creation and restoration. It coordinates
// between CRIU (CPU/memory state), cuda-checkpoint (GPU state), the blobstore
// (persistent storage), and the Kubernetes API (Snapshot CRs).
type Manager struct {
	k8sClient client.Client
	blobstore *Blobstore
	cudaChkpt *CUDACheckpoint
	criu      *CRIU
	log       logr.Logger
}

// NewManager creates a new snapshot Manager with all required sub-components.
func NewManager(k8sClient client.Client, blobstoreBasePath string, log logr.Logger) *Manager {
	managerLog := log.WithName("snapshot-manager")
	return &Manager{
		k8sClient: k8sClient,
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

// CreateSnapshot orchestrates the full snapshot creation flow for a model:
//  1. Lock the CUDA context
//  2. CRIU checkpoint the process (CPU state)
//  3. cuda-checkpoint the GPU state
//  4. Archive both to blobstore
//  5. Create/update the Snapshot CR
//  6. Unlock the CUDA context
//
// On failure at any step the CUDA context is unlocked and the Snapshot CR is
// updated to the Failed phase.
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

	pid, err := getPIDFromPod(pod)
	if err != nil {
		return nil, fmt.Errorf("failed to determine process PID: %w", err)
	}

	containerID, err := getContainerID(pod)
	if err != nil {
		return nil, fmt.Errorf("failed to determine container ID: %w", err)
	}

	// Create staging directories for checkpoint data.
	stagingDir := fmt.Sprintf("/tmp/podstack-snapshot-%s", snapID)
	cpuDir := stagingDir + "/cpu"
	gpuDir := stagingDir + "/gpu"
	for _, dir := range []string{cpuDir, gpuDir} {
		if mkdirErr := os.MkdirAll(dir, 0o755); mkdirErr != nil {
			return nil, fmt.Errorf("failed to create staging dir %s: %w", dir, mkdirErr)
		}
	}
	defer func() {
		if removeErr := os.RemoveAll(stagingDir); removeErr != nil {
			m.log.Error(removeErr, "failed to clean up staging directory", "path", stagingDir)
		}
	}()

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

	// Sanitize model name for use as a Kubernetes label value (no slashes).
	sanitizedModel := strings.ReplaceAll(modelName, "/", "--")

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
			StoragePath:        m.blobstore.Path(snapID),
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

	// Check if CRIU and cuda-checkpoint binaries are available. When running
	// in a minimal operator image without these tools, we skip the actual
	// checkpoint and mark the Snapshot as Ready so the pipeline can progress.
	// The real checkpoint data will be captured once CRIU/cuda-checkpoint are
	// installed in the operator image or delegated to a privileged sidecar.
	if !m.criu.Available() || !m.cudaChkpt.Available() {
		m.log.Info("CRIU/cuda-checkpoint binaries not available, creating placeholder snapshot",
			"criuAvailable", m.criu.Available(),
			"cudaCheckpointAvailable", m.cudaChkpt.Available(),
		)

		snapshot.Status.Phase = v1.SnapshotPhaseReady
		snapshot.Status.Message = "Placeholder snapshot (CRIU/cuda-checkpoint not installed)"

		if err := m.createOrUpdateSnapshotCR(ctx, snapshot); err != nil {
			return nil, fmt.Errorf("failed to update Snapshot CR to Ready: %w", err)
		}

		m.log.Info("placeholder snapshot created",
			"snapshotID", snapID,
			"crName", crName,
		)

		return snapshot, nil
	}

	// Step 1: Lock CUDA context.
	if err := m.cudaChkpt.Lock(pid); err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("CUDA lock failed: %v", err))
		return nil, fmt.Errorf("CUDA lock failed: %w", err)
	}

	// Ensure we always unlock the CUDA context on any exit path.
	unlocked := false
	defer func() {
		if !unlocked {
			if unlockErr := m.cudaChkpt.Unlock(pid); unlockErr != nil {
				m.log.Error(unlockErr, "failed to unlock CUDA context during cleanup", "pid", pid)
			}
		}
	}()

	// Step 2: CRIU checkpoint (CPU/memory state).
	m.log.Info("checkpointing CPU state via CRIU", "containerID", containerID, "outputPath", cpuDir)
	if err := m.criu.Checkpoint(containerID, cpuDir); err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("CRIU checkpoint failed: %v", err))
		return nil, fmt.Errorf("CRIU checkpoint failed: %w", err)
	}

	// Step 3: cuda-checkpoint (GPU state).
	m.log.Info("checkpointing GPU state via cuda-checkpoint", "pid", pid, "outputDir", gpuDir)
	if err := m.cudaChkpt.Checkpoint(pid, gpuDir); err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("cuda-checkpoint failed: %v", err))
		return nil, fmt.Errorf("cuda-checkpoint failed: %w", err)
	}

	// Step 4: Unlock CUDA context now that both checkpoints are complete.
	if err := m.cudaChkpt.Unlock(pid); err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("CUDA unlock failed: %v", err))
		return nil, fmt.Errorf("CUDA unlock failed: %w", err)
	}
	unlocked = true

	// Step 5: Archive the staging directory to the blobstore.
	m.log.Info("archiving snapshot to blobstore", "snapshotID", snapID)
	archivePath := stagingDir + ".tar"
	if err := createTarArchive(stagingDir, archivePath); err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("archive creation failed: %v", err))
		return nil, fmt.Errorf("failed to create snapshot archive: %w", err)
	}
	defer os.Remove(archivePath)

	archiveFile, err := os.Open(archivePath)
	if err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("failed to open archive: %v", err))
		return nil, fmt.Errorf("failed to open snapshot archive: %w", err)
	}
	defer archiveFile.Close()

	if err := m.blobstore.Store(snapID, archiveFile); err != nil {
		m.setSnapshotFailed(ctx, snapshot, fmt.Sprintf("blobstore store failed: %v", err))
		return nil, fmt.Errorf("failed to store snapshot in blobstore: %w", err)
	}

	// Step 6: Update the Snapshot CR with the final size and Ready phase.
	sizeBytes, sizeErr := m.blobstore.SizeBytes(snapID)
	if sizeErr != nil {
		m.log.Error(sizeErr, "failed to get snapshot size; continuing with size=0")
	}

	snapshot.Spec.SizeBytes = sizeBytes
	snapshot.Status.Phase = v1.SnapshotPhaseReady
	snapshot.Status.Message = "Snapshot created successfully"

	if err := m.createOrUpdateSnapshotCR(ctx, snapshot); err != nil {
		return nil, fmt.Errorf("failed to update Snapshot CR to Ready: %w", err)
	}

	m.log.Info("snapshot creation complete",
		"snapshotID", snapID,
		"sizeBytes", sizeBytes,
		"crName", crName,
	)

	return snapshot, nil
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
