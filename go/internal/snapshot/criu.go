package snapshot

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/go-logr/logr"
)

const defaultCRIUBinary = "criu"

// CRIU wraps the CRIU (Checkpoint/Restore In Userspace) tool for capturing
// and restoring process state. It is used in conjunction with CUDACheckpoint
// to perform a complete snapshot: CRIU handles CPU/memory state while
// cuda-checkpoint handles GPU state.
type CRIU struct {
	binaryPath string
	log        logr.Logger
}

// NewCRIU creates a new CRIU helper. It attempts to locate the criu binary
// on PATH; if not found, it falls back to the default binary name.
func NewCRIU(log logr.Logger) *CRIU {
	binaryPath := defaultCRIUBinary
	if resolved, err := exec.LookPath(defaultCRIUBinary); err == nil {
		binaryPath = resolved
	}
	return &CRIU{
		binaryPath: binaryPath,
		log:        log.WithName("criu"),
	}
}

// Checkpoint captures the full process state (CPU registers, memory, open file
// descriptors, etc.) for the container identified by containerID and writes
// the checkpoint images to outputPath.
//
// This uses criu's dump command with options suitable for container workloads:
//   - --leave-running: keep the process alive after checkpointing
//   - --shell-job: handle processes attached to a terminal
//   - --tcp-established: checkpoint TCP connections
//   - --file-locks: checkpoint file locks
//   - --ext-unix-sk: allow external UNIX sockets
//
// Equivalent to:
//
//	criu dump -t <containerID> -D <outputPath> --leave-running --shell-job \
//	  --tcp-established --file-locks --ext-unix-sk
func (c *CRIU) Checkpoint(containerID string, outputPath string) error {
	c.log.Info("checkpointing process state", "containerID", containerID, "outputPath", outputPath)

	args := []string{
		"dump",
		"-t", containerID,
		"-D", outputPath,
		"--leave-running",
		"--shell-job",
		"--tcp-established",
		"--file-locks",
		"--ext-unix-sk",
	}

	cmd := exec.Command(c.binaryPath, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("criu checkpoint failed for container %s: %w (output: %s)",
			containerID, err, string(output))
	}

	c.log.Info("process state checkpointed",
		"containerID", containerID,
		"outputPath", outputPath,
		"output", string(output),
	)
	return nil
}

// Restore restores a process from a CRIU checkpoint stored at snapshotPath.
// It returns the PID of the restored process as a string.
//
// This uses criu's restore command with options matching those used during
// checkpoint:
//   - --shell-job: handle processes attached to a terminal
//   - --tcp-established: restore TCP connections
//   - --file-locks: restore file locks
//   - --ext-unix-sk: allow external UNIX sockets
//   - --restore-detached: detach the restored process
//   - --pidfile: write the restored PID to a file for retrieval
//
// Equivalent to:
//
//	criu restore -D <snapshotPath> --shell-job --tcp-established \
//	  --file-locks --ext-unix-sk --restore-detached --pidfile /tmp/criu-restore.pid
func (c *CRIU) Restore(snapshotPath string) (string, error) {
	c.log.Info("restoring process from checkpoint", "snapshotPath", snapshotPath)

	pidFile := "/tmp/criu-restore.pid"

	args := []string{
		"restore",
		"-D", snapshotPath,
		"--shell-job",
		"--tcp-established",
		"--file-locks",
		"--ext-unix-sk",
		"--restore-detached",
		"--pidfile", pidFile,
	}

	cmd := exec.Command(c.binaryPath, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("criu restore failed from %s: %w (output: %s)",
			snapshotPath, err, string(output))
	}

	// Read the restored PID from the pidfile.
	pidOutput, err := os.ReadFile(pidFile)
	if err != nil {
		return "", fmt.Errorf("criu restore: failed to read pidfile %s: %w", pidFile, err)
	}

	pid := strings.TrimSpace(string(pidOutput))
	c.log.Info("process restored from checkpoint",
		"snapshotPath", snapshotPath,
		"restoredPID", pid,
		"output", string(output),
	)

	return pid, nil
}
