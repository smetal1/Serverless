package snapshot

import (
	"fmt"
	"os/exec"
	"strconv"

	"github.com/go-logr/logr"
)

const defaultCUDACheckpointBinary = "cuda-checkpoint"

// CUDACheckpoint wraps the NVIDIA cuda-checkpoint CLI tool for capturing and
// restoring GPU state during CUDA context checkpointing.
type CUDACheckpoint struct {
	binaryPath string // path to cuda-checkpoint binary
	log        logr.Logger
}

// NewCUDACheckpoint creates a new CUDACheckpoint helper. It attempts to locate
// the cuda-checkpoint binary on PATH; if not found, it falls back to the
// default binary name (will fail at runtime if unavailable).
func NewCUDACheckpoint(log logr.Logger) *CUDACheckpoint {
	binaryPath := defaultCUDACheckpointBinary
	if resolved, err := exec.LookPath(defaultCUDACheckpointBinary); err == nil {
		binaryPath = resolved
	}
	return &CUDACheckpoint{
		binaryPath: binaryPath,
		log:        log.WithName("cuda-checkpoint"),
	}
}

// Lock locks the CUDA context for the given process so that GPU state can be
// safely captured. This must be called before Checkpoint.
//
// Equivalent to: cuda-checkpoint --toggle --pid <pid>
func (c *CUDACheckpoint) Lock(pid int) error {
	c.log.Info("locking CUDA context", "pid", pid)

	cmd := exec.Command(c.binaryPath, "--toggle", "--pid", strconv.Itoa(pid))
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cuda-checkpoint lock failed for pid %d: %w (output: %s)", pid, err, string(output))
	}

	c.log.V(1).Info("CUDA context locked", "pid", pid, "output", string(output))
	return nil
}

// Checkpoint captures the GPU state for the given process and writes it to
// outputDir. The CUDA context must be locked before calling this method.
//
// Equivalent to: cuda-checkpoint --checkpoint --pid <pid> --dir <outputDir>
func (c *CUDACheckpoint) Checkpoint(pid int, outputDir string) error {
	c.log.Info("capturing GPU state", "pid", pid, "outputDir", outputDir)

	cmd := exec.Command(
		c.binaryPath,
		"--checkpoint",
		"--pid", strconv.Itoa(pid),
		"--dir", outputDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cuda-checkpoint capture failed for pid %d: %w (output: %s)", pid, err, string(output))
	}

	c.log.Info("GPU state captured", "pid", pid, "outputDir", outputDir, "output", string(output))
	return nil
}

// Restore restores the GPU state for the given process from the checkpoint
// stored in inputDir.
//
// Equivalent to: cuda-checkpoint --restore --pid <pid> --dir <inputDir>
func (c *CUDACheckpoint) Restore(pid int, inputDir string) error {
	c.log.Info("restoring GPU state", "pid", pid, "inputDir", inputDir)

	cmd := exec.Command(
		c.binaryPath,
		"--restore",
		"--pid", strconv.Itoa(pid),
		"--dir", inputDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cuda-checkpoint restore failed for pid %d: %w (output: %s)", pid, err, string(output))
	}

	c.log.Info("GPU state restored", "pid", pid, "inputDir", inputDir, "output", string(output))
	return nil
}

// Unlock unlocks the CUDA context for the given process after checkpoint or
// restore operations are complete. This allows the process to resume normal
// GPU operations.
//
// Equivalent to: cuda-checkpoint --toggle --pid <pid>
func (c *CUDACheckpoint) Unlock(pid int) error {
	c.log.Info("unlocking CUDA context", "pid", pid)

	cmd := exec.Command(c.binaryPath, "--toggle", "--pid", strconv.Itoa(pid))
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("cuda-checkpoint unlock failed for pid %d: %w (output: %s)", pid, err, string(output))
	}

	c.log.V(1).Info("CUDA context unlocked", "pid", pid, "output", string(output))
	return nil
}
