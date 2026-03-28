package nfs

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/go-logr/logr"
)

// NFSHealthChecker monitors the health of an NFS mount by performing
// periodic write-read-delete tests. This detects stale mounts, permission
// issues, and NFS server outages that would prevent model loading.
type NFSHealthChecker struct {
	mountPath string
	log       logr.Logger
}

// NewNFSHealthChecker creates a new health checker for the NFS mount at
// mountPath. The mount path should be the root of the NFS share (e.g.
// "/mnt/models").
func NewNFSHealthChecker(mountPath string, log logr.Logger) *NFSHealthChecker {
	return &NFSHealthChecker{
		mountPath: mountPath,
		log:       log.WithName("nfs-health"),
	}
}

// Check performs a health check by writing a sentinel file, reading it back,
// and then removing it. This exercises the full write path to detect issues
// with stale NFS mounts or permission problems. Returns nil if the mount is
// healthy, or an error describing the failure.
func (h *NFSHealthChecker) Check() error {
	// Verify the mount point directory exists and is accessible.
	info, err := os.Stat(h.mountPath)
	if err != nil {
		return fmt.Errorf("nfs health: mount path not accessible: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("nfs health: mount path is not a directory: %s", h.mountPath)
	}

	// Write a sentinel file with a timestamp to detect stale mounts.
	sentinelPath := filepath.Join(h.mountPath, ".podstack-health-check")
	timestamp := fmt.Sprintf("podstack-nfs-health:%d", time.Now().UnixNano())

	if err := os.WriteFile(sentinelPath, []byte(timestamp), 0o644); err != nil {
		return fmt.Errorf("nfs health: failed to write sentinel file: %w", err)
	}

	// Read back the sentinel and verify contents match.
	data, err := os.ReadFile(sentinelPath)
	if err != nil {
		return fmt.Errorf("nfs health: failed to read sentinel file: %w", err)
	}
	if string(data) != timestamp {
		return fmt.Errorf("nfs health: sentinel data mismatch (wrote %q, read %q)", timestamp, string(data))
	}

	// Clean up the sentinel file.
	if err := os.Remove(sentinelPath); err != nil {
		// Log but do not fail the check for cleanup failures.
		h.log.V(1).Info("failed to remove health check sentinel", "path", sentinelPath, "error", err)
	}

	h.log.V(2).Info("NFS health check passed", "mountPath", h.mountPath)
	return nil
}

// IsHealthy returns true if the NFS mount is healthy. This is a convenience
// wrapper around Check() that returns a boolean instead of an error.
func (h *NFSHealthChecker) IsHealthy() bool {
	if err := h.Check(); err != nil {
		h.log.Error(err, "NFS health check failed", "mountPath", h.mountPath)
		return false
	}
	return true
}
