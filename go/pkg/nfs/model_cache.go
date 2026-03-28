// Package nfs provides utilities for managing the NFS-backed model cache
// and monitoring NFS mount health in the Podstack Inference OS.
package nfs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"
)

// ModelCache manages the download-once model cache on NFS. Models are stored
// at well-known paths derived from their identifiers so that multiple pods
// can share the same read-only model weights without redundant downloads.
//
// Directory layout:
//
//	{basePath}/base/{org}--{model}/         -- base model weights
//	{basePath}/lora/{tenant}--{adapter}/    -- per-tenant LoRA adapters
//	{basePath}/snapshots/{model}--{gpu}--{cuda}.tar  -- CUDA checkpoints
type ModelCache struct {
	basePath string
	log      logr.Logger
}

// NewModelCache creates a new ModelCache rooted at basePath. The basePath
// should be the NFS mount point (e.g. "/mnt/models").
func NewModelCache(basePath string, log logr.Logger) *ModelCache {
	return &ModelCache{
		basePath: basePath,
		log:      log.WithName("model-cache"),
	}
}

// ModelPath converts a HuggingFace model ID (e.g. "meta-llama/Llama-3-8B")
// into the corresponding NFS path for base model weights. Slashes in the
// model name are replaced with double dashes to produce a flat directory.
func (c *ModelCache) ModelPath(modelName string) string {
	safeName := strings.ReplaceAll(modelName, "/", "--")
	return filepath.Join(c.basePath, "base", safeName)
}

// ModelExists checks whether the model directory exists on the NFS mount.
func (c *ModelCache) ModelExists(modelName string) bool {
	p := c.ModelPath(modelName)
	info, err := os.Stat(p)
	if err != nil {
		c.log.V(2).Info("model path stat failed", "model", modelName, "path", p, "error", err)
		return false
	}
	return info.IsDir()
}

// LoRAPath returns the NFS path for a tenant-specific LoRA adapter.
// The path is {basePath}/lora/{tenant}--{adapter}/.
func (c *ModelCache) LoRAPath(tenantName, adapterName string) string {
	safeName := fmt.Sprintf("%s--%s", tenantName, adapterName)
	return filepath.Join(c.basePath, "lora", safeName)
}

// LoRAExists checks whether the LoRA adapter directory exists on the NFS mount.
func (c *ModelCache) LoRAExists(tenantName, adapterName string) bool {
	p := c.LoRAPath(tenantName, adapterName)
	info, err := os.Stat(p)
	if err != nil {
		c.log.V(2).Info("lora path stat failed", "tenant", tenantName, "adapter", adapterName, "path", p, "error", err)
		return false
	}
	return info.IsDir()
}

// SnapshotPath returns the NFS path for a CUDA snapshot archive. Snapshots
// are GPU-type and CUDA-version specific because they contain device state
// that can only be restored on matching hardware.
// The path is {basePath}/snapshots/{model}--{gpu}--{cuda}.tar.
func (c *ModelCache) SnapshotPath(modelName, gpuType, cudaVersion string) string {
	safeName := fmt.Sprintf("%s--%s--%s.tar",
		strings.ReplaceAll(modelName, "/", "--"),
		gpuType,
		cudaVersion,
	)
	return filepath.Join(c.basePath, "snapshots", safeName)
}

// EnsureDirectories creates the base directory structure for the model cache.
// This is typically called once during controller startup to ensure that
// the NFS mount has the expected layout.
func (c *ModelCache) EnsureDirectories() error {
	dirs := []string{
		filepath.Join(c.basePath, "base"),
		filepath.Join(c.basePath, "lora"),
		filepath.Join(c.basePath, "snapshots"),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			c.log.Error(err, "failed to create cache directory", "path", dir)
			return fmt.Errorf("model cache: failed to create directory %s: %w", dir, err)
		}
		c.log.V(1).Info("ensured cache directory exists", "path", dir)
	}

	c.log.Info("model cache directories initialized", "basePath", c.basePath)
	return nil
}
