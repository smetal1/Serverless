package snapshot

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/go-logr/logr"
)

// Blobstore manages snapshot archives on an NVMe-backed or NFS-mounted filesystem.
// Each snapshot is stored as a single file identified by its snapshot ID.
type Blobstore struct {
	basePath string // NFS/NVMe mount path for snapshots
	log      logr.Logger
}

// NewBlobstore creates a new Blobstore rooted at basePath. The base directory is
// created if it does not already exist.
func NewBlobstore(basePath string, log logr.Logger) *Blobstore {
	// Ensure the base directory exists on construction so callers do not need
	// to handle lazy initialization errors later.
	if err := os.MkdirAll(basePath, 0o755); err != nil {
		log.Error(err, "failed to create blobstore base directory", "path", basePath)
	}
	return &Blobstore{
		basePath: basePath,
		log:      log.WithName("blobstore"),
	}
}

// Path returns the absolute filesystem path for a given snapshot ID.
func (b *Blobstore) Path(snapshotID string) string {
	return filepath.Join(b.basePath, snapshotID+".snap")
}

// Store writes snapshot data from the provided reader to a file named after
// snapshotID. The write is performed atomically via a temporary file to avoid
// leaving partial snapshots on disk in case of failure.
func (b *Blobstore) Store(snapshotID string, data io.Reader) error {
	destPath := b.Path(snapshotID)
	b.log.V(1).Info("storing snapshot", "id", snapshotID, "path", destPath)

	// Ensure the parent directory exists (handles nested snapshot IDs).
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return fmt.Errorf("blobstore: failed to create parent directory for %s: %w", snapshotID, err)
	}

	// Write to a temporary file first for atomicity.
	tmpFile, err := os.CreateTemp(filepath.Dir(destPath), ".snap-tmp-*")
	if err != nil {
		return fmt.Errorf("blobstore: failed to create temp file for %s: %w", snapshotID, err)
	}
	tmpPath := tmpFile.Name()

	// Clean up the temporary file on any error path.
	defer func() {
		if tmpPath != "" {
			_ = os.Remove(tmpPath)
		}
	}()

	written, err := io.Copy(tmpFile, data)
	if err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("blobstore: failed to write snapshot data for %s: %w", snapshotID, err)
	}

	if err := tmpFile.Sync(); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("blobstore: failed to fsync snapshot %s: %w", snapshotID, err)
	}

	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("blobstore: failed to close temp file for %s: %w", snapshotID, err)
	}

	// Atomically move the temporary file to the final destination.
	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("blobstore: failed to rename temp file to %s: %w", destPath, err)
	}

	// Prevent deferred removal of the (now-renamed) file.
	tmpPath = ""

	b.log.Info("snapshot stored", "id", snapshotID, "bytes", written, "path", destPath)
	return nil
}

// Load opens the snapshot file for reading. The caller is responsible for
// closing the returned ReadCloser.
func (b *Blobstore) Load(snapshotID string) (io.ReadCloser, error) {
	snapPath := b.Path(snapshotID)
	b.log.V(1).Info("loading snapshot", "id", snapshotID, "path", snapPath)

	f, err := os.Open(snapPath)
	if err != nil {
		return nil, fmt.Errorf("blobstore: failed to open snapshot %s: %w", snapshotID, err)
	}
	return f, nil
}

// Exists returns true if a snapshot file exists at the expected path.
func (b *Blobstore) Exists(snapshotID string) bool {
	snapPath := b.Path(snapshotID)
	info, err := os.Stat(snapPath)
	if err != nil {
		return false
	}
	return !info.IsDir()
}

// Delete removes the snapshot file from disk. It returns nil if the file
// does not exist (idempotent delete).
func (b *Blobstore) Delete(snapshotID string) error {
	snapPath := b.Path(snapshotID)
	b.log.V(1).Info("deleting snapshot", "id", snapshotID, "path", snapPath)

	if err := os.Remove(snapPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("blobstore: failed to delete snapshot %s: %w", snapshotID, err)
	}
	b.log.Info("snapshot deleted", "id", snapshotID)
	return nil
}

// SizeBytes returns the size of the snapshot file in bytes.
func (b *Blobstore) SizeBytes(snapshotID string) (int64, error) {
	snapPath := b.Path(snapshotID)
	info, err := os.Stat(snapPath)
	if err != nil {
		return 0, fmt.Errorf("blobstore: failed to stat snapshot %s: %w", snapshotID, err)
	}
	return info.Size(), nil
}
