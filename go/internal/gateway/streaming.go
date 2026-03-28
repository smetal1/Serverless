package gateway

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// proxySSE reads Server-Sent Events from the upstream response body and forwards
// them to the client. It sets the appropriate headers for SSE streaming and flushes
// after each event to ensure low-latency delivery.
//
// Headers set:
//   - Content-Type: text/event-stream
//   - Cache-Control: no-cache
//   - Connection: keep-alive
//   - X-Accel-Buffering: no (disables nginx buffering)
//
// The function reads line by line from the upstream, detecting SSE event boundaries
// (blank lines). Each complete event is flushed immediately to the client.
// The stream terminates when the upstream closes the connection or sends the
// "[DONE]" sentinel used by OpenAI-compatible APIs.
func proxySSE(w http.ResponseWriter, upstream io.ReadCloser) error {
	defer upstream.Close()

	// Verify the ResponseWriter supports flushing.
	flusher, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("response writer does not support flushing")
	}

	// Set SSE headers.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	scanner := bufio.NewScanner(upstream)

	// Increase the scanner buffer for large SSE payloads.
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		// Write the line followed by a newline.
		if _, err := fmt.Fprintf(w, "%s\n", line); err != nil {
			return fmt.Errorf("writing SSE line to client: %w", err)
		}

		// SSE events are delimited by blank lines. Flush after each blank line
		// to send the complete event to the client immediately.
		if line == "" {
			flusher.Flush()
		}

		// Check for the OpenAI-style stream termination sentinel.
		if strings.TrimSpace(line) == "data: [DONE]" {
			flusher.Flush()
			return nil
		}
	}

	// Final flush to ensure any trailing data is sent.
	flusher.Flush()

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("reading SSE from upstream: %w", err)
	}

	return nil
}
