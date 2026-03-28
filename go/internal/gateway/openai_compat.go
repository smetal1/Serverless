package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/client"

	v1 "github.com/podstack/serverless/api/v1"
)

// --------------------------------------------------------------------------
// OpenAI-compatible request types
// --------------------------------------------------------------------------

// ChatCompletionRequest represents an OpenAI-compatible chat completion request.
type ChatCompletionRequest struct {
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	Stream           bool      `json:"stream,omitempty"`
	MaxTokens        int       `json:"max_tokens,omitempty"`
	Temperature      float64   `json:"temperature,omitempty"`
	TopP             float64   `json:"top_p,omitempty"`
	N                int       `json:"n,omitempty"`
	Stop             []string  `json:"stop,omitempty"`
	PresencePenalty  float64   `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitempty"`
	User             string    `json:"user,omitempty"`
}

// Message represents a single message in a chat conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest represents an OpenAI-compatible text completion request.
type CompletionRequest struct {
	Model            string   `json:"model"`
	Prompt           string   `json:"prompt"`
	Stream           bool     `json:"stream,omitempty"`
	MaxTokens        int      `json:"max_tokens,omitempty"`
	Temperature      float64  `json:"temperature,omitempty"`
	TopP             float64  `json:"top_p,omitempty"`
	N                int      `json:"n,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	PresencePenalty  float64  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64  `json:"frequency_penalty,omitempty"`
	User             string   `json:"user,omitempty"`
}

// EmbeddingRequest represents an OpenAI-compatible embedding request.
type EmbeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
	User  string `json:"user,omitempty"`
}

// ImageRequest represents an OpenAI-compatible image generation request.
type ImageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
	User           string `json:"user,omitempty"`
}

// AudioTranscriptionRequest represents an OpenAI-compatible audio transcription request.
type AudioTranscriptionRequest struct {
	Model       string `json:"model"`
	Language    string `json:"language,omitempty"`
	Prompt      string `json:"prompt,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
}

// AudioSpeechRequest represents an OpenAI-compatible text-to-speech request.
type AudioSpeechRequest struct {
	Model string  `json:"model"`
	Input string  `json:"input"`
	Voice string  `json:"voice,omitempty"`
	Speed float64 `json:"speed,omitempty"`
}

// --------------------------------------------------------------------------
// OpenAI-compatible response types
// --------------------------------------------------------------------------

// ChatCompletionResponse represents an OpenAI-compatible chat completion response.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice represents a single completion choice in a response.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage tracks token consumption for a request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CompletionResponse represents an OpenAI-compatible text completion response.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`
}

// CompletionChoice represents a single choice in a completion response.
type CompletionChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// EmbeddingResponse represents an OpenAI-compatible embedding response.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

// EmbeddingData represents a single embedding vector.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

// --------------------------------------------------------------------------
// Model listing types and handler
// --------------------------------------------------------------------------

// ModelListResponse represents the OpenAI /v1/models response.
type ModelListResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ModelInfo represents a single model in the models list.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// HandleListModels returns an http.HandlerFunc that queries ModelDeployment CRs
// and returns them in OpenAI's /v1/models response format.
func HandleListModels(k8sClient client.Client, namespace string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()

		mdList := &v1.ModelDeploymentList{}
		if err := k8sClient.List(ctx, mdList, client.InNamespace(namespace)); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"failed to list models","type":"server_error"}}`))
			return
		}

		// Filter to models the authenticated tenant owns or that are public.
		tenant := TenantFromContext(r.Context())
		models := make([]ModelInfo, 0, len(mdList.Items))

		for i := range mdList.Items {
			md := &mdList.Items[i]

			// Only show models belonging to the tenant, or unowned (shared) models.
			if md.Spec.TenantRef != "" && md.Spec.TenantRef != tenant {
				continue
			}

			created := md.CreationTimestamp.Unix()
			ownedBy := "podstack"
			if md.Spec.TenantRef != "" {
				ownedBy = md.Spec.TenantRef
			}

			models = append(models, ModelInfo{
				ID:      md.Spec.ModelName,
				Object:  "model",
				Created: created,
				OwnedBy: ownedBy,
			})
		}

		resp := ModelListResponse{
			Object: "list",
			Data:   models,
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(resp)
	}
}

// --------------------------------------------------------------------------
// Request field extraction
// --------------------------------------------------------------------------

// modelRequest is a minimal struct used for extracting the model field
// from any inference request JSON payload.
type modelRequest struct {
	Model string `json:"model"`
}

// extractModel parses the "model" field from a JSON request body.
// It works with any OpenAI-compatible request type since they all share
// the top-level "model" field.
func extractModel(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var req modelRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	return req.Model
}

// --------------------------------------------------------------------------
// Streaming request detection
// --------------------------------------------------------------------------

// streamCheck is a minimal struct for detecting the "stream" field.
type streamCheck struct {
	Stream bool `json:"stream"`
}

// isStreamingRequest checks if the JSON request body has "stream": true.
func isStreamingRequest(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	var check streamCheck
	if err := json.Unmarshal(body, &check); err != nil {
		return false
	}
	return check.Stream
}
