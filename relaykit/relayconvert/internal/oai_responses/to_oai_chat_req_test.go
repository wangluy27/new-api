package oairesponses

import (
	"encoding/json"
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestResponsesRequestToChatCompletionsRequestInstructionsAndScalarInput(t *testing.T) {
	stream := true
	temperature := 0.0
	topP := 0.9
	maxOutputTokens := uint(128)
	parallelToolCalls := true

	got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
		Model:                "gpt-test",
		Instructions:         mustRawMessage(t, "system rules"),
		Input:                mustRawMessage(t, "hello"),
		Stream:               &stream,
		StreamOptions:        &dto.StreamOptions{IncludeUsage: true},
		MaxOutputTokens:      &maxOutputTokens,
		Temperature:          &temperature,
		TopP:                 &topP,
		User:                 mustRawMessage(t, "user-1"),
		Store:                mustRawMessage(t, false),
		Metadata:             mustRawMessage(t, map[string]any{"trace": "abc"}),
		ParallelToolCalls:    mustRawMessage(t, parallelToolCalls),
		PromptCacheKey:       mustRawMessage(t, "cache-key"),
		PromptCacheRetention: mustRawMessage(t, "24h"),
		Reasoning:            &dto.Reasoning{Effort: "medium"},
	})
	require.NoError(t, err)

	assert.Equal(t, "gpt-test", got.Model)
	require.Len(t, got.Messages, 2)
	assert.Equal(t, dto.Message{Role: "system", Content: "system rules"}, got.Messages[0])
	assert.Equal(t, dto.Message{Role: "user", Content: "hello"}, got.Messages[1])
	assert.Same(t, &stream, got.Stream)
	require.NotNil(t, got.StreamOptions)
	assert.True(t, got.StreamOptions.IncludeUsage)
	assert.Equal(t, maxOutputTokens, lo.FromPtr(got.MaxCompletionTokens))
	assert.Equal(t, 0.0, lo.FromPtr(got.Temperature))
	assert.Equal(t, 0.9, lo.FromPtr(got.TopP))
	assert.True(t, lo.FromPtr(got.ParallelTooCalls))
	assert.Equal(t, "cache-key", got.PromptCacheKey)
	assert.Equal(t, "medium", got.ReasoningEffort)
	assert.Equal(t, `"user-1"`, string(got.User))
	assert.Equal(t, `false`, string(got.Store))
	assert.Equal(t, "abc", gjson.GetBytes(got.Metadata, "trace").String())
}

func TestResponsesRequestToChatCompletionsRequestPreservesQwenThinkingBudget(t *testing.T) {
	tests := []struct {
		name   string
		budget json.RawMessage
		want   int64
	}{
		{name: "positive budget", budget: json.RawMessage(`128`), want: 128},
		{name: "zero budget", budget: json.RawMessage(`0`), want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
				Model:          "qwen-plus",
				Input:          mustRawMessage(t, "hello"),
				EnableThinking: json.RawMessage(`true`),
				ThinkingBudget: tt.budget,
			})
			require.NoError(t, err)
			assert.Equal(t, tt.budget, got.ThinkingBudget)

			encoded, err := kitutil.Marshal(got)
			require.NoError(t, err)

			assert.True(t, gjson.GetBytes(encoded, "enable_thinking").Bool())
			value := gjson.GetBytes(encoded, "thinking_budget")
			assert.True(t, value.Exists())
			assert.Equal(t, tt.want, value.Int())
		})
	}
}

func TestResponsesRequestToChatCompletionsRequestMultimodalInput(t *testing.T) {
	got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
		Model: "gpt-test",
		Input: mustRawMessage(t, []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "input_text", "text": "look"},
					{"type": "input_image", "image_url": "https://example.test/a.png", "detail": "low"},
					{"type": "input_file", "file_id": "file_1", "filename": "a.txt"},
					{"type": "input_audio", "input_audio": map[string]any{"data": "abc", "format": "wav"}},
					{"type": "input_video", "video_url": map[string]any{"url": "https://example.test/v.mp4"}},
				},
			},
		}),
	})
	require.NoError(t, err)

	require.Len(t, got.Messages, 1)
	assert.Equal(t, "user", got.Messages[0].Role)
	parts := got.Messages[0].ParseContent()
	require.Len(t, parts, 5)
	assert.Equal(t, dto.ContentTypeText, parts[0].Type)
	assert.Equal(t, "look", parts[0].Text)
	assert.Equal(t, dto.ContentTypeImageURL, parts[1].Type)
	assert.Equal(t, "https://example.test/a.png", parts[1].GetImageMedia().Url)
	assert.Equal(t, dto.ContentTypeFile, parts[2].Type)
	assert.Equal(t, "file_1", parts[2].GetFile().FileId)
	assert.Equal(t, dto.ContentTypeInputAudio, parts[3].Type)
	assert.Equal(t, "wav", parts[3].GetInputAudio().Format)
	assert.Equal(t, dto.ContentTypeVideoUrl, parts[4].Type)
	assert.Equal(t, "https://example.test/v.mp4", parts[4].GetVideoUrl().Url)

	// Responses allows a bare image_url string; chat completions only accepts the
	// object form, so the body actually sent upstream has to carry it wrapped.
	encoded, err := kitutil.Marshal(got.Messages[0])
	require.NoError(t, err)
	assert.JSONEq(t,
		`{"url":"https://example.test/a.png","detail":"low"}`,
		gjson.GetBytes(encoded, "content.1.image_url").Raw,
	)
}

func TestResponsesRequestToChatCompletionsRequestAssistantTextAndFunctionCallCoexist(t *testing.T) {
	got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
		Model: "gpt-test",
		Input: mustRawMessage(t, []map[string]any{
			{
				"role": "assistant",
				"content": []map[string]any{
					{"type": "output_text", "text": "I will call."},
				},
			},
			{
				"type":      "function_call",
				"call_id":   "call_1",
				"name":      "lookup",
				"arguments": map[string]any{"q": "x"},
			},
			{
				"type":    "function_call_output",
				"call_id": "call_1",
				"output":  map[string]any{"ok": true},
			},
		}),
	})
	require.NoError(t, err)

	require.Len(t, got.Messages, 2)
	assert.Equal(t, "assistant", got.Messages[0].Role)
	assert.Equal(t, "I will call.", got.Messages[0].StringContent())
	toolCalls := got.Messages[0].ParseToolCalls()
	require.Len(t, toolCalls, 1)
	assert.Equal(t, "call_1", toolCalls[0].ID)
	assert.Equal(t, "function", toolCalls[0].Type)
	assert.Equal(t, "lookup", toolCalls[0].Function.Name)
	assert.JSONEq(t, `{"q":"x"}`, toolCalls[0].Function.Arguments)
	assert.Equal(t, "tool", got.Messages[1].Role)
	assert.Equal(t, "call_1", got.Messages[1].ToolCallId)
	assert.JSONEq(t, `{"ok":true}`, got.Messages[1].StringContent())
}

func TestResponsesRequestToChatCompletionsRequestOnlyFunctionCallCreatesAssistant(t *testing.T) {
	got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
		Model: "gpt-test",
		Input: mustRawMessage(t, []map[string]any{
			{
				"type":      "function_call",
				"call_id":   "call_1",
				"name":      "lookup",
				"arguments": `{"q":"x"}`,
			},
		}),
	})
	require.NoError(t, err)

	require.Len(t, got.Messages, 1)
	assert.Equal(t, "assistant", got.Messages[0].Role)
	assert.Nil(t, got.Messages[0].Content)
	toolCalls := got.Messages[0].ParseToolCalls()
	require.Len(t, toolCalls, 1)
	assert.Equal(t, `{"q":"x"}`, toolCalls[0].Function.Arguments)
}

func TestResponsesRequestToChatCompletionsRequestToolsToolChoiceAndTextFormat(t *testing.T) {
	got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
		Model: "gpt-test",
		Input: mustRawMessage(t, "hello"),
		Tools: mustRawMessage(t, []map[string]any{
			{
				"type":        "function",
				"name":        "lookup",
				"description": "Lookup data",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"q": map[string]any{"type": "string"},
					},
				},
			},
		}),
		ToolChoice: mustRawMessage(t, map[string]any{
			"type": "function",
			"name": "lookup",
		}),
		Text: mustRawMessage(t, map[string]any{
			"format": map[string]any{
				"type":   "json_schema",
				"name":   "answer",
				"schema": map[string]any{"type": "object"},
				"strict": true,
			},
		}),
	})
	require.NoError(t, err)

	require.Len(t, got.Tools, 1)
	assert.Equal(t, "function", got.Tools[0].Type)
	assert.Equal(t, "lookup", got.Tools[0].Function.Name)
	assert.Equal(t, "Lookup data", got.Tools[0].Function.Description)
	assert.Equal(t, "object", got.Tools[0].Function.Parameters.(map[string]any)["type"])
	assert.Equal(t, map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "lookup",
		},
	}, got.ToolChoice)
	require.NotNil(t, got.ResponseFormat)
	assert.Equal(t, "json_schema", got.ResponseFormat.Type)
	assert.Equal(t, "answer", gjson.GetBytes(got.ResponseFormat.JsonSchema, "name").String())
	assert.True(t, gjson.GetBytes(got.ResponseFormat.JsonSchema, "strict").Bool())
}

func TestResponsesRequestToChatCompletionsRequestCustomToolCallPreservesRawShape(t *testing.T) {
	got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
		Model: "gpt-test",
		Input: mustRawMessage(t, []map[string]any{
			{
				"type":    "custom_tool_call",
				"call_id": "call_custom",
				"name":    "apply_patch",
				"input":   "patch body",
			},
		}),
	})
	require.NoError(t, err)

	require.Len(t, got.Messages, 1)
	toolCalls := got.Messages[0].ParseToolCalls()
	require.Len(t, toolCalls, 1)
	assert.Equal(t, dto.CustomType, toolCalls[0].Type)
	assert.Equal(t, "call_custom", toolCalls[0].ID)
	assert.Equal(t, "apply_patch", toolCalls[0].Function.Name)
	assert.Equal(t, "patch body", toolCalls[0].Function.Arguments)
	assert.Equal(t, "custom_tool_call", gjson.GetBytes(toolCalls[0].Custom, "type").String())
	assert.Equal(t, "patch body", gjson.GetBytes(toolCalls[0].Custom, "input").String())
}

func TestResponsesRequestToChatCompletionsRequestRejectsStatefulFields(t *testing.T) {
	tests := []struct {
		name string
		req  *dto.OpenAIResponsesRequest
		want string
	}{
		{
			name: "conversation",
			req:  &dto.OpenAIResponsesRequest{Model: "gpt-test", Conversation: mustRawMessage(t, "conv_1")},
			want: "conversation",
		},
		{
			name: "previous response",
			req:  &dto.OpenAIResponsesRequest{Model: "gpt-test", PreviousResponseID: "resp_1"},
			want: "previous_response_id",
		},
		{
			name: "prompt",
			req:  &dto.OpenAIResponsesRequest{Model: "gpt-test", Prompt: mustRawMessage(t, map[string]any{"id": "pmpt_1"})},
			want: "prompt",
		},
		{
			name: "context management",
			req:  &dto.OpenAIResponsesRequest{Model: "gpt-test", ContextManagement: mustRawMessage(t, map[string]any{"type": "auto"})},
			want: "context_management",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ResponsesRequestToChatCompletionsRequest(tt.req)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
			assert.Contains(t, err.Error(), "stateful fields")
		})
	}
}

func TestResponsesRequestToChatCompletionsRequestTools(t *testing.T) {
	t.Run("function and custom tools keep their chat completions shape", func(t *testing.T) {
		got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
			Model: "deepseek-chat",
			Input: mustRawMessage(t, "hi"),
			Tools: mustRawMessage(t, []map[string]any{
				{
					"type":        "function",
					"name":        "memory_search",
					"description": "search memory",
					"parameters":  map[string]any{"type": "object"},
				},
				{
					"type":   "custom",
					"name":   "freeform",
					"format": map[string]any{"type": "text"},
				},
			}),
		})
		require.NoError(t, err)

		encoded, err := kitutil.Marshal(got.Tools)
		require.NoError(t, err)
		assert.JSONEq(t, `[
			{"type":"function","function":{"name":"memory_search","description":"search memory","parameters":{"type":"object"}}},
			{"type":"custom","custom":{"name":"freeform","format":{"type":"text"}}}
		]`, string(encoded))
	})

	t.Run("namespace members are lifted to the top level under qualified names", func(t *testing.T) {
		got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
			Model: "deepseek-chat",
			Input: mustRawMessage(t, "hi"),
			Tools: mustRawMessage(t, []map[string]any{
				{"type": "function", "name": "memory_search", "parameters": map[string]any{"type": "object"}},
				{
					"type":        "namespace",
					"name":        "multi_agent_v1",
					"description": "Tools for spawning and managing sub-agents.",
					"tools": []map[string]any{
						{"type": "function", "name": "close_agent", "description": "close", "parameters": map[string]any{"type": "object"}},
						{"type": "function", "name": "resume_agent", "description": "resume", "parameters": map[string]any{"type": "object"}},
					},
				},
			}),
		})
		require.NoError(t, err)

		encoded, err := kitutil.Marshal(got.Tools)
		require.NoError(t, err)
		assert.JSONEq(t, `[
			{"type":"function","function":{"name":"memory_search","parameters":{"type":"object"}}},
			{"type":"function","function":{"name":"multi_agent_v1__close_agent","description":"close","parameters":{"type":"object"}}},
			{"type":"function","function":{"name":"multi_agent_v1__resume_agent","description":"resume","parameters":{"type":"object"}}}
		]`, string(encoded))
	})

	t.Run("same member name in two namespaces stays distinguishable", func(t *testing.T) {
		got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
			Model: "deepseek-chat",
			Input: mustRawMessage(t, "hi"),
			Tools: mustRawMessage(t, []map[string]any{
				{
					"type":  "namespace",
					"name":  "container",
					"tools": []map[string]any{{"type": "function", "name": "js", "parameters": map[string]any{"type": "object"}}},
				},
				{
					"type":  "namespace",
					"name":  "browser",
					"tools": []map[string]any{{"type": "function", "name": "js", "parameters": map[string]any{"type": "object"}}},
				},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got.Tools, 2)
		assert.Equal(t, "container__js", got.Tools[0].Function.Name)
		assert.Equal(t, "browser__js", got.Tools[1].Function.Name)
	})

	t.Run("mcp_server groups are flattened like namespaces", func(t *testing.T) {
		got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
			Model: "deepseek-chat",
			Input: mustRawMessage(t, "hi"),
			Tools: mustRawMessage(t, []map[string]any{
				{
					"type":  "mcp_server",
					"name":  "mcp__zai_vision",
					"tools": []map[string]any{{"type": "function", "name": "analyze_image", "parameters": map[string]any{"type": "object"}}},
				},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got.Tools, 1)
		assert.Equal(t, "mcp__zai_vision__analyze_image", got.Tools[0].Function.Name)
	})

	t.Run("a replayed function_call is re-qualified with its namespace", func(t *testing.T) {
		got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
			Model: "deepseek-chat",
			Input: mustRawMessage(t, []map[string]any{
				{"role": "user", "content": "hi"},
				{
					"type":      "function_call",
					"call_id":   "call_1",
					"name":      "close_agent",
					"namespace": "multi_agent_v1",
					"arguments": `{"target":"a1"}`,
				},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got.Messages, 2)
		toolCalls := got.Messages[1].ParseToolCalls()
		require.Len(t, toolCalls, 1)
		assert.Equal(t, "multi_agent_v1__close_agent", toolCalls[0].Function.Name)
	})

	t.Run("a namespace member colliding with a top-level tool is reported", func(t *testing.T) {
		_, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
			Model: "deepseek-chat",
			Input: mustRawMessage(t, "hi"),
			Tools: mustRawMessage(t, []map[string]any{
				{"type": "function", "name": "multi_agent_v1__close_agent", "parameters": map[string]any{"type": "object"}},
				{
					"type": "namespace",
					"name": "multi_agent_v1",
					"tools": []map[string]any{
						{"type": "function", "name": "close_agent", "parameters": map[string]any{"type": "object"}},
					},
				},
			}),
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), `"multi_agent_v1__close_agent"`)
		assert.Contains(t, err.Error(), "tools[1].tools[0]")
		assert.Contains(t, err.Error(), "tools[0]")
	})

	for _, toolType := range []string{"web_search", "mcp", "image_generation"} {
		t.Run("drops hosted tool "+toolType+" instead of failing the request", func(t *testing.T) {
			// ChatGPT attaches web_search to every request; failing on it would
			// make the whole channel unusable, so the capability is dropped and
			// the rest of the request goes through.
			got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
				Model: "deepseek-chat",
				Input: mustRawMessage(t, "hi"),
				Tools: mustRawMessage(t, []map[string]any{
					{"type": "function", "name": "memory_search", "parameters": map[string]any{"type": "object"}},
					{"type": toolType, "name": "hosted"},
				}),
			})
			require.NoError(t, err)

			require.Len(t, got.Tools, 1)
			assert.Equal(t, "memory_search", got.Tools[0].Function.Name)
		})
	}

	t.Run("drops a hosted tool nested inside a namespace", func(t *testing.T) {
		got, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
			Model: "deepseek-chat",
			Input: mustRawMessage(t, "hi"),
			Tools: mustRawMessage(t, []map[string]any{
				{
					"type": "namespace",
					"name": "multi_agent_v1",
					"tools": []map[string]any{
						{"type": "web_search"},
						{"type": "function", "name": "close_agent", "parameters": map[string]any{"type": "object"}},
					},
				},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got.Tools, 1)
		assert.Equal(t, "multi_agent_v1__close_agent", got.Tools[0].Function.Name)
	})

	t.Run("still fails on a namespace without a tools array", func(t *testing.T) {
		_, err := ResponsesRequestToChatCompletionsRequest(&dto.OpenAIResponsesRequest{
			Model: "deepseek-chat",
			Input: mustRawMessage(t, "hi"),
			Tools: mustRawMessage(t, []map[string]any{
				{"type": "namespace", "name": "multi_agent_v1"},
			}),
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "tools[0]")
	})
}

func mustRawMessage(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := kitutil.Marshal(value)
	require.NoError(t, err)
	return raw
}
