package oairesponses

import (
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatMessagesFromResponsesRequest(t *testing.T) {
	t.Run("a replayed reasoning item does not reach the upstream", func(t *testing.T) {
		// Codex asks for reasoning.encrypted_content and replays the reasoning
		// item on the next turn. It has no role, so it converts into a user
		// message whose parts the upstream rejects outright:
		//   messages[4]: unknown variant `reasoning_text`
		got, err := ChatMessagesFromResponsesRequest(&dto.OpenAIResponsesRequest{
			Model: "glm-5.3-flash",
			Input: mustRawMessage(t, []map[string]any{
				{"role": "user", "content": []map[string]any{{"type": "input_text", "text": "hi"}}},
				{
					"type":    "reasoning",
					"id":      "rs_1",
					"summary": []map[string]any{{"type": "summary_text", "text": "thinking"}},
					"content": []map[string]any{{"type": "reasoning_text", "text": "private chain of thought"}},
				},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got, 1)
		assert.Equal(t, "user", got[0].Role)
		assert.Equal(t, "hi", got[0].Content)
	})

	t.Run("unsupported parts are dropped without losing the rest of the message", func(t *testing.T) {
		got, err := ChatMessagesFromResponsesRequest(&dto.OpenAIResponsesRequest{
			Model: "glm-5.3-flash",
			Input: mustRawMessage(t, []map[string]any{
				{
					"role": "user",
					"content": []map[string]any{
						{"type": "input_text", "text": "look at this"},
						{"type": "reasoning_text", "text": "private chain of thought"},
						{"type": "input_image", "image_url": "https://example.test/a.png"},
					},
				},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got, 1)
		parts := got[0].ParseContent()
		require.Len(t, parts, 2)
		assert.Equal(t, dto.ContentTypeText, parts[0].Type)
		assert.Equal(t, dto.ContentTypeImageURL, parts[1].Type)
	})

	t.Run("an assistant message keeps its tool calls when it has no content", func(t *testing.T) {
		// Dropping a message that has no parts left must not take a replayed
		// tool call with it: the tool result that follows would then reference a
		// call the upstream never saw.
		got, err := ChatMessagesFromResponsesRequest(&dto.OpenAIResponsesRequest{
			Model: "glm-5.3-flash",
			Input: mustRawMessage(t, []map[string]any{
				{"role": "user", "content": "hi"},
				{"type": "function_call", "call_id": "call_1", "name": "js", "arguments": "{}"},
				{"type": "function_call_output", "call_id": "call_1", "output": "42"},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got, 3)
		assert.Equal(t, "assistant", got[1].Role)
		require.Len(t, got[1].ParseToolCalls(), 1)
		assert.Equal(t, "tool", got[2].Role)
		assert.Equal(t, "call_1", got[2].ToolCallId)
	})

	t.Run("a scalar image_url is wrapped into the object form", func(t *testing.T) {
		// Codex pastes a screenshot as a bare data: URI, which Responses allows
		// and chat completions does not. Sending the string fails the content
		// union and the error that surfaces names the other branch - "Input
		// should be a valid string" - pointing at content, not at the part.
		got, err := ChatMessagesFromResponsesRequest(&dto.OpenAIResponsesRequest{
			Model: "glm-5.3-flash",
			Input: mustRawMessage(t, []map[string]any{
				{
					"role": "user",
					"content": []map[string]any{
						{"type": "input_text", "text": "what is in this image?"},
						{"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"},
					},
				},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got, 1)
		encoded, err := kitutil.Marshal(got[0].Content)
		require.NoError(t, err)
		assert.Contains(t, string(encoded), `"image_url":{"url":"data:image/jpeg;base64,AAAA"}`)
	})

	t.Run("an object image_url is left alone", func(t *testing.T) {
		got, err := ChatMessagesFromResponsesRequest(&dto.OpenAIResponsesRequest{
			Model: "glm-5.3-flash",
			Input: mustRawMessage(t, []map[string]any{
				{
					"role": "user",
					"content": []map[string]any{
						{"type": "input_image", "image_url": map[string]any{"url": "https://example.test/a.png", "detail": "low"}},
					},
				},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got, 1)
		encoded, err := kitutil.Marshal(got[0].Content)
		require.NoError(t, err)
		assert.Contains(t, string(encoded), `"detail":"low"`)
		assert.Contains(t, string(encoded), `"url":"https://example.test/a.png"`)
	})

	t.Run("a message left with only tool calls sends a null content", func(t *testing.T) {
		// Not an empty array: an upstream that accepts a string or null there
		// answers "Input should be a valid string" for [].
		got, err := ChatMessagesFromResponsesRequest(&dto.OpenAIResponsesRequest{
			Model: "glm-5.3-flash",
			Input: mustRawMessage(t, []map[string]any{
				{"role": "assistant", "content": []map[string]any{{"type": "reasoning_text", "text": "private"}}},
				{"type": "function_call", "call_id": "call_1", "name": "js", "arguments": "{}"},
			}),
		})
		require.NoError(t, err)

		require.Len(t, got, 1)
		assert.Equal(t, "assistant", got[0].Role)
		assert.Nil(t, got[0].Content)
		require.Len(t, got[0].ParseToolCalls(), 1)

		encoded, err := kitutil.Marshal(got[0])
		require.NoError(t, err)
		assert.Contains(t, string(encoded), `"content":null`)
	})
}
