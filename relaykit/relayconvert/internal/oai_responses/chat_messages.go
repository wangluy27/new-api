package oairesponses

import (
	"fmt"
	"strings"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
)

const (
	responsesInputTypeReasoning = "reasoning"
	// The two shapes a replayed reasoning item can carry its text in: the
	// upstream's own items use content parts, while the ones this gateway
	// produces from a chat completions reasoning_content carry only a summary.
	responsesReasoningTextPart = "reasoning_text"
	responsesSummaryTextPart   = "summary_text"
)

// chatContentPartTypes are the content part types chat completions understands.
// Anything else is a Responses-only shape that the upstream will reject.
var chatContentPartTypes = map[string]struct{}{
	dto.ContentTypeText:       {},
	dto.ContentTypeImageURL:   {},
	dto.ContentTypeFile:       {},
	dto.ContentTypeInputAudio: {},
	dto.ContentTypeVideoUrl:   {},
}

// ChatMessagesFromResponsesRequest converts the Responses input into chat
// completions messages and reshapes what chat completions cannot express.
//
// Three Responses-only shapes are handled here, each of which the upstream
// refuses outright:
//
//   - A reasoning item, which has no role and would convert into a user message
//     full of parts no chat upstream knows. Its text is moved onto the assistant
//     turn it belongs to, as reasoning_content, because a thinking-mode upstream
//     requires the reasoning it produced to come back.
//   - A tool result answering no call, which ChatGPT injects to open an
//     automation turn.
//   - Content parts outside the set chat completions understands, and an
//     image_url given as a bare string.
func ChatMessagesFromResponsesRequest(req *dto.OpenAIResponsesRequest) ([]dto.Message, error) {
	messages, err := responsesRequestMessagesToChat(req)
	if err != nil {
		return nil, err
	}

	// Read from the request rather than the converted messages: the items this
	// gateway produces carry their text in summary, and the request converter
	// only ever looks at content, so by this point the text would be gone.
	reasoningTexts := responsesReasoningTexts(req)

	// A tool result is only valid next to the call it answers. Collecting the
	// ids first lets the loop below tell a real result from one the client
	// injected on its own.
	answeredCalls := make(map[string]struct{})
	for _, message := range messages {
		for _, toolCall := range message.ParseToolCalls() {
			if toolCall.ID != "" {
				answeredCalls[toolCall.ID] = struct{}{}
			}
		}
	}

	kept := make([]dto.Message, 0, len(messages))
	pendingReasoning := ""
	for i, message := range messages {
		// The converted stand-in for a reasoning item carries nothing worth
		// sending; its text belongs to the assistant turn that follows.
		if len(reasoningTexts) > 0 && isReasoningPlaceholder(message) {
			pendingReasoning = reasoningTexts[0]
			reasoningTexts = reasoningTexts[1:]
			continue
		}

		message = adoptOrphanToolResult(message, answeredCalls, i)

		if parts, isParts := message.Content.([]any); isParts {
			supported := make([]any, 0, len(parts))
			for j, rawPart := range parts {
				part, isObject := rawPart.(map[string]any)
				if !isObject {
					supported = append(supported, rawPart)
					continue
				}
				partType := strings.TrimSpace(kitutil.Interface2String(part["type"]))
				if _, ok := chatContentPartTypes[partType]; ok {
					supported = append(supported, normalizeChatContentPart(part))
					continue
				}
				kitutil.LogError(fmt.Sprintf("responses to chat conversion dropped unsupported content part type %q at messages[%d].content[%d]", partType, i, j))
			}
			// Tool calls are carried outside content, so a message can
			// legitimately end up with no parts left and still be worth
			// sending. null rather than an empty array: that is the shape an
			// assistant message with only tool calls has always been sent
			// with, and an upstream that accepts a string or null rejects [].
			if len(supported) == 0 {
				if len(message.ToolCalls) == 0 {
					continue
				}
				message.Content = nil
			} else {
				message.Content = supported
			}
		}

		if message.Role == "assistant" {
			if pendingReasoning != "" && message.ReasoningContent == nil {
				reasoning := pendingReasoning
				message.ReasoningContent = &reasoning
			}
			pendingReasoning = ""
		} else {
			// Reasoning belongs to the turn that directly follows it. Anything
			// else in between means the turn it described never made it here.
			pendingReasoning = ""
		}

		kept = append(kept, message)
	}
	return kept, nil
}

// responsesReasoningTexts returns the text of every reasoning item in the
// request, in order.
//
// A thinking-mode upstream requires the reasoning it produced to be passed back
// and refuses the request otherwise:
//
//	The `reasoning_content` in the thinking mode must be passed back to the API.
//
// The text lives in one of two places. An upstream that speaks Responses
// natively fills content with reasoning_text parts; an item this gateway built
// from a chat completions reasoning_content carries only a summary, because that
// is all the response converter has to work with.
func responsesReasoningTexts(req *dto.OpenAIResponsesRequest) []string {
	if req == nil || !rawJSONPresent(req.Input) || kitutil.GetJsonType(req.Input) != "array" {
		return nil
	}
	var items []map[string]any
	if err := kitutil.Unmarshal(req.Input, &items); err != nil {
		return nil
	}

	texts := make([]string, 0)
	for _, item := range items {
		if strings.TrimSpace(kitutil.Interface2String(item["type"])) != responsesInputTypeReasoning {
			continue
		}
		text := reasoningPartsText(item["content"], responsesReasoningTextPart)
		if text == "" {
			text = reasoningPartsText(item["summary"], responsesSummaryTextPart)
		}
		texts = append(texts, text)
	}
	return texts
}

func reasoningPartsText(raw any, partType string) string {
	parts, isList := raw.([]any)
	if !isList {
		return ""
	}
	var text strings.Builder
	for _, rawPart := range parts {
		part, isObject := rawPart.(map[string]any)
		if !isObject {
			continue
		}
		if strings.TrimSpace(kitutil.Interface2String(part["type"])) != partType {
			continue
		}
		text.WriteString(kitutil.Interface2String(part["text"]))
	}
	return text.String()
}

// isReasoningPlaceholder reports whether a converted message is all that is
// left of a reasoning item. Such an item has no role, so it becomes a user
// message holding either nothing at all (a summary-only item, whose summary the
// request converter never reads) or reasoning_text parts.
func isReasoningPlaceholder(message dto.Message) bool {
	if message.Role != "user" || len(message.ToolCalls) > 0 {
		return false
	}
	switch content := message.Content.(type) {
	case string:
		return strings.TrimSpace(content) == ""
	case []any:
		if len(content) == 0 {
			return false
		}
		for _, rawPart := range content {
			part, isObject := rawPart.(map[string]any)
			if !isObject {
				return false
			}
			if strings.TrimSpace(kitutil.Interface2String(part["type"])) != responsesReasoningTextPart {
				return false
			}
		}
		return true
	}
	return false
}

// normalizeChatContentPart repairs part shapes that Responses allows and chat
// completions does not.
//
// Responses accepts a bare URL string for image_url - Codex sends a pasted
// screenshot that way, as a data: URI - while chat completions only takes the
// object form. Sending the string makes the whole content array fail its union,
// and the error that surfaces names the other branch:
//
//	[invalid_request_error] Input should be a valid string
//
// which points at content rather than at the part that caused it.
func normalizeChatContentPart(part map[string]any) map[string]any {
	if strings.TrimSpace(kitutil.Interface2String(part["type"])) != dto.ContentTypeImageURL {
		return part
	}
	imageURL, isString := part[dto.ContentTypeImageURL].(string)
	if !isString || imageURL == "" {
		return part
	}
	repaired := make(map[string]any, len(part))
	for key, value := range part {
		repaired[key] = value
	}
	repaired[dto.ContentTypeImageURL] = map[string]any{"url": imageURL}
	return repaired
}

// adoptOrphanToolResult turns a tool result that answers no call into an
// ordinary user message.
//
// ChatGPT opens an automation turn with a function_call_output that has no
// call_id and no preceding function_call: the "call" never happened, the
// platform injected the event. Chat completions has no such shape - a tool
// message needs a tool_call_id, and an assistant message with tool_calls has to
// precede it - so the request is refused, as "Field required" when the id is
// missing or as a complaint about an unanswered tool call when it is present
// but unmatched.
//
// The output text is the only real input of that turn, so it is carried over
// verbatim rather than dropped, and nothing is invented: no fabricated call id,
// no fabricated assistant tool call.
func adoptOrphanToolResult(message dto.Message, answeredCalls map[string]struct{}, index int) dto.Message {
	if message.Role != "tool" {
		return message
	}
	if _, answers := answeredCalls[message.ToolCallId]; answers && message.ToolCallId != "" {
		return message
	}
	kitutil.LogError(fmt.Sprintf("responses to chat conversion turned an unmatched tool result into a user message at messages[%d] (tool_call_id %q)", index, message.ToolCallId))
	return dto.Message{Role: "user", Content: message.Content}
}
