package oairesponses

import (
	"fmt"
	"strings"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
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
// completions messages and removes what chat completions cannot express.
//
// A Responses "reasoning" item carries the model's own prior thinking as
// content parts of type reasoning_text, and it has no role, so it converts into
// a user message full of parts the upstream rejects outright:
//
//	messages[4]: unknown variant `reasoning_text`, expected one of `text`,
//	`image_url`, `file`
//
// Reasoning is provider-internal and cannot be replayed to a different provider
// in any case, so those parts are dropped, and a message left with nothing else
// is dropped with them rather than sent empty.
func ChatMessagesFromResponsesRequest(req *dto.OpenAIResponsesRequest) ([]dto.Message, error) {
	messages, err := responsesRequestMessagesToChat(req)
	if err != nil {
		return nil, err
	}

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
	for i, message := range messages {
		message = adoptOrphanToolResult(message, answeredCalls, i)
		parts, ok := message.Content.([]any)
		if !ok {
			// A string content, or a message carrying only tool calls.
			kept = append(kept, message)
			continue
		}

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

		// Tool calls are carried outside content, so a message can legitimately
		// end up with no parts left and still be worth sending.
		if len(supported) == 0 {
			if len(message.ToolCalls) == 0 {
				continue
			}
			// null, not an empty array: an assistant message that only carries
			// tool calls has always been sent with a null content, and an
			// upstream that accepts a string or null there rejects [].
			message.Content = nil
			kept = append(kept, message)
			continue
		}
		message.Content = supported
		kept = append(kept, message)
	}
	return kept, nil
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
