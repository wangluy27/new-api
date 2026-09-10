package oairesponses

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/QuantumNous/new-api/relaykit/dto"
	sharedresponses "github.com/QuantumNous/new-api/relaykit/relayconvert/internal/shared/responses"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
)

const (
	responsesInputTypeFunctionCall       = "function_call"
	responsesInputTypeFunctionCallOutput = "function_call_output"
	responsesInputTypeCustomToolCall     = "custom_tool_call"
	responsesInputTypeCustomToolOutput   = "custom_tool_call_output"

	// ChatGPT wraps a group of function tools in these Responses tool types.
	responsesToolTypeNamespace = "namespace"
	responsesToolTypeMCPServer = "mcp_server"

	// Cap on the dropped tool JSON echoed into the log; tool schemas can be tens
	// of kilobytes and only the head identifies the shape.
	droppedToolLogBytes = 2048
)

const (
	ResponsesInputTypeFunctionCall       = responsesInputTypeFunctionCall
	ResponsesInputTypeFunctionCallOutput = responsesInputTypeFunctionCallOutput
	ResponsesInputTypeCustomToolCall     = responsesInputTypeCustomToolCall
	ResponsesInputTypeCustomToolOutput   = responsesInputTypeCustomToolOutput
)

func ResponsesRequestToChatCompletionsRequest(req *dto.OpenAIResponsesRequest) (*dto.GeneralOpenAIRequest, error) {
	if req == nil {
		return nil, errors.New("request is nil")
	}
	if req.Model == "" {
		return nil, errors.New("model is required")
	}
	if err := validateResponsesRequestChatUnsupportedFields(req); err != nil {
		return nil, err
	}

	messages, err := responsesRequestMessagesToChat(req)
	if err != nil {
		return nil, err
	}

	tools, err := responsesRequestToolsToChat(req.Tools)
	if err != nil {
		return nil, err
	}

	toolChoice, err := responsesRequestToolChoiceToChat(req.ToolChoice)
	if err != nil {
		return nil, err
	}

	responseFormat, err := responsesRequestTextToChatResponseFormat(req.Text)
	if err != nil {
		return nil, err
	}

	out := &dto.GeneralOpenAIRequest{
		Model:                req.Model,
		Messages:             messages,
		Stream:               req.Stream,
		StreamOptions:        req.StreamOptions,
		MaxCompletionTokens:  req.MaxOutputTokens,
		Temperature:          req.Temperature,
		TopP:                 req.TopP,
		TopLogProbs:          req.TopLogProbs,
		ResponseFormat:       responseFormat,
		Tools:                tools,
		ToolChoice:           toolChoice,
		User:                 req.User,
		Store:                req.Store,
		Metadata:             req.Metadata,
		SafetyIdentifier:     req.SafetyIdentifier,
		PromptCacheRetention: req.PromptCacheRetention,
		EnableThinking:       req.EnableThinking,
		ThinkingBudget:       req.ThinkingBudget,
	}

	if req.Reasoning != nil {
		out.ReasoningEffort = req.Reasoning.Effort
	}
	if req.ServiceTier != "" {
		out.ServiceTier, _ = kitutil.Marshal(req.ServiceTier)
	}
	if len(req.ParallelToolCalls) > 0 && kitutil.GetJsonType(req.ParallelToolCalls) == "boolean" {
		var parallelToolCalls bool
		if err := kitutil.Unmarshal(req.ParallelToolCalls, &parallelToolCalls); err == nil {
			out.ParallelTooCalls = &parallelToolCalls
		}
	}
	if len(req.PromptCacheKey) > 0 && kitutil.GetJsonType(req.PromptCacheKey) == "string" {
		var promptCacheKey string
		if err := kitutil.Unmarshal(req.PromptCacheKey, &promptCacheKey); err == nil {
			out.PromptCacheKey = promptCacheKey
		}
	}

	return out, nil
}

func validateResponsesRequestChatUnsupportedFields(req *dto.OpenAIResponsesRequest) error {
	unsupported := make([]string, 0, 4)
	if rawJSONPresent(req.Conversation) {
		unsupported = append(unsupported, "conversation")
	}
	if strings.TrimSpace(req.PreviousResponseID) != "" {
		unsupported = append(unsupported, "previous_response_id")
	}
	if rawJSONPresent(req.Prompt) {
		unsupported = append(unsupported, "prompt")
	}
	if rawJSONPresent(req.ContextManagement) {
		unsupported = append(unsupported, "context_management")
	}
	if len(unsupported) > 0 {
		return fmt.Errorf("responses to chat conversion does not support stateful fields: %s", strings.Join(unsupported, ", "))
	}
	return nil
}

func ValidateRequestChatUnsupportedFields(req *dto.OpenAIResponsesRequest) error {
	return validateResponsesRequestChatUnsupportedFields(req)
}

func responsesRequestMessagesToChat(req *dto.OpenAIResponsesRequest) ([]dto.Message, error) {
	messages := make([]dto.Message, 0)
	if rawJSONPresent(req.Instructions) {
		instructions, err := responsesJSONString(req.Instructions)
		if err != nil {
			return nil, fmt.Errorf("invalid instructions: %w", err)
		}
		if strings.TrimSpace(instructions) != "" {
			messages = append(messages, dto.Message{Role: "system", Content: instructions})
		}
	}

	if !rawJSONPresent(req.Input) {
		return messages, nil
	}

	switch kitutil.GetJsonType(req.Input) {
	case "string":
		input, err := responsesJSONString(req.Input)
		if err != nil {
			return nil, fmt.Errorf("invalid input string: %w", err)
		}
		messages = append(messages, dto.Message{Role: "user", Content: input})
		return messages, nil
	case "array":
		var items []map[string]any
		if err := kitutil.Unmarshal(req.Input, &items); err != nil {
			return nil, fmt.Errorf("invalid input array: %w", err)
		}
		for _, item := range items {
			nextMessages, err := responsesInputItemToChatMessages(item, messages)
			if err != nil {
				return nil, err
			}
			messages = nextMessages
		}
		return messages, nil
	default:
		return nil, fmt.Errorf("unsupported responses input type %q", kitutil.GetJsonType(req.Input))
	}
}

func responsesInputItemToChatMessages(item map[string]any, messages []dto.Message) ([]dto.Message, error) {
	itemType := strings.TrimSpace(kitutil.Interface2String(item["type"]))
	switch itemType {
	case responsesInputTypeFunctionCall:
		toolCall, err := responsesFunctionCallItemToChatToolCall(item)
		if err != nil {
			return nil, err
		}
		return appendToolCallToLastAssistant(messages, toolCall), nil
	case responsesInputTypeCustomToolCall:
		toolCall, err := responsesCustomToolCallItemToChatToolCall(item)
		if err != nil {
			return nil, err
		}
		return appendToolCallToLastAssistant(messages, toolCall), nil
	case responsesInputTypeFunctionCallOutput:
		callID := strings.TrimSpace(kitutil.Interface2String(item["call_id"]))
		content := responseToolOutputToChatContent(item["output"])
		return append(messages, dto.Message{Role: "tool", ToolCallId: callID, Content: content}), nil
	}

	role := strings.TrimSpace(kitutil.Interface2String(item["role"]))
	if role == "" {
		role = "user"
	}
	content, err := responsesInputContentToChatContent(item["content"])
	if err != nil {
		return nil, err
	}
	return append(messages, dto.Message{Role: role, Content: content}), nil
}

func responsesInputContentToChatContent(content any) (any, error) {
	if content == nil {
		return "", nil
	}

	switch value := content.(type) {
	case string:
		return value, nil
	case []any:
		return responsesContentPartsToChatContent(value)
	case []map[string]any:
		parts := make([]any, 0, len(value))
		for _, part := range value {
			parts = append(parts, part)
		}
		return responsesContentPartsToChatContent(parts)
	default:
		return content, nil
	}
}

func responsesContentPartsToChatContent(parts []any) (any, error) {
	chatParts := make([]any, 0, len(parts))
	var textOnly strings.Builder
	onlyText := true

	for _, rawPart := range parts {
		part, ok := rawPart.(map[string]any)
		if !ok {
			onlyText = false
			chatParts = append(chatParts, rawPart)
			continue
		}

		partType := strings.TrimSpace(kitutil.Interface2String(part["type"]))
		switch partType {
		case "input_text", "output_text", "text":
			text := kitutil.Interface2String(part["text"])
			textOnly.WriteString(text)
			chatParts = append(chatParts, map[string]any{
				"type": dto.ContentTypeText,
				"text": text,
			})
		case "input_image":
			onlyText = false
			chatParts = append(chatParts, map[string]any{
				"type":      dto.ContentTypeImageURL,
				"image_url": responsesImagePartToChatImageURL(part),
			})
		case "input_file":
			onlyText = false
			chatParts = append(chatParts, map[string]any{
				"type": dto.ContentTypeFile,
				"file": responsesFilePartToChatFile(part),
			})
		case "input_audio":
			onlyText = false
			chatParts = append(chatParts, map[string]any{
				"type":        dto.ContentTypeInputAudio,
				"input_audio": responsesPartPayload(part, "input_audio"),
			})
		case "input_video":
			onlyText = false
			chatParts = append(chatParts, map[string]any{
				"type":      dto.ContentTypeVideoUrl,
				"video_url": responsesVideoPartToChatVideoURL(part),
			})
		default:
			onlyText = false
			chatParts = append(chatParts, part)
		}
	}

	if onlyText {
		return textOnly.String(), nil
	}
	return chatParts, nil
}

func responsesFunctionCallItemToChatToolCall(item map[string]any) (dto.ToolCallRequest, error) {
	name := strings.TrimSpace(kitutil.Interface2String(item["name"]))
	if name == "" {
		return dto.ToolCallRequest{}, errors.New("function_call item is missing name")
	}
	// The client replays the call it received, namespace and name split apart;
	// the upstream only knows the flattened name it was offered.
	name = sharedresponses.JoinNamespacedTool(strings.TrimSpace(kitutil.Interface2String(item["namespace"])), name)
	return dto.ToolCallRequest{
		ID:   responsesCallID(item),
		Type: "function",
		Function: dto.FunctionRequest{
			Name:      name,
			Arguments: responsesArgumentsString(item["arguments"]),
		},
	}, nil
}

func responsesCustomToolCallItemToChatToolCall(item map[string]any) (dto.ToolCallRequest, error) {
	raw, err := kitutil.Marshal(item)
	if err != nil {
		return dto.ToolCallRequest{}, err
	}
	return dto.ToolCallRequest{
		ID:     responsesCallID(item),
		Type:   dto.CustomType,
		Custom: raw,
		Function: dto.FunctionRequest{
			Name:      strings.TrimSpace(kitutil.Interface2String(item["name"])),
			Arguments: responsesArgumentsString(item["input"]),
		},
	}, nil
}

func appendToolCallToLastAssistant(messages []dto.Message, toolCall dto.ToolCallRequest) []dto.Message {
	if len(messages) == 0 || messages[len(messages)-1].Role != "assistant" {
		messages = append(messages, dto.Message{Role: "assistant"})
	}

	idx := len(messages) - 1
	toolCalls := messages[idx].ParseToolCalls()
	toolCalls = append(toolCalls, toolCall)
	toolCallsRaw, _ := kitutil.Marshal(toolCalls)
	messages[idx].ToolCalls = toolCallsRaw
	return messages
}

// positionedTool pairs a Responses tool with the request path it came from, so
// errors and logs can point at the exact entry the client sent. name is the
// tool's effective chat completions name, which for a namespace member is
// qualified with its namespace.
type positionedTool struct {
	position string
	name     string
	tool     map[string]any
}

func responsesRequestToolsToChat(raw json.RawMessage) ([]dto.ToolCallRequest, error) {
	if !rawJSONPresent(raw) {
		return nil, nil
	}

	var tools []map[string]any
	if err := kitutil.Unmarshal(raw, &tools); err != nil {
		return nil, fmt.Errorf("invalid tools: %w", err)
	}

	// ChatGPT groups related tools under a "namespace" tool. Chat completions has
	// no grouping, so namespace members are lifted to the top level and keep their
	// own names, which is what the model calls them by.
	flattened := make([]positionedTool, 0, len(tools))
	for i, tool := range tools {
		position := fmt.Sprintf("tools[%d]", i)
		name := strings.TrimSpace(kitutil.Interface2String(tool["name"]))
		toolType := strings.TrimSpace(kitutil.Interface2String(tool["type"]))
		if toolType != responsesToolTypeNamespace && toolType != responsesToolTypeMCPServer {
			flattened = append(flattened, positionedTool{position: position, name: name, tool: tool})
			continue
		}
		members, ok := tool["tools"].([]any)
		if !ok {
			return nil, fmt.Errorf("%s declares tool type %q without a \"tools\" array", position, toolType)
		}
		for j, rawMember := range members {
			member, ok := rawMember.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("%s.tools[%d] is not a tool object", position, j)
			}
			// Member names are unique only inside their namespace — two
			// namespaces in the same request can both expose a "js" tool — so
			// the namespace has to survive into the flat name and be split back
			// out when the model calls the tool.
			memberName := strings.TrimSpace(kitutil.Interface2String(member["name"]))
			flattened = append(flattened, positionedTool{
				position: fmt.Sprintf("%s.tools[%d]", position, j),
				name:     sharedresponses.JoinNamespacedTool(name, memberName),
				tool:     member,
			})
		}
	}

	out := make([]dto.ToolCallRequest, 0, len(flattened))
	declaredAt := make(map[string]string, len(flattened))
	for _, entry := range flattened {
		toolType := strings.TrimSpace(kitutil.Interface2String(entry.tool["type"]))
		switch toolType {
		case "function":
			out = append(out, dto.ToolCallRequest{
				Type: "function",
				Function: dto.FunctionRequest{
					Name:        entry.name,
					Description: kitutil.Interface2String(entry.tool["description"]),
					Parameters:  entry.tool["parameters"],
				},
			})
		case dto.CustomType:
			// Chat completions keeps the discriminator in the outer "type" and
			// nests the rest of the custom tool under "custom".
			payload := make(map[string]any, len(entry.tool))
			for key, value := range entry.tool {
				if key != "type" {
					payload[key] = value
				}
			}
			if entry.name != "" {
				payload["name"] = entry.name
			}
			rawTool, err := kitutil.Marshal(payload)
			if err != nil {
				return nil, err
			}
			out = append(out, dto.ToolCallRequest{
				Type:   dto.CustomType,
				Custom: rawTool,
			})
		default:
			// Hosted Responses tools (web_search, file_search, code_interpreter,
			// ...) run on the provider side and have no chat completions
			// representation. Forwarding them makes the upstream reject the whole
			// request, and failing here would too: ChatGPT attaches web_search to
			// every request, so the client would never get a single answer
			// through. Drop the capability, keep the request, and log what was
			// lost so the gap is visible.
			if rawTool, err := kitutil.Marshal(entry.tool); err == nil {
				if len(rawTool) > droppedToolLogBytes {
					rawTool = rawTool[:droppedToolLogBytes]
				}
				kitutil.LogError(fmt.Sprintf("responses to chat conversion dropped unsupported tool type %q at %s: %s", toolType, entry.position, strings.ToValidUTF8(string(rawTool), "")))
			}
			continue
		}

		// Namespace qualification keeps members apart, but a top-level tool can
		// still claim the same name, and chat completions has no way to tell
		// them apart.
		if previous, exists := declaredAt[entry.name]; exists {
			return nil, fmt.Errorf("tool name %q at %s duplicates %s; chat completions requires unique tool names", entry.name, entry.position, previous)
		}
		declaredAt[entry.name] = entry.position
	}
	return out, nil
}

func responsesRequestToolChoiceToChat(raw json.RawMessage) (any, error) {
	if !rawJSONPresent(raw) {
		return nil, nil
	}
	if kitutil.GetJsonType(raw) == "string" {
		var choice string
		if err := kitutil.Unmarshal(raw, &choice); err != nil {
			return nil, fmt.Errorf("invalid tool_choice: %w", err)
		}
		return choice, nil
	}

	var choice map[string]any
	if err := kitutil.Unmarshal(raw, &choice); err != nil {
		return nil, fmt.Errorf("invalid tool_choice: %w", err)
	}
	if kitutil.Interface2String(choice["type"]) == "function" {
		name := strings.TrimSpace(kitutil.Interface2String(choice["name"]))
		if name != "" {
			return map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": name,
				},
			}, nil
		}
	}
	return choice, nil
}

func RequestToolChoiceToChat(raw json.RawMessage) (any, error) {
	return responsesRequestToolChoiceToChat(raw)
}

func responsesRequestTextToChatResponseFormat(raw json.RawMessage) (*dto.ResponseFormat, error) {
	if !rawJSONPresent(raw) {
		return nil, nil
	}

	var textConfig map[string]any
	if err := kitutil.Unmarshal(raw, &textConfig); err != nil {
		return nil, fmt.Errorf("invalid text config: %w", err)
	}
	format, ok := textConfig["format"].(map[string]any)
	if !ok {
		return nil, nil
	}

	formatType := strings.TrimSpace(kitutil.Interface2String(format["type"]))
	if formatType == "" {
		return nil, nil
	}

	out := &dto.ResponseFormat{Type: formatType}
	if formatType == "json_schema" {
		schemaRaw, err := kitutil.Marshal(format)
		if err != nil {
			return nil, err
		}
		out.JsonSchema = schemaRaw
	}
	return out, nil
}

func RequestTextToChatResponseFormat(raw json.RawMessage) (*dto.ResponseFormat, error) {
	return responsesRequestTextToChatResponseFormat(raw)
}

func responsesImagePartToChatImageURL(part map[string]any) any {
	if imageURL, ok := part["image_url"]; ok {
		if object, isObject := imageURL.(map[string]any); isObject {
			return object
		}
		// Responses allows image_url to be the bare URL string (a data: URI, for
		// instance); chat completions only accepts the object form.
		wrapped := map[string]any{"url": imageURL}
		if detail, hasDetail := part["detail"]; hasDetail {
			wrapped["detail"] = detail
		}
		return wrapped
	}
	imageURL := map[string]any{}
	for _, key := range []string{"url", "file_id", "detail"} {
		if value, ok := part[key]; ok {
			imageURL[key] = value
		}
	}
	if len(imageURL) == 0 {
		return part
	}
	return imageURL
}

func responsesFilePartToChatFile(part map[string]any) any {
	if file, ok := part["file"]; ok {
		return file
	}
	file := map[string]any{}
	for _, key := range []string{"file_id", "file_data", "filename", "file_url"} {
		if value, ok := part[key]; ok {
			file[key] = value
		}
	}
	if len(file) == 0 {
		return part
	}
	return file
}

func responsesVideoPartToChatVideoURL(part map[string]any) any {
	if videoURL, ok := part["video_url"]; ok {
		if videoURLMap, ok := videoURL.(map[string]any); ok {
			if url := kitutil.Interface2String(videoURLMap["url"]); url != "" {
				return url
			}
		}
		return videoURL
	}
	if url := kitutil.Interface2String(part["url"]); url != "" {
		return url
	}
	return responsesPartPayload(part, "video_url")
}

func responsesPartPayload(part map[string]any, key string) any {
	if value, ok := part[key]; ok {
		return value
	}
	payload := make(map[string]any, len(part))
	for k, value := range part {
		if k == "type" {
			continue
		}
		payload[k] = value
	}
	return payload
}

func responsesCallID(item map[string]any) string {
	callID := strings.TrimSpace(kitutil.Interface2String(item["call_id"]))
	if callID != "" {
		return callID
	}
	return strings.TrimSpace(kitutil.Interface2String(item["id"]))
}

func CallID(item map[string]any) string {
	return responsesCallID(item)
}

func responsesArgumentsString(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		raw, err := kitutil.Marshal(v)
		if err != nil {
			return kitutil.Interface2String(v)
		}
		return string(raw)
	}
}

func responseToolOutputToChatContent(value any) any {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		raw, err := kitutil.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(raw)
	}
}

func responsesJSONString(raw json.RawMessage) (string, error) {
	if kitutil.GetJsonType(raw) != "string" {
		return string(raw), nil
	}
	var value string
	if err := kitutil.Unmarshal(raw, &value); err != nil {
		return "", err
	}
	return value, nil
}

func rawJSONPresent(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	return kitutil.GetJsonType(raw) != "null"
}

func JSONString(raw json.RawMessage) (string, error) {
	return responsesJSONString(raw)
}

func RawJSONPresent(raw json.RawMessage) bool {
	return rawJSONPresent(raw)
}
