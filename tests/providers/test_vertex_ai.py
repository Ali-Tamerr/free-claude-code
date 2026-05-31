"""Tests for the Vertex AI provider and its request/response handling."""

from providers.vertex_ai.request import _openai_messages_to_contents


def test_openai_messages_to_contents_alternating_and_merging():
    # 1. Simple alternating roles
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    contents = _openai_messages_to_contents(messages)
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Hello"}]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [{"text": "Hi there!"}]
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"] == [{"text": "How are you?"}]

    # 2. Consecutive user messages (should be merged)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Are you there?"},
    ]
    contents = _openai_messages_to_contents(messages)
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Hello"}, {"text": "Are you there?"}]

    # 3. Consecutive tool response messages (should be merged)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "list_dir",
                        "arguments": '{"DirectoryPath": "."}',
                    },
                },
                {
                    "id": "tc2",
                    "type": "function",
                    "function": {
                        "name": "view_file",
                        "arguments": '{"AbsolutePath": "CLAUDE.md"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": "file1, file2",
        },
        {
            "role": "tool",
            "tool_call_id": "tc2",
            "content": "Line 1\nLine 2",
        },
    ]
    contents = _openai_messages_to_contents(messages)
    # The assistant message generates 1 model turn with tool calls
    # The two tool messages generate 1 user turn with two merged functionResponse parts
    assert len(contents) == 2
    assert contents[0]["role"] == "model"
    assert len(contents[0]["parts"]) == 2  # two functionCalls
    assert "functionCall" in contents[0]["parts"][0]
    assert "functionCall" in contents[0]["parts"][1]

    assert contents[1]["role"] == "user"
    assert len(contents[1]["parts"]) == 2  # two merged functionResponses
    assert contents[1]["parts"][0]["functionResponse"]["name"] == "list_dir"
    assert contents[1]["parts"][0]["functionResponse"]["response"] == {
        "content": "file1, file2"
    }
    assert contents[1]["parts"][1]["functionResponse"]["name"] == "view_file"
    assert contents[1]["parts"][1]["functionResponse"]["response"] == {
        "content": "Line 1\nLine 2"
    }

    # 4. Mixed user and tool message mapping safely (should not merge user text with tool functionResponse)
    messages = [
        {"role": "user", "content": "Text message"},
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": "some tool output",
        },
    ]
    contents = _openai_messages_to_contents(messages)
    # Since one has functionResponse and the other has text, they should NOT merge into the same turn
    # even though both map to role 'user'. This avoids Gemini API 400 part mixing restrictions.
    assert len(contents) == 2
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Text message"}]
    assert contents[1]["role"] == "user"
    assert "functionResponse" in contents[1]["parts"][0]


def test_openai_messages_to_contents_strips_interrupted_and_no_content():
    # Test stripping of "[Tool use interrupted]" and "(no content)" from strings
    messages = [
        {"role": "user", "content": "[Tool use interrupted] drain context"},
        {"role": "assistant", "content": "(no content)"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello [Tool use interrupted]"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "world (no content)"}],
        },
    ]
    contents = _openai_messages_to_contents(messages)
    assert len(contents) == 4

    # 1. "[Tool use interrupted] drain context" -> "drain context"
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "drain context"}]

    # 2. "(no content)" -> "" -> will be cleaned to [{"text": ""}]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [{"text": ""}]

    # 3. List with text: "hello [Tool use interrupted]" -> "hello"
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"] == [{"text": "hello"}]

    # 4. List with text: "world (no content)" -> "world"
    assert contents[3]["role"] == "model"
    assert contents[3]["parts"] == [{"text": "world"}]
