import { ChatMessage, ConversationSummary, ConversationDetail } from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

export async function saveConversation(
  conversationId: string,
  title: string,
  messages: ChatMessage[]
): Promise<void> {
  const payload = {
    conversation_id: conversationId,
    title,
    messages: messages
      .filter((m) => !m.isStreaming)
      .map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        thinking: m.thinking || null,
        thinking_tokens: m.thinkingTokens || null,
        thinking_duration_ms: m.thinkingDurationMs || null,
        tool_calls: m.toolCalls || null,
        error: m.error || null,
        timestamp: m.timestamp instanceof Date ? m.timestamp.toISOString() : m.timestamp,
      })),
  };

  await fetch(`${API_BASE}/history`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload),
  });
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await fetch(`${API_BASE}/history`, {
    credentials: "include",
  });
  if (!response.ok) return [];
  return response.json();
}

export async function getConversation(id: string): Promise<ConversationDetail | null> {
  const response = await fetch(`${API_BASE}/history/${id}`, {
    credentials: "include",
  });
  if (!response.ok) return null;
  return response.json();
}

export async function deleteConversation(id: string): Promise<void> {
  await fetch(`${API_BASE}/history/${id}`, {
    method: "DELETE",
    credentials: "include",
  });
}

export async function renameConversation(id: string, title: string): Promise<void> {
  await fetch(`${API_BASE}/history/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ title }),
  });
}
