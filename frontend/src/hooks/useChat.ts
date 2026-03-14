import { useCallback, useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  ChatMessage,
  CoTConfig,
  ExecutionStep,
  MoERequestTrace,
  ThinkingMode,
  ThinkingVisibility,
  WebSocketEvent,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "/api";
const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/api/chat/ws`;
const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 5;

interface UseChatReturn {
  messages: ChatMessage[];
  isConnected: boolean;
  isStreaming: boolean;
  sessionId: string;
  sendMessage: (text: string, images?: File[], cotConfig?: CoTConfig) => void;
  clearMessages: () => void;
  reconnect: () => void;
  activeToolCalls: ExecutionStep[];
}

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error(`Failed to read ${file.name}`));
        return;
      }
      const commaIndex = result.indexOf(",");
      resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
    };
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function encodeImages(images?: File[]): Promise<string[] | undefined> {
  if (!images?.length) {
    return undefined;
  }
  return Promise.all(images.map(fileToBase64));
}

export function useChat(initialSessionId?: string): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId] = useState<string>(initialSessionId || uuidv4());
  const [activeToolCalls, setActiveToolCalls] = useState<ExecutionStep[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<number | null>(null);

  // Streaming state
  const streamingMessageId = useRef<string | null>(null);
  const thinkingBuffer = useRef<string>("");
  const responseBuffer = useRef<string>("");
  const thinkingStartTime = useRef<number | null>(null);
  const toolCallsBuffer = useRef<ExecutionStep[]>([]);

  const appendErrorMessage = useCallback((error: string) => {
    const errorId = streamingMessageId.current || uuidv4();
    streamingMessageId.current = null;
    setMessages((prev) => {
      const exists = prev.some((message) => message.id === errorId);
      if (exists) {
        return prev.map((message) =>
          message.id === errorId
            ? { ...message, isStreaming: false, error }
            : message
        );
      }

      return [
        ...prev,
        {
          id: errorId,
          role: "assistant",
          content: "",
          error,
          isStreaming: false,
          timestamp: new Date(),
        },
      ];
    });
  }, []);

  const resetStreamingState = useCallback((error?: string) => {
    setIsStreaming(false);
    setActiveToolCalls([]);
    toolCallsBuffer.current = [];
    responseBuffer.current = "";
    thinkingBuffer.current = "";
    thinkingStartTime.current = null;

    if (error) {
      appendErrorMessage(error);
    } else {
      streamingMessageId.current = null;
    }
  }, [appendErrorMessage]);

  const sendViaRest = useCallback(
    async (text: string, images?: File[], cotConfig?: CoTConfig) => {
      setIsStreaming(true);

      const responseId = uuidv4();
      streamingMessageId.current = responseId;
      setMessages((prev) => [
        ...prev,
        {
          id: responseId,
          role: "assistant",
          content: "",
          isStreaming: true,
          timestamp: new Date(),
        },
      ]);

      try {
        const encodedImages = await encodeImages(images);
        const response = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "include",
          body: JSON.stringify({
            message: text,
            images: encodedImages,
            session_id: sessionId,
            cot_config: cotConfig,
          }),
        });

        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(payload?.detail || `HTTP ${response.status}`);
        }

        const payload = (await response.json()) as {
          answer: string;
          thinking?: {
            content?: string;
            tokens_used?: number;
            duration_ms?: number;
            was_budget_enforced?: boolean;
          };
          execution_trace?: ExecutionStep[];
          moe_trace?: MoERequestTrace;
        };

        setMessages((prev) =>
          prev.map((message) =>
            message.id === responseId
              ? {
                  ...message,
                  content: payload.answer || "",
                  thinking: payload.thinking?.content,
                  thinkingTokens: payload.thinking?.tokens_used,
                  thinkingDurationMs: payload.thinking?.duration_ms,
                  wasThinkingEnforced: payload.thinking?.was_budget_enforced,
                  toolCalls: payload.execution_trace || [],
                  moeTrace: payload.moe_trace,
                  isStreaming: false,
                }
              : message
          )
        );
      } catch (error) {
        appendErrorMessage(error instanceof Error ? error.message : "Request failed");
      } finally {
        setIsStreaming(false);
        setActiveToolCalls([]);
        toolCallsBuffer.current = [];
        streamingMessageId.current = null;
      }
    },
    [appendErrorMessage, sessionId]
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttempts.current = 0;
      if (reconnectTimer.current !== null) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      if (streamingMessageId.current) {
        resetStreamingState("Connection lost while generating a response. Please try again.");
      }
      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.current += 1;
        reconnectTimer.current = window.setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      setIsConnected(false);
      if (streamingMessageId.current) {
        resetStreamingState("The chat connection encountered an error. Please try again.");
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data: WebSocketEvent = JSON.parse(event.data);
        handleWsEvent(data);
      } catch {
        // ignore parse errors
      }
    };
  }, [resetStreamingState]);

  const handleWsEvent = useCallback((event: WebSocketEvent) => {
    switch (event.type) {
      case "session_start":
        break;

      case "thinking_start":
        setIsStreaming(true);
        thinkingBuffer.current = "";
        thinkingStartTime.current = Date.now();

        // Create streaming message placeholder
        const thinkingMsgId = uuidv4();
        streamingMessageId.current = thinkingMsgId;
        toolCallsBuffer.current = [];
        setMessages((prev) => [
          ...prev,
          {
            id: thinkingMsgId,
            role: "assistant",
            content: "",
            thinking: "",
            isStreaming: true,
            timestamp: new Date(),
          },
        ]);
        break;

      case "thinking_delta":
        thinkingBuffer.current += event.content || "";
        if (streamingMessageId.current) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === streamingMessageId.current
                ? { ...m, thinking: thinkingBuffer.current }
                : m
            )
          );
        }
        break;

      case "thinking_end": {
        const durationMs = thinkingStartTime.current
          ? Date.now() - thinkingStartTime.current
          : event.duration_ms || 0;

        if (streamingMessageId.current) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === streamingMessageId.current
                ? {
                    ...m,
                    thinking: event.full_thinking_content || thinkingBuffer.current,
                    thinkingTokens: event.thinking_tokens || event.tokens_used || 0,
                    thinkingDurationMs: durationMs,
                  }
                : m
            )
          );
        }
        thinkingBuffer.current = "";
        break;
      }

      case "response_start":
        responseBuffer.current = "";
        // If no streaming message yet, create one
        if (!streamingMessageId.current) {
          const msgId = uuidv4();
          streamingMessageId.current = msgId;
          setMessages((prev) => [
            ...prev,
            {
              id: msgId,
              role: "assistant",
              content: "",
              isStreaming: true,
              timestamp: new Date(),
            },
          ]);
        }
        break;

      case "response_delta":
        responseBuffer.current += event.content || "";
        if (streamingMessageId.current) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === streamingMessageId.current
                ? { ...m, content: responseBuffer.current }
                : m
            )
          );
        }
        break;

      case "response_end":
        if (streamingMessageId.current) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === streamingMessageId.current
                ? {
                    ...m,
                    content: event.full_response || responseBuffer.current,
                    isStreaming: false,
                    toolCalls: toolCallsBuffer.current,
                  }
                : m
            )
          );
        }
        setIsStreaming(false);
        setActiveToolCalls([]);
        streamingMessageId.current = null;
        responseBuffer.current = "";
        toolCallsBuffer.current = [];
        break;

      case "tool_call": {
        const step: ExecutionStep = {
          step: toolCallsBuffer.current.length,
          tool_name:
            (event.tool_calls as any)?.[0]?.function?.name || "unknown",
          tool_input: {},
          tool_output: null,
          duration_ms: 0,
          success: true,
        };
        toolCallsBuffer.current = [...toolCallsBuffer.current, step];
        setActiveToolCalls([...toolCallsBuffer.current]);
        break;
      }

      case "done":
        setIsStreaming(false);
        streamingMessageId.current = null;
        break;

      case "error":
        setIsStreaming(false);
        appendErrorMessage(event.error || "Request failed");
        break;

      case "pong":
        break;
    }
  }, []);

  const sendMessage = useCallback(
    (text: string, images?: File[], cotConfig?: CoTConfig) => {
      if (isStreaming) return;

      // Add user message to history
      const userMsg: ChatMessage = {
        id: uuidv4(),
        role: "user",
        content: text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);

      if (!isConnected || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        void sendViaRest(text, images, cotConfig);
        return;
      }

      void (async () => {
        try {
          const encodedImages = await encodeImages(images);
          const payload = {
            type: "message",
            content: text,
            images: encodedImages,
            session_id: sessionId,
            cot_config: cotConfig
              ? {
                  mode: cotConfig.mode,
                  budget: cotConfig.budget,
                  visibility: cotConfig.visibility,
                  enable_thinking: cotConfig.enable_thinking,
                }
              : { mode: "on", budget: "standard", visibility: "collapsed", enable_thinking: true },
          };

          if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            await sendViaRest(text, images, cotConfig);
            return;
          }

          wsRef.current.send(JSON.stringify(payload));
        } catch (error) {
          appendErrorMessage(error instanceof Error ? error.message : "Image upload failed");
        }
      })();
    },
    [appendErrorMessage, isConnected, isStreaming, sendViaRest, sessionId]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    setActiveToolCalls([]);
  }, []);

  const reconnect = useCallback(() => {
    wsRef.current?.close();
    reconnectAttempts.current = 0;
    connect();
  }, [connect]);

  // Connect on mount
  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current !== null) {
        clearTimeout(reconnectTimer.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  // Ping every 30s to keep connection alive
  useEffect(() => {
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 30_000);
    return () => clearInterval(interval);
  }, []);

  return {
    messages,
    isConnected,
    isStreaming,
    sessionId,
    sendMessage,
    clearMessages,
    reconnect,
    activeToolCalls,
  };
}
