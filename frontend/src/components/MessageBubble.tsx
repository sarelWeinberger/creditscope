import React, { useState } from "react";
import { ChatMessage, ThinkingVisibility } from "../types";
import ThinkingPanel from "./ThinkingPanel";

interface MessageBubbleProps {
  message: ChatMessage;
  cotVisibility?: ThinkingVisibility;
}

function MarkdownText({ content }: { content: string }) {
  // Simple markdown rendering: bold, code, lists
  const lines = content.split("\n");
  return (
    <div className="space-y-1">
      {lines.map((line, i) => {
        if (line.startsWith("# "))
          return (
            <h1 key={i} className="text-lg font-bold text-white mt-2">
              {line.slice(2)}
            </h1>
          );
        if (line.startsWith("## "))
          return (
            <h2 key={i} className="text-base font-semibold text-gray-100 mt-2">
              {line.slice(3)}
            </h2>
          );
        if (line.startsWith("### "))
          return (
            <h3 key={i} className="text-sm font-semibold text-gray-200 mt-1">
              {line.slice(4)}
            </h3>
          );
        if (line.startsWith("- ") || line.startsWith("* "))
          return (
            <div key={i} className="flex gap-2 text-sm text-gray-300">
              <span className="text-blue-400 mt-0.5">•</span>
              <span>{renderInline(line.slice(2))}</span>
            </div>
          );
        if (line.startsWith("**") && line.endsWith("**"))
          return (
            <p key={i} className="text-sm font-semibold text-gray-100">
              {line.slice(2, -2)}
            </p>
          );
        if (line === "") return <div key={i} className="h-2" />;
        return (
          <p key={i} className="text-sm text-gray-300 leading-relaxed">
            {renderInline(line)}
          </p>
        );
      })}
    </div>
  );
}

function renderInline(text: string): React.ReactNode {
  // Handle **bold** and `code`
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return (
        <strong key={i} className="text-white font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    if (part.startsWith("`") && part.endsWith("`"))
      return (
        <code key={i} className="bg-gray-700 text-green-400 px-1 rounded text-xs font-mono">
          {part.slice(1, -1)}
        </code>
      );
    return part;
  });
}

function MessageBubbleInner({ message, cotVisibility }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const [showToolDetails, setShowToolDetails] = useState(false);

  const toolCount = message.toolCalls?.length || 0;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} group`}>
      <div className={`max-w-[85%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-1`}>
        {/* Thinking panel (assistant only) */}
        {!isUser && message.thinking !== undefined && cotVisibility !== ThinkingVisibility.Hidden && (
          <ThinkingPanel
            content={message.thinking}
            tokensUsed={message.thinkingTokens}
            durationMs={message.thinkingDurationMs}
            wasEnforced={message.wasThinkingEnforced}
            isStreaming={message.isStreaming && !message.content}
            visibility={cotVisibility || ThinkingVisibility.Collapsed}
          />
        )}

        {/* Message bubble */}
        <div
          className={`px-4 py-3 rounded-2xl ${
            isUser
              ? "bg-blue-600 text-white rounded-br-sm"
              : "bg-gray-800 text-gray-100 rounded-bl-sm"
          } ${message.error ? "border border-red-500" : ""}`}
        >
          {message.isStreaming && !message.content ? (
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <span className="animate-spin inline-block w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full" />
              <span>Analyzing...</span>
            </div>
          ) : message.error ? (
            <p className="text-sm text-red-400">{message.error}</p>
          ) : isUser ? (
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          ) : (
            <MarkdownText content={message.content} />
          )}
        </div>

        {/* Footer row: timestamp, tool badges */}
        <div className={`flex items-center gap-2 px-1 ${isUser ? "justify-end" : "justify-start"}`}>
          <span className="text-xs text-gray-600">
            {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>

          {toolCount > 0 && (
            <button
              onClick={() => setShowToolDetails((v) => !v)}
              className="flex items-center gap-1 text-xs bg-gray-800 border border-gray-700 rounded-full px-2 py-0.5 text-blue-400 hover:text-blue-300 transition-colors"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18"
                />
              </svg>
              {toolCount} tool{toolCount > 1 ? "s" : ""}
            </button>
          )}
        </div>

        {/* Tool call detail (expandable) */}
        {showToolDetails && toolCount > 0 && (
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-3 text-xs w-full max-w-lg">
            {message.toolCalls?.map((step, i) => (
              <div key={i} className="mb-2 last:mb-0">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-blue-400">{step.tool_name}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-xs ${
                      step.success ? "bg-green-900 text-green-400" : "bg-red-900 text-red-400"
                    }`}
                  >
                    {step.duration_ms.toFixed(0)}ms
                  </span>
                </div>
                {step.error && <p className="text-red-400">{step.error}</p>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const MessageBubble = React.memo(MessageBubbleInner);
export default MessageBubble;
