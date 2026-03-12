import React, { useRef, useEffect, useState, useCallback } from "react";
import { CoTConfig, ThinkingMode, ThinkingVisibility, Customer } from "../types";
import { useChat } from "../hooks/useChat";
import MessageBubble from "./MessageBubble";
import ThinkingBudgetSlider from "./ThinkingBudgetSlider";
import ToolExecutionPanel from "./ToolExecutionPanel";
import MoEExpertPanel from "./MoEExpertPanel";
import ImageUpload from "./ImageUpload";
import CustomerCard from "./CustomerCard";

interface ChatInterfaceProps {
  selectedCustomer?: Customer | null;
}

const DEFAULT_COT: CoTConfig = {
  mode: ThinkingMode.On,
  budget: "standard",
  visibility: ThinkingVisibility.Collapsed,
  enable_thinking: true,
};

export default function ChatInterface({ selectedCustomer }: ChatInterfaceProps) {
  const { messages, isConnected, isStreaming, sessionId, sendMessage, clearMessages } =
    useChat();

  const [cotConfig, setCotConfig] = useState<CoTConfig>(DEFAULT_COT);
  const [input, setInput] = useState("");
  const [pendingImages, setPendingImages] = useState<File[]>([]);
  const [showToolPanel, setShowToolPanel] = useState(false);
  const [showMoEPanel, setShowMoEPanel] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom (use "auto" to avoid expensive smooth-scroll
  // animations that fire on every streaming delta)
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    scrollTimeoutRef.current = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
    }, 80);
    return () => {
      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current);
    };
  }, [messages]);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text || isStreaming) return;
    sendMessage(text, pendingImages.length ? pendingImages : undefined, cotConfig);
    setInput("");
    setPendingImages([]);
    inputRef.current?.focus();
  }, [input, isStreaming, pendingImages, cotConfig, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const allToolCalls = messages.flatMap((m) => m.toolCalls || []);
  const lastMoETrace = messages
    .slice()
    .reverse()
    .find((m) => m.moeTrace)?.moeTrace;

  return (
    <div className="flex h-full bg-gray-950">
      {/* Left: Customer sidebar */}
      {selectedCustomer && (
        <div className="w-72 flex-shrink-0 border-r border-gray-800 overflow-y-auto bg-gray-900 p-4">
          <CustomerCard customer={selectedCustomer} compact />
        </div>
      )}

      {/* Center: Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-gray-900">
          <div className="flex items-center gap-3">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? "bg-green-400" : "bg-red-400"
              }`}
            />
            <span className="text-sm text-gray-400">
              {isConnected ? "Connected" : "Disconnected"} · Session{" "}
              <code className="text-xs text-blue-400">{sessionId.slice(0, 8)}</code>
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowToolPanel((v) => !v)}
              className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                showToolPanel
                  ? "bg-blue-600 border-blue-500 text-white"
                  : "border-gray-700 text-gray-400 hover:text-white"
              }`}
            >
              Tools {allToolCalls.length > 0 && `(${allToolCalls.length})`}
            </button>
            <button
              onClick={() => setShowMoEPanel((v) => !v)}
              className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                showMoEPanel
                  ? "bg-purple-600 border-purple-500 text-white"
                  : "border-gray-700 text-gray-400 hover:text-white"
              }`}
            >
              MoE
            </button>
            <button
              onClick={clearMessages}
              className="px-3 py-1 text-xs rounded-md border border-gray-700 text-gray-400 hover:text-white transition-colors"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Message list */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="text-4xl mb-4">🏦</div>
              <h2 className="text-xl font-semibold text-gray-300 mb-2">
                CreditScope AI
              </h2>
              <p className="text-gray-500 max-w-md text-sm">
                Ask about customer credit profiles, loan analysis, risk assessments, or
                upload financial documents for processing.
              </p>
              <div className="mt-6 grid grid-cols-2 gap-2 max-w-sm">
                {[
                  "Analyze customer #5's credit score",
                  "What's the DTI for customer #12?",
                  "Review loan eligibility for $50k personal loan",
                  "Show risk factors for customer #23",
                ].map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => setInput(prompt)}
                    className="text-xs text-left p-2 rounded-lg border border-gray-700 text-gray-400 hover:text-white hover:border-gray-600 transition-colors"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} cotVisibility={cotConfig.visibility} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Tool panel (inline) */}
        {showToolPanel && allToolCalls.length > 0 && (
          <div className="border-t border-gray-800 max-h-64 overflow-y-auto">
            <ToolExecutionPanel steps={allToolCalls} />
          </div>
        )}

        {/* MoE panel (inline) */}
        {showMoEPanel && (
          <div className="border-t border-gray-800 max-h-48">
            <MoEExpertPanel requestTrace={lastMoETrace} />
          </div>
        )}

        {/* Input area */}
        <div className="border-t border-gray-800 bg-gray-900 px-6 py-4">
          {/* CoT slider */}
          <div className="mb-3">
            <ThinkingBudgetSlider config={cotConfig} onChange={setCotConfig} />
          </div>

          {/* Image preview */}
          {pendingImages.length > 0 && (
            <div className="mb-2">
              <ImageUpload
                images={pendingImages}
                onChange={setPendingImages}
                compact
              />
            </div>
          )}

          {/* Text input row */}
          <div className="flex items-end gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about credit scores, loan analysis, risk assessment..."
                rows={3}
                disabled={isStreaming}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-500 resize-none focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
              />
            </div>
            <div className="flex flex-col gap-2">
              <ImageUpload
                images={pendingImages}
                onChange={setPendingImages}
                trigger
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isStreaming}
                className="px-4 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-white font-medium text-sm transition-colors"
              >
                {isStreaming ? (
                  <span className="flex items-center gap-2">
                    <span className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                    Thinking
                  </span>
                ) : (
                  "Send"
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Right: MoE Expert Panel (full) */}
      {showMoEPanel && (
        <div className="w-80 flex-shrink-0 border-l border-gray-800 overflow-y-auto bg-gray-900">
          <MoEExpertPanel requestTrace={lastMoETrace} fullView />
        </div>
      )}
    </div>
  );
}
