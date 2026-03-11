import React from "react";

interface AppErrorBoundaryState {
  errorMessage: string | null;
  errorStack: string | null;
}

export default class AppErrorBoundary extends React.Component<
  React.PropsWithChildren,
  AppErrorBoundaryState
> {
  state: AppErrorBoundaryState = {
    errorMessage: null,
    errorStack: null,
  };

  private handleWindowError = (event: ErrorEvent) => {
    this.setState({
      errorMessage: event.message || "Unknown browser error",
      errorStack: event.error?.stack || null,
    });
  };

  private handleUnhandledRejection = (event: PromiseRejectionEvent) => {
    const reason = event.reason;
    this.setState({
      errorMessage:
        reason instanceof Error
          ? reason.message
          : typeof reason === "string"
          ? reason
          : "Unhandled promise rejection",
      errorStack: reason instanceof Error ? reason.stack || null : null,
    });
  };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return {
      errorMessage: error.message,
      errorStack: error.stack || null,
    };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("CreditScope frontend error", error, info);
  }

  componentDidMount() {
    window.addEventListener("error", this.handleWindowError);
    window.addEventListener("unhandledrejection", this.handleUnhandledRejection);
  }

  componentWillUnmount() {
    window.removeEventListener("error", this.handleWindowError);
    window.removeEventListener("unhandledrejection", this.handleUnhandledRejection);
  }

  private handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.errorMessage) {
      return (
        <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center p-6">
          <div className="max-w-3xl w-full rounded-2xl border border-red-900 bg-gray-900 p-6 shadow-2xl">
            <p className="text-xs uppercase tracking-[0.2em] text-red-400 mb-2">
              Frontend Runtime Error
            </p>
            <h1 className="text-2xl font-semibold text-white mb-3">
              CreditScope failed to render
            </h1>
            <p className="text-sm text-gray-300 mb-4">
              The browser hit an exception while loading the UI. The details are shown below.
            </p>
            <div className="rounded-xl border border-gray-800 bg-black/30 p-4 mb-4">
              <p className="text-sm text-red-300 font-medium break-words">
                {this.state.errorMessage}
              </p>
              {this.state.errorStack && (
                <pre className="mt-3 whitespace-pre-wrap text-xs text-gray-400 overflow-x-auto">
                  {this.state.errorStack}
                </pre>
              )}
            </div>
            <button
              onClick={this.handleReload}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-sm font-medium text-white transition-colors"
            >
              Reload App
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}