/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Banking dark theme colors
        banking: {
          50: "#f0f4ff",
          100: "#dce9ff",
          200: "#b9d4ff",
          300: "#84b5ff",
          400: "#468aff",
          500: "#1a5fff",
          600: "#0040f5",
          700: "#0031d1",
          800: "#002aaa",
          900: "#002185",
          950: "#001452",
        },
        risk: {
          excellent: "#22c55e",
          good: "#84cc16",
          fair: "#eab308",
          poor: "#f97316",
          very_poor: "#ef4444",
          extreme: "#dc2626",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "Cascadia Code",
          "monospace",
        ],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "typewriter": "typewriter 0.05s steps(1) infinite",
      },
      keyframes: {
        typewriter: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
      },
      boxShadow: {
        "glow-blue": "0 0 20px rgba(59, 130, 246, 0.3)",
        "glow-purple": "0 0 20px rgba(168, 85, 247, 0.3)",
        "glow-green": "0 0 20px rgba(34, 197, 94, 0.3)",
      },
    },
  },
  plugins: [],
};
