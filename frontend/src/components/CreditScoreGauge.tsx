import React, { useEffect, useRef, useState } from "react";

interface CreditScoreGaugeProps {
  score: number;
  grade?: string;
  size?: "sm" | "md" | "lg";
  animated?: boolean;
}

const SCORE_MIN = 300;
const SCORE_MAX = 850;

function scoreToColor(score: number): string {
  if (score >= 750) return "#22c55e";       // green-500
  if (score >= 700) return "#84cc16";       // lime-500
  if (score >= 650) return "#eab308";       // yellow-500
  if (score >= 600) return "#f97316";       // orange-500
  if (score >= 550) return "#ef4444";       // red-500
  return "#dc2626";                          // red-600
}

function gradeLabel(score: number): string {
  if (score >= 750) return "Excellent";
  if (score >= 700) return "Good";
  if (score >= 650) return "Fair";
  if (score >= 600) return "Poor";
  if (score >= 550) return "Very Poor";
  return "Extremely Poor";
}

function scoreToAngle(score: number): number {
  const pct = (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN);
  // Arc from 225° to -45° (270° sweep)
  return 225 - pct * 270;
}

const SIZE_CONFIG = {
  sm: { width: 140, height: 80, radius: 56, strokeWidth: 10, fontSize: 22, labelSize: 10 },
  md: { width: 200, height: 115, radius: 80, strokeWidth: 14, fontSize: 30, labelSize: 12 },
  lg: { width: 260, height: 150, radius: 105, strokeWidth: 18, fontSize: 40, labelSize: 14 },
};

export default function CreditScoreGauge({
  score,
  grade,
  size = "md",
  animated = true,
}: CreditScoreGaugeProps) {
  const { width, height, radius, strokeWidth, fontSize, labelSize } = SIZE_CONFIG[size];
  const cx = width / 2;
  const cy = height * 0.88;
  const [displayScore, setDisplayScore] = useState(animated ? SCORE_MIN : score);

  // Animate score on mount / change
  useEffect(() => {
    if (!animated) {
      setDisplayScore(score);
      return;
    }
    const duration = 800;
    const start = Date.now();
    const from = displayScore;
    const to = score;
    const tick = () => {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayScore(Math.round(from + (to - from) * eased));
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [score]);

  const color = scoreToColor(displayScore);
  const label = gradeLabel(displayScore);

  // Build SVG arc path
  function polarToCart(angleDeg: number, r: number) {
    const rad = ((angleDeg - 90) * Math.PI) / 180;
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad),
    };
  }

  function arcPath(startAngle: number, endAngle: number, r: number) {
    const s = polarToCart(startAngle, r);
    const e = polarToCart(endAngle, r);
    const large = Math.abs(endAngle - startAngle) > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
  }

  const startAngle = -45; // right end (300 score)
  const fullEndAngle = 225; // left end (850 score)
  const progressEndAngle = scoreToAngle(displayScore);

  const bgPath = arcPath(-45, 225, radius);
  const progressPath = arcPath(-45, progressEndAngle, radius);

  return (
    <div className="flex flex-col items-center select-none">
      <svg width={width} height={height} overflow="visible">
        {/* Background track */}
        <path
          d={bgPath}
          fill="none"
          stroke="#374151"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Colored progress arc */}
        <path
          d={progressPath}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          style={{ transition: "d 0.1s ease-out, stroke 0.3s ease" }}
        />

        {/* Score label */}
        <text
          x={cx}
          y={cy - 8}
          textAnchor="middle"
          fontSize={fontSize}
          fontWeight="700"
          fill="white"
        >
          {displayScore}
        </text>

        {/* Grade letter */}
        {grade && (
          <text
            x={cx}
            y={cy + labelSize + 2}
            textAnchor="middle"
            fontSize={labelSize + 2}
            fontWeight="600"
            fill={color}
          >
            {grade}
          </text>
        )}
      </svg>

      {/* Label below */}
      <p
        className="text-xs font-medium mt-1"
        style={{ color }}
      >
        {label}
      </p>

      {/* Score range labels */}
      <div
        className="flex justify-between w-full mt-0.5 text-xs text-gray-600"
        style={{ maxWidth: width }}
      >
        <span>300</span>
        <span>850</span>
      </div>
    </div>
  );
}
