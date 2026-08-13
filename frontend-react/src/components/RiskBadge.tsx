import { clsx } from "clsx";
import type { RiskLevel } from "../types";

const STYLES: Record<RiskLevel, string> = {
  LOW: "bg-green-500/10 text-[var(--green)] border-green-400/30",
  MEDIUM: "bg-amber-500/10 text-[var(--amber)] border-amber-400/30",
  HIGH: "bg-red-500/10 text-[var(--red)] border-red-400/30",
};

const DOT: Record<RiskLevel, string> = {
  LOW: "bg-[var(--green)]",
  MEDIUM: "bg-[var(--amber)]",
  HIGH: "bg-[var(--red)]",
};

export function RiskBadge({ risk, pulse = false }: { risk: RiskLevel; pulse?: boolean }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider",
        STYLES[risk]
      )}
    >
      <span className={clsx("h-1.5 w-1.5 rounded-full", DOT[risk], pulse && risk === "HIGH" && "pulse-dot")} />
      {risk}
    </span>
  );
}
