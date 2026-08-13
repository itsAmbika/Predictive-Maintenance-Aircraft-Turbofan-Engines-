import { useMemo } from "react";
import { motion } from "framer-motion";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, ArrowDownRight, ArrowUpRight, Wrench } from "lucide-react";
import { clsx } from "clsx";
import type { EnginePrediction } from "../types";
import { RiskBadge } from "./RiskBadge";
import { AnimatedNumber } from "./AnimatedNumber";

const RISK_HEX = { LOW: "#35e28f", MEDIUM: "#ffb547", HIGH: "#ff5470" } as const;

function StatTile({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-[var(--border)] bg-white/[0.03] px-4 py-3">
      <p className="text-[11px] uppercase tracking-wider text-[var(--text-dim)]">{label}</p>
      <div className="mt-1 font-mono-num text-lg font-semibold text-[var(--text-bright)]">{children}</div>
    </div>
  );
}

export function EngineDetail({ engine }: { engine: EnginePrediction }) {
  const color = RISK_HEX[engine.risk];
  const chartData = useMemo(
    () => engine.health_trend.map((p) => ({ cycle: p.cycle, health: p.health_indicator })),
    [engine.health_trend]
  );
  const failProba = engine.fail_within_20_proba;

  return (
    <motion.div
      key={engine.unit_number}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={clsx("glass rounded-2xl p-5", engine.risk === "HIGH" && "glow-red")}
    >
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[var(--border)] pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-lg font-semibold text-[var(--text-bright)]">Engine #{engine.unit_number}</h2>
            <RiskBadge risk={engine.risk} pulse />
          </div>
          <p className="mt-1 flex items-center gap-1.5 text-xs text-[var(--text-dim)]">
            <Wrench className="h-3.5 w-3.5" /> {engine.risk_action}
          </p>
        </div>
        <p className="text-xs text-[var(--text-dim)]">Last observed cycle: <span className="font-mono-num text-[var(--text-bright)]">{engine.last_cycle}</span></p>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatTile label="Predicted RUL">
          <span style={{ color }}>{engine.predicted_rul.toFixed(1)}</span> <span className="text-xs text-[var(--text-dim)]">cyc</span>
          {engine.rul_low != null && engine.rul_high != null && (
            <p className="mt-0.5 text-[11px] font-normal text-[var(--text-dim)]">
              [{engine.rul_low.toFixed(0)}–{engine.rul_high.toFixed(0)}] 80% interval
            </p>
          )}
        </StatTile>
        <StatTile label="Health score">
          <AnimatedNumber value={engine.health_score} decimals={0} suffix="%" />
        </StatTile>
        <StatTile label="P(fail ≤ 20 cyc)">
          {failProba != null ? (
            <span className={failProba > 0.5 ? "text-[var(--red)]" : "text-[var(--text-bright)]"}>
              <AnimatedNumber value={failProba * 100} decimals={0} suffix="%" />
            </span>
          ) : (
            <span className="text-[var(--text-dim)]">n/a</span>
          )}
        </StatTile>
        <StatTile label="Risk category">
          <span style={{ color }}>{engine.risk}</span>
        </StatTile>
      </div>

      <div className="mt-5">
        <p className="mb-2 text-[11px] uppercase tracking-wider text-[var(--text-dim)]">Health indicator over time</p>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="healthFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(148,178,255,0.08)" vertical={false} />
            <XAxis dataKey="cycle" tick={{ fill: "#7c88ab", fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#7c88ab", fontSize: 11 }} axisLine={false} tickLine={false} width={40} />
            <Tooltip
              contentStyle={{ background: "#0b0f1a", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#7c88ab" }}
              itemStyle={{ color: "#cdd6f0" }}
            />
            <Area type="monotone" dataKey="health" stroke={color} strokeWidth={2} fill="url(#healthFill)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {engine.top_factors.length > 0 && (
        <div className="mt-5">
          <p className="mb-2 flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-[var(--text-dim)]">
            <AlertTriangle className="h-3.5 w-3.5" /> Top factors driving this prediction (SHAP)
          </p>
          <div className="space-y-2">
            {(() => {
              const maxAbs = Math.max(...engine.top_factors.map((f) => Math.abs(f.impact)), 0.001);
              return engine.top_factors.map((f) => (
                <div key={f.feature} className="flex items-center gap-3">
                  <div className="w-44 shrink-0 truncate text-xs text-[var(--text)]" title={f.feature}>
                    {f.feature}
                  </div>
                  <div className="relative flex-1">
                    <div className="h-2 rounded-full bg-white/5" />
                    <div
                      className={clsx(
                        "absolute inset-y-0 rounded-full",
                        f.direction === "lowers_rul" ? "bg-[var(--red)]" : "bg-[var(--green)]"
                      )}
                      style={{ width: `${(Math.abs(f.impact) / maxAbs) * 100}%` }}
                    />
                  </div>
                  <div className="flex w-16 shrink-0 items-center justify-end gap-1 font-mono-num text-xs">
                    {f.direction === "lowers_rul" ? (
                      <ArrowDownRight className="h-3 w-3 text-[var(--red)]" />
                    ) : (
                      <ArrowUpRight className="h-3 w-3 text-[var(--green)]" />
                    )}
                    {f.impact.toFixed(1)}
                  </div>
                </div>
              ));
            })()}
          </div>
        </div>
      )}
    </motion.div>
  );
}
