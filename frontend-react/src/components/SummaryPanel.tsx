import { useMemo } from "react";
import { motion } from "framer-motion";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import type { EnginePrediction } from "../types";
import { AnimatedNumber } from "./AnimatedNumber";
import { Gauge, HeartPulse, ShieldAlert, ShieldCheck, ShieldQuestion } from "lucide-react";

const RISK_COLOR = { LOW: "#35e28f", MEDIUM: "#ffb547", HIGH: "#ff5470" } as const;

function KpiCard({
  icon,
  label,
  value,
  decimals = 0,
  suffix = "",
  accent,
  delay = 0,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  decimals?: number;
  suffix?: string;
  accent: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: "easeOut" }}
      className="glass flex items-center gap-3 rounded-xl px-4 py-3.5"
    >
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg" style={{ background: `${accent}1a`, color: accent }}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-[11px] uppercase tracking-wider text-[var(--text-dim)]">{label}</p>
        <p className="font-mono-num text-xl font-semibold text-[var(--text-bright)]">
          <AnimatedNumber value={value} decimals={decimals} suffix={suffix} />
        </p>
      </div>
    </motion.div>
  );
}

export function SummaryPanel({ engines }: { engines: EnginePrediction[] }) {
  const stats = useMemo(() => {
    const counts = { LOW: 0, MEDIUM: 0, HIGH: 0 };
    let rulSum = 0;
    engines.forEach((e) => {
      counts[e.risk] += 1;
      rulSum += e.predicted_rul;
    });
    return {
      counts,
      avgRul: engines.length ? rulSum / engines.length : 0,
      total: engines.length,
    };
  }, [engines]);

  const pieData = [
    { name: "Low", value: stats.counts.LOW, color: RISK_COLOR.LOW },
    { name: "Medium", value: stats.counts.MEDIUM, color: RISK_COLOR.MEDIUM },
    { name: "High", value: stats.counts.HIGH, color: RISK_COLOR.HIGH },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_180px]">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <KpiCard icon={<Gauge className="h-4.5 w-4.5" />} label="Engines scored" value={stats.total} accent="#4f8cff" delay={0} />
        <KpiCard icon={<HeartPulse className="h-4.5 w-4.5" />} label="Avg predicted RUL" value={stats.avgRul} decimals={1} suffix=" cyc" accent="#38f2e0" delay={0.05} />
        <KpiCard icon={<ShieldCheck className="h-4.5 w-4.5" />} label="Low risk" value={stats.counts.LOW} accent="#35e28f" delay={0.1} />
        <KpiCard icon={<ShieldQuestion className="h-4.5 w-4.5" />} label="Medium risk" value={stats.counts.MEDIUM} accent="#ffb547" delay={0.15} />
      </div>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2, duration: 0.4 }}
        className="glass relative flex items-center justify-center rounded-xl p-2"
      >
        {stats.counts.HIGH > 0 && (
          <div className="absolute left-3 top-3 flex items-center gap-1.5 text-[11px] font-semibold text-[var(--red)]">
            <ShieldAlert className="h-3.5 w-3.5" /> {stats.counts.HIGH} high
          </div>
        )}
        <ResponsiveContainer width="100%" height={150}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={42} outerRadius={62} paddingAngle={3} strokeWidth={0}>
              {pieData.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: "#0b0f1a", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
              itemStyle={{ color: "#cdd6f0" }}
            />
          </PieChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );
}
