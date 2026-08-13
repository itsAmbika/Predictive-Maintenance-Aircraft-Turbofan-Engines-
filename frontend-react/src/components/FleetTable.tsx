import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { clsx } from "clsx";
import { ArrowDown, ArrowUp, Search } from "lucide-react";
import type { EnginePrediction, RiskFilter, SortDir, SortKey } from "../types";
import { RiskBadge } from "./RiskBadge";

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: "unit_number", label: "Engine" },
  { key: "last_cycle", label: "Last cycle" },
  { key: "predicted_rul", label: "Predicted RUL" },
  { key: "health_score", label: "Health" },
  { key: "risk", label: "Risk" },
];

const FILTERS: RiskFilter[] = ["ALL", "LOW", "MEDIUM", "HIGH"];

export function FleetTable({
  engines,
  selectedUnit,
  onSelect,
}: {
  engines: EnginePrediction[];
  selectedUnit: number | null;
  onSelect: (engine: EnginePrediction) => void;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("predicted_rul");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("ALL");
  const [search, setSearch] = useState("");

  const rows = useMemo(() => {
    let out = engines;
    if (riskFilter !== "ALL") out = out.filter((e) => e.risk === riskFilter);
    if (search.trim()) out = out.filter((e) => String(e.unit_number).includes(search.trim()));
    const dir = sortDir === "asc" ? 1 : -1;
    return [...out].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "string") return av.localeCompare(bv as string) * dir;
      return ((av as number) - (bv as number)) * dir;
    });
  }, [engines, riskFilter, search, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  return (
    <div className="glass flex min-h-0 flex-1 flex-col rounded-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] px-5 py-4">
        <h2 className="text-sm font-semibold text-[var(--text-bright)]">
          Fleet overview
          <span className="ml-2 text-xs font-normal text-[var(--text-dim)]">
            {rows.length === engines.length ? `(${engines.length})` : `(${rows.length} of ${engines.length})`}
          </span>
        </h2>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[var(--text-dim)]" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search engine #"
              className="w-36 rounded-lg border border-[var(--border)] bg-white/5 py-1.5 pl-8 pr-2.5 text-xs text-[var(--text-bright)] outline-none placeholder:text-[var(--text-dim)] focus:border-[var(--cyan)]/50"
            />
          </div>
          <div className="flex gap-1 rounded-lg border border-[var(--border)] bg-white/5 p-1">
            {FILTERS.map((f) => (
              <button
                key={f}
                onClick={() => setRiskFilter(f)}
                className={clsx(
                  "rounded-md px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide transition-colors",
                  riskFilter === f ? "bg-[var(--blue)]/25 text-[var(--text-bright)]" : "text-[var(--text-dim)] hover:text-[var(--text)]"
                )}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-h-[440px] flex-1 overflow-y-auto">
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-[#0b0f1ae6] backdrop-blur">
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => toggleSort(col.key)}
                  className="cursor-pointer select-none whitespace-nowrap px-5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-[var(--text-dim)] hover:text-[var(--text-bright)]"
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    {sortKey === col.key &&
                      (sortDir === "asc" ? <ArrowUp className="h-3 w-3 text-[var(--cyan)]" /> : <ArrowDown className="h-3 w-3 text-[var(--cyan)]" />)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <AnimatePresence initial={false}>
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-5 py-10 text-center text-[var(--text-dim)]">
                    No engines match the current filter.
                  </td>
                </tr>
              ) : (
                rows.map((e) => (
                  <motion.tr
                    key={e.unit_number}
                    layout
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={() => onSelect(e)}
                    className={clsx(
                      "cursor-pointer border-b border-[var(--border)]/60 transition-colors hover:bg-white/[0.04]",
                      selectedUnit === e.unit_number && "bg-[var(--blue)]/10"
                    )}
                  >
                    <td className="px-5 py-2.5 font-mono-num text-[var(--text-bright)]">#{e.unit_number}</td>
                    <td className="px-5 py-2.5 font-mono-num text-[var(--text-dim)]">{e.last_cycle}</td>
                    <td className="px-5 py-2.5 font-mono-num text-[var(--text-bright)]">{e.predicted_rul.toFixed(1)} cyc</td>
                    <td className="px-5 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.max(0, Math.min(100, e.health_score))}%`,
                              background:
                                e.risk === "HIGH" ? "var(--red)" : e.risk === "MEDIUM" ? "var(--amber)" : "var(--green)",
                            }}
                          />
                        </div>
                        <span className="font-mono-num text-xs text-[var(--text-dim)]">{e.health_score.toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="px-5 py-2.5">
                      <RiskBadge risk={e.risk} pulse />
                    </td>
                  </motion.tr>
                ))
              )}
            </AnimatePresence>
          </tbody>
        </table>
      </div>
    </div>
  );
}
