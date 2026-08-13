import { motion } from "framer-motion";
import { Download, Plane, Radio } from "lucide-react";
import type { ModelInfo } from "../types";

export function Header({
  modelInfo,
  apiOnline,
  onDownload,
  downloadDisabled,
}: {
  modelInfo: ModelInfo | null;
  apiOnline: boolean;
  onDownload: () => void;
  downloadDisabled: boolean;
}) {
  return (
    <header className="sticky top-0 z-20 border-b border-[var(--border)] bg-[#05070dcc] backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-6 py-4">
        <div className="flex items-center gap-3">
          <motion.div
            initial={{ rotate: -12, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            className="flex h-10 w-10 items-center justify-center rounded-xl border border-[var(--border-bright)] bg-gradient-to-br from-[var(--blue)]/20 to-[var(--violet)]/20"
          >
            <Plane className="h-5 w-5 text-[var(--cyan)]" />
          </motion.div>
          <div>
            <h1 className="text-[15px] font-semibold tracking-tight text-[var(--text-bright)]">
              Aircraft Engine RUL Prognostics
            </h1>
            <div className="flex items-center gap-2 text-[11px] text-[var(--text-dim)]">
              <span className="inline-flex items-center gap-1">
                <Radio className={`h-2.5 w-2.5 ${apiOnline ? "text-[var(--green)]" : "text-[var(--red)]"}`} />
                {apiOnline ? "API online" : "API unreachable"}
              </span>
              {modelInfo && (
                <span className="font-mono-num">
                  · {modelInfo.subset} · {modelInfo.model_name} · MAE {modelInfo.metrics.MAE.toFixed(1)} · R²{" "}
                  {modelInfo.metrics.R2.toFixed(2)}
                </span>
              )}
            </div>
          </div>
        </div>

        <button
          onClick={onDownload}
          disabled={downloadDisabled}
          className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] bg-white/5 px-3 py-1.5 text-xs font-medium text-[var(--text)] transition-colors hover:border-[var(--cyan)]/50 hover:text-[var(--text-bright)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Download className="h-3.5 w-3.5" /> Export CSV
        </button>
      </div>
    </header>
  );
}
