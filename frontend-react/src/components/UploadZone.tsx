import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { clsx } from "clsx";
import { UploadCloud, CheckCircle2, XCircle, Loader2, Sparkles } from "lucide-react";

type Status = "idle" | "dragging" | "uploading" | "success" | "error";

export function UploadZone({
  onFile,
  statusText,
  errorText,
}: {
  onFile: (file: File) => void;
  statusText: string | null;
  errorText: string | null;
}) {
  const [status, setStatus] = useState<Status>("idle");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;
      setStatus("uploading");
      onFile(file);
    },
    [onFile]
  );

  // Lets a first-time visitor score real data without having to find the NASA
  // dataset first. Goes through the same onFile path as a manual upload.
  const loadSample = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation();
      setStatus("uploading");
      try {
        const res = await fetch("/api/sample");
        if (!res.ok) throw new Error(`sample unavailable (${res.status})`);
        const blob = await res.blob();
        onFile(new File([blob], "test_FD001.txt", { type: "text/plain" }));
      } catch {
        setStatus("error");
      }
    },
    [onFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const effectiveStatus: Status = errorText ? "error" : statusText?.startsWith("Scored") ? "success" : status;

  return (
    <motion.div
      layout
      onDragOver={(e) => {
        e.preventDefault();
        setStatus("dragging");
      }}
      onDragLeave={() => setStatus("idle")}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className={clsx(
        "glass relative flex cursor-pointer items-center gap-4 rounded-2xl px-6 py-5 transition-colors",
        effectiveStatus === "dragging" && "border-[var(--cyan)]/60 bg-cyan-400/5",
        effectiveStatus === "success" && "glow-green",
        effectiveStatus === "error" && "glow-red"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".txt"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />

      <div className="relative flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-[var(--border)] bg-white/5">
        <AnimatePresence mode="wait">
          {effectiveStatus === "uploading" ? (
            <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <Loader2 className="h-5 w-5 animate-spin text-[var(--blue)]" />
            </motion.div>
          ) : effectiveStatus === "success" ? (
            <motion.div key="ok" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
              <CheckCircle2 className="h-5 w-5 text-[var(--green)]" />
            </motion.div>
          ) : effectiveStatus === "error" ? (
            <motion.div key="err" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
              <XCircle className="h-5 w-5 text-[var(--red)]" />
            </motion.div>
          ) : (
            <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <UploadCloud className="h-5 w-5 text-[var(--text-dim)]" />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <p className="text-sm font-medium text-[var(--text-bright)]">
            Drop a raw C-MAPSS sensor file, or click to browse
          </p>
          <button
            type="button"
            onClick={loadSample}
            className="inline-flex items-center gap-1 rounded-lg border border-[var(--border)] bg-white/5 px-2 py-0.5 text-xs text-[var(--text-dim)] transition-colors hover:border-[var(--cyan)]/60 hover:text-[var(--text-bright)]"
          >
            <Sparkles className="h-3 w-3" />
            Try sample data
          </button>
        </div>
        <p className="truncate text-xs text-[var(--text-dim)]">
          {errorText ?? statusText ?? "Whitespace-delimited .txt — e.g. test_FD001.txt — scored instantly, on-device."}
        </p>
      </div>
    </motion.div>
  );
}
