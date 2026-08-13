import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Satellite } from "lucide-react";
import { fetchHealth, fetchModelInfo, uploadSensorFile } from "./api/client";
import type { EnginePrediction, ModelInfo } from "./types";
import { Header } from "./components/Header";
import { UploadZone } from "./components/UploadZone";
import { SummaryPanel } from "./components/SummaryPanel";
import { FleetTable } from "./components/FleetTable";
import { EngineDetail } from "./components/EngineDetail";

function downloadCsv(engines: EnginePrediction[]) {
  if (!engines.length) return;
  const header = "unit_number,last_cycle,predicted_rul,rul_low,rul_high,health_score,fail_within_20_proba,risk,risk_action";
  const rows = engines.map(
    (e) =>
      `${e.unit_number},${e.last_cycle},${e.predicted_rul},${e.rul_low ?? ""},${e.rul_high ?? ""},${e.health_score},${e.fail_within_20_proba ?? ""},${e.risk},"${e.risk_action}"`
  );
  const blob = new Blob([header + "\n" + rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "rul_predictions.csv";
  a.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [apiOnline, setApiOnline] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [engines, setEngines] = useState<EnginePrediction[]>([]);
  const [selectedUnit, setSelectedUnit] = useState<number | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
    fetchModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null));
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setErrorText(null);
    setStatusText("Scoring…");
    try {
      const result = await uploadSensorFile(file);
      setEngines(result.engines);
      setSelectedUnit(result.engines[0]?.unit_number ?? null);
      setStatusText(`Scored ${result.n_engines} engines from ${file.name}.`);
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : "Upload failed");
      setStatusText(null);
    }
  }, []);

  const selectedEngine = engines.find((e) => e.unit_number === selectedUnit) ?? null;

  return (
    <div className="min-h-screen">
      <Header
        modelInfo={modelInfo}
        apiOnline={apiOnline}
        onDownload={() => downloadCsv(engines)}
        downloadDisabled={engines.length === 0}
      />

      <main className="mx-auto max-w-7xl space-y-5 px-6 py-6">
        <UploadZone onFile={handleFile} statusText={statusText} errorText={errorText} />

        <AnimatePresence mode="wait">
          {engines.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="glass flex flex-col items-center gap-3 rounded-2xl px-6 py-20 text-center"
            >
              <Satellite className="h-8 w-8 text-[var(--text-dim)]" />
              <p className="text-sm text-[var(--text-dim)]">
                Upload a raw C-MAPSS sensor file to score the fleet and see live predictions.
              </p>
            </motion.div>
          ) : (
            <motion.div key="data" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">
              <SummaryPanel engines={engines} />

              <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.35fr_1fr]">
                <FleetTable engines={engines} selectedUnit={selectedUnit} onSelect={(e) => setSelectedUnit(e.unit_number)} />
                <div className="lg:sticky lg:top-24 lg:self-start">
                  {selectedEngine && <EngineDetail engine={selectedEngine} />}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="mx-auto max-w-7xl px-6 py-8 text-center text-[11px] text-[var(--text-dim)]">
        Predictions are model estimates from NASA C-MAPSS data — application-level risk cut points, not a certified
        maintenance standard.
      </footer>
    </div>
  );
}
