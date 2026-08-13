const API_BASE = "";

let allEngines = [];
let selectedUnit = null;
let sortKey = "predicted_rul";
let sortDir = "asc";
let activeRiskFilter = "ALL";
let searchTerm = "";

let healthChart = null;
let riskDonut = null;

const RISK_COLORS = { LOW: "#2e7d32", MEDIUM: "#b8860b", HIGH: "#c62828" };

function showError(message) {
  const banner = document.getElementById("error-banner");
  banner.textContent = message;
  banner.hidden = false;
}

function clearError() {
  document.getElementById("error-banner").hidden = true;
}

async function loadModelInfo() {
  const el = document.getElementById("model-info");
  try {
    const res = await fetch(`${API_BASE}/api/model-info`);
    if (!res.ok) throw new Error(res.statusText);
    const info = await res.json();
    el.textContent =
      `Subset ${info.subset} · Model: ${info.model_name} · ${info.n_features} features · ` +
      `MAE ${info.metrics.MAE.toFixed(1)} · RMSE ${info.metrics.RMSE.toFixed(1)} · R² ${info.metrics.R2.toFixed(2)}`;
  } catch (err) {
    el.textContent = `Could not load model info (${err.message})`;
  }
}

function riskBadge(risk) {
  return `<span class="risk-badge risk-${risk}">${risk}</span>`;
}

function updateSummary(engines) {
  const panel = document.getElementById("summary-panel");
  if (!engines.length) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  const counts = { LOW: 0, MEDIUM: 0, HIGH: 0 };
  let rulSum = 0;
  engines.forEach((e) => {
    counts[e.risk] = (counts[e.risk] || 0) + 1;
    rulSum += e.predicted_rul;
  });

  document.getElementById("kpi-total").textContent = engines.length;
  document.getElementById("kpi-avg-rul").textContent = `${(rulSum / engines.length).toFixed(1)} cyc`;
  document.getElementById("kpi-low").textContent = counts.LOW;
  document.getElementById("kpi-medium").textContent = counts.MEDIUM;
  document.getElementById("kpi-high").textContent = counts.HIGH;

  const ctx = document.getElementById("risk-donut");
  const data = [counts.LOW, counts.MEDIUM, counts.HIGH];
  if (riskDonut) riskDonut.destroy();
  riskDonut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Low", "Medium", "High"],
      datasets: [{ data, backgroundColor: [RISK_COLORS.LOW, RISK_COLORS.MEDIUM, RISK_COLORS.HIGH], borderWidth: 0 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 11 } } } },
      cutout: "60%",
    },
  });
}

function applyFiltersAndSort() {
  let rows = allEngines;

  if (activeRiskFilter !== "ALL") {
    rows = rows.filter((e) => e.risk === activeRiskFilter);
  }
  if (searchTerm) {
    rows = rows.filter((e) => String(e.unit_number).includes(searchTerm));
  }

  rows = [...rows].sort((a, b) => {
    const dir = sortDir === "asc" ? 1 : -1;
    const av = a[sortKey];
    const bv = b[sortKey];
    if (typeof av === "string") return av.localeCompare(bv) * dir;
    return (av - bv) * dir;
  });

  renderFleetTable(rows);
  updateSortIndicators();

  const countEl = document.getElementById("row-count");
  countEl.textContent = rows.length === allEngines.length ? `(${allEngines.length})` : `(${rows.length} of ${allEngines.length})`;
}

function updateSortIndicators() {
  document.querySelectorAll("th[data-sort]").forEach((th) => {
    const arrow = th.querySelector(".sort-arrow");
    if (th.dataset.sort === sortKey) {
      arrow.textContent = sortDir === "asc" ? "▲" : "▼";
    } else {
      arrow.textContent = "";
    }
  });
}

function renderFleetTable(engines) {
  const body = document.getElementById("fleet-table-body");
  if (!engines.length) {
    body.innerHTML = `<tr><td colspan="5" class="empty-row">No engines match the current filter.</td></tr>`;
    return;
  }
  body.innerHTML = engines
    .map(
      (e) => `
      <tr data-unit="${e.unit_number}" class="${e.unit_number === selectedUnit ? "selected" : ""}">
        <td>#${e.unit_number}</td>
        <td>${e.last_cycle}</td>
        <td>${e.predicted_rul.toFixed(1)} cycles</td>
        <td>${e.health_score.toFixed(0)}%</td>
        <td>${riskBadge(e.risk)}</td>
      </tr>`
    )
    .join("");

  body.querySelectorAll("tr[data-unit]").forEach((row) => {
    row.addEventListener("click", () => {
      const unit = Number(row.dataset.unit);
      const engine = allEngines.find((e) => e.unit_number === unit);
      if (engine) showDetail(engine);
    });
  });
}

function showDetail(engine) {
  selectedUnit = engine.unit_number;
  document.querySelectorAll("#fleet-table-body tr[data-unit]").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.unit) === selectedUnit);
  });

  const panel = document.getElementById("detail-panel");
  panel.hidden = false;
  document.getElementById("detail-title").textContent = `Engine #${engine.unit_number}`;

  document.getElementById("detail-stats").innerHTML = `
    <div><div class="label">Predicted RUL</div><div class="value">${engine.predicted_rul.toFixed(1)} cycles</div></div>
    <div><div class="label">Health score</div><div class="value">${engine.health_score.toFixed(0)}%</div></div>
    <div><div class="label">Risk</div><div class="value">${riskBadge(engine.risk)}</div></div>
    <div><div class="label">Recommended action</div><div class="value">${engine.risk_action}</div></div>
    <div><div class="label">Last observed cycle</div><div class="value">${engine.last_cycle}</div></div>
  `;

  const ctx = document.getElementById("health-chart");
  const labels = engine.health_trend.map((p) => p.cycle);
  const data = engine.health_trend.map((p) => p.health_indicator);

  if (healthChart) healthChart.destroy();
  healthChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Health indicator (higher = more degraded)",
          data,
          borderColor: RISK_COLORS[engine.risk] || "#10223d",
          backgroundColor: "rgba(16,34,61,0.08)",
          pointRadius: 0,
          borderWidth: 2,
          fill: true,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "cycle" } },
        y: { title: { display: true, text: "health indicator" } },
      },
      plugins: { legend: { display: false } },
    },
  });

  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function downloadCsv() {
  if (!allEngines.length) return;
  const header = "unit_number,last_cycle,predicted_rul,health_score,risk,risk_action";
  const rows = allEngines.map(
    (e) => `${e.unit_number},${e.last_cycle},${e.predicted_rul},${e.health_score},${e.risk},"${e.risk_action}"`
  );
  const blob = new Blob([header + "\n" + rows.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "rul_predictions.csv";
  a.click();
  URL.revokeObjectURL(url);
}

async function handleFileUpload(file) {
  const status = document.getElementById("upload-status");
  clearError();
  status.textContent = "Scoring…";
  document.getElementById("file-input").disabled = true;
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_BASE}/api/predict/upload`, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || res.statusText);
    }
    const result = await res.json();
    allEngines = result.engines;
    selectedUnit = null;
    status.textContent = `Scored ${result.n_engines} engines from ${file.name}.`;
    document.getElementById("download-csv").disabled = false;

    updateSummary(allEngines);
    applyFiltersAndSort();
    if (allEngines.length) showDetail(allEngines[0]);
  } catch (err) {
    showError(`Upload failed: ${err.message}`);
    status.textContent = "";
  } finally {
    document.getElementById("file-input").disabled = false;
  }
}

document.getElementById("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFileUpload(file);
});

document.getElementById("download-csv").addEventListener("click", downloadCsv);

document.getElementById("engine-search").addEventListener("input", (e) => {
  searchTerm = e.target.value.trim();
  applyFiltersAndSort();
});

document.getElementById("risk-filter").addEventListener("click", (e) => {
  const btn = e.target.closest(".filter-chip");
  if (!btn) return;
  activeRiskFilter = btn.dataset.risk;
  document.querySelectorAll(".filter-chip").forEach((c) => c.classList.toggle("active", c === btn));
  applyFiltersAndSort();
});

document.querySelectorAll("th[data-sort]").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.sort;
    if (sortKey === key) {
      sortDir = sortDir === "asc" ? "desc" : "asc";
    } else {
      sortKey = key;
      sortDir = "asc";
    }
    applyFiltersAndSort();
  });
});

loadModelInfo();
