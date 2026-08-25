// AI Lab Optimization Center (Phase 11 Dashboard)
// Handles Runs, Diagnostics/Step Audits, Visual Diff/Approval, and Revision Hash Chain Rollback

let currentSelectedRunId = null;
let currentPendingApprovalId = null;
let activeOptTab = "runs";
let runsRefreshTimer = null;
let apiKeyPromptShown = false;
let aiLabApiKey = "";

// Initialize when DOM loaded
document.addEventListener("DOMContentLoaded", () => {
  initOptimizerPage();
});

function initOptimizerPage() {
  bindTabs();
  loadOptimizationRuns();
  loadRevisions();
  loadPermissions();
  switchOptTab("runs");

  // Auto-refresh every 10s if on runs tab
  if (runsRefreshTimer) clearInterval(runsRefreshTimer);
  runsRefreshTimer = setInterval(() => {
    if (activeOptTab === "runs") {
      loadOptimizationRuns(true);
    } else if (activeOptTab === "revisions") {
      loadRevisions(true);
    }
  }, 10000);
}

// 1. Tab Switching Logic
function bindTabs() {
  const tabs = document.querySelectorAll(".opt-tab-btn");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab || btn.dataset.target;
      if (!target) return;
      switchOptTab(target);
    });
  });
}

function switchOptTab(tabId) {
  activeOptTab = tabId;
  document.querySelectorAll(".opt-tab-btn").forEach((b) => {
    b.classList.toggle("active", (b.dataset.tab || b.dataset.target) === tabId);
  });
  document.querySelectorAll(".tab-pane").forEach((p) => {
    p.style.display = p.id === `tab-${tabId}` ? "block" : "none";
  });

  if ((tabId === "detail" || tabId === "timeline" || tabId === "candidates" || tabId === "shadow" || tabId === "errors" || tabId === "audit") && currentSelectedRunId) {
    loadRunDetail(currentSelectedRunId);
  } else if (tabId === "approval" && currentSelectedRunId) {
    loadApprovalView(currentSelectedRunId);
  } else if (tabId === "revisions") {
    loadRevisions();
  } else if (tabId === "deployments") {
    loadRevisions();
  } else if (tabId === "permissions") {
    loadPermissions();
  }
}

// Helper: Format badges
function formatStatusBadge(status) {
  const s = (status || "").toUpperCase();
  let badgeClass = "badge-neutral";
  if (s === "ACTIVE" || s === "COMPLETED" || s === "APPROVED") badgeClass = "badge-success";
  else if (s === "RUNNING" || s === "SHADOW" || s === "PLANNING" || s === "TRAINING" || s === "EVALUATING")
    badgeClass = "badge-running";
  else if (s === "PENDING_APPROVAL") badgeClass = "badge-pending";
  else if (s === "FAILED" || s === "CANCELLED" || s === "REJECTED") badgeClass = "badge-danger";
  else if (s === "SUPERSEDED" || s === "ROLLED_BACK") badgeClass = "badge-neutral";

  return `<span class="badge ${badgeClass}">${escapeHtml(s)}</span>`;
}

// Compatibility aliases for the existing template contract. Stage 9 keeps
// these read-only and deliberately does not expose deployment mutations.
function loadRuns(silent = false) { return loadOptimizationRuns(silent); }
function closeModal(id) { const modal = document.getElementById(id); if (modal) modal.classList.remove("show"); }

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function getAuthHeaders() {
  let apiKey = "";
  let storageRead = false;
  try {
    apiKey = localStorage.getItem("polyflip_api_key") || "";
    storageRead = true;
  } catch (err) {
    console.warn("localStorage unavailable", err);
  }
  if (!storageRead && !apiKey) {
    apiKey = aiLabApiKey;
  }

  if (!apiKey && !apiKeyPromptShown) {
    apiKeyPromptShown = true;
    const enteredKey = window.prompt("Введите API key для AI Lab");
    if (enteredKey && enteredKey.trim()) {
      aiLabApiKey = enteredKey.trim();
      apiKey = aiLabApiKey;
      apiKeyPromptShown = false;
    }
  } else if (apiKey) {
    aiLabApiKey = apiKey;
    apiKeyPromptShown = false;
  }

  return {
    "Content-Type": "application/json",
    "X-API-Key": aiLabApiKey,
  };
}

// 2. Optimization Runs List
async function loadOptimizationRuns(silent = false) {
  const tbody = document.getElementById("runs-table-body");
  if (!silent) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">Загрузка запусков оптимизатора...</td></tr>`;
  }

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs?limit=50`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const runs = data.runs || [];

    if (runs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">Запусков оптимизатора не найдено. Нажмите «Новый PAPER/RESEARCH запуск» для старта.</td></tr>`;
      return;
    }

    tbody.innerHTML = runs
      .map((r) => {
        const created = r.created_at ? new Date(r.created_at).toLocaleString("ru-RU") : "—";
        const asset = (r.scope && r.scope.asset) || "BTCUSDT";
        const objective = r.objective || "Автономная оптимизация";
        const statusBadge = formatStatusBadge(r.status);
        const actionBtn = `<button class="btn btn-secondary btn-sm" onclick="selectRun(${r.id})">🔍 Детали / Анализ</button>`;

        return `
        <tr class="run-row ${currentSelectedRunId === r.id ? "selected-row" : ""}" onclick="selectRun(${r.id})">
          <td style="font-weight: 700;">#${r.id}</td>
          <td>
            <div>${escapeHtml(objective)}</div>
            <small style="color: var(--text-muted);">${escapeHtml(asset)} · ${escapeHtml(r.mode || "RESEARCH")} · ${escapeHtml(r.llm_provider || "mock")} / ${escapeHtml(r.llm_research_model || "default")}</small>
          </td>
          <td>${statusBadge}</td>
          <td><span class="badge badge-neutral" style="font-size: 0.75rem;">${escapeHtml(r.autonomy_level || "EXPERIMENT")}</span></td>
          <td>${escapeHtml(r.budget_experiments ?? "—")}</td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem;">${escapeHtml(r.agent_thread_id || "—")}</td>
          <td>${created}</td>
          <td>${actionBtn}</td>
        </tr>
      `;
      })
      .join("");
  } catch (err) {
    console.error("loadOptimizationRuns error:", err);
    if (!silent) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--color-failed); padding: 2rem;">Ошибка загрузки запусков: ${escapeHtml(err.message)}</td></tr>`;
    }
  }
}

// Select Run & Load Detail
async function selectRun(runId) {
  currentSelectedRunId = runId;
  currentPendingApprovalId = null; // Сброс ID согласования предыдущего запуска
  switchOptTab("timeline");
  await loadRunDetail(runId);
}

// 3. Run Details & Diagnostics
async function loadRunDetail(runId) {
  if (!runId) return;

  const headerEl = document.getElementById("detail-run-header") || document.getElementById("detail-run-objective");
  const scopeEl = document.getElementById("detail-run-scope");
  const stepsEl = document.getElementById("detail-run-steps") || document.getElementById("steps-timeline");
  const auditsEl = document.getElementById("detail-run-audits") || document.getElementById("stage9-audit");
  if (!headerEl || !stepsEl || !auditsEl) return;

  headerEl.innerHTML = `<div style="color: var(--text-muted);">Загрузка деталей запуска #${runId}...</div>`;

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs/${runId}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const run = data.run;
    const steps = data.steps || [];
    const audits = data.audits || [];
    renderStage9Detail(run, steps, data.results || [], audits);
    updateRunControls(run);
    const timelineRun = document.getElementById("timeline-selected-run");
    if (timelineRun) {
      timelineRun.textContent = "#" + run.id + " · " + String(run.status || "DRAFT");
    }

    // Header Info
    const created = run.created_at ? new Date(run.created_at).toLocaleString("ru-RU") : "—";
    const completed = run.completed_at ? new Date(run.completed_at).toLocaleString("ru-RU") : "В процессе";

    headerEl.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
          <h3 style="margin: 0 0 0.5rem 0; font-size: 1.25rem;">Запуск #${run.id}: ${escapeHtml(run.objective)}</h3>
          <div style="color: var(--text-muted); font-size: 0.85rem;">
            Создан: <strong>${created}</strong> | Завершён: <strong>${completed}</strong> | Инициатор: <strong>${escapeHtml(run.created_by || "system")}</strong><br>LLM: <strong>${escapeHtml(run.llm_provider || "mock")}</strong> · research <strong>${escapeHtml(run.llm_research_model || "default")}</strong> · summary <strong>${escapeHtml(run.llm_summary_model || "default")}</strong>
          </div>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center;">
          ${formatStatusBadge(run.status)}
          <button class="btn btn-warning btn-sm" onclick="switchOptTab('approval')">⚖️ Visual Diff / Audit</button>
        </div>
      </div>
    `;

    // Scope & Settings
    if (scopeEl) scopeEl.innerHTML = `
      <pre style="background: var(--bg-primary); padding: 1rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; overflow-x: auto; color: var(--text-secondary); border: 1px solid var(--border-color);">${escapeHtml(JSON.stringify(run.scope || {}, null, 2))}</pre>
    `;

    // Timeline Steps
    if (steps.length === 0) {
      stepsEl.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">Шаги пайплайна ещё не зафиксированы.</div>`;
    } else {
      stepsEl.innerHTML = `
        <div class="timeline-container">
          ${steps
            .map(
              (st) => `
            <div class="timeline-item">
              <div class="timeline-badge">${st.sequence}</div>
              <div class="timeline-content">
                <div style="display: flex; justify-content: space-between;">
                  <strong>${escapeHtml(st.step_type)}</strong>
                  ${formatStatusBadge(st.status)}
                </div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">
                  Создан: ${st.started_at ? new Date(st.started_at).toLocaleTimeString("ru-RU") : "—"}
                </div>
              </div>
            </div>
          `
            )
            .join("")}
        </div>
      `;
    }

    // Step Audits & Guardrails
    if (audits.length === 0) {
      auditsEl.innerHTML = `<div style="color: var(--text-muted); padding: 1rem;">Журнал аудита пуст.</div>`;
    } else {
      auditsEl.innerHTML = `
        <table class="table" style="font-size: 0.85rem;">
          <thead>
            <tr>
              <th>Время</th>
              <th>Действие</th>
              <th>Решение / Причина</th>
              <th>Guardrails</th>
            </tr>
          </thead>
          <tbody>
            ${audits
              .map((a) => {
                const ts = a.created_at ? new Date(a.created_at).toLocaleTimeString("ru-RU") : "—";
                const passBadge = a.passed_checks
                  ? `<span class="badge badge-success" style="font-size: 0.7rem;">PASSED</span>`
                  : `<span class="badge badge-danger" style="font-size: 0.7rem;">VIOLATION</span>`;
                const fails = (a.guardrail_failures || []).map((f) => `<div style="color: var(--color-failed); font-size: 0.75rem;">⚠️ ${escapeHtml(f)}</div>`).join("");

                return `
                <tr>
                  <td>${ts}</td>
                  <td><strong>${escapeHtml(a.action)}</strong></td>
                  <td>${escapeHtml(a.decision_reason || "—")}</td>
                  <td>${passBadge}${fails}</td>
                </tr>
              `;
              })
              .join("")}
          </tbody>
        </table>
      `;
    }
  } catch (err) {
    console.error("loadRunDetail error:", err);
    headerEl.innerHTML = `<div style="color: var(--color-failed);">Ошибка загрузки деталей запуска: ${escapeHtml(err.message)}</div>`;
  }
}

function renderStage9Detail(run, steps, results, audits) {
  const timeline = document.getElementById("stage9-timeline");
  const candidates = document.getElementById("stage9-candidates");
  const shadow = document.getElementById("stage9-shadow");
  const errors = document.getElementById("stage9-errors");
  const audit = document.getElementById("stage9-audit");
  const items = steps || [];
  if (timeline) timeline.innerHTML = items.length ? items.map((step) => `<div class="timeline-item"><div class="timeline-content"><strong>#${escapeHtml(step.step_index)} ${escapeHtml(step.step_type)}</strong> ${formatStatusBadge(step.status)}<div class="diff-label">${escapeHtml(step.summary || step.hypothesis || "—")}</div></div></div>`).join("") : '<div class="unavailable">Шаги отсутствуют.</div>';
  if (candidates) candidates.innerHTML = results.length ? `<table class="opt-table"><thead><tr><th>Config</th><th>Type</th><th>Status</th><th>Trades</th><th>Net PnL</th><th>Drawdown</th></tr></thead><tbody>${results.map((r) => `<tr><td>#${escapeHtml(r.config_id)}</td><td>${escapeHtml(r.evaluation_kind)}</td><td>${formatStatusBadge(r.status)}</td><td>${escapeHtml(r.trade_count ?? "—")}</td><td>${escapeHtml(r.net_pnl ?? "—")}</td><td>${escapeHtml(r.max_drawdown ?? "—")}</td></tr>`).join("")}</tbody></table>` : '<div class="unavailable">Результаты отсутствуют.</div>';
  if (shadow) shadow.innerHTML = String(run.status).toUpperCase() === "SHADOW" ? '<div class="status-badge badge-shadow">SHADOW</div><p>Пассивное наблюдение. LIVE-активация отключена.</p>' : `<div class="unavailable">Текущий статус: ${escapeHtml(run.status)}. SHADOW assignment не найден.</div>`;
  const failed = [...items.filter((s) => ["FAILED", "ERROR"].includes(String(s.status).toUpperCase())), ...(audits || []).filter((a) => a.error_code || a.error_message)];
  if (errors) errors.innerHTML = failed.length ? failed.map((e) => `<div class="timeline-content"><strong>${escapeHtml(e.error_code || e.status || "ERROR")}</strong><div>${escapeHtml(e.error_message || e.error || "—")}</div></div>`).join("") : '<div class="unavailable">Ошибок не найдено.</div>';
  if (audit) audit.innerHTML = audits.length ? audits.map((a) => `<div class="timeline-content"><strong>${escapeHtml(a.action || "AUDIT")}</strong><div>${escapeHtml(a.created_at || "—")} — ${escapeHtml(a.error_message || a.reason || "OK")}</div></div>`).join("") : '<div class="unavailable">Аудит пуст.</div>';
}

async function loadPermissions() {
  const target = document.getElementById("stage9-permissions");
  if (!target) return;
  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/permissions`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const rows = data.permissions || [];
    target.innerHTML = rows.length ? `<table class="opt-table"><thead><tr><th>Profile</th><th>Version</th><th>Enabled</th><th>Allowed actions</th></tr></thead><tbody>${rows.map((p) => `<tr><td>${escapeHtml(p.profile_name)}</td><td>${escapeHtml(p.version)}</td><td>${p.enabled ? "yes" : "no"}</td><td>${escapeHtml((p.allowed_actions || []).join(", ") || "—")}</td></tr>`).join("")}</tbody></table>` : '<div class="unavailable">Разрешения отсутствуют.</div>';
  } catch (err) { target.innerHTML = `<div class="unavailable">Permissions недоступны: ${escapeHtml(err.message)}</div>`; }
}

// 4. Approval & Visual Diff
async function loadApprovalView(runId) {
  if (!runId) return;

  const badgeEl = document.getElementById("approval-status-badge");
  const bannerEl = document.getElementById("approval-action-banner");
  const decisionBox = document.getElementById("approval-decision-box");

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs/${runId}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const run = data.run;
    const approvals = data.approvals || [];

    badgeEl.innerHTML = formatStatusBadge(run.status);

    const pendingApproval = approvals.find((a) => a.status === "PENDING");
    const latestApproval = approvals.length > 0 ? approvals[0] : null;

    if (pendingApproval) {
      currentPendingApprovalId = pendingApproval.id;
    } else {
      currentPendingApprovalId = null;
    }

    if (run.status === "SHADOW") {
      bannerEl.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div>
            <strong>Модель находится в пассивном режиме SHADOW</strong>
            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">
               Deployment actions are disabled in this research dashboard. Only read-only diff and audit inspection is available.
            </div>
          </div>
        </div>
      `;
      decisionBox.style.display = "none";
    } else if (run.status === "PENDING_APPROVAL") {
      bannerEl.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div>
            <strong style="color: #FBBF24;">⏳ Требуется согласование оператора (Human Approval Required)</strong>
            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">
              Внимательно проверьте серверный Visual Diff параметров и метрик кандидата перед утверждением.
            </div>
          </div>
        </div>
      `;
       decisionBox.style.display = "none";
    } else if (run.status === "ACTIVE") {
      bannerEl.innerHTML = `
        <div style="color: var(--poly-green); font-weight: 700;">
           Deployment revision is recorded; LIVE activation is disabled in this dashboard.
        </div>
      `;
      decisionBox.style.display = "none";
    } else {
      bannerEl.innerHTML = `
        <div style="color: var(--text-muted);">
          Текущий статус запуска (${escapeHtml(run.status)}) не требует согласования.
        </div>
      `;
      decisionBox.style.display = "none";
    }

    // Load actual server-generated diff from AIApprovalRequest
    const targetApproval = pendingApproval || latestApproval;
    const diff = targetApproval && targetApproval.diff && Object.keys(targetApproval.diff).length > 0
      ? targetApproval.diff
      : null;

    renderDiffTables(diff);
  } catch (err) {
    console.error("loadApprovalView error:", err);
  }
}

function renderDiffTables(diff) {
  const candEl = document.getElementById("diff-candidate-content");
  const baseEl = document.getElementById("diff-baseline-content");

  if (!diff || !diff.candidate) {
     const emptyMsg = `<div style="color: var(--text-muted); padding: 1.5rem; text-align: center;">Серверный diff еще не сформирован. Deployment actions are disabled in RESEARCH.</div>`;
    candEl.innerHTML = emptyMsg;
    baseEl.innerHTML = emptyMsg;
    return;
  }

  const cand = diff.candidate || {};
  const base = diff.baseline || {};
  const metrics = diff.metrics || {};

  const pnlVal = metrics.median_pnl !== undefined && metrics.median_pnl !== null
    ? `${Number(metrics.median_pnl) > 0 ? "+" : ""}${Number(metrics.median_pnl).toFixed(2)}%`
    : "—";
  const pnlColor = (metrics.median_pnl || 0) >= 0 ? "var(--poly-green)" : "var(--color-rejected)";

  candEl.innerHTML = `
    <div class="diff-row"><span class="diff-label">Config ID:</span><span class="diff-val">#${cand.config_id || "—"}</span></div>
    <div class="diff-row"><span class="diff-label">Artifact ID:</span><span class="diff-val">${cand.artifact_id ? "#" + cand.artifact_id : "—"}</span></div>
    <div class="diff-row"><span class="diff-label">Семейство модели:</span><span class="diff-val">${escapeHtml(cand.model_family || "—")}</span></div>
    <div class="diff-row"><span class="diff-label">Набор признаков:</span><span class="diff-val">${escapeHtml(cand.feature_set || "—")} (v${escapeHtml(cand.feature_pipeline_version || "1.0")})</span></div>
    <div class="diff-row"><span class="diff-label">Порог UP (Threshold):</span><span class="diff-val" style="color: var(--poly-green); font-weight: 700;">${cand.decision_threshold !== undefined && cand.decision_threshold !== null ? cand.decision_threshold : "—"}</span></div>
    <div class="diff-row"><span class="diff-label">Порог DOWN (Threshold):</span><span class="diff-val" style="color: var(--color-failed); font-weight: 700;">${cand.decision_threshold_down !== undefined && cand.decision_threshold_down !== null ? cand.decision_threshold_down : "—"}</span></div>
    <div class="diff-row"><span class="diff-label">Медианный PnL OOT:</span><span class="diff-val" style="color: ${pnlColor}; font-size: 1.1rem; font-weight: 700;">${pnlVal}</span></div>
    <div class="diff-row"><span class="diff-label">Объем сделок OOT:</span><span class="diff-val">${metrics.total_trades || "0"}</span></div>
    <div class="diff-row"><span class="diff-label">Макс. просадка OOT:</span><span class="diff-val" style="color: var(--color-failed);">${metrics.max_drawdown !== undefined && metrics.max_drawdown !== null ? Number(metrics.max_drawdown).toFixed(2) + "%" : "—"}</span></div>
  `;

  baseEl.innerHTML = `
    <div class="diff-row"><span class="diff-label">Model Registry ID:</span><span class="diff-val">${base.model_registry_id ? "#" + base.model_registry_id : "Active Default"}</span></div>
    <div class="diff-row"><span class="diff-label">Версия модели:</span><span class="diff-val">${base.version ? "v" + base.version : "—"}</span></div>
    <div class="diff-row"><span class="diff-label">Тип модели:</span><span class="diff-val">${escapeHtml(base.model_type || "—")}</span></div>
    <div class="diff-row"><span class="diff-label">Набор признаков:</span><span class="diff-val">${escapeHtml(base.features || "—")}</span></div>
    <div class="diff-row"><span class="diff-label">Порог UP (Threshold):</span><span class="diff-val">${base.decision_threshold !== undefined && base.decision_threshold !== null ? base.decision_threshold : "—"}</span></div>
    <div class="diff-row"><span class="diff-label">Порог DOWN (Threshold):</span><span class="diff-val">${base.decision_threshold_down !== undefined && base.decision_threshold_down !== null ? base.decision_threshold_down : "—"}</span></div>
    <div class="diff-row"><span class="diff-label">Исторический PnL:</span><span class="diff-val">${base.backtest_pnl !== null && base.backtest_pnl !== undefined ? Number(base.backtest_pnl).toFixed(2) + "%" : "—"}</span></div>
    <div class="diff-row"><span class="diff-label">Исторические сделки:</span><span class="diff-val">${base.backtest_trades || "—"}</span></div>
  `;
}

// 5. Deployment Revisions & Hash Chain Rollback
async function loadRevisions(silent = false) {
  const tbody = document.getElementById("revisions-table-body");
  if (!silent) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">Загрузка ревизий развертывания...</td></tr>`;
  }

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/deployments/revisions`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const revs = Array.isArray(data) ? data : (data.revisions || []);
    const stageDeployments = document.getElementById("stage9-deployments");
    if (stageDeployments) stageDeployments.innerHTML = revs.length ? `<table class="opt-table"><thead><tr><th>ID</th><th>Key</th><th>Status</th><th>Created</th></tr></thead><tbody>${revs.map((r) => `<tr><td>#${escapeHtml(r.id)}</td><td>${escapeHtml(r.revision_key || "—")}</td><td>${formatStatusBadge(r.status)}</td><td>${escapeHtml(r.created_at || "—")}</td></tr>`).join("")}</tbody></table>` : '<div class="unavailable">Ревизии отсутствуют.</div>';

    if (revs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">Ревизий развертывания пока нет.</td></tr>`;
      return;
    }

    tbody.innerHTML = revs
      .map((r) => {
        const created = r.created_at ? new Date(r.created_at).toLocaleString("ru-RU") : "—";
        const activated = r.activated_at ? new Date(r.activated_at).toLocaleString("ru-RU") : "—";
        const statusBadge = formatStatusBadge(r.status);
        const shortHash = r.manifest_hash ? r.manifest_hash.substring(0, 12) + "..." : "—";
        const shortKey = escapeHtml(r.revision_key || `rev-${r.id}`);

        let rollbackBtn = "";
        if (r.status === "ACTIVE") {
          rollbackBtn = `<span style="color: var(--poly-green); font-weight: 700; font-size: 0.85rem;">Recorded active revision</span>`;
        }

        return `
        <tr>
          <td style="font-weight: 700;">#${r.id}</td>
          <td><code>${shortKey}</code></td>
          <td><code title="${r.manifest_hash}">${shortHash}</code></td>
          <td>${created}</td>
          <td>${activated}</td>
          <td>${statusBadge}</td>
          <td>
            <button class="btn btn-secondary btn-sm" onclick="showEventsChain(${r.id})" style="margin-right: 0.5rem;">🔗 Хеш-цепь</button>
            ${rollbackBtn}
          </td>
        </tr>
      `;
      })
      .join("");
  } catch (err) {
    console.error("loadRevisions error:", err);
    if (!silent) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--color-failed); padding: 2rem;">Ошибка загрузки ревизий: ${escapeHtml(err.message)}</td></tr>`;
    }
  }
}

// Show Hash Chain Modal
async function showEventsChain(revisionId) {
  const modal = document.getElementById("events-modal");
  const listEl = document.getElementById("events-chain-list");
  listEl.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 1.5rem;">Загрузка криптографической хеш-цепочки...</div>`;
  modal.style.display = "flex";

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/deployments/revisions/${revisionId}/events`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const events = data.events || [];

    if (events.length === 0) {
      listEl.innerHTML = `<div style="color: var(--text-muted); padding: 1.5rem; text-align: center;">Для ревизии #${revisionId} событий не найдено.</div>`;
      return;
    }

    listEl.innerHTML = events
      .map((ev) => {
        const ts = ev.created_at ? new Date(ev.created_at).toLocaleString("ru-RU") : "—";
        const shortPrev = ev.previous_hash ? ev.previous_hash.substring(0, 16) + "..." : "0000000000000000...";
        const shortHash = ev.event_hash ? ev.event_hash.substring(0, 16) + "..." : "—";

        return `
        <div class="hash-chain-card">
          <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <strong>${formatStatusBadge(ev.event_type)} &nbsp; Инициатор: <code>${escapeHtml(ev.actor || "system")}</code></strong>
            <span style="font-size: 0.8rem; color: var(--text-muted);">${ts}</span>
          </div>
          <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
            Причина: <em>${escapeHtml(ev.reason || "—")}</em>
          </div>
          <div class="hash-block">
            <div><span class="hash-label">Prev Hash:</span> <code>${shortPrev}</code></div>
            <div><span class="hash-label">Event Hash:</span> <code style="color: var(--poly-green); font-weight: 700;">${shortHash}</code></div>
          </div>
        </div>
      `;
      })
      .join("");
  } catch (err) {
    listEl.innerHTML = `<div style="color: var(--color-failed); padding: 1.5rem;">Ошибка загрузки цепочки событий: ${escapeHtml(err.message)}</div>`;
  }
}

function closeEventsModal() {
  document.getElementById("events-modal").style.display = "none";
}

// 6. Modal: Create New PAPER/RESEARCH Run
async function loadResearchPermissions() {
  const select = document.getElementById("new-run-permission");
  if (!select) return;
  select.disabled = true;
  select.innerHTML = '<option value="">Загрузка permission-профилей...</option>';
  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/permissions`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error(await responseError(res));
    const data = await res.json();
    const required = new Set(["CREATE_EXPERIMENT", "TRAIN_MODEL"]);
    const profiles = (data.permissions || []).filter((profile) => {
      const actions = new Set((profile.allowed_actions || []).map((item) => String(item).toUpperCase()));
      return profile.enabled && [...required].every((action) => actions.has(action));
    });
    if (!profiles.length) {
      select.innerHTML = '<option value="">Нет включённого research permission-профиля</option>';
      return;
    }
    select.innerHTML = profiles.map((profile) =>
      '<option value="' + escapeHtml(profile.id) + '">' +
      escapeHtml(profile.profile_name) + ' v' + escapeHtml(profile.version) + '</option>'
    ).join("");
    select.disabled = false;
  } catch (error) {
    select.innerHTML = '<option value="">Ошибка загрузки профилей</option>';
    showToast("Не удалось загрузить permission-профили: " + error.message, "danger");
  }
}

let llmCatalogCache = null;

function renderLLMModelOptions() {
  const researchSelect = document.getElementById("new-run-research-model");
  const summarySelect = document.getElementById("new-run-summary-model");
  const metaEl = document.getElementById("llm-catalog-meta");
  if (!researchSelect || !summarySelect || !llmCatalogCache) return;
  const search = (document.getElementById("new-run-model-search")?.value || "").trim().toLowerCase();
  const models = (llmCatalogCache.models || []).filter((item) => {
    if (!search) return true;
    return String(item.id).toLowerCase().includes(search)
      || String(item.label || "").toLowerCase().includes(search);
  });
  const options = models.map((item) => {
    const available = item.is_available !== false;
    const checked = item.last_check;
    let badge = available ? "✔" : "✖ unavailable";
    if (checked && checked.available) {
      badge = `✔ ${checked.latency_ms}ms`;
    } else if (checked && checked.available === false) {
      badge = "✖ check failed";
    }
    return '<option value="' + escapeHtml(item.id) + '"' + (available ? "" : " disabled") + '>'
      + escapeHtml(item.label || item.id) + " · " + escapeHtml(badge) + "</option>";
  }).join("");
  const emptyHtml = '<option value="">Нет доступных моделей</option>';
  const previousResearch = researchSelect.value;
  const previousSummary = summarySelect.value;
  researchSelect.innerHTML = options || emptyHtml;
  summarySelect.innerHTML = options || emptyHtml;
  if (previousResearch && models.some((item) => item.id === previousResearch)) {
    researchSelect.value = previousResearch;
  }
  if (previousSummary && models.some((item) => item.id === previousSummary)) {
    summarySelect.value = previousSummary;
  }
  if (metaEl) {
    const parts = [];
    if (llmCatalogCache.checked_at) {
      parts.push("Каталог обновлён: " + escapeHtml(String(llmCatalogCache.checked_at).replace("T", " ").slice(0, 19)));
    }
    if (llmCatalogCache.source) parts.push("источник: " + escapeHtml(llmCatalogCache.source));
    if (llmCatalogCache.stale) parts.push("⚠ STALE (endpoint недоступен)");
    metaEl.textContent = parts.join(" • ");
  }
}

async function loadLLMModels(provider = "", refresh = false) {
  const providerSelect = document.getElementById("new-run-llm-provider");
  const researchSelect = document.getElementById("new-run-research-model");
  const summarySelect = document.getElementById("new-run-summary-model");
  if (!providerSelect || !researchSelect || !summarySelect) return;
  try {
    const params = new URLSearchParams();
    if (provider) params.set("provider", provider);
    if (refresh) params.set("refresh", "true");
    const query = params.toString() ? ("?" + params.toString()) : "";
    const res = await fetch(`${window.API_BASE}/api/ai-lab/llm/models${query}`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error(await responseError(res));
    const data = await res.json();
    if (!provider) {
      providerSelect.innerHTML = (data.providers || []).map((item) =>
        '<option value="' + escapeHtml(item.id) + '">' +
        escapeHtml(item.label || item.id) + (item.configured ? "" : " (не настроен)") +
        "</option>"
      ).join("");
      provider = data.provider || providerSelect.value;
      providerSelect.value = provider;
      return loadLLMModels(provider, refresh);
    }
    llmCatalogCache = {
      provider: data.provider,
      models: data.models || [],
      checked_at: data.checked_at,
      source: data.source,
      stale: !!data.stale,
      error: data.error,
    };
    renderLLMModelOptions();
    const defaults = data.defaults || {};
    if (!researchSelect.value && defaults.research_model) researchSelect.value = defaults.research_model;
    if (!summarySelect.value && defaults.summary_model) summarySelect.value = defaults.summary_model;
  } catch (error) {
    providerSelect.innerHTML = '<option value="">Ошибка загрузки провайдеров</option>';
    researchSelect.innerHTML = '<option value="">Ошибка загрузки моделей</option>';
    summarySelect.innerHTML = '<option value="">Ошибка загрузки моделей</option>';
    showToast("Не удалось загрузить каталог LLM: " + error.message, "danger");
  }
}

async function refreshLLMCatalog() {
  const providerSelect = document.getElementById("new-run-llm-provider");
  const provider = providerSelect?.value || llmCatalogCache?.provider || "";
  await loadLLMModels(provider, true);
}

async function checkSelectedLLMModel() {
  const providerSelect = document.getElementById("new-run-llm-provider");
  const researchSelect = document.getElementById("new-run-research-model");
  const modelId = researchSelect?.value;
  const provider = providerSelect?.value || llmCatalogCache?.provider || "";
  if (!modelId) {
    showToast("Выберите модель для проверки.", "warning");
    return;
  }
  try {
    const url = `${window.API_BASE}/api/ai-lab/llm/models/${encodeURIComponent(provider)}/${encodeURIComponent(modelId)}/check`;
    const res = await fetch(url, { method: "POST", headers: getAuthHeaders() });
    if (!res.ok) throw new Error(await responseError(res));
    const data = await res.json();
    if (llmCatalogCache) {
      const entry = (llmCatalogCache.models || []).find((item) => item.id === data.model_id);
      if (entry) {
        entry.is_available = data.available;
        entry.last_check = {
          available: data.available,
          latency_ms: data.latency_ms,
          protocol: data.protocol,
          error: data.error,
        };
      }
      renderLLMModelOptions();
      researchSelect.value = data.model_id;
    }
    if (data.available) {
      showToast(`Модель ${data.model_id} доступна (${data.protocol}, ${data.latency_ms}ms).`, "success");
    } else {
      showToast(`Модель ${data.model_id} недоступна: ${data.error || "нет ответа"}`, "danger");
    }
  } catch (error) {
    showToast("Проверка модели не удалась: " + error.message, "danger");
  }
}

function filterLLMModels() {
  renderLLMModelOptions();
}
function openNewRunModal() {
  const modal = document.getElementById("modal-new-run");
  if (modal) modal.classList.add("show");
  loadResearchPermissions();
  loadLLMModels();
}

function closeNewRunModal() {
  const modal = document.getElementById("modal-new-run");
  if (modal) modal.classList.remove("show");
}

async function submitCreateRun() {
  const objective = document.getElementById("new-run-objective")?.value.trim();
  const asset = document.getElementById("new-run-asset")?.value;
  const family = document.getElementById("new-run-family")?.value || "LOGREG";
  const autonomy = document.getElementById("new-run-autonomy")?.value || "EXPERIMENT";
  const permissionId = Number(document.getElementById("new-run-permission")?.value || 0);
  const budget = Number(document.getElementById("new-run-budget")?.value || 1);
  const minTrades = Number(document.getElementById("new-run-min-trades")?.value || 50);
  const maxDrawdown = Number(document.getElementById("new-run-max-dd")?.value || -5);
  const llmProvider = document.getElementById("new-run-llm-provider")?.value || null;
  const researchModel = document.getElementById("new-run-research-model")?.value || null;
  const summaryModel = document.getElementById("new-run-summary-model")?.value || null;

  if (!objective) {
    showToast("Укажите цель исследовательского запуска.", "danger");
    return;
  }
  if (!permissionId) {
    showToast("Выберите включённый research permission-профиль.", "danger");
    return;
  }

  const submit = document.querySelector("#modal-new-run .btn-primary:last-child");
  if (submit) { submit.disabled = true; submit.textContent = "Создание..."; }
  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        objective: objective,
        mode: "RESEARCH",
        llm_provider: llmProvider,
        research_model: researchModel,
        summary_model: summaryModel,
        autonomy_level: autonomy,
        budget_experiments: Math.max(1, Math.min(10000, budget)),
        created_by: "optimizer-ui",
        permission_id: permissionId,
        scope: {
          asset: asset,
          model_family: family,
          environment: "PAPER",
          min_trades: minTrades,
          max_drawdown_pct: maxDrawdown,
        },
      }),
    });
    if (!res.ok) throw new Error(await responseError(res));
    const data = await res.json();
    closeNewRunModal();
    await loadOptimizationRuns();
    const runId = data.id || data.run_id;
    if (runId) await selectRun(runId);
    showToast("Исследовательский запуск #" + (runId || "создан") + " создан. LIVE не затрагивается.", "success");
  } catch (error) {
    showToast("Ошибка создания запуска: " + error.message, "danger");
  } finally {
    if (submit) { submit.disabled = false; submit.textContent = "🚀 Запустить"; }
  }
}

function responseError(response) {
  return response.json()
    .then((payload) => payload.detail || payload.message || response.statusText)
    .catch(() => response.statusText);
}

function showToast(message, level = "info") {
  const prefix = level === "danger" ? "Ошибка: " : "";
  window.alert(prefix + message);
}

function updateRunControls(run) {
  const status = String(run?.status || "").toUpperCase();
  const terminal = new Set(["COMPLETED", "INSUFFICIENT_DATA", "INSUFFICIENT_EVIDENCE", "TECHNICAL_INVALID", "FAILED", "REJECTED", "CANCELLED", "ROLLED_BACK", "ACTIVE"]);
  const iterate = document.getElementById("btn-run-iterate");
  const pause = document.getElementById("btn-run-pause");
  const resume = document.getElementById("btn-run-resume");
  const cancel = document.getElementById("btn-run-cancel");
  // Queueing is valid only for a fresh draft or an evaluating run.
  // Never expose it while the worker already owns the run.
  if (iterate) iterate.style.display = ["DRAFT", "EVALUATING"].includes(status) ? "inline-flex" : "none";
  if (pause) pause.style.display = ["PLANNING", "RUNNING", "EVALUATING", "SHADOW"].includes(status) ? "inline-flex" : "none";
  if (resume) resume.style.display = status === "PAUSED" ? "inline-flex" : "none";
  if (cancel) cancel.style.display = terminal.has(status) ? "none" : "inline-flex";
}

async function triggerRunIterate() {
  const runId = currentSelectedRunId;
  if (!runId) return;
  const btn = document.getElementById("btn-run-iterate");
  if (btn) { btn.disabled = true; btn.textContent = "⚡ Выполняется..."; }
  try {
    const resp = await fetch(`${window.API_BASE}/api/ai-lab/runs/${runId}/iterate`, { method: "POST", headers: getAuthHeaders() });
    if (!resp.ok) throw new Error(await responseError(resp));
    const data = await resp.json();
    showToast("Шаг агента выполнен: " + (data.decision || "OK"), "success");
    await loadRunDetail(runId);
  } catch (error) {
    showToast("Ошибка шага агента: " + error.message, "danger");
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "⚡ Шаг агента"; }
  }
}

async function pauseRun() {
  const runId = currentSelectedRunId;
  if (!runId) return;
  try {
    const resp = await fetch(`${window.API_BASE}/api/ai-lab/runs/${runId}/pause`, { method: "POST", headers: getAuthHeaders() });
    if (!resp.ok) throw new Error(await responseError(resp));
    showToast("Запуск приостановлен", "info");
    await loadRunDetail(runId);
  } catch (error) {
    showToast("Ошибка паузы: " + error.message, "danger");
  }
}

async function resumeRun() {
  const runId = currentSelectedRunId;
  if (!runId) return;
  try {
    const resp = await fetch(`${window.API_BASE}/api/ai-lab/runs/${runId}/resume`, { method: "POST", headers: getAuthHeaders() });
    if (!resp.ok) throw new Error(await responseError(resp));
    showToast("Запуск возобновлён", "success");
    await loadRunDetail(runId);
  } catch (error) {
    showToast("Ошибка возобновления: " + error.message, "danger");
  }
}

async function cancelRun() {
  const runId = currentSelectedRunId;
  if (!runId) return;
  if (!window.confirm("Отменить исследовательский запуск #" + runId + "?")) return;
  try {
    const resp = await fetch(`${window.API_BASE}/api/ai-lab/runs/${runId}/cancel`, { method: "POST", headers: getAuthHeaders() });
    if (!resp.ok) throw new Error(await responseError(resp));
    showToast("Запуск отменён", "warning");
    await loadRunDetail(runId);
  } catch (error) {
    showToast("Ошибка отмены: " + error.message, "danger");
  }
}
