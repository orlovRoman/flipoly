// AI Lab Optimization Center (Phase 11 Dashboard)
// Handles Runs, Diagnostics/Step Audits, Visual Diff/Approval, and Revision Hash Chain Rollback

let currentSelectedRunId = null;
let currentPendingApprovalId = null;
let activeOptTab = "runs";
let runsRefreshTimer = null;

// Initialize when DOM loaded
document.addEventListener("DOMContentLoaded", () => {
  initOptimizerPage();
});

function initOptimizerPage() {
  bindTabs();
  loadOptimizationRuns();
  loadRevisions();

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
      const target = btn.dataset.target;
      switchOptTab(target);
    });
  });
}

function switchOptTab(tabId) {
  activeOptTab = tabId;
  document.querySelectorAll(".opt-tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.target === tabId);
  });
  document.querySelectorAll(".opt-tab-pane").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${tabId}`);
  });

  if (tabId === "detail" && currentSelectedRunId) {
    loadRunDetail(currentSelectedRunId);
  } else if (tabId === "approval" && currentSelectedRunId) {
    loadApprovalView(currentSelectedRunId);
  } else if (tabId === "revisions") {
    loadRevisions();
  }
}

// Helper: Format badges
function formatStatusBadge(status) {
  const s = (status || "").toUpperCase();
  let badgeClass = "badge-neutral";
  if (s === "ACTIVE" || s === "COMPLETED" || s === "APPROVED") badgeClass = "badge-success";
  else if (s === "RUNNING" || s === "SHADOW" || s === "PLANNING" || s === "TRAINING" || s === "EVALUATING" || s === "RESEARCH")
    badgeClass = "badge-running";
  else if (s === "RESEARCH_PROVISIONAL" || s === "INSUFFICIENT_DATA") badgeClass = "badge-pending";
  else if (s === "PENDING_APPROVAL") badgeClass = "badge-pending";
  else if (s === "FAILED" || s === "CANCELLED" || s === "REJECTED") badgeClass = "badge-danger";
  else if (s === "SUPERSEDED" || s === "ROLLED_BACK") badgeClass = "badge-neutral";

  return `<span class="badge ${badgeClass}">${escapeHtml(s)}</span>`;
}

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
  const apiKey = localStorage.getItem("polyflip_api_key") || "";
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { "X-API-Key": apiKey } : {}),
  };
}

// 2. Optimization Runs List
async function loadOptimizationRuns(silent = false) {
  const tbody = document.getElementById("runs-table-body");
  if (!silent) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">Загрузка запусков оптимизатора...</td></tr>`;
  }

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs?limit=50`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const runs = data.runs || [];

    if (runs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 2rem;">Запусков оптимизатора не найдено. Нажмите «Запустить новый поиск» для старта.</td></tr>`;
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
          <td>${created}</td>
          <td><span style="font-weight: 600; color: var(--poly-green);">${escapeHtml(asset)}</span></td>
          <td>${escapeHtml(objective)}</td>
          <td><span class="badge badge-neutral" style="font-size: 0.75rem;">${escapeHtml(r.autonomy_level || "AUTONOMOUS_SHADOW")}</span></td>
          <td>${statusBadge}</td>
          <td>${actionBtn}</td>
        </tr>
      `;
      })
      .join("");
  } catch (err) {
    console.error("loadOptimizationRuns error:", err);
    if (!silent) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--color-failed); padding: 2rem;">Ошибка загрузки запусков: ${escapeHtml(err.message)}</td></tr>`;
    }
  }
}

// Select Run & Load Detail
async function selectRun(runId) {
  currentSelectedRunId = runId;
  currentPendingApprovalId = null; // Сброс ID согласования предыдущего запуска
  switchOptTab("detail");
  await loadRunDetail(runId);
}

// 3. Run Details & Diagnostics
async function loadRunDetail(runId) {
  if (!runId) return;

  const headerEl = document.getElementById("detail-run-header");
  const scopeEl = document.getElementById("detail-run-scope");
  const stepsEl = document.getElementById("detail-run-steps");
  const auditsEl = document.getElementById("detail-run-audits");

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

    // Header Info
    const created = run.created_at ? new Date(run.created_at).toLocaleString("ru-RU") : "—";
    const completed = run.completed_at ? new Date(run.completed_at).toLocaleString("ru-RU") : "В процессе";

    headerEl.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
          <h3 style="margin: 0 0 0.5rem 0; font-size: 1.25rem;">Запуск #${run.id}: ${escapeHtml(run.objective)}</h3>
          <div style="color: var(--text-muted); font-size: 0.85rem;">
            Создан: <strong>${created}</strong> | Завершён: <strong>${completed}</strong> | Инициатор: <strong>${escapeHtml(run.created_by || "system")}</strong>
          </div>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center;">
          ${formatStatusBadge(run.status)}
          <button class="btn btn-warning btn-sm" onclick="switchOptTab('approval')">⚖️ Visual Diff / Approval</button>
        </div>
      </div>
    `;

    // Scope & Settings
    scopeEl.innerHTML = `
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
              Вы можете запросить серверный расчёт diff и создать ревизию для согласования перед переводом в LIVE.
            </div>
          </div>
          <button class="btn btn-warning" onclick="proposeLiveApproval(${run.id})">
            📝 Запросить допуск в LIVE (Request Approval)
          </button>
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
      decisionBox.style.display = "block";
    } else if (run.status === "ACTIVE") {
      bannerEl.innerHTML = `
        <div style="color: var(--poly-green); font-weight: 700;">
          ✅ Ревизия данного запуска утверждена и активна в LIVE
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
    const emptyMsg = `<div style="color: var(--text-muted); padding: 1.5rem; text-align: center;">Серверный diff еще не сформирован для данного запуска. Нажмите «Запросить допуск в LIVE» для генерации.</div>`;
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

// Propose Live Activation
async function proposeLiveApproval(runId) {
  if (!confirm(`Создать запрос на согласование LIVE активации для запуска #${runId}?`)) return;

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs/${runId}/approval`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        requested_action: "ACTIVATE",
        actor: "operator",
        reason: "Manual proposal from Web UI",
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка создания запроса: ${err.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    alert(`Запрос на согласование #${data.id} успешно создан!`);
    loadApprovalView(runId);
  } catch (e) {
    alert(`Ошибка: ${e.message}`);
  }
}

// Approve Live Activation Modal Trigger
function openApproveModal() {
  if (!currentPendingApprovalId) {
    alert("Нет активного запроса на согласование для данного запуска.");
    return;
  }
  document.getElementById("approve-modal").style.display = "flex";
}

function closeApproveModal() {
  document.getElementById("approve-modal").style.display = "none";
}

async function executeApproveLive() {
  const reason = document.getElementById("approve-reason-input").value.trim();
  const actor = document.getElementById("approve-actor-input").value.trim() || "admin";

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/approvals/${currentPendingApprovalId}/approve`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        actor: actor,
        reason: reason || "Manual operator approval via Web UI",
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка утверждения: ${err.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    alert(`✅ Ревизия успешно утверждена и активирована в LIVE!\nRevision Key: ${data.revision_key}`);
    closeApproveModal();
    loadApprovalView(currentSelectedRunId);
    loadOptimizationRuns(true);
  } catch (e) {
    alert(`Ошибка утверждения: ${e.message}`);
  }
}

// Reject Approval Modal Trigger
function openRejectModal() {
  if (!currentPendingApprovalId) {
    alert("Нет активного запроса на согласование для данного запуска.");
    return;
  }
  document.getElementById("reject-modal").style.display = "flex";
}

function closeRejectModal() {
  document.getElementById("reject-modal").style.display = "none";
}

async function executeRejectApproval() {
  const reason = document.getElementById("reject-reason-input").value.trim();
  const actor = document.getElementById("reject-actor-input").value.trim() || "admin";

  if (!reason) {
    alert("Пожалуйста, укажите причину отклонения предложения.");
    return;
  }

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/approvals/${currentPendingApprovalId}/reject`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        actor: actor,
        reason: reason,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка отклонения: ${err.detail || res.statusText}`);
      return;
    }

    alert(`❌ Предложение активации отклонено.`);
    closeRejectModal();
    loadApprovalView(currentSelectedRunId);
    loadOptimizationRuns(true);
  } catch (e) {
    alert(`Ошибка: ${e.message}`);
  }
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
    const revs = data.revisions || [];

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
        if (r.status === "SUPERSEDED") {
          rollbackBtn = `<button class="btn btn-danger btn-sm" onclick="openRollbackModal(${r.id}, '${shortKey}')">⏪ Откатить к этой</button>`;
        } else if (r.status === "ACTIVE") {
          rollbackBtn = `<span style="color: var(--poly-green); font-weight: 700; font-size: 0.85rem;">Текущая LIVE</span>`;
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

// Rollback Modal Trigger
let currentRollbackTargetId = null;
function openRollbackModal(revId, revKey) {
  currentRollbackTargetId = revId;
  document.getElementById("rollback-target-label").innerText = `#${revId} (${revKey})`;
  document.getElementById("rollback-modal").style.display = "flex";
}

function closeRollbackModal() {
  document.getElementById("rollback-modal").style.display = "none";
}

async function executeRollback() {
  if (!currentRollbackTargetId) return;

  const actor = document.getElementById("rollback-actor-input").value.trim() || "admin";
  const reason = document.getElementById("rollback-reason-input").value.trim() || "Emergency Operator Rollback";

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/deployments/rollback`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        target_revision_id: currentRollbackTargetId,
        actor: actor,
        reason: reason,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка отката: ${err.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    alert(`⏪ Откат успешно выполнен!\nВосстановлена ревизия: ${data.restored_revision_key}\nСтатус: ACTIVE\nБиржевые позиции сохранены.`);
    closeRollbackModal();
    loadRevisions();
  } catch (e) {
    alert(`Ошибка отката: ${e.message}`);
  }
}

// 6. Modal: Create New Experiment Run
function openNewRunModal() {
  document.getElementById("new-run-modal").style.display = "flex";
}

function closeNewRunModal() {
  document.getElementById("new-run-modal").style.display = "none";
}

async function executeCreateRun() {
  const objective = document.getElementById("run-objective-input").value.trim();
  const asset = document.getElementById("run-asset-select").value;
  const family = document.getElementById("run-family-select").value;
  const autonomy = document.getElementById("run-autonomy-select").value;

  if (!objective) {
    alert("Пожалуйста, опишите цель оптимизационного запуска.");
    return;
  }

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        objective: objective,
        autonomy_level: autonomy,
        scope: {
          asset: asset,
          model_family: family,
        },
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка создания запуска: ${err.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    alert(`🚀 Оптимизационный запуск #${data.id} успешно инициализирован!`);
    closeNewRunModal();
    loadOptimizationRuns();
  } catch (e) {
    alert(`Ошибка: ${e.message}`);
  }
}

/* -------------------------------------------------------------------------
 * Phase 10: Autonomous Agent Controls & Overlays
 * ------------------------------------------------------------------------- */
async function triggerRunIterate() {
  if (!currentRunId) return;
  const btn = document.getElementById("btn-run-iterate");
  if (btn) { btn.disabled = true; btn.textContent = "⚡ Выполняется..."; }
  try {
    const resp = await fetch(`/api/ai-lab/runs/${currentRunId}/iterate`, {
      method: "POST",
      headers: getAuthHeaders()
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Iteration failed");
    }
    const data = await resp.json();
    showToast("Шаг агента успешно выполнен: " + (data.decision || "OK"), "success");
    await loadRunDetail(currentRunId);
  } catch (e) {
    showToast("Ошибка шага агента: " + e.message, "danger");
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "⚡ Шаг агента"; }
  }
}

async function pauseRun() {
  if (!currentRunId) return;
  try {
    const resp = await fetch(`/api/ai-lab/runs/${currentRunId}/pause`, { method: "POST", headers: getAuthHeaders() });
    if (!resp.ok) throw new Error("Pause failed");
    showToast("Запуск приостановлен", "info");
    await loadRunDetail(currentRunId);
  } catch (e) {
    showToast("Ошибка паузы: " + e.message, "danger");
  }
}

async function resumeRun() {
  if (!currentRunId) return;
  try {
    const resp = await fetch(`/api/ai-lab/runs/${currentRunId}/resume`, { method: "POST", headers: getAuthHeaders() });
    if (!resp.ok) throw new Error("Resume failed");
    showToast("Запуск возобновлен", "success");
    await loadRunDetail(currentRunId);
  } catch (e) {
    showToast("Ошибка возобновления: " + e.message, "danger");
  }
}

async function cancelRun() {
  if (!currentRunId) return;
  if (!confirm("Вы уверены, что хотите отменить запуск #" + currentRunId + "?")) return;
  try {
    const resp = await fetch(`/api/ai-lab/runs/${currentRunId}/cancel`, { method: "POST", headers: getAuthHeaders() });
    if (!resp.ok) throw new Error("Cancel failed");
    showToast("Запуск отменен", "warning");
    await loadRunDetail(currentRunId);
  } catch (e) {
    showToast("Ошибка отмены: " + e.message, "danger");
  }
}
