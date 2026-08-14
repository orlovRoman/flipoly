/**
 * PolyFlip AI Lab — Optimizer Dashboard Client Logic
 */

let currentSelectedRunId = null;
let currentPendingApprovalId = null;
let pollTimer = null;

function getAuthHeaders() {
  let key = "test-key";
  try {
    key = localStorage.getItem("polyflip_api_key") || "test-key";
  } catch (e) {}
  return {
    "Content-Type": "application/json",
    "X-API-Key": key,
  };
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

function formatStatusBadge(status) {
  const s = String(status || "UNKNOWN").toUpperCase();
  const cls = `badge-${s.toLowerCase()}`;
  return `<span class="status-badge ${cls}">${escapeHtml(s)}</span>`;
}

function formatDate(isoStr) {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    return d.toLocaleString("ru-RU", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch (e) {
    return isoStr;
  }
}

function switchOptTab(tabName) {
  document.querySelectorAll(".opt-tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tabName);
  });
  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.style.display = pane.id === `tab-${tabName}` ? "block" : "none";
  });

  if (tabName === "runs") {
    loadRuns();
  } else if (tabName === "detail" && currentSelectedRunId) {
    loadRunDetail(currentSelectedRunId);
  } else if (tabName === "approval" && currentSelectedRunId) {
    loadApprovalView(currentSelectedRunId);
  } else if (tabName === "revisions") {
    loadRevisions();
  }
}

// Modal Helpers
function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("show");
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("show");
}

function openNewRunModal() {
  openModal("modal-new-run");
}

function openRollbackModal(targetId = null) {
  if (targetId) {
    document.getElementById("rollback-target-id").value = targetId;
  }
  openModal("modal-rollback");
}

// 1. Active Revision Hero Banner
async function loadActiveRevision() {
  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/deployments/revisions?limit=10`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) return;
    const revisions = await res.json();
    const active = revisions.find((r) => r.status === "ACTIVE");

    const titleEl = document.getElementById("hero-active-title");
    const metaEl = document.getElementById("hero-active-meta");

    if (active) {
      titleEl.innerHTML = `Ревизия #${active.id} <span class="status-badge badge-active">LIVE</span>`;
      metaEl.innerHTML = `Ключ: <strong>${escapeHtml(active.revision_key)}</strong> &bull; Хеш: <span class="mono-hash">${active.manifest_hash.substring(0, 16)}...</span> &bull; Активирована: ${formatDate(active.activated_at)}`;
    } else {
      titleEl.innerHTML = `Нет активной AI Lab ревизии`;
      metaEl.innerHTML = `Инференс использует базовые модели по умолчанию`;
    }
  } catch (err) {
    console.error("loadActiveRevision error:", err);
  }
}

// 2. Runs List
async function loadRuns() {
  const tbody = document.getElementById("runs-table-body");
  const statusFilter = document.getElementById("filter-run-status").value;

  try {
    let url = `${window.API_BASE}/api/ai-lab/runs?limit=50`;
    if (statusFilter) {
      url += `&status=${encodeURIComponent(statusFilter)}`;
    }

    const res = await fetch(url, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const runs = data.runs || [];

    if (runs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">Нет запусков по выбранному фильтру</td></tr>`;
      return;
    }

    tbody.innerHTML = runs
      .map((run) => {
        const threadId = run.agent_thread_id
          ? `<span class="mono-hash">${escapeHtml(run.agent_thread_id)}</span>`
          : `<span style="color: var(--text-muted); font-size: 0.85rem;">—</span>`;

        return `
        <tr>
          <td><strong>#${run.id}</strong></td>
          <td>
            <div style="font-weight: 600; color: var(--text-main);">${escapeHtml(run.objective)}</div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">Создатель: ${escapeHtml(run.created_by || "system")}</div>
          </td>
          <td>${formatStatusBadge(run.status)}</td>
          <td><span style="font-size: 0.8rem; font-family: var(--font-mono);">${escapeHtml(run.autonomy_level)}</span></td>
          <td>${run.experiments_completed || 0} / ${run.budget_experiments || 0}</td>
          <td>${threadId}</td>
          <td>${formatDate(run.created_at)}</td>
          <td>
            <button class="btn btn-outline" style="padding: 0.35rem 0.7rem; font-size: 0.8rem;" onclick="selectRun(${run.id})">
              Открыть &rarr;
            </button>
          </td>
        </tr>
      `;
      })
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--color-failed);">Ошибка загрузки: ${escapeHtml(err.message)}</td></tr>`;
  }
}

// Select Run & Load Detail
async function selectRun(runId) {
  currentSelectedRunId = runId;
  currentPendingApprovalId = null; // Сброс ID согласования предыдущего запуска
  switchOptTab("detail");
  await loadRunDetail(runId);
}

// 3. Run Detail, Timeline & Finalization Gate
async function loadRunDetail(runId) {
  if (!runId) return;

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs/${runId}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const run = data.run;
    const steps = data.steps || [];
    const results = data.results || [];

    // Header info
    document.getElementById("detail-run-id").innerText = run.id;
    document.getElementById("detail-run-objective").innerText = run.objective;
    document.getElementById("detail-run-badge").innerHTML = formatStatusBadge(run.status);
    document.getElementById("detail-autonomy").innerText = run.autonomy_level;
    document.getElementById("detail-budget").innerText = `${run.experiments_completed || 0} / ${run.budget_experiments || 0}`;
    document.getElementById("detail-thread-id").innerText = run.agent_thread_id || "—";
    document.getElementById("detail-created-by").innerText = run.created_by || "system";

    // Strict Finalization Gate summary parse
    const gatePanel = document.getElementById("gate-report-panel");
    const gateBadge = document.getElementById("gate-status-badge");
    const gateContent = document.getElementById("gate-report-content");

    if (run.summary) {
      try {
        const summaryObj = typeof run.summary === "string" ? JSON.parse(run.summary) : run.summary;
        const report = summaryObj.report || summaryObj;
        const recStatus = report.recommendation_status || summaryObj.status;

        if (recStatus) {
          gatePanel.style.display = "block";
          gateBadge.innerHTML = recStatus === "READY_FOR_SHADOW"
            ? `<span class="status-badge badge-active">🛡️ ГОТОВ К SHADOW</span>`
            : `<span class="status-badge badge-rejected">⚠️ ОТКЛОНЕН ГЕЙТОМ: ${escapeHtml(recStatus)}</span>`;

          const pnlStr = report.median_pnl !== undefined && report.median_pnl !== null ? `${report.median_pnl > 0 ? "+" : ""}${Number(report.median_pnl).toFixed(2)}%` : "—";
          const windowsStr = report.window_count !== undefined ? report.window_count : (report.min_windows || 3);
          const tradesStr = report.total_trades !== undefined ? report.total_trades : (report.min_trades || 50);

          gateContent.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.75rem;">
              <div class="diff-column">
                <div class="diff-label">OOT Окон (min 3)</div>
                <div class="diff-val" style="color: ${windowsStr >= 3 ? 'var(--poly-green)' : 'var(--color-rejected)'}">${windowsStr} / 3</div>
              </div>
              <div class="diff-column">
                <div class="diff-label">Сделок OOT (min 50)</div>
                <div class="diff-val" style="color: ${tradesStr >= 50 ? 'var(--poly-green)' : 'var(--color-rejected)'}">${tradesStr} / 50</div>
              </div>
              <div class="diff-column">
                <div class="diff-label">Медианный OOT PnL</div>
                <div class="diff-val" style="color: ${report.median_pnl > 0 ? 'var(--poly-green)' : 'var(--color-rejected)'}">${pnlStr}</div>
              </div>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.85rem; color: var(--text-muted);">
              ${escapeHtml(report.recommendation_reason || "")}
            </div>
          `;
        } else {
          gatePanel.style.display = "none";
        }
      } catch (e) {
        gatePanel.style.display = "none";
      }
    } else {
      gatePanel.style.display = "none";
    }

    // Steps Timeline
    const timelineEl = document.getElementById("steps-timeline");
    if (steps.length === 0) {
      timelineEl.innerHTML = `<div style="color: var(--text-muted);">Нет запланированных шагов</div>`;
    } else {
      timelineEl.innerHTML = steps
        .map((st) => {
          const dotClass = st.status.toLowerCase();
          const actionText = st.action ? `<span class="mono-hash">[${escapeHtml(st.action)}]</span>` : "";
          const hypothesis = st.hypothesis ? `<div style="font-size: 0.85rem; color: var(--text-main); margin-top: 0.35rem;">💡 ${escapeHtml(st.hypothesis)}</div>` : "";
          const errText = st.error_message ? `<div style="font-size: 0.8rem; color: var(--color-failed); margin-top: 0.35rem;">⚠️ ${escapeHtml(st.error_message)}</div>` : "";
          const summary = st.summary ? `<div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.35rem;">${escapeHtml(st.summary)}</div>` : "";

          return `
          <div class="timeline-item">
            <div class="timeline-dot ${dotClass}">#${st.step_index}</div>
            <div class="timeline-content">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                  <strong>${escapeHtml(st.step_type)}</strong> ${actionText}
                </div>
                <div>${formatStatusBadge(st.status)}</div>
              </div>
              ${hypothesis}
              ${summary}
              ${errText}
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.4rem;">
                Создан: ${formatDate(st.created_at)}
              </div>
            </div>
          </div>
        `;
        })
        .join("");
    }

    // Results Table
    const resTbody = document.getElementById("results-table-body");
    if (results.length === 0) {
      resTbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">Результаты еще не получены</td></tr>`;
    } else {
      resTbody.innerHTML = results
        .map((r) => {
          const pnlColor = (r.net_pnl || 0) > 0 ? "var(--poly-green)" : (r.net_pnl || 0) < 0 ? "var(--color-rejected)" : "inherit";
          const pnlText = r.net_pnl !== null && r.net_pnl !== undefined ? `${r.net_pnl > 0 ? "+" : ""}${Number(r.net_pnl).toFixed(2)}%` : "—";
          const ddText = r.max_drawdown !== null && r.max_drawdown !== undefined ? `${Number(r.max_drawdown).toFixed(2)}%` : "—";
          const metrics = r.metrics || {};
          const eceAuc = metrics.auc ? `AUC: ${Number(metrics.auc).toFixed(3)}` : (metrics.ece ? `ECE: ${Number(metrics.ece).toFixed(3)}` : "—");

          return `
          <tr>
            <td><strong>Config #${r.config_id}</strong></td>
            <td><span class="mono-hash">${escapeHtml(r.evaluation_kind)}</span></td>
            <td>${r.trade_count || 0}</td>
            <td style="color: ${pnlColor}; font-weight: 700;">${pnlText}</td>
            <td style="color: var(--color-failed);">${ddText}</td>
            <td style="font-size: 0.8rem; font-family: var(--font-mono);">${escapeHtml(eceAuc)}</td>
          </tr>
        `;
        })
        .join("");
    }
  } catch (err) {
    console.error("loadRunDetail error:", err);
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
    currentPendingApprovalId = data.id;
    alert(`Запрос на согласование #${data.id} успешно создан!`);
    await loadRunDetail(runId);
    await loadApprovalView(runId);
  } catch (err) {
    alert(`Ошибка сети: ${err.message}`);
  }
}

// Approve Live
async function executeApproveLive() {
  const actor = document.getElementById("approval-actor").value.trim() || "operator";
  const reason = document.getElementById("approval-reason").value.trim() || "Approved via Web UI";

  if (!confirm(`Вы уверены, что хотите АКТИВИРОВАТЬ эту ревизию в LIVE? Указатели моделей в ModelRegistry будут переключены.`)) return;

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/approvals/${currentPendingApprovalId || 1}/approve`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ actor, reason }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка активации: ${err.detail || res.statusText}`);
      return;
    }

    alert("✅ Ревизия успешно АКТИВИРОВАНА в LIVE!");
    await loadActiveRevision();
    if (currentSelectedRunId) {
      await loadRunDetail(currentSelectedRunId);
      await loadApprovalView(currentSelectedRunId);
    }
  } catch (err) {
    alert(`Ошибка сети: ${err.message}`);
  }
}

// Reject Approval
async function executeRejectApproval() {
  const actor = document.getElementById("approval-actor").value.trim() || "operator";
  const reason = document.getElementById("approval-reason").value.trim() || "Rejected via Web UI";

  if (!confirm(`Отклонить запрос на активацию?`)) return;

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/approvals/${currentPendingApprovalId || 1}/reject`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ actor, reason }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка отклонения: ${err.detail || res.statusText}`);
      return;
    }

    alert("Ревизия отклонена.");
    if (currentSelectedRunId) {
      await loadRunDetail(currentSelectedRunId);
      await loadApprovalView(currentSelectedRunId);
    }
  } catch (err) {
    alert(`Ошибка сети: ${err.message}`);
  }
}

// 5. Revisions & Events Hash Chain
async function loadRevisions() {
  const tbody = document.getElementById("revisions-table-body");

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/deployments/revisions?limit=50`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const revisions = await res.json();

    if (revisions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">Нет сохраненных ревизий</td></tr>`;
      return;
    }

    tbody.innerHTML = revisions
      .map((rev) => {
        const hashPreview = rev.manifest_hash ? `<span class="mono-hash">${rev.manifest_hash.substring(0, 16)}...</span>` : "—";
        const parentText = rev.parent_id ? `<strong>#${rev.parent_id}</strong>` : `<span style="color: var(--text-muted);">Root</span>`;

        return `
        <tr>
          <td><strong>#${rev.id}</strong></td>
          <td><span style="font-family: var(--font-mono); font-size: 0.85rem;">${escapeHtml(rev.revision_key)}</span></td>
          <td>${hashPreview}</td>
          <td>${parentText}</td>
          <td>${formatStatusBadge(rev.status)}</td>
          <td>${formatDate(rev.created_at)}</td>
          <td>${formatDate(rev.activated_at)}</td>
          <td>
            <button class="btn btn-outline" style="padding: 0.35rem 0.7rem; font-size: 0.8rem;" onclick="viewRevisionEvents(${rev.id})">
              🔗 События
            </button>
          </td>
        </tr>
      `;
      })
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--color-failed);">Ошибка: ${escapeHtml(err.message)}</td></tr>`;
  }
}

// View Revision Event Chain
async function viewRevisionEvents(revId) {
  const panel = document.getElementById("revision-events-panel");
  const revIdEl = document.getElementById("events-rev-id");
  const timelineEl = document.getElementById("events-timeline");

  panel.style.display = "block";
  revIdEl.innerText = revId;
  timelineEl.innerHTML = `<div style="color: var(--text-muted);">Загрузка событий...</div>`;

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/deployments/revisions/${revId}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const events = data.events || [];

    if (events.length === 0) {
      timelineEl.innerHTML = `<div style="color: var(--text-muted);">Нет зафиксированных событий</div>`;
      return;
    }

    timelineEl.innerHTML = events
      .map((ev, idx) => {
        const isRoot = ev.previous_hash === "0".repeat(64);
        const hashMatch = idx === 0 || events[idx - 1].event_hash === ev.previous_hash;

        return `
        <div class="timeline-item">
          <div class="timeline-dot succeeded">⛓️</div>
          <div class="timeline-content">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <strong>${escapeHtml(ev.event_type)}</strong>
              <span style="font-size: 0.8rem; color: var(--text-muted);">Актор: <strong>${escapeHtml(ev.actor || "system")}</strong></span>
            </div>
            <div style="font-size: 0.85rem; color: var(--text-main); margin-top: 0.35rem;">
              ${escapeHtml(ev.reason || "—")}
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.75rem; font-family: var(--font-mono); color: var(--text-muted);">
              <div>Prev Hash: <span style="color: var(--text-muted);">${ev.previous_hash.substring(0, 24)}...</span> ${isRoot ? "(GENESIS)" : ""}</div>
              <div>Event Hash: <span style="color: var(--poly-green);">${ev.event_hash.substring(0, 24)}...</span></div>
              <div>Целостность цепи: ${hashMatch ? '<span style="color: var(--poly-green);">✓ ВЕРИФИЦИРОВАНА</span>' : '<span style="color: var(--color-failed);">✕ ОШИБКА ХЕША</span>'}</div>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.4rem;">
              Время: ${formatDate(ev.created_at)}
            </div>
          </div>
        </div>
      `;
      })
      .join("");
  } catch (err) {
    timelineEl.innerHTML = `<div style="color: var(--color-failed);">Ошибка загрузки: ${escapeHtml(err.message)}</div>`;
  }
}

// 6. Submit New Run
async function submitCreateRun() {
  const objective = document.getElementById("new-run-objective").value.trim();
  const asset = document.getElementById("new-run-asset").value;
  const budget = parseInt(document.getElementById("new-run-budget").value, 10) || 10;
  const autonomy = document.getElementById("new-run-autonomy").value;
  const minTrades = parseInt(document.getElementById("new-run-min-trades").value, 10) || 50;
  const maxDd = parseFloat(document.getElementById("new-run-max-dd").value) || -5.0;

  if (!objective) {
    alert("Укажите цель оптимизации");
    return;
  }

  try {
    // Get permissions first
    const permRes = await fetch(`${window.API_BASE}/api/ai-lab/permissions`, {
      headers: getAuthHeaders(),
    });
    let permissionId = null;
    if (permRes.ok) {
      const pData = await permRes.json();
      if (pData.permissions && pData.permissions.length > 0) {
        permissionId = pData.permissions[0].id;
      }
    }

    const payload = {
      objective,
      scope: {
        asset,
        min_trades: minTrades,
        max_drawdown: maxDd,
      },
      autonomy_level: autonomy,
      budget_experiments: budget,
      created_by: "web-ui",
      permission_id: permissionId,
    };

    const res = await fetch(`${window.API_BASE}/api/ai-lab/runs`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка создания запуска: ${err.detail || res.statusText}`);
      return;
    }

    const run = await res.json();
    closeModal("modal-new-run");
    alert(`Запуск #${run.id} успешно создан!`);
    await loadRuns();
    selectRun(run.id);
  } catch (err) {
    alert(`Ошибка сети: ${err.message}`);
  }
}

// 7. Submit Rollback
async function submitRollback() {
  const targetIdRaw = document.getElementById("rollback-target-id").value.trim();
  const actor = document.getElementById("rollback-actor").value.trim() || "admin";
  const reason = document.getElementById("rollback-reason").value.trim() || "Emergency Rollback via Web UI";

  const targetRevisionId = targetIdRaw ? parseInt(targetIdRaw, 10) : null;

  try {
    const res = await fetch(`${window.API_BASE}/api/ai-lab/deployments/rollback`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({
        target_revision_id: targetRevisionId,
        actor,
        reason,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка отката: ${err.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    closeModal("modal-rollback");
    alert(`✅ Откат успешно выполнен! Активная ревизия переключена на #${data.active_revision_id}`);
    await loadActiveRevision();
    await loadRevisions();
  } catch (err) {
    alert(`Ошибка сети: ${err.message}`);
  }
}

// Init on Load
document.addEventListener("DOMContentLoaded", () => {
  loadActiveRevision();
  loadRuns();

  // Periodic polling every 5s for active tasks
  pollTimer = setInterval(() => {
    loadActiveRevision();
    if (currentSelectedRunId) {
      loadRunDetail(currentSelectedRunId);
    }
  }, 5000);
});
