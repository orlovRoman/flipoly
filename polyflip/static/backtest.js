// polyflip/static/backtest.js
'use strict';

// ─── CONFIG ─────────────────────────────────────────────────────────────
const API_KEY = window.API_KEY ?? null;
if (!API_KEY) console.warn('[backtest.js] window.API_KEY not set — requests will fail auth');
const HEADERS  = { 'Content-Type': 'application/json', 'X-API-Key': API_KEY ?? '' };

// ─── STATE ────────────────────────────────────────────────────────────────
let currentResult   = null;   // BacktestResult
let equityChart     = null;   // Chart.js instance
let strategyChart   = null;
let assetChart      = null;
let allTrades       = [];     // equity_curve flattened for table

// ─── API ──────────────────────────────────────────────────────────────────
const API = {
  async run(config)         { return fetchJSON('/api/backtest/run', 'POST', config); },
  async getResult(runId)    { return fetchJSON(`/api/backtest/result/${runId}`); },
  async getHistory()        { return fetchJSON('/api/backtest/history'); },
  async getModels()         { return fetchJSON('/api/backtest/models'); },
  async getDatasetStats()   { return fetchJSON('/api/backtest/dataset_stats'); },
  async getLiveSettings()   { return fetchJSON('/api/settings'); },
};

async function fetchJSON(url, method = 'GET', body = null) {
  const opts = { method, headers: HEADERS };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(url, opts);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
  return data;
}

function toUTCDateISO(dateStr, endOfDay = false) {
  const [y, m, d] = dateStr.split('-');
  return new Date(Date.UTC(+y, +m - 1, +d, endOfDay ? 23 : 0, endOfDay ? 59 : 0, endOfDay ? 59 : 0)).toISOString();
}

// ─── READ CONFIG FROM FORM ────────────────────────────────────────────────
function readConfig() {
  const assets = document.getElementById('cfg-assets').value
    .split(',').map(a => a.trim().toUpperCase()).filter(Boolean);
  const dateFrom = document.getElementById('cfg-date-from').value;
  const dateTo   = document.getElementById('cfg-date-to').value;

  return {
    assets,
    date_from:  dateFrom  ? toUTCDateISO(dateFrom, false) : null,
    date_to:    dateTo    ? toUTCDateISO(dateTo, true)    : null,
    min_snapshots_per_market: parseInt(document.getElementById('cfg-min-snaps').value) || 3,
    model_id: parseInt(document.getElementById('cfg-model-id').value) || null,
    strategy_mode:        document.getElementById('cfg-strategy-mode').value,
    min_time_left_min:    parseFloat(document.getElementById('cfg-min-time').value),
    max_time_left_min:    parseFloat(document.getElementById('cfg-max-time').value),
    no_flip_threshold:    parseFloat(document.getElementById('cfg-no-flip').value),
    flip_threshold:       parseFloat(document.getElementById('cfg-flip').value),
    trade_on_flip:        document.getElementById('cfg-trade-on-flip').checked,
    favorite_threshold:   parseFloat(document.getElementById('cfg-fav-thresh').value),
    auto_dead_zone_width: parseFloat(document.getElementById('cfg-dead-zone').value),
    favorite_min_price:   parseFloat(document.getElementById('cfg-fav-min').value),
    favorite_max_price:   parseFloat(document.getElementById('cfg-fav-max').value),

    initial_capital:      parseFloat(document.getElementById('cfg-capital').value),
    bet_sizing_mode:      document.getElementById('cfg-bet-sizing-mode').value,
    trade_bet_size_usdc:  parseFloat(document.getElementById('cfg-min-bet').value),
    max_bet_size_usdc:    parseFloat(document.getElementById('cfg-max-bet').value),
    min_edge:             parseFloat(document.getElementById('cfg-min-edge').value),
    max_edge:             parseFloat(document.getElementById('cfg-max-edge').value),
    slippage_pct:         parseFloat(document.getElementById('cfg-slippage').value) / 100,
  };
}

// ─── MAIN: RUN BACKTEST ───────────────────────────────────────────────────
async function runBacktest() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.textContent = '⏳ Submitting...';
  hideAlert();

  try {
    const config = readConfig();
    const res = await fetch('/api/backtest/submit', {
        method: "POST",
        headers: HEADERS,
        body: JSON.stringify(config)
    });
    
    if (!res.ok) {
        const errorData = await res.json();
        console.error("Backtest API error:", errorData);
        let msg = errorData.detail;
        if (Array.isArray(msg) || typeof msg === 'object') {
            msg = JSON.stringify(msg);
        }
        throw new Error(msg || `HTTP ${res.status}`);
    }
    const data = await res.json();
    
    await pollJobStatus(data.run_id, btn);
  } catch (err) {
    showAlert(`❌ Error: ${err.message}`, 'error');
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.textContent = '▶ Run Backtest';
  }
}

async function pollJobStatus(runId, btn) {
    const INTERVAL_MS = 2000;
    const MAX_WAIT_SEC = 300;   // 5 минут максимум
    const started = Date.now();

    return new Promise((resolve, reject) => {
        const timer = setInterval(async () => {
            if (Date.now() - started > MAX_WAIT_SEC * 1000) {
                clearInterval(timer);
                reject(new Error("Timeout after 5 minutes"));
                return;
            }

            try {
                const r = await fetch(`/api/backtest/status/${runId}`, {
                    headers: HEADERS
                });
                if (!r.ok) return;
                const data = await r.json();

                btn.textContent = `⏳ Running... ${data.progress}% (${Math.round(data.elapsed_sec)}s)`;

                if (data.status === "completed") {
                    clearInterval(timer);
                    currentResult = data.result;
                    allTrades = data.result.equity_curve || [];
                    renderAll(data.result);
                    switchTab('equity');
                    showAlert(`✅ Completed: ${data.result.total_trades} trades`, 'success');
                    loadHistory();
                    
                    btn.disabled = false;
                    btn.classList.remove('loading');
                    btn.textContent = '▶ Run Backtest';
                    resolve(data.result);
                } else if (data.status === "failed") {
                    clearInterval(timer);
                    reject(new Error(data.error || "Unknown error"));
                }
            } catch (e) {
                console.warn("poll error:", e);
            }
        }, INTERVAL_MS);
    });
}

// ─── RENDER ALL ───────────────────────────────────────────────────────────
function renderAll(result) {
  renderKPIs(result);
  renderEquityCurve(result.equity_curve);
  renderStrategyBreakdown(result.strategies);
  renderAssetBreakdown(result.assets);
  renderTradesTable();
}

// ─── KPI CARDS ────────────────────────────────────────────────────────────
function renderKPIs(r) {
  const pnlSign = r.net_profit >= 0;
  document.getElementById('kpi-trades').textContent  = r.total_trades;
  document.getElementById('kpi-pnl').textContent     = (pnlSign ? '+' : '') + r.net_profit.toFixed(2);
  document.getElementById('kpi-roi').textContent     = (r.roi_pct >= 0 ? '+' : '') + r.roi_pct.toFixed(1) + '%';
  document.getElementById('kpi-winrate').textContent = r.win_rate_pct.toFixed(1) + '%';
  document.getElementById('kpi-sharpe').textContent  = r.sharpe_ratio != null ? r.sharpe_ratio.toFixed(2) : '—';
  document.getElementById('kpi-pf').textContent      = r.profit_factor >= 999 ? '∞' : r.profit_factor.toFixed(2);
  document.getElementById('kpi-dd').textContent      = r.max_drawdown_pct.toFixed(1) + '%';
  document.getElementById('kpi-duration').textContent = r.duration_sec.toFixed(2) + 's';

  // Цвет PnL и ROI
  setCardColor('kpi-pnl-card', pnlSign);
  setCardColor('kpi-roi-card', r.roi_pct >= 0);
  setCardColor('kpi-winrate-card', r.win_rate_pct >= 50);
  setCardColor('kpi-dd-card', r.max_drawdown_pct < 20, true); // inverse: low = good
}

function setCardColor(cardId, isPositive, inverse = false) {
  const card = document.getElementById(cardId);
  if (!card) return;
  card.classList.remove('positive', 'negative');
  card.classList.add(isPositive ? 'positive' : 'negative');
}

// ─── EQUITY CURVE CHART ──────────────────────────────────────────────────
function renderEquityCurve(curve) {
  const ctx = document.getElementById('equity-chart');
  if (!ctx) return;
  if (equityChart) equityChart.destroy();

  const labels    = curve.map(p => `#${p.trade_index + 1}`);
  const cumulData = curve.map(p => p.cumulative_pnl);
  const colors    = curve.map(p => p.outcome === 'WIN' ? '#00d084' : '#ff4d4d');

  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Cumulative P&L ($)',
          data: cumulData,
          borderColor: '#667eea',
          backgroundColor: 'rgba(102,126,234,0.08)',
          borderWidth: 2,
          tension: 0.3,
          fill: true,
          pointBackgroundColor: colors,
          pointRadius: curve.length < 100 ? 4 : 2,
          pointHoverRadius: 6,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = curve[ctx.dataIndex];
              return [
                `Cumulative: $${p.cumulative_pnl.toFixed(2)}`,
                `Trade P&L: ${p.trade_pnl >= 0 ? '+' : ''}$${p.trade_pnl.toFixed(2)}`,
                `Asset: ${p.asset} | ${p.strategy}`,
                p.p_flip != null ? `P(flip): ${(p.p_flip*100).toFixed(1)}%` : '',
                p.edge != null ? `Edge: ${(p.edge*100).toFixed(1)}%` : '',
              ].filter(Boolean);
            }
          }
        }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 10, color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' },
             title: { display: true, text: 'P&L ($)', color: '#888' } }
      }
    }
  });
}

// ─── STRATEGY BREAKDOWN ──────────────────────────────────────────────────
function renderStrategyBreakdown(strategies) {
  const ctx = document.getElementById('strategy-chart');
  if (strategyChart) strategyChart.destroy();

  const labels  = strategies.map(s => s.strategy);
  const pnlData = strategies.map(s => s.net_pnl);
  const colors  = pnlData.map(v => v >= 0 ? '#00d084' : '#ff4d4d');

  strategyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Net P&L ($)', data: pnlData, backgroundColor: colors, borderRadius: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });

  const badgeClass = { ML_TREND: 'badge-ml', PURE_FAVORITE: 'badge-fav', OUTSIDER: 'badge-out', SKIP: '' };
  const html = strategies.map(s => `
    <div class="strategy-row">
      <span class="strategy-badge ${badgeClass[s.strategy] || ''}">${s.strategy}</span>
      <span style="flex:1; font-size:0.83rem;">${s.trades} trades</span>
      <span style="color:${s.net_pnl>=0?'#00d084':'#ff4d4d'}; font-weight:600; font-size:0.9rem;">
        ${s.net_pnl>=0?'+':''}$${s.net_pnl.toFixed(2)}
      </span>
      <span style="color:var(--text-muted); font-size:0.8rem;">${s.win_rate_pct.toFixed(1)}% WR</span>
      ${s.avg_edge != null ? `<span style="color:var(--text-muted); font-size:0.75rem;">Edge: ${(s.avg_edge*100).toFixed(1)}%</span>` : ''}
    </div>
  `).join('');
  document.getElementById('strategy-breakdown').innerHTML = html || '<div class="no-data">No strategy data</div>';
}

// ─── ASSET BREAKDOWN ──────────────────────────────────────────────────────
function renderAssetBreakdown(assets) {
  const ctx = document.getElementById('asset-chart');
  if (assetChart) assetChart.destroy();

  const labels  = assets.map(a => a.asset);
  const pnlData = assets.map(a => a.net_pnl);
  const colors  = ['#667eea','#00d084','#ffa500','#ff6b9d','#a78bfa'];

  assetChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Net P&L ($)', data: pnlData,
        backgroundColor: labels.map((_, i) => colors[i % colors.length]),
        borderRadius: 6
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });

  const html = assets.map(a => `
    <div class="strategy-row">
      <span style="font-weight:600; font-size:0.9rem; min-width:60px;">${a.asset}</span>
      <span style="flex:1; font-size:0.83rem; color:var(--text-muted);">${a.trades} trades</span>
      <span style="color:${a.net_pnl>=0?'#00d084':'#ff4d4d'}; font-weight:600;">
        ${a.net_pnl>=0?'+':''}$${a.net_pnl.toFixed(2)}
      </span>
      <span style="color:var(--text-muted); font-size:0.8rem;">${a.win_rate_pct.toFixed(1)}% WR</span>
    </div>
  `).join('');
  document.getElementById('asset-breakdown').innerHTML = html || '<div class="no-data">No asset data</div>';
}

// ─── TRADES TABLE ────────────────────────────────────────────────────────
function renderTradesTable() {
  if (!allTrades.length) return;

  const outcomeFilter   = document.getElementById('trades-filter-outcome').value;
  const strategyFilter  = document.getElementById('trades-filter-strategy').value;
  const sortMode        = document.getElementById('trades-sort').value;

  let filtered = allTrades.filter(t => {
    if (outcomeFilter  !== 'ALL' && t.outcome   !== outcomeFilter)   return false;
    if (strategyFilter !== 'ALL' && t.strategy  !== strategyFilter)  return false;
    return true;
  });

  if (sortMode === 'pnl_desc') filtered.sort((a, b) => b.trade_pnl - a.trade_pnl);
  if (sortMode === 'pnl_asc')  filtered.sort((a, b) => a.trade_pnl - b.trade_pnl);

  document.getElementById('trades-count').textContent = `${filtered.length} trades`;

  const badgeClass = { ML_TREND: 'badge-ml', PURE_FAVORITE: 'badge-fav', OUTSIDER: 'badge-out' };
  const rows = filtered.map(t => `
    <tr>
      <td style="color:var(--text-muted);">#${t.trade_index + 1}</td>
      <td style="max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:0.75rem; color:var(--text-muted);"
          title="${t.market_id}">${t.market_id.slice(0, 12)}…</td>
      <td><strong>${t.asset}</strong></td>
      <td><span class="strategy-badge ${badgeClass[t.strategy] || ''}">${t.strategy.replace('_', ' ')}</span></td>
      <td style="font-size:0.8rem;">${t.outcome === 'WIN' ? '📈 BUY' : '📉 BUY'}</td>
      <td>${t.executed_price.toFixed(3)}</td>
      <td>$${t.bet_size.toFixed(2)}</td>
      <td>${t.p_flip != null ? (t.p_flip * 100).toFixed(1) + '%' : '—'}</td>
      <td>${t.edge != null ? (t.edge * 100).toFixed(1) + '%' : '—'}</td>
      <td class="${t.trade_pnl >= 0 ? 'win-badge' : 'loss-badge'}">
        ${t.trade_pnl >= 0 ? '+' : ''}$${t.trade_pnl.toFixed(2)}
      </td>
      <td class="${t.outcome === 'WIN' ? 'win-badge' : 'loss-badge'}">${t.outcome}</td>
    </tr>
  `).join('');

  document.getElementById('trades-tbody').innerHTML =
    rows || '<tr><td colspan="11" class="no-data">No trades match filter</td></tr>';
}

// ─── HISTORY ──────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const data = await API.getHistory();
    const list = document.getElementById('history-list');
    if (!data.runs.length) {
      list.innerHTML = '<div class="no-data">No runs yet</div>';
      return;
    }
    list.innerHTML = data.runs.map(r => `
      <div class="history-item ${currentResult?.run_id === r.run_id ? 'active' : ''}"
           onclick="loadHistoricRun('${r.run_id}')">
        <div style="display:flex; justify-content:space-between;">
          <span style="font-size:0.83rem; font-weight:600;">${r.assets.join(', ')} — ${r.strategy_mode}</span>
          <span style="${r.net_profit>=0?'color:#00d084':'color:#ff4d4d'}; font-size:0.83rem; font-weight:600;">
            ${r.net_profit>=0?'+':''}$${r.net_profit.toFixed(2)}
          </span>
        </div>
        <div class="history-meta">
          ${r.total_trades} trades · ROI ${r.roi_pct.toFixed(1)}% · WR ${r.win_rate_pct.toFixed(1)}%
          · ${r.duration_sec.toFixed(1)}s · ${new Date(r.started_at).toLocaleTimeString()}
        </div>
      </div>
    `).join('');
  } catch (e) { /* silent */ }
}

async function loadHistoricRun(runId) {
  try {
    const data = await API.getResult(runId);
    if (data.result) {
      currentResult = data.result;
      allTrades = data.result.equity_curve || [];
      renderAll(data.result);
      switchTab('equity');
      
      // Заполняем форму параметрами из этого прогона
      const cfg = data.result.config;
      if (cfg) {
        document.getElementById('cfg-assets').value     = (cfg.assets || []).join(', ');
        if (cfg.date_from) {
          document.getElementById('cfg-date-from').value = cfg.date_from.split('T')[0];
        }
        if (cfg.date_to) {
          document.getElementById('cfg-date-to').value   = cfg.date_to.split('T')[0];
        }
        document.getElementById('cfg-min-snaps').value = cfg.min_snapshots_per_market || 3;
        if (cfg.model_id) {
          document.getElementById('cfg-model-id').value   = cfg.model_id;
        } else {
          document.getElementById('cfg-model-id').value   = "";
        }
        document.getElementById('cfg-strategy-mode').value = cfg.strategy_mode || 'ML';
        document.getElementById('cfg-min-time').value   = cfg.min_time_left_min || 1;
        document.getElementById('cfg-max-time').value   = cfg.max_time_left_min || 60;
        document.getElementById('cfg-no-flip').value    = cfg.no_flip_threshold || 0.35;
        document.getElementById('cfg-flip').value       = cfg.flip_threshold || 0.60;
        document.getElementById('cfg-fav-thresh').value = cfg.favorite_threshold != null ? cfg.favorite_threshold : 0.65;
        document.getElementById('cfg-dead-zone').value  = cfg.auto_dead_zone_width != null ? cfg.auto_dead_zone_width : 0.10;
        document.getElementById('cfg-fav-min').value    = cfg.favorite_min_price != null ? cfg.favorite_min_price : 0.55;
        document.getElementById('cfg-fav-max').value    = cfg.favorite_max_price != null ? cfg.favorite_max_price : 0.95;

        document.getElementById('cfg-capital').value    = cfg.initial_capital != null ? cfg.initial_capital : 1000;
        document.getElementById('cfg-bet-sizing-mode').value = cfg.bet_sizing_mode || 'scaled';
        document.getElementById('cfg-min-bet').value    = cfg.trade_bet_size_usdc != null ? cfg.trade_bet_size_usdc : 5;
        document.getElementById('cfg-max-bet').value    = cfg.max_bet_size_usdc != null ? cfg.max_bet_size_usdc : 50;
        document.getElementById('cfg-min-edge').value   = cfg.min_edge != null ? cfg.min_edge : -0.05;
        document.getElementById('cfg-max-edge').value   = cfg.max_edge != null ? cfg.max_edge : 0.50;
        document.getElementById('cfg-slippage').value   = (cfg.slippage_pct || 0.005) * 100;
        document.getElementById('cfg-trade-on-flip').checked = cfg.trade_on_flip === true;
        
        onStrategyChange();

      }
      
      loadHistory(); // обновить подсветку активного
    }
  } catch (e) {
    showAlert(`❌ ${e.message}`, 'error');
  }
}

// ─── DATASET STATS ────────────────────────────────────────────────────────
async function loadDatasetStats() {
  try {
    const data = await API.getDatasetStats();
    const stats = data.stats;
    const infoEl = document.getElementById('dataset-info');
    const totalMarkets = Object.values(stats).reduce((acc, outcomes) => {
      return acc + Object.values(outcomes).reduce((a, o) => a + (o.markets || 0), 0);
    }, 0);
    infoEl.innerHTML = `📦 ${totalMarkets} resolved markets available`;

    // Dataset tab content
    const rows = Object.entries(stats).map(([asset, outcomes]) => {
      const resolved = outcomes['YES'] || outcomes['NO'] ? Object.values(outcomes).reduce((a, o) => ({
        markets: (a.markets||0) + (o.markets||0),
        snapshots: (a.snapshots||0) + (o.snapshots||0),
        date_from: a.date_from || o.date_from,
        date_to: o.date_to || a.date_to,
      }), {}) : null;
      if (!resolved) return '';
      const dateFrom = resolved.date_from ? new Date(resolved.date_from).toLocaleDateString() : '—';
      const dateTo   = resolved.date_to   ? new Date(resolved.date_to).toLocaleDateString()   : '—';
      return `
        <div style="padding:0.8rem 0; border-bottom:1px solid var(--border);">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong>${asset}</strong>
            <span style="color:var(--text-muted); font-size:0.8rem;">${dateFrom} → ${dateTo}</span>
          </div>
          <div style="display:flex; gap:1.5rem; margin-top:0.3rem; font-size:0.8rem; color:var(--text-muted);">
            <span>🏪 ${resolved.markets} markets</span>
            <span>📸 ${resolved.snapshots} snapshots</span>
            <span>YES: ${outcomes.YES?.markets||0} / NO: ${outcomes.NO?.markets||0}</span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar" style="width:${Math.min(resolved.markets/10,100)}%;"></div>
          </div>
        </div>
      `;
    }).join('');
    document.getElementById('dataset-breakdown').innerHTML = rows || '<div class="no-data">No data</div>';
  } catch (e) {
    document.getElementById('dataset-info').textContent = 'Dataset stats unavailable';
  }
}

// ─── MODELS ───────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await API.getModels();
    const sel = document.getElementById('cfg-model-id');
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = `${m.asset} v${m.version}${m.is_active ? ' (активная)' : ''}`;
      sel.appendChild(opt);
    });
  } catch (e) { /* silent */ }
}

// ─── UI HELPERS ──────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.bt-tab').forEach((btn, i) => {
    const names = ['equity','strategies','assets','trades','dataset'];
    btn.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.bt-tab-content').forEach(el => {
    el.classList.toggle('active', el.id === `tab-${name}`);
  });
}

function onStrategyChange() {
  const mode = document.getElementById('cfg-strategy-mode').value;
  document.getElementById('ml-section').style.display = mode === 'ML' ? '' : 'none';
}



function resetConfig() {
  document.getElementById('cfg-assets').value     = 'BTC,ETH';
  document.getElementById('cfg-min-time').value   = '1';
  document.getElementById('cfg-max-time').value   = '60';
  document.getElementById('cfg-strategy-mode').value = 'ML';
  document.getElementById('cfg-no-flip').value    = '0.35';
  document.getElementById('cfg-flip').value       = '0.60';
  document.getElementById('cfg-fav-thresh').value = '0.65';
  document.getElementById('cfg-dead-zone').value  = '0.10';
  document.getElementById('cfg-yes-min').value    = '0.55';
  document.getElementById('cfg-yes-max').value    = '0.95';
  document.getElementById('cfg-no-min').value     = '0.55';
  document.getElementById('cfg-no-max').value     = '0.95';

  document.getElementById('cfg-capital').value    = '1000';
  document.getElementById('cfg-bet-sizing-mode').value = 'scaled';
  document.getElementById('cfg-min-bet').value    = '5';
  document.getElementById('cfg-max-bet').value    = '50';
  document.getElementById('cfg-min-edge').value   = '-0.05';
  document.getElementById('cfg-max-edge').value   = '0.50';
  document.getElementById('cfg-slippage').value   = '0.5';
  onStrategyChange();

}

async function applyLiveSettings() {
  try {
    const s = await API.getLiveSettings();
    document.getElementById('cfg-assets').value     = s.TRADE_ASSETS || 'BTC,ETH';
    document.getElementById('cfg-min-time').value   = s.TRADE_MIN_TIME_LEFT_SEC != null ? (parseInt(s.TRADE_MIN_TIME_LEFT_SEC) / 60).toFixed(1) : '1';
    document.getElementById('cfg-max-time').value   = s.TRADE_MAX_TIME_LEFT_SEC != null ? (parseInt(s.TRADE_MAX_TIME_LEFT_SEC) / 60).toFixed(0) : '60';
    document.getElementById('cfg-strategy-mode').value = s.TRADING_MODE === 'ml' ? 'ML' : 'PURE_FAVORITE';
    document.getElementById('cfg-no-flip').value    = s.TRADE_NO_FLIP_THRESHOLD != null ? parseFloat(s.TRADE_NO_FLIP_THRESHOLD).toFixed(2) : '0.35';
    document.getElementById('cfg-flip').value       = s.FLIP_THRESHOLD != null ? parseFloat(s.FLIP_THRESHOLD).toFixed(2) : '0.60';
    document.getElementById('cfg-fav-thresh').value = s.FAVORITE_THRESHOLD != null ? parseFloat(s.FAVORITE_THRESHOLD).toFixed(2) : '0.65';
    document.getElementById('cfg-dead-zone').value  = s.DEAD_ZONE_WIDTH != null ? parseFloat(s.DEAD_ZONE_WIDTH).toFixed(2) : '0.10';
    
    document.getElementById('cfg-fav-min').value    = s.FAVORITE_MIN_PRICE  != null ? s.FAVORITE_MIN_PRICE  : '0.55';
    document.getElementById('cfg-fav-max').value    = s.FAVORITE_MAX_PRICE  != null ? s.FAVORITE_MAX_PRICE  : '0.95';
    

      
    document.getElementById('cfg-capital').value    = s.INITIAL_CAPITAL != null ? s.INITIAL_CAPITAL : '1000';
    document.getElementById('cfg-bet-sizing-mode').value = s.BET_SIZING_MODE || 'scaled';
    document.getElementById('cfg-min-bet').value    = s.TRADE_BET_SIZE_USDC != null ? s.TRADE_BET_SIZE_USDC : '5';
    
    document.getElementById('cfg-max-bet').value    = s.MAX_BET_SIZE_USDC != null
        ? s.MAX_BET_SIZE_USDC
        // Если MAX_BET_SIZE_USDC не задан — дефолт = bet_size * 5 (эвристика: лимит = 5 ставок)
        : (s.TRADE_BET_SIZE_USDC != null
            ? (parseFloat(s.TRADE_BET_SIZE_USDC) * 5).toFixed(0)
            : '50');
      
    document.getElementById('cfg-min-edge').value   = s.MIN_EDGE !== undefined ? parseFloat(s.MIN_EDGE).toFixed(3) : '-0.05';
    document.getElementById('cfg-max-edge').value   = s.MAX_EDGE !== undefined ? parseFloat(s.MAX_EDGE).toFixed(3) : '0.50';
    document.getElementById('cfg-trade-on-flip').checked = s.TRADE_ON_FLIP === 'true';
    
    onStrategyChange();

    
    const minTime = parseFloat(document.getElementById('cfg-min-time').value) || 0;
    const maxTime = parseFloat(document.getElementById('cfg-max-time').value) || 0;
    if (minTime >= maxTime) {
      document.getElementById('cfg-max-time').value = String(Math.ceil(minTime) + 1);
      showAlert('Настройки загружены. max_time скорректирован (конфликт с min_time)', 'warning');
      return;
    }
    
    showAlert('Загружены текущие настройки торгового бота', 'success');
  } catch (e) {
    showAlert(`Не удалось загрузить настройки бота: ${e.message}`, 'error');
  }
}

function showAlert(msg, type = 'info') {
  const el = document.getElementById('bt-alert');
  el.className = `alert alert-${type}`;
  el.textContent = msg;
  el.style.display = 'block';
  if (type === 'success') setTimeout(hideAlert, 5000);
}

function hideAlert() {
  document.getElementById('bt-alert').style.display = 'none';
}

// ─── INIT ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Устанавливаем дефолтные даты бэктеста на последние 5 дней
  const today = new Date();
  const dateTo = today.toISOString().split('T')[0];
  const dateFrom = new Date(today.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  document.getElementById('cfg-date-from').value = dateFrom;
  document.getElementById('cfg-date-to').value = dateTo;

  loadModels();
  loadDatasetStats();
  loadHistory();
  onStrategyChange();
  function updateBetSizingUI() {
    const mode = document.getElementById('cfg-bet-sizing-mode').value;
    const group = document.getElementById('cfg-max-bet-group');
    if (group) {
      group.style.display = (mode === 'fixed') ? 'none' : 'block';
    }
  }
  document.getElementById('cfg-bet-sizing-mode').addEventListener('change', updateBetSizingUI);
  updateBetSizingUI();
});
