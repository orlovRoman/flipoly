document.addEventListener("DOMContentLoaded", () => {
  function parseFormattedFloat(val) {
    if (val === null || val === undefined || val === "") return NaN;
    return parseFloat(String(val).trim().replace(",", "."));
  }

  let apiKey = "test-key";
  let currentMinEdge = 0.05;
  try {
    apiKey = localStorage.getItem("polyflip_api_key") || "test-key";
  } catch (e) {
    console.warn("localStorage unavailable, using default key");
  }

  const elements = {
    capital: document.getElementById("stat-capital"),
    pnl: document.getElementById("stat-pnl"),
    winrate: document.getElementById("stat-winrate"),
    wl: document.getElementById("stat-wl"),
    assetTable: document.querySelector("#asset-stats-table tbody"),
    dailyPnlTable: document.querySelector("#daily-pnl-table tbody"),
    dailyPnlLoader: document.getElementById("daily-pnl-loader"),
    pnlTimeframeSelect: document.getElementById("pnl-timeframe-select"),
    refreshBtn: document.getElementById("btn-refresh-trading"),
  };


  let pnlChart = null;
  let wlChart = null;

  let currentPage = 1;
  let totalPages = 1;
  const PAGE_SIZE = 25;

  let _statsFetchToken = 0;

  async function fetchStats(tf) {
    const tfSelect = document.getElementById("asset-stats-tf-select");
    const timeframe = tf || (tfSelect ? tfSelect.value : "all");

    const myToken = ++_statsFetchToken;
    if (tfSelect) tfSelect.disabled = true;

    try {
      const response = await fetch(
        `${window.API_BASE}/api/trading/stats?timeframe=${encodeURIComponent(timeframe)}`,
        { headers: { "X-API-Key": apiKey } }
      );
      if (response.status === 401) {
        alert(
          "Неверный API ключ. Введите его на вкладке 'Настройки' в основном дашборде.",
        );
        return;
      }
      if (myToken !== _statsFetchToken) return;

      const data = await response.json();
      updateUI(data);

      const chartsTfSelect = document.getElementById("charts-tf-select");
      const chartsTimeframe = chartsTfSelect ? chartsTfSelect.value : "all";
      if (chartsTimeframe === timeframe && data.daily_pnl) {
        await updateCharts(data.daily_pnl);
      }
    } catch (error) {
      if (myToken !== _statsFetchToken) return;
      console.error("fetchStats error:", error);
    } finally {
      if (tfSelect) tfSelect.disabled = false;
    }
  }

  function updateUI(data) {
    if (!data) return;
    // Update KPIs
    if (elements.capital) elements.capital.textContent = `${(data.capital ?? 0).toFixed(2)} USDC`;
    if (elements.pnl) {
      const pnlVal = data.overall_pnl ?? 0;
      elements.pnl.textContent = `${pnlVal > 0 ? "+" : ""}${pnlVal.toFixed(2)} USDC`;
      elements.pnl.style.color = pnlVal >= 0 ? "#00ff88" : "#ff3366";
    }

    if (elements.winrate) elements.winrate.textContent = `${data.winrate ?? 0}%`;
    if (elements.wl && data.wins_vs_losses) {
      elements.wl.textContent = `${data.wins_vs_losses.wins ?? 0} / ${data.wins_vs_losses.losses ?? 0}`;
    }

    // Update Asset Table
    if (elements.assetTable && data.assets) {
      elements.assetTable.innerHTML = "";
      for (const [asset, stat] of Object.entries(data.assets)) {
        const winrate =
          stat.trades > 0 ? ((stat.wins / stat.trades) * 100).toFixed(1) : 0;
        const pnlColor = stat.pnl >= 0 ? "#00ff88" : "#ff3366";
        const tr = document.createElement("tr");
        tr.innerHTML = `
                  <td>${asset}</td>
                  <td>${stat.trades}</td>
                  <td>${winrate}%</td>
                  <td style="color: ${pnlColor}">${stat.pnl > 0 ? "+" : ""}${stat.pnl.toFixed(2)}</td>
              `;
        elements.assetTable.appendChild(tr);
      }
    }
  }

  async function updateCharts(dailyData) {
    try {
      const sortedDates = Object.keys(dailyData).sort();

      let cumulativePnl = 0;
      const pnlData = [];
      const dailyPnlData = [];
      const dailyColors = [];

      for (const date of sortedDates) {
        const dayPnl = dailyData[date].pnl;
        cumulativePnl += dayPnl;
        pnlData.push(cumulativePnl);
        dailyPnlData.push(dayPnl);
        dailyColors.push(dayPnl >= 0 ? "#00ff88" : "#ff3366");
      }

      const displayDates = [...sortedDates];
      // Если дат слишком мало (меньше 7), дополняем будущими днями, чтобы первый день отображался слева, а не растягивался
      if (displayDates.length > 0 && displayDates.length < 7) {
        const lastDateStr = displayDates[displayDates.length - 1];
        const lastDate = new Date(lastDateStr);
        const lastPnl = pnlData[pnlData.length - 1] ?? 0;
        while (displayDates.length < 7) {
          lastDate.setUTCDate(lastDate.getUTCDate() + 1);
          const nextDateStr = lastDate.toISOString().split("T")[0];
          displayDates.push(nextDateStr);
          pnlData.push(lastPnl);
          dailyPnlData.push(null);
          dailyColors.push("#00ff88");
        }
      }

      const pnlOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, labels: { color: "white" } },
        },
        scales: {
          x: {
            offset: true,
            ticks: { color: "rgba(255, 255, 255, 0.7)" },
            grid: { color: "rgba(255, 255, 255, 0.1)" },
          },
          y: {
            ticks: { color: "rgba(255, 255, 255, 0.7)" },
            grid: { color: "rgba(255, 255, 255, 0.1)" },
          },
        },
      };

      const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: true, labels: { color: "white" } } },
        scales: {
          x: {
            offset: true,
            ticks: { color: "rgba(255, 255, 255, 0.7)" },
            grid: { color: "rgba(255, 255, 255, 0.1)" },
          },
          y: {
            ticks: { color: "rgba(255, 255, 255, 0.7)" },
            grid: { color: "rgba(255, 255, 255, 0.1)" },
          },
        },
      };

      if (pnlChart) pnlChart.destroy();
      pnlChart = new Chart(
        document.getElementById("chart-daily-pnl").getContext("2d"),
        {
          type: "line",
          data: {
            labels: displayDates,
            datasets: [
              {
                label: "Кумулятивный PnL (USDC)",
                data: pnlData,
                borderColor: "#4facfe",
                backgroundColor: "rgba(79, 172, 254, 0.2)",
                fill: true,
                tension: 0.4,
              },
            ],
          },
          options: pnlOptions,
        },
      );

      if (wlChart) wlChart.destroy();
      wlChart = new Chart(
        document.getElementById("chart-daily-wl").getContext("2d"),
        {
          type: "bar",
          data: {
            labels: displayDates,
            datasets: [
              {
                label: "Дневной PnL (USDC)",
                data: dailyPnlData,
                backgroundColor: dailyColors,
                maxBarThickness: 30,
              },
            ],
          },
          options: {
            ...commonOptions,
            plugins: {
              legend: { display: true, labels: { color: "white" } },
              tooltip: {
                callbacks: {
                  label: function(context) {
                    const val = context.raw;
                    if (val === null || val === undefined) return "";
                    return ` Дневной PnL: ${val >= 0 ? "+" : ""}${val.toFixed(2)} USDC`;
                  }
                }
              }
            },
            scales: {
              x: {
                stacked: false,
                ticks: { color: "rgba(255, 255, 255, 0.7)" },
                grid: { color: "rgba(255, 255, 255, 0.1)" },
              },
              y: {
                stacked: false,
                ticks: {
                  color: "rgba(255, 255, 255, 0.7)",
                  callback: function(val) {
                    return (val >= 0 ? "+" : "") + val.toFixed(2) + " USDC";
                  }
                },
                grid: { color: "rgba(255, 255, 255, 0.1)" },
              }
            }
          },
        }
      );
    } catch (err) {
      console.error("updateCharts_error", err);
    }
  }

  if (elements.refreshBtn) {
    elements.refreshBtn.addEventListener("click", () => {
      fetchStats();
      const assetTf = document.getElementById("asset-stats-tf-select")?.value ?? "all";
      const chartsTf = document.getElementById("charts-tf-select")?.value ?? "all";
      if (chartsTf !== assetTf) {
        fetchChartsData();
      }
    });
  }

  // ----------------------------------------------------
  // Trading Settings Logic
  // ----------------------------------------------------
  let currentFlipThreshold = 0.70;

  const settingsElements = {
    apiKeyInput: document.getElementById("API_KEY"),
    favorMinTimeLeft: document.getElementById("FAVOR_MIN_TIME_LEFT_SEC"),
    favorMaxTimeLeft: document.getElementById("FAVOR_MAX_TIME_LEFT_SEC"),
    outsMinTimeLeft: document.getElementById("OUTS_MIN_TIME_LEFT_SEC"),
    outsMaxTimeLeft: document.getElementById("OUTS_MAX_TIME_LEFT_SEC"),
    betSizingMode: document.getElementById("BET_SIZING_MODE"),
    maxBetSizeGroup: document.getElementById("max-bet-size-group"),
    maxBetSize: document.getElementById("MAX_BET_SIZE_USDC"),
    betSize: document.getElementById("TRADE_BET_SIZE_USDC"),

    minDirectionProb: document.getElementById("MIN_DIRECTION_PROB"),
    minWinProb: document.getElementById("MIN_WIN_PROB"),
    tradeFlipThreshold: document.getElementById("TRADE_FLIP_THRESHOLD"),
    dailyLossLimit: document.getElementById("DAILY_LOSS_LIMIT_USDC"),
    tradingEnabled: document.getElementById("TRADING_ENABLED"),
    initialCapital: document.getElementById("INITIAL_CAPITAL"),
    minPrice: document.getElementById("TRADE_MIN_PRICE"),
    maxPrice: document.getElementById("TRADE_MAX_PRICE"),
    maxSpreadPct: document.getElementById("MAX_SPREAD_PCT"),
    outsiderPwinDiscount: document.getElementById("OUTSIDER_PWIN_DISCOUNT"),
    combinedDirDiscountWeight: document.getElementById("COMBINED_DIR_DISCOUNT_WEIGHT"),
    combinedDirStrongThreshold: document.getElementById("COMBINED_DIR_STRONG_THRESHOLD"),
    combinedRequireConsensus: document.getElementById("COMBINED_REQUIRE_CONSENSUS"),
    combinedFallbackToLogregOnNone: document.getElementById("COMBINED_FALLBACK_TO_LOGREG_ON_NONE"),
    invertLgbmSignal: document.getElementById("INVERT_LGBM_SIGNAL"),
    enableEceCorrection: document.getElementById("ENABLE_ECE_CORRECTION"),
    combinedLogregAbstainBand: document.getElementById("COMBINED_LOGREG_ABSTAIN_BAND"),
    combinedCostBuffer: document.getElementById("COMBINED_COST_BUFFER"),
    stopLossEnabled: document.getElementById("STOP_LOSS_ENABLED"),
    stopLossPctFavorite: document.getElementById("STOP_LOSS_PCT_FAVORITE"),
    stopLossPctOutsider: document.getElementById("STOP_LOSS_PCT_OUTSIDER"),
    stopLossCheckSec: document.getElementById("STOP_LOSS_CHECK_SEC"),
    takeProfitEnabled: document.getElementById("TAKE_PROFIT_ENABLED"),
    takeProfitMultiplier: document.getElementById("TAKE_PROFIT_MULTIPLIER"),
    takeProfitOrderMode: document.getElementById("TAKE_PROFIT_ORDER_MODE"),
    takeProfitCheckIntervalSec: document.getElementById("TAKE_PROFIT_CHECK_INTERVAL_SEC"),
    tradingModeRadios: document.querySelectorAll('input[name="trading_mode"]'),
    tradingModeBadge: document.getElementById('trading-mode-badge'),
    pollIntervalInput: document.getElementById("LIVE_POLL_INTERVAL_SECONDS"),

    favoriteThreshold: document.getElementById("FAVORITE_THRESHOLD"),
    tradeOnFavorite: document.getElementById("TRADE_ON_FAVORITE"),
    tradeOnFlip: document.getElementById("TRADE_ON_FLIP"),
    flipThreshold: document.getElementById("FLIP_THRESHOLD"),
    outsMinEdge: document.getElementById("OUTS_MIN_EDGE"),
    favoriteMinEdge: document.getElementById("FAVORITE_MIN_EDGE"),

    favoriteMinPrice: document.getElementById("FAVORITE_MIN_PRICE"),
    favoriteMaxPrice: document.getElementById("FAVORITE_MAX_PRICE"),
    outsiderMaxPrice: document.getElementById("OUTSIDER_MAX_PRICE"),
    bypassBetSizeCheck: document.getElementById("BYPASS_BET_SIZE_CHECK"),
    liquidityFraction: document.getElementById("LIQUIDITY_FRACTION"),
    maxPriceDrift: document.getElementById("MAX_PRICE_DRIFT"),
    combinedModeSettings: document.getElementById('combined-mode-settings'),

    // MRF (Market Regime Filter)
    mrfModeRadios: document.querySelectorAll('input[name="mrf_mode"]'),
    mrfModeBadge: document.getElementById('mrf-mode-badge'),
    mrfEfficiencyThreshold: document.getElementById('MARKET_REGIME_EFFICIENCY_THRESHOLD'),
    mrfBreadthThreshold: document.getElementById('MARKET_REGIME_BREADTH_THRESHOLD'),
    mrfMinHistory: document.getElementById('MARKET_REGIME_MIN_HISTORY'),
    mrfUnknownMultiplier: document.getElementById('MARKET_REGIME_UNKNOWN_MULTIPLIER'),
    mrfOutsiderTrendMultiplier: document.getElementById('MARKET_REGIME_OUTSIDER_TREND_MULTIPLIER'),
    mrfFilterVersion: document.getElementById('MARKET_REGIME_FILTER_VERSION'),
    mrfStatusPanel: document.getElementById('mrf-status-panel'),
    mrfCurrentRegime: document.getElementById('mrf-current-regime'),
    mrfStatusDetails: document.getElementById('mrf-status-details'),
    mrfStatEvaluated: document.getElementById('mrf-stat-evaluated'),
    mrfStatBlocked: document.getElementById('mrf-stat-blocked'),
    mrfStatMultiplier: document.getElementById('mrf-stat-multiplier'),
    mrfStatStrength: document.getElementById('mrf-stat-strength'),
    mrfPerAssetCards: document.getElementById('mrf-per-asset-cards'),
  };

  function updateDeadZoneInfo() {}

  
  function updateSizingModeUI() {
    if (settingsElements.betSizingMode && settingsElements.maxBetSizeGroup) {
      if (settingsElements.betSizingMode.value === "fixed") {
        settingsElements.maxBetSizeGroup.style.display = "none";
      } else {
        settingsElements.maxBetSizeGroup.style.display = "block";
      }
    }
  }

  if (settingsElements.betSizingMode) {
    settingsElements.betSizingMode.addEventListener("change", updateSizingModeUI);
    updateSizingModeUI();
  }


  function updateOutsiderStrategyStatus() {
    const statusSpan = document.getElementById("outsider-strategy-status");
    if (!statusSpan || !settingsElements.tradeOnFlip) return;
    if (settingsElements.tradeOnFlip.checked) {
      statusSpan.innerHTML = `<span style="background: rgba(0, 255, 136, 0.12); border: 1px solid #00ff88; color: #00ff88; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: bold; margin-left: 8px;">Активна</span>`;
    } else {
      statusSpan.innerHTML = `<span style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.1); color: var(--text-muted); padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: bold; margin-left: 8px;">Отключена</span>`;
    }
  }

  if (settingsElements.tradeOnFlip) {
    settingsElements.tradeOnFlip.addEventListener("change", updateOutsiderStrategyStatus);
    updateOutsiderStrategyStatus();
  }

  if (settingsElements.apiKeyInput) {
    settingsElements.apiKeyInput.value = apiKey;
  }

  async function loadRecommendedThresholds() {
    try {

      const res = await fetch(`${window.API_BASE}/api/settings/recommended_thresholds`, {
        headers: { "X-API-Key": apiKey }
      });
      const data = await res.json();
      const g = data.global;

      // Текущее значение no_flip берем из API
      const currentNoFlipVal = g.current_no_flip || 0.45;
      const currentNoFlipPct = Math.round(currentNoFlipVal * 100);
      
      const firstAsset = Object.keys(data.per_asset)[0];
      const recPct = firstAsset ? Math.round(data.per_asset[firstAsset].recommended_no_flip * 100) : Math.round(currentNoFlipVal * 100);

      // Подсказка под полем no_flip
      const hint = document.getElementById("no-flip-hint");
      if (hint) {
        hint.innerHTML = `
            Текущее значение: <strong>${currentNoFlipPct}%</strong>
            ${firstAsset ? `&nbsp;(Рекомендовано для ${firstAsset}: <strong style="color:#00ff88">${recPct}%</strong>)` : ""}
        `;
      }

      // Кнопка "Применить рекомендованное"
      const btn = document.getElementById("btn-apply-recommended-no-flip");
      if (btn) {
        if (firstAsset) {
          btn.style.display = "none";
        } else {
          btn.style.display = "none";
        }
      }

      // Per-asset пороги (автокалиброванные trainer'ом)
      const perAssetDiv = document.getElementById("per-asset-thresholds");
      if (perAssetDiv) {
        if (Object.keys(data.per_asset).length > 0) {
          perAssetDiv.innerHTML = Object.entries(data.per_asset).map(([asset, v]) =>
            `<span style="margin-right:1rem;">
                ${asset}: <strong>${Math.round(v.recommended_no_flip * 100)}%</strong>
                <span style="color:#00ff88; font-size:0.78rem;">(auto)</span>
            </span>`
          ).join("");
        } else {
          perAssetDiv.innerHTML = "";
        }
      }
    } catch (e) {
      console.warn("Failed to load recommended thresholds", e);
    }
  }

  function onTradingModeChange(mode) {
    if (settingsElements.combinedModeSettings) {
      settingsElements.combinedModeSettings.style.display = 'block';
    }
    if (settingsElements.tradingModeBadge) {
      settingsElements.tradingModeBadge.textContent = '⚖️ Режим: ML + LightGBM (Combined)';
      settingsElements.tradingModeBadge.className = `mode-badge mode-combined`;
    }
  }

  
  // ── MRF Mode Change Handler ──
  function updateMrfModeBadge() {
    const badge = settingsElements.mrfModeBadge;
    if (!badge) return;
    const checked = document.querySelector('input[name="mrf_mode"]:checked');
    const mode = checked ? checked.value : 'OFF';
    badge.textContent = mode;
    badge.className = 'mrf-regime-badge mrf-regime-' + (mode === 'ACTIVE' ? 'TREND' : mode === 'SHADOW' ? 'MIXED' : 'UNKNOWN');
    // Show/hide status panel
    if (settingsElements.mrfStatusPanel) {
      settingsElements.mrfStatusPanel.style.display = mode !== 'OFF' ? 'block' : 'none';
    }
  }

  if (settingsElements.mrfModeRadios) {
    settingsElements.mrfModeRadios.forEach(radio => {
      radio.addEventListener('change', () => {
        updateMrfModeBadge();
        loadMrfStatus();
      });
    });
    updateMrfModeBadge();
  }

  // ── MRF Live Status Polling ──
  async function loadMrfStatus() {
    try {
      const checked = document.querySelector('input[name="mrf_mode"]:checked');
      const mode = checked ? checked.value : 'OFF';
      if (mode === 'OFF') return;

      const res = await fetch(window.API_BASE + '/api/mrf/status?hours=24', {
        headers: { 'X-API-Key': apiKey }
      });
      if (!res.ok) return;
      const data = await res.json();

      const globalPhaseKnown =
        data.phase_available === true ||
        (data.phase_available === undefined &&
          data.latest_regime &&
          data.latest_regime !== 'UNKNOWN');
      if (settingsElements.mrfCurrentRegime) {
        const regime = globalPhaseKnown ? data.latest_regime : 'UNKNOWN';
        settingsElements.mrfCurrentRegime.textContent = globalPhaseKnown ? regime : 'НЕТ ДАННЫХ';
        settingsElements.mrfCurrentRegime.className = 'mrf-regime-badge mrf-regime-' + regime;
      }
      if (settingsElements.mrfStatusDetails) {
        if (!globalPhaseKnown) {
          settingsElements.mrfStatusDetails.textContent =
            'Нет актуальной MRF-оценки за выбранный период (' + data.hours + 'ч окно)';
        } else {
          const parts = [];
          if (data.latest_asset) parts.push(data.latest_asset);
          if (data.latest_strength != null) parts.push('сила: ' + (data.latest_strength * 100).toFixed(0) + '%');
          if (data.latest_confidence != null) parts.push('увер: ' + (data.latest_confidence * 100).toFixed(0) + '%');
          settingsElements.mrfStatusDetails.textContent =
            parts.join(' · ') + ' (' + data.hours + 'ч окно)';
        }
      }
      if (settingsElements.mrfStatEvaluated) {
        settingsElements.mrfStatEvaluated.textContent = data.total_evaluated != null ? data.total_evaluated : '—';
      }
      if (settingsElements.mrfStatBlocked) {
        settingsElements.mrfStatBlocked.textContent = data.total_blocked != null ? data.total_blocked : '—';
      }
      if (settingsElements.mrfStatMultiplier) {
        const m = data.avg_multiplier != null ? data.avg_multiplier : 1.0;
        settingsElements.mrfStatMultiplier.textContent = '×' + m.toFixed(2);
      }
      if (settingsElements.mrfStatStrength) {
        const s = globalPhaseKnown && data.latest_strength != null
          ? (data.latest_strength * 100).toFixed(0) + '%'
          : '—';
        settingsElements.mrfStatStrength.textContent = s;
      }
      if (settingsElements.mrfPerAssetCards && data.per_asset) {
        const container = settingsElements.mrfPerAssetCards;
        container.innerHTML = '';
        const assets = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP'];
        for (const asset of assets) {
          const info = data.per_asset[asset];
          const card = document.createElement('div');
          card.className = 'mrf-asset-card';
          const phaseKnown = !!info && (
            info.phase_available === true ||
            (info.phase_available === undefined && info.phase && info.phase !== 'UNKNOWN')
          );
          const phase = phaseKnown ? info.phase : 'UNKNOWN';
          const phaseLabel = phaseKnown ? phase : 'НЕТ ДАННЫХ';
          const strength = info ? (info.blocked > 0 ? ' (забл.)' : '') : ' (нет данных)';
          const strengthVal = phaseKnown && info.strength != null ? (info.strength * 100).toFixed(0) + '%' : '—';
          const confVal = phaseKnown && info.confidence != null ? (info.confidence * 100).toFixed(0) + '%' : '—';
          const pnl = info && info.pnl != null ? (info.pnl >= 0 ? '+' : '') + info.pnl.toFixed(2) : '—';
          const winRate = info ? info.win_rate_pct.toFixed(0) + '%' : '—';
          const mult = phaseKnown && info.avg_multiplier != null ? '\u00d7' + info.avg_multiplier.toFixed(2) : '—';
          card.innerHTML =
            '<div style="display:flex;justify-content:space-between;align-items:center;">' +
              '<span class="asset-name">' + asset + '</span>' +
              '<span class="asset-phase mrf-regime-badge mrf-regime-' + phase + '">' + phaseLabel + '</span>' +
            '</div>' +
            '<div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;display:grid;grid-template-columns:1fr 1fr;gap:2px 8px;">' +
              '<span>\u0441\u0438\u043b\u0430: ' + strengthVal + '</span>' +
              '<span>\u0443\u0432\u0435\u0440: ' + confVal + '</span>' +
              '<span>\u0431\u043b\u043e\u043a: ' + (info ? info.blocked : 0) + strength + '</span>' +
              '<span>mult: ' + mult + '</span>' +
              '<span>PnL: <span style="color:' + (info && info.pnl >= 0 ? '#4ade80' : '#f87171') + '">' + pnl + '</span></span>' +
              '<span>win: ' + winRate + '</span>' +
            '</div>';
          container.appendChild(card);
        }
      }
    } catch (e) {
      console.warn('MRF status load failed:', e);
    }
  }

  loadMrfStatus();
  setInterval(loadMrfStatus, 30000);


  if (settingsElements.tradingModeRadios) {
    settingsElements.tradingModeRadios.forEach(radio => {
      radio.addEventListener('change', (e) => onTradingModeChange(e.target.value));
    });
  }

  function getPerAssetFields() { return []; }

  async function checkCalibrationWarnings() {
    try {
      const res = await fetch(window.API_BASE + "/api/analytics/models", {
        headers: { "X-API-Key": apiKey },
      });
      if (!res.ok) return;
      const models = await res.json();
      
      const perAssetNames = getPerAssetFields();
      perAssetNames.forEach((asset) => {
        const warnSpan = document.getElementById(`calibration-warning-${asset}`);
        if (warnSpan) {
          warnSpan.style.display = "none";
          warnSpan.textContent = "";
        }
      });

      models.forEach((m) => {
        if (m.is_active && m.ece !== null && m.ece > 0.10) {
          const warnSpan = document.getElementById(`calibration-warning-${m.asset.toUpperCase()}`);
          if (warnSpan) {
            warnSpan.textContent = `⚠️ Калибровка: Плохая (ECE: ${m.ece.toFixed(4)})`;
            warnSpan.style.display = "inline-block";
          }
        }
      });
    } catch (e) {
      console.warn("Failed to check calibration warnings", e);
    }
  }

  async function loadSettings() {
    try {
      const res = await fetch(window.API_BASE + "/api/settings", {
        headers: { "X-API-Key": apiKey },
      });
      const data = await res.json();

      if (settingsElements.favorMinTimeLeft && data.FAVOR_MIN_TIME_LEFT_SEC)
        settingsElements.favorMinTimeLeft.value = data.FAVOR_MIN_TIME_LEFT_SEC;
      if (settingsElements.favorMaxTimeLeft && data.FAVOR_MAX_TIME_LEFT_SEC)
        settingsElements.favorMaxTimeLeft.value = data.FAVOR_MAX_TIME_LEFT_SEC;
      if (settingsElements.outsMinTimeLeft && data.OUTS_MIN_TIME_LEFT_SEC)
        settingsElements.outsMinTimeLeft.value = data.OUTS_MIN_TIME_LEFT_SEC;
      if (settingsElements.outsMaxTimeLeft && data.OUTS_MAX_TIME_LEFT_SEC)
        settingsElements.outsMaxTimeLeft.value = data.OUTS_MAX_TIME_LEFT_SEC;
      if (settingsElements.betSizingMode && data.BET_SIZING_MODE)
        settingsElements.betSizingMode.value = data.BET_SIZING_MODE;
      if (settingsElements.maxBetSize && data.MAX_BET_SIZE_USDC)
        settingsElements.maxBetSize.value = data.MAX_BET_SIZE_USDC;
      if (settingsElements.betSize && data.TRADE_BET_SIZE_USDC)
        settingsElements.betSize.value = data.TRADE_BET_SIZE_USDC;
      
      if (settingsElements.betSizingMode) {
        settingsElements.betSizingMode.dispatchEvent(new Event("change"));
      }
      if (settingsElements.minDirectionProb && data.MIN_DIRECTION_PROB !== undefined) {
        let val = parseFloat(data.MIN_DIRECTION_PROB);
        if (val <= 1) val *= 100;
        settingsElements.minDirectionProb.value = val.toFixed(1);
      }
      if (settingsElements.minWinProb && data.MIN_WIN_PROB !== undefined) {
        let val = parseFloat(data.MIN_WIN_PROB);
        if (val <= 1) val *= 100;
        settingsElements.minWinProb.value = val.toFixed(1);
      }
      if (settingsElements.tradeFlipThreshold && data.TRADE_FLIP_THRESHOLD !== undefined) {
        let val = parseFloat(data.TRADE_FLIP_THRESHOLD);
        if (val > 1) val /= 100;
        settingsElements.tradeFlipThreshold.value = Math.round(val * 100);
      }


      if (settingsElements.dailyLossLimit && data.DAILY_LOSS_LIMIT_USDC !== undefined) {
        settingsElements.dailyLossLimit.value = data.DAILY_LOSS_LIMIT_USDC;
      }
      if (settingsElements.stopLossEnabled && data.STOP_LOSS_ENABLED !== undefined) {
        settingsElements.stopLossEnabled.checked = data.STOP_LOSS_ENABLED === "true";
      }
      if (settingsElements.stopLossPctFavorite && data.STOP_LOSS_PCT_FAVORITE !== undefined) {
        settingsElements.stopLossPctFavorite.value = data.STOP_LOSS_PCT_FAVORITE;
      }
      if (settingsElements.stopLossPctOutsider && data.STOP_LOSS_PCT_OUTSIDER !== undefined) {
        settingsElements.stopLossPctOutsider.value = data.STOP_LOSS_PCT_OUTSIDER;
      }
      if (settingsElements.stopLossCheckSec && data.STOP_LOSS_CHECK_SEC !== undefined) {
        settingsElements.stopLossCheckSec.value = data.STOP_LOSS_CHECK_SEC;
      }
      if (settingsElements.takeProfitEnabled && data.TAKE_PROFIT_ENABLED !== undefined) {
        settingsElements.takeProfitEnabled.checked = data.TAKE_PROFIT_ENABLED === "true";
      }
      if (settingsElements.takeProfitMultiplier && data.TAKE_PROFIT_MULTIPLIER !== undefined) {
        settingsElements.takeProfitMultiplier.value = data.TAKE_PROFIT_MULTIPLIER;
      }
      if (settingsElements.takeProfitOrderMode && data.TAKE_PROFIT_ORDER_MODE !== undefined) {
        settingsElements.takeProfitOrderMode.value = data.TAKE_PROFIT_ORDER_MODE.toUpperCase();
      }
      if (settingsElements.takeProfitCheckIntervalSec && data.TAKE_PROFIT_CHECK_INTERVAL_SEC !== undefined) {
        settingsElements.takeProfitCheckIntervalSec.value = data.TAKE_PROFIT_CHECK_INTERVAL_SEC;
      }
      if (settingsElements.tradingEnabled && data.TRADING_ENABLED) {
        settingsElements.tradingEnabled.checked =
          data.TRADING_ENABLED === "true";
        // Setup direct toggle listener (added here to ensure element exists)
        if (!settingsElements.tradingEnabled.hasAttribute('data-toggle-bound')) {
          settingsElements.tradingEnabled.setAttribute('data-toggle-bound', 'true');
          settingsElements.tradingEnabled.addEventListener("change", async (e) => {
            const val = e.target.checked ? "true" : "false";
            try {
              const res = await fetch(window.API_BASE + "/api/settings/security/TRADING_ENABLED", {
                method: "PUT",
                headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
                body: JSON.stringify({ value: val })
              });
              if (!res.ok) {
                e.target.checked = !e.target.checked; // Revert
                const err = await res.json();
                alert("Ошибка: " + (err.detail || "Не удалось изменить статус торговли"));
              }
            } catch (err) {
              e.target.checked = !e.target.checked; // Revert
              console.error(err);
              alert("Ошибка сети при изменении статуса торговли");
            }
          });
        }
      }

      const statusBadge = document.getElementById("trading-status-badge");
      if (statusBadge && data.TRADING_ENABLED !== undefined) {
        const isEnabled = data.TRADING_ENABLED === "true";
        statusBadge.textContent = isEnabled ? "Торговля: ВКЛЮЧЕНА" : "Торговля: ВЫКЛЮЧЕНА";
        statusBadge.className = "status-indicator " + (isEnabled ? "online" : "offline");
      }
      if (settingsElements.initialCapital && data.INITIAL_CAPITAL)
        settingsElements.initialCapital.value = data.INITIAL_CAPITAL;
      if (settingsElements.minPrice && data.TRADE_MIN_PRICE)
        settingsElements.minPrice.value = data.TRADE_MIN_PRICE;
      if (settingsElements.maxPrice && data.TRADE_MAX_PRICE)
        settingsElements.maxPrice.value = data.TRADE_MAX_PRICE;

      if (settingsElements.favoriteThreshold && data.FAVORITE_THRESHOLD !== undefined) {
        let val = parseFloat(data.FAVORITE_THRESHOLD);
        settingsElements.favoriteThreshold.value = val;
      }
      if (settingsElements.tradeOnFavorite && data.TRADE_ON_FAVORITE !== undefined) {
        settingsElements.tradeOnFavorite.checked = data.TRADE_ON_FAVORITE === "true";
      }
      if (settingsElements.tradeOnFlip && data.TRADE_ON_FLIP !== undefined) {
        settingsElements.tradeOnFlip.checked = data.TRADE_ON_FLIP === "true";
        updateOutsiderStrategyStatus();
      }
      if (settingsElements.flipThreshold && data.FLIP_THRESHOLD !== undefined) {
        let val = parseFloat(data.FLIP_THRESHOLD);
        settingsElements.flipThreshold.value = Math.round(val * 100);
        currentFlipThreshold = val;
      }
      if (settingsElements.favoriteMaxPrice && data.FAVORITE_MAX_PRICE !== undefined) {
        settingsElements.favoriteMaxPrice.value = data.FAVORITE_MAX_PRICE;
      }
      if (settingsElements.outsMinEdge && data.OUTS_MIN_EDGE !== undefined) {
        let val = parseFloat(data.OUTS_MIN_EDGE);
        settingsElements.outsMinEdge.value = (val * 100).toFixed(1);
      }
      if (settingsElements.favoriteMinEdge && data.FAVORITE_MIN_EDGE !== undefined) {
        let val = parseFloat(data.FAVORITE_MIN_EDGE);
        settingsElements.favoriteMinEdge.value = (val * 100).toFixed(1);
      }


      if (settingsElements.favoriteMinPrice && data.FAVORITE_MIN_PRICE !== undefined) {
        settingsElements.favoriteMinPrice.value = data.FAVORITE_MIN_PRICE;
      }
      if (settingsElements.outsiderMaxPrice && data.OUTSIDER_MAX_PRICE !== undefined) {
        settingsElements.outsiderMaxPrice.value = data.OUTSIDER_MAX_PRICE;
      }
      if (settingsElements.bypassBetSizeCheck && data.BYPASS_BET_SIZE_CHECK) {
        settingsElements.bypassBetSizeCheck.checked = data.BYPASS_BET_SIZE_CHECK === "true";
      }
      if (settingsElements.liquidityFraction && data.LIQUIDITY_FRACTION !== undefined) {
        settingsElements.liquidityFraction.value = data.LIQUIDITY_FRACTION;
      }
      if (settingsElements.maxPriceDrift && data.MAX_PRICE_DRIFT !== undefined) {
        settingsElements.maxPriceDrift.value = data.MAX_PRICE_DRIFT;
      }
      if (settingsElements.maxSpreadPct && data.MAX_SPREAD_PCT !== undefined) {
        let val = parseFloat(data.MAX_SPREAD_PCT);
        settingsElements.maxSpreadPct.value = (val * 100).toFixed(1);
      }
      if (settingsElements.outsiderPwinDiscount && data.OUTSIDER_PWIN_DISCOUNT !== undefined) {
        let val = parseFloat(data.OUTSIDER_PWIN_DISCOUNT);
        settingsElements.outsiderPwinDiscount.value = (val * 100).toFixed(1);
      }
      if (settingsElements.combinedDirDiscountWeight && data.COMBINED_DIR_DISCOUNT_WEIGHT !== undefined) {
        let val = parseFloat(data.COMBINED_DIR_DISCOUNT_WEIGHT);
        settingsElements.combinedDirDiscountWeight.value = isNaN(val) ? "0" : Math.round(val * 100);
      }
      if (settingsElements.combinedDirStrongThreshold && data.COMBINED_DIR_STRONG_THRESHOLD !== undefined) {
        let val = parseFloat(data.COMBINED_DIR_STRONG_THRESHOLD);
        settingsElements.combinedDirStrongThreshold.value = isNaN(val) ? "65.0" : (val * 100).toFixed(1);
      }
      if (settingsElements.combinedRequireConsensus && data.COMBINED_REQUIRE_CONSENSUS !== undefined) {
        settingsElements.combinedRequireConsensus.checked = data.COMBINED_REQUIRE_CONSENSUS === "true";
      }
      if (settingsElements.combinedFallbackToLogregOnNone && data.COMBINED_FALLBACK_TO_LOGREG_ON_NONE !== undefined) {
        settingsElements.combinedFallbackToLogregOnNone.checked = data.COMBINED_FALLBACK_TO_LOGREG_ON_NONE === "true";
      }
      if (settingsElements.invertLgbmSignal && data.INVERT_LGBM_SIGNAL !== undefined) {
        settingsElements.invertLgbmSignal.checked = data.INVERT_LGBM_SIGNAL === "true";
      }
      if (settingsElements.enableEceCorrection && data.ENABLE_ECE_CORRECTION !== undefined) {
        settingsElements.enableEceCorrection.checked = data.ENABLE_ECE_CORRECTION === "true";
      }
      if (settingsElements.combinedLogregAbstainBand && data.COMBINED_LOGREG_ABSTAIN_BAND !== undefined) {
        const raw = parseFloat(data.COMBINED_LOGREG_ABSTAIN_BAND);
        settingsElements.combinedLogregAbstainBand.value = isNaN(raw) ? "5.0" : (raw * 100).toFixed(1);
      }
      if (settingsElements.combinedCostBuffer && data.COMBINED_COST_BUFFER !== undefined) {
        let val = parseFloat(data.COMBINED_COST_BUFFER);
        settingsElements.combinedCostBuffer.value = isNaN(val) ? "0.020" : val.toFixed(3);
      }


      // MRF settings
      if (settingsElements.mrfEfficiencyThreshold && data.MARKET_REGIME_EFFICIENCY_THRESHOLD !== undefined)
        settingsElements.mrfEfficiencyThreshold.value = data.MARKET_REGIME_EFFICIENCY_THRESHOLD;
      if (settingsElements.mrfBreadthThreshold && data.MARKET_REGIME_BREADTH_THRESHOLD !== undefined)
        settingsElements.mrfBreadthThreshold.value = data.MARKET_REGIME_BREADTH_THRESHOLD;
      if (settingsElements.mrfMinHistory && data.MARKET_REGIME_MIN_HISTORY !== undefined)
        settingsElements.mrfMinHistory.value = data.MARKET_REGIME_MIN_HISTORY;
      if (settingsElements.mrfUnknownMultiplier && data.MARKET_REGIME_UNKNOWN_MULTIPLIER !== undefined)
        settingsElements.mrfUnknownMultiplier.value = data.MARKET_REGIME_UNKNOWN_MULTIPLIER;
      if (settingsElements.mrfOutsiderTrendMultiplier && data.MARKET_REGIME_OUTSIDER_TREND_MULTIPLIER !== undefined)
        settingsElements.mrfOutsiderTrendMultiplier.value = data.MARKET_REGIME_OUTSIDER_TREND_MULTIPLIER;
      if (settingsElements.mrfFilterVersion && data.MARKET_REGIME_FILTER_VERSION !== undefined)
        settingsElements.mrfFilterVersion.value = data.MARKET_REGIME_FILTER_VERSION;
      if (data.MARKET_REGIME_FILTER_MODE) {
        const mode = data.MARKET_REGIME_FILTER_MODE.toUpperCase();
        const radio = document.querySelector('input[name="mrf_mode"][value="' + mode + '"]');
        if (radio) radio.checked = true;
        updateMrfModeBadge();
      }

      if (data.TRADING_MODE) {
        const mode = data.TRADING_MODE;
        const radio = document.querySelector(`input[name="trading_mode"][value="${mode}"]`);
        if (radio) radio.checked = true;
        onTradingModeChange(mode);
      }
      if (settingsElements.pollIntervalInput && data.LIVE_POLL_INTERVAL_SECONDS !== undefined) {
        settingsElements.pollIntervalInput.value = data.LIVE_POLL_INTERVAL_SECONDS;
      }

      if (data.TRADE_ASSETS) {
        const assets = data.TRADE_ASSETS.split(",");
        document.querySelectorAll(".asset-checkbox").forEach((cb) => {
          cb.checked = assets.includes(cb.value);
        });
      }

      // Заполняем индивидуальные настройки по активам
      const perAssetNames = getPerAssetFields();
      perAssetNames.forEach((asset) => {
        // Individual asset settings removed.
      });

      await loadRecommendedThresholds();
      await checkCalibrationWarnings();
    } catch (e) {
      console.error("Failed to load settings", e);
    }
  }

  const btnSaveSettings = document.getElementById("btn-save-trading-settings");
  if (btnSaveSettings) {
    btnSaveSettings.addEventListener("click", async (e) => {
      e.preventDefault();

      const tradeAssets = Array.from(
        document.querySelectorAll(".asset-checkbox:checked"),
      )
        .map((cb) => cb.value)
        .join(",");

      if (settingsElements.apiKeyInput) {
        const newKey = settingsElements.apiKeyInput.value.trim() || "test-key";
        localStorage.setItem("polyflip_api_key", newKey);
        apiKey = newKey; // update local scope variable
      }

      // ВАЖНО: нормализация значений перед сохранением в Redis:
      // FAVORITE_MIN_EDGE   → / 100  (хранится как float, напр. -0.01)
      // LIQUIDITY_FRACTION  → as-is  (хранится как float, напр. 0.05)
      // MAX_PRICE_DRIFT     → as-is  (хранится как float, напр. 0.03)
      // NO_MIN_PRICE        → as-is  (хранится как float, напр. 0.55)
      // OUTSIDER_MAX_PRICE   → as-is  (хранится как float)
      // Если меняешь формат хранения — обнови loadSettings() симметрично.
      const settingsToSave = {};
      if (settingsElements.favorMinTimeLeft) settingsToSave.FAVOR_MIN_TIME_LEFT_SEC = settingsElements.favorMinTimeLeft.value;
      if (settingsElements.favorMaxTimeLeft) settingsToSave.FAVOR_MAX_TIME_LEFT_SEC = settingsElements.favorMaxTimeLeft.value;
      if (settingsElements.outsMinTimeLeft) settingsToSave.OUTS_MIN_TIME_LEFT_SEC = settingsElements.outsMinTimeLeft.value;
      if (settingsElements.outsMaxTimeLeft) settingsToSave.OUTS_MAX_TIME_LEFT_SEC = settingsElements.outsMaxTimeLeft.value;
      if (settingsElements.betSizingMode) settingsToSave.BET_SIZING_MODE = settingsElements.betSizingMode.value;
      if (settingsElements.maxBetSize) settingsToSave.MAX_BET_SIZE_USDC = settingsElements.maxBetSize.value;
      if (settingsElements.betSize) settingsToSave.TRADE_BET_SIZE_USDC = settingsElements.betSize.value;
      if (settingsElements.minDirectionProb) settingsToSave.MIN_DIRECTION_PROB = parseFloat(settingsElements.minDirectionProb.value) / 100;
      if (settingsElements.minWinProb) settingsToSave.MIN_WIN_PROB = parseFloat(settingsElements.minWinProb.value) / 100;
      if (settingsElements.outsiderPwinDiscount) settingsToSave.OUTSIDER_PWIN_DISCOUNT = parseFloat(settingsElements.outsiderPwinDiscount.value) / 100;
      if (settingsElements.combinedDirDiscountWeight) settingsToSave.COMBINED_DIR_DISCOUNT_WEIGHT = parseFormattedFloat(settingsElements.combinedDirDiscountWeight.value) / 100;
      if (settingsElements.combinedDirStrongThreshold) settingsToSave.COMBINED_DIR_STRONG_THRESHOLD = parseFormattedFloat(settingsElements.combinedDirStrongThreshold.value) / 100;
      if (settingsElements.combinedRequireConsensus) settingsToSave.COMBINED_REQUIRE_CONSENSUS = settingsElements.combinedRequireConsensus.checked ? "true" : "false";
      if (settingsElements.combinedFallbackToLogregOnNone) settingsToSave.COMBINED_FALLBACK_TO_LOGREG_ON_NONE = settingsElements.combinedFallbackToLogregOnNone.checked ? "true" : "false";
      if (settingsElements.invertLgbmSignal) settingsToSave.INVERT_LGBM_SIGNAL = settingsElements.invertLgbmSignal.checked ? "true" : "false";
      if (settingsElements.enableEceCorrection) settingsToSave.ENABLE_ECE_CORRECTION = settingsElements.enableEceCorrection.checked ? "true" : "false";
      if (settingsElements.combinedLogregAbstainBand) settingsToSave.COMBINED_LOGREG_ABSTAIN_BAND = (parseFormattedFloat(settingsElements.combinedLogregAbstainBand.value) / 100).toString();
      if (settingsElements.combinedCostBuffer) settingsToSave.COMBINED_COST_BUFFER = parseFormattedFloat(settingsElements.combinedCostBuffer.value).toString();
      if (settingsElements.tradeFlipThreshold) settingsToSave.TRADE_FLIP_THRESHOLD = parseFloat(settingsElements.tradeFlipThreshold.value) / 100;

      if (settingsElements.dailyLossLimit) settingsToSave.DAILY_LOSS_LIMIT_USDC = settingsElements.dailyLossLimit.value;
      if (settingsElements.stopLossEnabled) settingsToSave.STOP_LOSS_ENABLED = settingsElements.stopLossEnabled.checked ? "true" : "false";
      if (settingsElements.stopLossPctFavorite) {
        const val = parseFloat(settingsElements.stopLossPctFavorite.value);
        if (isNaN(val) || val <= 0 || val >= 100) {
          alert("Стоп-лосс % (фаворит) должен быть от 1 до 99");
          return;
        }
        settingsToSave.STOP_LOSS_PCT_FAVORITE = val.toString();
      }
      if (settingsElements.stopLossPctOutsider) {
        const val = parseFloat(settingsElements.stopLossPctOutsider.value);
        if (isNaN(val) || val <= 0 || val >= 100) {
          alert("Стоп-лосс % (аутсайдер) должен быть от 1 до 99");
          return;
        }
        settingsToSave.STOP_LOSS_PCT_OUTSIDER = val.toString();
      }
      if (settingsElements.stopLossCheckSec) {
        const val = parseInt(settingsElements.stopLossCheckSec.value);
        if (isNaN(val) || val < 10 || val > 300) {
          alert("Интервал проверки стоп-лосса должен быть от 10 до 300 секунд");
          return;
        }
        settingsToSave.STOP_LOSS_CHECK_SEC = val.toString();
      }
      if (settingsElements.takeProfitEnabled) {
        settingsToSave.TAKE_PROFIT_ENABLED = settingsElements.takeProfitEnabled.checked ? "true" : "false";
      }
      if (settingsElements.takeProfitMultiplier) {
        const val = parseFloat(settingsElements.takeProfitMultiplier.value);
        if (isNaN(val) || val <= 1.0) {
          alert("Мультипликатор тейк-профита должен быть больше 1.0");
          return;
        }
        settingsToSave.TAKE_PROFIT_MULTIPLIER = val.toString();
      }
      if (settingsElements.takeProfitOrderMode) {
        const mode = settingsElements.takeProfitOrderMode.value.toUpperCase();
        if (!['GTD', 'TRIGGERED'].includes(mode)) {
          alert("Take-profit order mode must be GTD or TRIGGERED");
          return;
        }
        settingsToSave.TAKE_PROFIT_ORDER_MODE = mode;
      }
      if (settingsElements.takeProfitCheckIntervalSec) {
        const val = parseInt(settingsElements.takeProfitCheckIntervalSec.value);
        if (isNaN(val) || val < 10 || val > 300) {
          alert("Интервал проверки тейк-профита должен быть от 10 до 300 секунд");
          return;
        }
        settingsToSave.TAKE_PROFIT_CHECK_INTERVAL_SEC = val.toString();
      }
      if (settingsElements.initialCapital) settingsToSave.INITIAL_CAPITAL = settingsElements.initialCapital.value;
      if (settingsElements.minPrice) settingsToSave.TRADE_MIN_PRICE = settingsElements.minPrice.value;
      if (settingsElements.maxPrice) settingsToSave.TRADE_MAX_PRICE = settingsElements.maxPrice.value;
      if (settingsElements.maxSpreadPct) settingsToSave.MAX_SPREAD_PCT = (parseFloat(settingsElements.maxSpreadPct.value) / 100).toString();
      settingsToSave.TRADING_MODE = 'combined';

      if (settingsElements.pollIntervalInput) {
        let parsedInt = parseInt(settingsElements.pollIntervalInput.value, 10);
        if (isNaN(parsedInt) || parsedInt < 2) parsedInt = 2;
        if (parsedInt > 300) parsedInt = 300;
        settingsToSave.LIVE_POLL_INTERVAL_SECONDS = parsedInt.toString();
      }

      if (settingsElements.favoriteThreshold) settingsToSave.FAVORITE_THRESHOLD = parseFormattedFloat(settingsElements.favoriteThreshold.value);
      if (settingsElements.tradeOnFavorite) settingsToSave.TRADE_ON_FAVORITE = settingsElements.tradeOnFavorite.checked ? "true" : "false";
      if (settingsElements.tradeOnFlip) settingsToSave.TRADE_ON_FLIP = settingsElements.tradeOnFlip.checked ? "true" : "false";
      if (settingsElements.flipThreshold) settingsToSave.FLIP_THRESHOLD = parseFormattedFloat(settingsElements.flipThreshold.value) / 100;
      if (settingsElements.outsMinEdge) settingsToSave.OUTS_MIN_EDGE = parseFormattedFloat(settingsElements.outsMinEdge.value) / 100;
      if (settingsElements.favoriteMinEdge) settingsToSave.FAVORITE_MIN_EDGE = parseFormattedFloat(settingsElements.favoriteMinEdge.value) / 100;

      if (settingsElements.favoriteMinPrice) settingsToSave.FAVORITE_MIN_PRICE = parseFormattedFloat(settingsElements.favoriteMinPrice.value);
      if (settingsElements.favoriteMaxPrice) settingsToSave.FAVORITE_MAX_PRICE = parseFormattedFloat(settingsElements.favoriteMaxPrice.value);
      if (settingsElements.outsiderMaxPrice) settingsToSave.OUTSIDER_MAX_PRICE = parseFormattedFloat(settingsElements.outsiderMaxPrice.value);
      
      const bypassValue = settingsElements.bypassBetSizeCheck ? (settingsElements.bypassBetSizeCheck.checked ? "true" : "false") : null;
      
      if (settingsElements.liquidityFraction) settingsToSave.LIQUIDITY_FRACTION = parseFormattedFloat(settingsElements.liquidityFraction.value);
      if (settingsElements.maxPriceDrift) settingsToSave.MAX_PRICE_DRIFT = parseFormattedFloat(settingsElements.maxPriceDrift.value);
      if (settingsElements.combinedFallbackToMlOnNone) {
      }
      settingsToSave.TRADE_ASSETS = tradeAssets;

      // MRF settings
      if (settingsElements.mrfEfficiencyThreshold) settingsToSave.MARKET_REGIME_EFFICIENCY_THRESHOLD = parseFloat(settingsElements.mrfEfficiencyThreshold.value);
      if (settingsElements.mrfBreadthThreshold) settingsToSave.MARKET_REGIME_BREADTH_THRESHOLD = parseFloat(settingsElements.mrfBreadthThreshold.value);
      if (settingsElements.mrfMinHistory) settingsToSave.MARKET_REGIME_MIN_HISTORY = parseInt(settingsElements.mrfMinHistory.value);
      if (settingsElements.mrfUnknownMultiplier) settingsToSave.MARKET_REGIME_UNKNOWN_MULTIPLIER = parseFloat(settingsElements.mrfUnknownMultiplier.value);
      if (settingsElements.mrfOutsiderTrendMultiplier) settingsToSave.MARKET_REGIME_OUTSIDER_TREND_MULTIPLIER = parseFloat(settingsElements.mrfOutsiderTrendMultiplier.value);
      if (settingsElements.mrfFilterVersion) settingsToSave.MARKET_REGIME_FILTER_VERSION = parseInt(settingsElements.mrfFilterVersion.value);
      const mrfModeChecked = document.querySelector('input[name="mrf_mode"]:checked');
      if (mrfModeChecked) settingsToSave.MARKET_REGIME_FILTER_MODE = mrfModeChecked.value;


      // Считываем индивидуальные настройки по активам
      const perAssetNames = getPerAssetFields();
      perAssetNames.forEach((asset) => {
        
        if (modeSelect) {
        }
        if (minEdgeInput) {
          const val = minEdgeInput.value.trim();
        }
        if (maxPriceInput) {
          const val = maxPriceInput.value.trim();
        }
        if (flipThresholdInput) {
          const val = flipThresholdInput.value.trim();
        }
      });

      try {
        if (bypassValue !== null) {
            await fetch(window.API_BASE + "/api/settings/security/BYPASS_BET_SIZE_CHECK", {
                method: "PUT",
                headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
                body: JSON.stringify({ value: bypassValue }),
            });
        }

        const res = await fetch(window.API_BASE + "/api/settings/bulk", {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": apiKey,
          },
          body: JSON.stringify({ settings: settingsToSave }),
        });
        if (!res.ok) {
          alert("Не удалось сохранить настройки (ошибка сервера).");
          return;
        }
        const data = await res.json();
        if (data.errors && Object.keys(data.errors).length > 0) {
          alert(`Сохранено частично. Ошибки в следующих полях:\n` + 
                Object.entries(data.errors).map(([k, v]) => `- ${k}: ${v}`).join("\n"));
        } else {
          alert("Настройки торговли успешно сохранены!");
        }
        await loadSettings();
        fetchStats(); // Update capital based on new initial_capital
      } catch (err) {
        alert("Ошибка при сохранении настроек: " + err.message);
      }
    });
  }

  // ----------------------------------------------------
  // Trade Logs Logic
  // ----------------------------------------------------
  const escapeHtml = (unsafe) => {
    return String(unsafe)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  };

  async function loadLogs(page = 1) {
    if (typeof page !== 'number') {
      page = currentPage;
    }
    try {
      const res = await fetch(window.API_BASE + `/api/dashboard/trade_logs?page=${page}&page_size=${PAGE_SIZE}`, {
        headers: { "X-API-Key": apiKey },
      });
      const data = await res.json();
      
      currentPage = data.page;
      totalPages = data.pages;

      const tbody = document.querySelector("#trade-logs-table tbody");
      tbody.innerHTML = "";

      if (!data.items || data.items.length === 0) {
        tbody.innerHTML =
          '<tr><td colspan="15" style="text-align:center; padding: 1rem;">Нет событий</td></tr>';
        renderPagination(currentPage, totalPages, data.total || 0);
        return;
      }

      const rows = [];

      data.items.forEach((log) => {
        const displayTime = log.updated_at ? log.updated_at : log.created_at;
        const timeStr = new Date(displayTime).toLocaleTimeString();
        const flipColor = log.predicted_flip_prob > 0.5 ? "#00ff88" : "#ff3366";
        let statusColor = "#8F9BB3"; // SKIPPED
        let displayStatus = log.status;
        if (log.status === "SUCCESS") statusColor = "#00ff88";
        if (log.status === "FAILED") statusColor = "#ff3366";
        
        if (log.status === "SUCCESS" && (log.stop_loss_status === "TRIGGERED" || log.exit_reason === "STOP_LOSS")) {
            displayStatus = "Закрыто по стоп-лоссу";
            statusColor = "#ffb020"; // Yellow/orange for stop-loss
        }
        
        const isTpClosed = log.take_profit_status === "TRIGGERED" || log.take_profit_status === "QUEUED" || log.exit_reason === "TAKE_PROFIT";
        if (log.status === "SUCCESS" && isTpClosed) {
            const sellPrice = log.take_profit_sell_price || log.close_price;
            const price = sellPrice ? ` @ $${parseFloat(sellPrice).toFixed(3)}` : "";
            displayStatus = `Закрыто по тейк-профиту${price}`;
            statusColor = "#00ff88"; // Green for take-profit
        }

        window.funnelLogs = window.funnelLogs || {};
        if (log.funnel_log) {
            window.funnelLogs[log.id] = log.funnel_log;
        }

        let reasonHtml = escapeHtml(log.error_msg || "-");
        if (log.status === "SKIPPED") {
            reasonHtml = `<span style="color: #ffb020">${escapeHtml(log.error_msg || "-")}</span>`;
        }
        const isPureFav = log.active_features && log.active_features.includes("PURE_FAVORITE");
        const isCrypto = log.active_features && (log.active_features.includes("LIGHTGBM_TREND") || log.active_features.includes("CRYPTO_TREND"));
        const isCombined = log.active_features && log.active_features.includes("COMBINED_ML_LGBM");
        
        let rawPhase = "";
        let phaseSuffix = "";
        // phaseSuffix вычисляется для ml и combined (оба используют phase-модели)
        if (!isPureFav && !isCrypto && log.executed_price > 0) {
            const dev = Math.abs(log.executed_price - 0.5);
            if (dev < 0.10) { phaseSuffix = " <span style='font-size:0.85em; color:var(--text-muted);'>(contested)</span>"; rawPhase = "contested"; }
            else if (dev < 0.25) { phaseSuffix = " <span style='font-size:0.85em; color:var(--text-muted);'>(leaning)</span>"; rawPhase = "leaning"; }
            else { phaseSuffix = " <span style='font-size:0.85em; color:var(--text-muted);'>(decided)</span>"; rawPhase = "decided"; }
        }
        let lgbmName = "LightGBM";
        if (log.funnel_log && log.funnel_log.direction_model_key) {
            lgbmName = log.funnel_log.direction_model_key;
            if (log.funnel_log.direction_model_version) {
                lgbmName += ` v${log.funnel_log.direction_model_version}`;
            }
        }

        let lrNameBase = "";
        if (!isCrypto || isCombined) {
             lrNameBase = log.model_version ? `v${log.model_version}${phaseSuffix}` : "";
        }
        
        let logregDisplayHTML = lrNameBase || "-";
        if (!lrNameBase) {
             if (isPureFav) {
                 logregDisplayHTML = `<span title="Стратегия PureFav" style="cursor:help; color:var(--text-muted);">- (PureFav)</span>`;
             } else if (log.status === "SUCCESS" && !log.funnel_log) {
                 logregDisplayHTML = `<span title="Старые данные" style="cursor:help; color:var(--text-muted);">- (Legacy)</span>`;
             } else if (isCrypto && !isCombined) {
                 logregDisplayHTML = `<span title="Режим CryptoOnly не использует LogReg" style="cursor:help; color:var(--text-muted);">-</span>`;
             } else {
                 logregDisplayHTML = `<span style="color:var(--text-muted);">-</span>`;
             }
        } else if (isCombined) {
             const errMsg = log.error_msg || "";
             if (errMsg.includes("Consensus failed: CONFLICT")) {
                 logregDisplayHTML = `${lrNameBase} <br><span style="color: #ffb020; font-size:0.85em;">(⚠ conflict)</span>`;
             } else if (errMsg.includes("Consensus failed: BOTH_ABSTAIN") || errMsg.includes("Consensus failed")) {
                 logregDisplayHTML = `${lrNameBase} <br><span style="color: #ffb020; font-size:0.85em;">(⚠ abstain)</span>`;
             }
        }

        let pnlText = "-";
        let pnlColor = "var(--text-main)";
        if (log.status === "SUCCESS" && log.pnl !== null && log.pnl !== undefined) {
          const pnlVal = parseFloat(log.pnl);
          pnlText = (pnlVal >= 0 ? "+" : "") + pnlVal.toFixed(2) + " USDC";
          pnlColor = pnlVal >= 0 ? "#00ff88" : "#ff3366";
        }



        const isOutsider =
          log.market_role === "OUTSIDER" ||
          (
            !log.market_role &&
            Number(log.executed_price) > 0 &&
            Number(log.executed_price) < 0.5
          );

        const isLightGBM =
          log.strategy_type === "LIGHTGBM_TREND" ||
          (log.active_features && log.active_features.includes("LIGHTGBM"));

        let betTypeHtml = "";
        if (isOutsider) {
          betTypeHtml = isLightGBM
            ? `<span class="bet-outsider" style="color: #ffb020; font-weight: 500;">
                 Ставка на аутсайдера
                 <small style="font-size: 0.8em; color: #aaa;">(сигнал ${lgbmName})</small>
               </span>`
            : `<span class="bet-outsider" style="color: #ffb020; font-weight: 500;">
                 Ставка на аутсайдера
               </span>`;
        } else {
          betTypeHtml = isLightGBM
            ? `<span class="bet-favorite" style="color: #00ff88; font-weight: 500;">
                 Ставка на фаворита
                 <small style="font-size: 0.8em; color: #aaa;">(сигнал ${lgbmName})</small>
               </span>`
            : `<span class="bet-favorite" style="color: #00ff88; font-weight: 500;">
                 Ставка на фаворита
               </span>`;
        }

        let outcomeBadge = "";
        if (log.outcome_bought === "YES") {
          outcomeBadge = `<span style="color: #00ff88; font-size: 0.8em; margin-right: 6px; padding: 2px 4px; background: rgba(0,255,136,0.1); border-radius: 4px;">UP</span>`;
        } else if (log.outcome_bought === "NO") {
          outcomeBadge = `<span style="color: #ff3366; font-size: 0.8em; margin-right: 6px; padding: 2px 4px; background: rgba(255,51,102,0.1); border-radius: 4px;">DOWN</span>`;
        }

        let directionBadge = "-";
        let dirVal = log.direction_display || "NONE";

        if (dirVal === "UP") {
          directionBadge = `<span style="color: #00ff88; font-size: 0.8em; padding: 2px 6px; background: rgba(0,255,136,0.1); border-radius: 4px; font-weight: bold;">UP</span>`;
        } else if (dirVal === "DOWN") {
          directionBadge = `<span style="color: #ff3366; font-size: 0.8em; padding: 2px 6px; background: rgba(255,51,102,0.1); border-radius: 4px; font-weight: bold;">DOWN</span>`;
        } else if (dirVal === "UNAVAILABLE") {
          directionBadge = `<span style="color: #ffb020; font-size: 0.8em; padding: 2px 6px; background: rgba(255,176,32,0.1); border-radius: 4px; font-weight: bold;">UNAVAILABLE</span>`;
        } else {
          directionBadge = `<span style="color: #999; font-size: 0.8em; padding: 2px 6px; background: rgba(153,153,153,0.1); border-radius: 4px; font-weight: bold;">NONE</span>`;
        }

        const betText = log.amount_usdc > 0 ? `${outcomeBadge}$${parseFloat(log.amount_usdc).toFixed(2)}` : "-";

        const logDateObj = new Date(displayTime);
        let timeLeftStr = "-";
        if (log.end_time_est) {
            const endDateObj = new Date(log.end_time_est);
            let diffSec = Math.floor((endDateObj - logDateObj) / 1000);
            if (diffSec < 0) diffSec = 0;
            const m = Math.floor(diffSec / 60);
            const s = diffSec % 60;
            timeLeftStr = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
        } else {
            // Фолбэк, если данных об end_time_est нет
            const minutes = (logDateObj.getUTCMinutes() % 15) + 1;
            timeLeftStr = `${String(minutes).padStart(2, '0')}:00 (offset)`;
        }

        const LIVE_COLOR = "#e53e3e";
        const LIVE_BG_COLOR = "rgba(229, 62, 62, 0.16)";
        const LIVE_BORDER_COLOR = "rgba(229, 62, 62, 0.3)";

        const isLiveTrade = log.mode === "LIVE";
        const trStyle = isLiveTrade
            ? `background: ${LIVE_BG_COLOR}; border-left: 4px solid ${LIVE_COLOR}; border-bottom: 1px solid ${LIVE_BORDER_COLOR};`
            : 'border-bottom: 1px solid rgba(255,255,255,0.05);';
        
        const liveBadge = isLiveTrade
            ? `<span style="background: ${LIVE_COLOR}; color: #ffffff; padding: 2px 6px; border-radius: 4px; font-weight: 800; font-size: 0.75rem; margin-right: 6px; letter-spacing: 0.5px; box-shadow: 0 0 8px rgba(229,62,62,0.8);">🔥 LIVE</span>`
            : '';

        rows.push(`
                    <tr style="${trStyle}">
                        <td style="padding: 8px; color: var(--text-muted); font-family: monospace;">${timeLeftStr}</td>
                        <td style="padding: 8px; color: var(--text-muted);">${timeStr}</td>
                        <td style="padding: 8px;"><a href="#" class="market-link" data-market-id="${log.market_id}" data-asset="${escapeHtml(log.asset)}" style="color: var(--text-main); text-decoration: underline; cursor: pointer;">${escapeHtml(log.question)}</a></td>
                        <td style="padding: 8px;">
                            ${directionBadge}
                            ${(lgbmName && lgbmName !== "LightGBM") ? `<br><span style="font-size:0.85em; color:var(--poly-blue);">${lgbmName}</span>` : (isCrypto && log.model_version ? `<br><span style="font-size:0.85em; color:var(--poly-blue);">LightGBM v${log.model_version}</span>` : "")}
                        </td>
                        <td style="padding: 8px; color: var(--poly-blue);">${logregDisplayHTML}</td>
                        <td style="padding: 8px; color: ${statusColor};">${liveBadge}${displayStatus}</td>
                        <td style="padding: 8px;">${betTypeHtml}</td>
                        <td style="padding: 8px; font-weight: bold; color: var(--text-main);">${betText}</td>
                        <td style="padding: 8px;">${parseFloat(log.executed_price) > 0 ? "$" + parseFloat(log.executed_price).toFixed(3) : "-"}</td>
                        <td style="padding: 8px; color: ${pnlColor}; font-weight: 600;">${pnlText}</td>
                        <td style="padding: 8px; color: ${flipColor};">${(log.predicted_flip_prob * 100).toFixed(1)}%</td>
                        <td style="padding: 8px;">${
                          log.edge !== null && log.edge !== undefined
                            ? `<span style="color: ${
                                parseFloat(log.edge) >= currentMinEdge
                                  ? "#00ff88"
                                  : parseFloat(log.edge) >= 0.0
                                  ? "#ffb020"
                                  : "#ff3366"
                              }; font-weight: bold;">${(parseFloat(log.edge) * 100).toFixed(1)}%</span>`
                            : "-"
                        }</td>
                        <td style="padding: 8px;">${
                          (() => {
                            const fl = log.funnel_log;
                            const a = fl && (fl.mrf_audit || fl.mrf);
                            if (!a) return '<span style="color:var(--text-muted);">—</span>';
                            const p = a.global_phase || a.global_regime || 'UNKNOWN';
                            const pol = a.policy || {};
                            const multiplier = pol.multiplier ?? a.multiplier;
                            const dotColor = pol.allow === false ? '#ff3366' : pol.allow === true ? '#00ff88' : '#ffb020';
                            const title = a.failure_reason ? ` title="${escapeHtml(a.failure_reason)}"` : '';
                            return `<span class="mrf-regime-badge mrf-regime-${p}" style="font-size:0.78rem;"${title}>${escapeHtml(p)}</span>` +
                              (multiplier != null && Number(multiplier) !== 1.0 ? `<br><span style="font-size:0.75rem;color:${dotColor};">x${Number(multiplier).toFixed(2)}</span>` : '');
                          })()
                        }</td>
                        <td style="padding: 8px;">${reasonHtml}</td>
                        <td style="padding: 8px; text-align: center;">${log.funnel_log ? `<span style="cursor: pointer; font-size: 1.2em;" title="Детали инференса" onclick="showFunnelDiagnostic(${log.id})">🔍</span>` : ""}</td>
                    </tr>
                `);
      });

      tbody.innerHTML = rows.join("");

      if (rows.length > 0) {
        const tr = tbody.firstElementChild;
        // Ищем th именно в той же таблице, что и tbody (в данном случае #trade-logs-table)
        const thCount = document.querySelectorAll(
          "#trade-logs-table th",
        ).length;
        const tdCount = tr ? tr.querySelectorAll("td").length : 0;
        console.assert(
          thCount === tdCount,
          `Колонки рассинхронизированы: ${thCount} th vs ${tdCount} td`,
        );
      }
      renderPagination(currentPage, totalPages, data.total || 0);
    } catch (e) {
      console.error("Failed to load trade logs", e);
    }
  }

  function renderPagination(page, pages, total) {
    const btnPrev = document.getElementById("btn-prev");
    const btnNext = document.getElementById("btn-next");
    const pageInfo = document.getElementById("page-info");
    
    if (pageInfo) {
      pageInfo.textContent = `Стр. ${page} из ${pages || 1} (${total} записей)`;
    }
    if (btnPrev) {
      btnPrev.disabled = page <= 1;
    }
    if (btnNext) {
      btnNext.disabled = page >= pages;
    }
  }



  const btnRefreshLogs = document.getElementById("btn-refresh-logs");
  if (btnRefreshLogs) {
    btnRefreshLogs.addEventListener("click", loadLogs);
  }

  // Обработчик клика по ссылке на Polymarket
  const logsTable = document.getElementById("trade-logs-table");
  if (logsTable) {
    logsTable.addEventListener("click", async (e) => {
      const link = e.target.closest(".market-link");
      if (!link) return;
      e.preventDefault();
      
      const marketId = link.getAttribute("data-market-id");
      const asset = link.getAttribute("data-asset");
      if (!marketId) return;
      
      const originalText = link.textContent;
      link.textContent = "⏳...";
      link.style.pointerEvents = "none";
      
      try {
        const response = await fetch(`https://gamma-api.polymarket.com/markets/${marketId}`);
        if (!response.ok) throw new Error("Failed to fetch");
        const marketData = await response.json();
        
        let slug = marketData.slug;
        if (!slug && marketData.event) {
          slug = marketData.event.slug;
        }
        
        if (slug) {
          window.open(`https://polymarket.com/market/${slug}`, "_blank");
        } else if (marketData.eventSlug) {
          window.open(`https://polymarket.com/event/${marketData.eventSlug}`, "_blank");
        } else {
          window.open(`https://polymarket.com/?search=${encodeURIComponent(asset)}`, "_blank");
        }
      } catch (err) {
        console.error("Error fetching market slug from Polymarket:", err);
        window.open(`https://polymarket.com/?search=${encodeURIComponent(asset)}`, "_blank");
      } finally {
        link.textContent = originalText;
        link.style.pointerEvents = "auto";
      }
    });
  }

  const flipInput = document.getElementById("FLIP_THRESHOLD") || document.getElementById("TRADE_FLIP_THRESHOLD");
  if (flipInput) {
    flipInput.addEventListener("input", () => {
      const val = parseFloat(flipInput.value) / 100;
      if (!isNaN(val)) loadRecommendedThresholds(val);
    });
  }

  const noFlipInput = document.getElementById("NO_FLIP_THRESHOLD");
  if (noFlipInput) {
    noFlipInput.addEventListener("input", () => {
      const flipVal = flipInput ? parseFloat(flipInput.value) / 100 : 0.85;
      if (!isNaN(flipVal)) loadRecommendedThresholds(flipVal);
    });
  }

  // Bind pagination buttons
  const btnPrev = document.getElementById("btn-prev");
  const btnNext = document.getElementById("btn-next");
  if (btnPrev) {
    btnPrev.addEventListener("click", () => {
      if (currentPage > 1) {
        loadLogs(currentPage - 1);
      }
    });
  }
  if (btnNext) {
    btnNext.addEventListener("click", () => {
      if (currentPage < totalPages) {
        loadLogs(currentPage + 1);
      }
    });
  }

  let _chartsFetchToken = 0;

  async function fetchChartsData(tf) {
    const tfSelect = document.getElementById("charts-tf-select");
    const timeframe = tf || (tfSelect ? tfSelect.value : "all");
    const myToken = ++_chartsFetchToken;
    if (tfSelect) tfSelect.disabled = true;

    try {
      const response = await fetch(
        `${window.API_BASE}/api/trading/stats?timeframe=${encodeURIComponent(timeframe)}`,
        { headers: { "X-API-Key": apiKey } }
      );
      if (!response.ok) return;
      if (myToken !== _chartsFetchToken) return;

      const data = await response.json();
      if (data.daily_pnl) {
        await updateCharts(data.daily_pnl);
      }
    } catch (e) {
      if (myToken !== _chartsFetchToken) return;
      console.error("fetchChartsData error:", e);
    } finally {
      if (tfSelect) tfSelect.disabled = false;
    }
  }

  const chartsTfSelect = document.getElementById("charts-tf-select");
  if (chartsTfSelect) {
    chartsTfSelect.addEventListener("change", () => {
      fetchChartsData(chartsTfSelect.value);
    });
  }

  const assetStatsTfSelect = document.getElementById("asset-stats-tf-select");
  if (assetStatsTfSelect) {
    assetStatsTfSelect.addEventListener("change", () => {
      fetchStats(assetStatsTfSelect.value);
    });
  }

  async function fetchDailyPnL(tf) {
    const timeframe = tf || (elements.pnlTimeframeSelect ? elements.pnlTimeframeSelect.value : "24h");
    if (elements.dailyPnlLoader) {
      elements.dailyPnlLoader.style.display = "inline";
    }
    try {
      const response = await fetch(`${window.API_BASE}/api/dashboard/daily_pnl?timeframe=${encodeURIComponent(timeframe)}`, {
        headers: { "X-API-Key": apiKey },
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === "success" && elements.dailyPnlTable) {
          elements.dailyPnlTable.innerHTML = "";
          if (!result.data || result.data.length === 0) {
            elements.dailyPnlTable.innerHTML = `<tr><td colspan="4" style="text-align:center; color:var(--text-muted); padding: 1rem;">Сделок за выбранный период не найдено</td></tr>`;
          } else {
            result.data.forEach(item => {
              const tr = document.createElement("tr");
              const pnlColor = item.pnl > 0 ? "#00ff88" : (item.pnl < 0 ? "#ff3366" : "inherit");
              let modeBadge = "";
              if (item.mode === "LIVE") {
                  modeBadge = `<span style="background: rgba(255, 51, 102, 0.2); color: #ff3366; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; margin-left: 5px;">LIVE</span>`;
              } else {
                  modeBadge = `<span style="background: rgba(0, 255, 136, 0.2); color: #00ff88; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; margin-left: 5px;">PAPER</span>`;
              }
              tr.innerHTML = `
                <td>${item.asset} <span style="opacity:0.7;font-size:0.9em;margin-left:0.5rem">${item.strategy}</span>${modeBadge}</td>
                <td>${item.trades}</td>
                <td>${item.win_rate}%</td>
                <td style="color: ${pnlColor}">${item.pnl > 0 ? "+" : ""}${item.pnl.toFixed(2)}</td>
              `;
              elements.dailyPnlTable.appendChild(tr);
            });
          }
        }
      }
    } catch (e) {
      console.error("Error fetching daily PnL", e);
    } finally {
      if (elements.dailyPnlLoader) {
        elements.dailyPnlLoader.style.display = "none";
      }
    }
  }

  // Initial fetch (isolated try-catch for complete fault tolerance)
  try { fetchStats(); } catch (e) { console.error("fetchStats_init_error", e); }
  try { loadSettings(); } catch (e) { console.error("loadSettings_init_error", e); }
  try { loadLogs(); } catch (e) { console.error("loadLogs_init_error", e); }
  try { fetchDailyPnL(); } catch (e) { console.error("fetchDailyPnL_init_error", e); }
  try { if (typeof loadPresetsListUI === 'function') loadPresetsListUI(); } catch (e) { console.error("loadPresetsListUI_init_error", e); }

  // Auto refresh every 5 min for stats, every 30 sec for logs (only if tab is active)
  if (window.statsIntervalId) clearInterval(window.statsIntervalId);
  if (window.logsIntervalId) clearInterval(window.logsIntervalId);
  if (window.dailyPnlIntervalId) clearInterval(window.dailyPnlIntervalId);

  window.statsIntervalId = setInterval(() => {
    if (document.hidden) return;
    fetchStats();
    const assetTf = document.getElementById("asset-stats-tf-select")?.value ?? "all";
    const chartsTf = document.getElementById("charts-tf-select")?.value ?? "all";
    if (chartsTf !== assetTf) {
      fetchChartsData();
    }
  }, 5 * 60 * 1000);
  
  window.logsIntervalId = setInterval(() => {
    if (document.hidden) return;
    loadLogs(currentPage);
  }, 30000);

  window.dailyPnlIntervalId = setInterval(() => {
    if (document.hidden) return;
    fetchDailyPnL();
  }, 60000);

  if (elements.pnlTimeframeSelect) {
    elements.pnlTimeframeSelect.addEventListener("change", () => {
      const tf = elements.pnlTimeframeSelect.value;
      const titleEl = document.getElementById("pnl-analytics-title");
      if (titleEl) {
        const titles = {
          "24h": "PnL за 24 часа (по стратегиям)",
          "7d": "PnL за 7 дней (по стратегиям)",
          "30d": "PnL за 30 дней (по стратегиям)",
          "all": "PnL за все время (по стратегиям)"
        };
        titleEl.textContent = titles[tf] || "PnL (по стратегиям)";
      }
      fetchDailyPnL(tf);
    });
  }

});


// --- Управление пресетами (Config Presets) ---

async function loadPresetsListUI() {
  const container = document.getElementById("presets-container");
  if (!container) return;

  try {
    const res = await fetch(`${window.API_BASE}/api/presets/`);
    if (!res.ok) return;
    const presets = await res.json();
    const activePresetId = localStorage.getItem('active_preset_id');

    if (!presets || presets.length === 0) {
      container.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem; margin:0;">Нет сохраненных пресетов</p>';
      return;
    }

    container.innerHTML = presets.map(p => `
      <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 16px; background: ${activePresetId == p.id ? 'rgba(40,167,69,0.1)' : 'rgba(255,255,255,0.03)'}; border:1px solid ${activePresetId == p.id ? '#28a745' : 'rgba(255,255,255,0.08)'}; border-radius:8px;">
        <div style="display:flex; align-items:center; gap:16px;">
          <span style="font-size:1.2rem;">${p.preset_type === 'manual' ? '📌' : '🏆'}</span>
          <div>
            <div style="font-weight:600; font-size:0.95rem; color:#fff; display:flex; align-items:center; gap:8px;">
              <span>${p.name}</span>
              ${activePresetId == p.id ? '<span style="font-size:0.75rem; background:#28a745; color:white; padding:2px 8px; border-radius:12px; font-weight:600;">Активен</span>' : ''}
              <span style="font-size:0.75rem; padding:2px 8px; border-radius:12px; background:rgba(255,255,255,0.1); color:#cbd5e1; font-weight:500;">${p.param_count} параметров</span>
            </div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">
              Сохранён: ${new Date(p.created_at).toLocaleString()} ${p.capital_at_save ? `· Капитал: $${p.capital_at_save.toFixed(2)}` : ''} ${p.pnl_at_save ? `· PnL: $${p.pnl_at_save.toFixed(2)}` : ''}
            </div>
          </div>
        </div>
        <div style="display:flex; gap:8px;">
          <button type="button" onclick="showPresetDiffUI(${p.id})" style="padding:6px 12px; font-size:0.82rem; border-radius:6px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:#fff; cursor:pointer; font-weight:500;">Diff</button>
          <button type="button" onclick="restorePresetUI(${p.id}, '${p.name}')" style="padding:6px 14px; font-size:0.82rem; border-radius:6px; background:#4f46e5; border:none; color:#fff; cursor:pointer; font-weight:600;">Применить</button>
          <button type="button" onclick="deletePresetUI(${p.id})" style="padding:6px 10px; font-size:0.82rem; border-radius:6px; background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3); color:#ef4444; cursor:pointer;">✕</button>
        </div>
      </div>
    `).join("");
  } catch (err) {
    console.error("failed_to_load_presets_ui", err);
  }
}

async function savePresetFromUI() {
  const nameInput = document.getElementById("preset-name-input");
  const name = nameInput ? nameInput.value.trim() : "";
  if (!name) {
    alert("Введите имя пресета!");
    return;
  }

  try {
    const res = await fetch(`${window.API_BASE}/api/presets/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка сохранения: ${err.detail || 'Не удалось сохранить'}`);
      return;
    }
    const data = await res.json();
    if (nameInput) nameInput.value = "";
    loadPresetsListUI();
  } catch (err) {
    alert(`Ошибка подключения: ${err.message}`);
  }
}

async function restorePresetUI(presetId, presetName) {
  if (!confirm(`Вы действительно хотите применить пресет "${presetName}"?\nТекущие настраиваемые параметры торговли будут заменены.`)) {
    return;
  }

  try {
    const res = await fetch(`${window.API_BASE}/api/presets/${presetId}/restore`, {
      method: "POST"
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`Ошибка применения: ${err.detail || 'Не удалось применить'}`);
      return;
    }
    const data = await res.json();
    let msg = `✅ Пресет "${presetName}" успешно применен!\n`;
    if (data.updated_params && Object.keys(data.updated_params).length > 0) {
      const paramLines = Object.entries(data.updated_params)
        .map(([k, v]) => `• ${k}: ${v}`)
        .join("\n");
      msg += `\nИзменённые параметры (${data.changed_keys}):\n${paramLines}`;
    } else {
      msg += `\nОбновлено параметров: ${data.changed_keys || 0}`;
    }
    alert(msg);
    window.location.reload();
  } catch (err) {
    alert(`Ошибка восстановления: ${err.message}`);
  }
}

async function showPresetDiffUI(presetId) {
  try {
    const res = await fetch(`${window.API_BASE}/api/presets/${presetId}/diff`);
    if (!res.ok) return;
    const data = await res.json();

    if (data.diff_count === 0) {
      alert("Пресет полностью совпадает с текущими настройками!");
      return;
    }

    const lines = Object.entries(data.diff).map(([k, v]) => `${k}:\n  В пресете: ${v.preset}\n  Сейчас:    ${v.current}`).join("\n\n");
    alert(`Различий: ${data.diff_count}\n\n${lines}`);
  } catch (err) {
    alert(`Ошибка загрузки diff: ${err.message}`);
  }
}

async function deletePresetUI(presetId) {
  if (!confirm("Удалить этот пресет?")) return;
  try {
    await fetch(`${window.API_BASE}/api/presets/${presetId}`, { method: "DELETE" });
    loadPresetsListUI();
  } catch (err) {
    console.error("failed_to_delete_preset", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadPresetsListUI();
});

window.showFunnelDiagnostic = function(logId) {
    const funnel = window.funnelLogs && window.funnelLogs[logId];
    if (!funnel) return;

    const fmt = (v) => (v !== null && v !== undefined) ? Number(v).toFixed(4) : "—";

    const entryStatusMap = {
        "FAVORITE_DISABLED": "Пропущен: Вход по тренду выкл.",
        "OUTSIDER_DISABLED": "Пропущен: Торговля на отскок выкл.",
        "INVALID_TIME": "Время вне диапазона",
        "PRICE_OUT_OF_BOUNDS": "Цена вне диапазона",
        "INSUFFICIENT_NET_EDGE": "Недостаточный Edge"
    };
    const mappedEntryStatus = entryStatusMap[funnel.entry_status] || funnel.entry_status || "—";

    const gateLabels = {
        g1_model_loaded:   `G1: Модели (LGBM + LogReg)`,
        g2_price_fetched:  `G2: API котировки (цена: ${fmt(funnel.fresh_price)})`,
        g3_dead_zone:      `G3: Сигнал (статус: ${funnel.direction_status || "—"})`,
        g4_no_flip:        `G4: Консенсус (p_flip: ${fmt(funnel.p_flip)}${funnel.threshold_lower ? ` вне [${fmt(Math.min(funnel.threshold_lower, funnel.threshold_upper))} - ${fmt(Math.max(funnel.threshold_lower, funnel.threshold_upper))}]` : ""})`,
        g5_min_edge:       `G5: Вероятность победы (p_win: ${fmt(funnel.p_candidate_win)})`,
        g6_price_range:    `G6: Цена покупки (ask: ${fmt(funnel.candidate_ask)} в рамках лимитов)`,
        g7_crypto_confirm: `G7: Net Edge (${fmt(funnel.net_edge)} ≥ ${fmt(funnel.min_edge_used)})`,
        g8_combined_vote:  `G8: Итоговый ордер`,
    };

    const gates = funnel.gates || {};
    const gatesHtml = Object.entries(gateLabels).map(([key, label]) => {
        const val = gates[key];
        let color = "#8F9BB3"; let icon = "⚪";
        let extra = "";
        if (val === true)  { color = "#00ff88"; icon = "✅"; }
        if (val === false) { 
            color = "#ff3366"; icon = "❌"; 
            extra = funnel.reason ? `<br><span style="color:#ff3366;font-size:0.8em; margin-left:24px;">➔ ${funnel.reason}</span>` : "";
        }
        return `<li style="color:${color}; padding: 6px 0;">${icon} ${label}${extra}</li>`;
    }).join("");

    const html = `
    <div style="font-family: monospace; display:flex; flex-direction:column; gap:12px; overflow-y:auto; max-height:65vh;">
        <div>
            <div style="font-weight:600; color:#fff; margin-bottom:4px;">Direction Model</div>
            <div>Key: <span style="color:#e2e8f0;">${funnel.direction_model_key || "—"}</span></div>
            <div>Status: <span style="color:${funnel.direction_status === 'READY' ? '#00ff88' : '#ff3366'};">${funnel.direction_status || "—"}</span></div>
            <div>calibrated p_up: <b>${fmt(funnel.direction_p_up)}</b>
                 (UP при raw p_up ≥ ${funnel.direction_threshold_up ?? "—"}; DOWN при raw p_up ≤ ${funnel.direction_threshold_down ?? "—"})</div>
            <div>direction_probability: <b>${fmt(funnel.direction_probability)}</b></div>
            <div>Raw opinion: <b>${funnel.raw_opinion || "—"}</b>
                 (raw p_up=${fmt(funnel.direction_p_up_raw)}, raw p_down=${fmt(funnel.direction_p_down_raw)})</div>
            <div>Actionable signal: <b>${funnel.actionable_signal || funnel.direction_value || "NONE"}</b></div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.1); margin:0;">
        <div>
            <div style="font-weight:600; color:#fff; margin-bottom:4px;">Entry Model</div>
            <div>Key: <span style="color:#e2e8f0;">${funnel.entry_model_key || "—"}</span></div>
            <div>Status: <span style="color:${funnel.entry_status === 'READY' ? '#00ff88' : '#ff3366'};">${mappedEntryStatus}</span></div>
            <div>p_flip: <b>${fmt(funnel.p_flip)}</b>&nbsp;&nbsp;
                 edge: <b>${fmt(funnel.edge)}</b>&nbsp;(min_edge=${funnel.min_edge_used ?? "—"})</div>
            <div>threshold_lower: ${funnel.threshold_lower ?? "—"}&nbsp; threshold_upper: ${funnel.threshold_upper ?? "—"}</div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.1); margin:0;">
        <div>
            <div style="font-weight:600; color:#fff; margin-bottom:6px;">Гейты решения</div>
            <ul style="list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:4px; font-size:0.85rem;">
                ${gatesHtml}
            </ul>
        </div>
        ${(() => {
            const mrf = funnel.mrf_audit;
            if (!mrf) return '';
            const phase = mrf.global_phase || mrf.global_regime || 'UNKNOWN';
            const pol = mrf.policy || {};
            const polStatus = pol.allow ? '<span style="color:#00ff88;">PASS</span>' : '<span style="color:#ff3366;">BLOCK</span>';
            let assetsHtml = '';
            const assets = mrf.assets || {};
            for (const [sym, a] of Object.entries(assets)) {
                const c = a.confidence != null ? (a.confidence * 100).toFixed(0) + '%' : '—';
                const s = a.strength != null ? (a.strength * 100).toFixed(0) + '%' : '—';
                assetsHtml += `<div style="display:flex;gap:8px;align-items:center;font-size:0.82rem;">
                    <span style="color:#e2e8f0;min-width:40px;">${sym}</span>
                    <span class="mrf-regime-badge mrf-regime-${a.phase}" style="font-size:0.75rem;">${a.phase}</span>
                    <span>str:${s}</span><span>conf:${c}</span>
                </div>`;
            }
            const basket = mrf.basket || {};
            const basketInfo = basket.ready_count != null ? `${basket.ready_count}/${basket.total_count}` : '—';
            const reasonCodes = (mrf.reason_codes && mrf.reason_codes.length > 0)
                ? `<div style="margin-top:6px;font-size:0.78rem;color:#ffb020;">reasons: ${mrf.reason_codes.join(', ')}</div>` : '';
            return `
        <hr style="border-color:rgba(255,255,255,0.1); margin:0;">
        <div>
            <div style="font-weight:600; color:#fff; margin-bottom:4px;">Market Regime Filter (${mrf.mode})</div>
            <div>Phase: <span class="mrf-regime-badge mrf-regime-${phase}">${phase}</span>
                 &nbsp;confidence: <b>${mrf.global_confidence != null ? (mrf.global_confidence * 100).toFixed(1) + '%' : '—'}</b>
                 &nbsp;strength: <b>${mrf.global_strength != null ? (mrf.global_strength * 100).toFixed(1) + '%' : '—'}</b></div>
            <div>Policy: ${polStatus} &nbsp; multiplier: <b>${pol.multiplier != null ? 'x' + Number(pol.multiplier).toFixed(2) : '—'}</b>
                 &nbsp;reason: <span style="color:#e2e8f0;">${pol.reason || '—'}</span></div>
            <div>Basket: ${basketInfo} ready
                 ${basket.median_ret_24h != null ? ` &nbsp;ret24h: ${basket.median_ret_24h >= 0 ? '+' : ''}${(basket.median_ret_24h * 100).toFixed(2)}%` : ''}
                 ${basket.efficiency != null ? ` &nbsp;eff: ${Number(basket.efficiency).toFixed(2)}` : ''}
                 ${basket.breadth_up_24h != null ? ` &nbsp;breadth: ${(basket.breadth_up_24h * 100).toFixed(0)}%` : ''}
            </div>
            ${assetsHtml ? `<div style="margin-top:6px;font-size:0.85rem;"><div style="color:var(--text-muted);margin-bottom:2px;">Assets:</div>${assetsHtml}</div>` : ''}
            ${reasonCodes}
        </div>`;
        })()}
        ${funnel.fallback_reason ? `<div style="padding:10px; background:rgba(255,51,102,0.12); color:#ff3366; border-radius:6px; font-size:0.85rem;"><b>Fallback reason:</b> ${funnel.fallback_reason}</div>` : ""}
    </div>`;

    document.getElementById("funnel-diagnostic-content").innerHTML = html;
    document.getElementById("funnel-diagnostic-modal").style.display = "flex";
};
