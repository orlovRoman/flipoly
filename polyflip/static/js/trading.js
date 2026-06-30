document.addEventListener("DOMContentLoaded", () => {
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
    avgWinPrice: document.getElementById("avg-win-price"),
    avgWinProb: document.getElementById("avg-win-prob"),
    avgLossPrice: document.getElementById("avg-loss-price"),
    avgLossProb: document.getElementById("avg-loss-prob"),
    refreshBtn: document.getElementById("btn-refresh-trading"),
  };

  let pnlChart = null;
  let wlChart = null;

  let currentPage = 1;
  let totalPages = 1;
  const PAGE_SIZE = 25;

  async function fetchStats() {
    try {
      const response = await fetch(`${window.API_BASE}/api/trading/stats`, {
        headers: { "X-API-Key": apiKey },
      });
      if (response.status === 401) {
        alert(
          "Неверный API ключ. Введите его на вкладке 'Настройки' в основном дашборде.",
        );
        return;
      }
      const data = await response.json();
      updateUI(data);
      loadActiveModels();
    } catch (error) {
      console.error("Ошибка при загрузке данных:", error);
    }
  }

  function updateUI(data) {
    // Update KPIs
    elements.capital.textContent = `${data.capital.toFixed(2)} USDC`;
    elements.pnl.textContent = `${data.overall_pnl > 0 ? "+" : ""}${data.overall_pnl.toFixed(2)} USDC`;
    elements.pnl.style.color = data.overall_pnl >= 0 ? "#00ff88" : "#ff3366";

    elements.winrate.textContent = `${data.winrate}%`;
    elements.wl.textContent = `${data.wins_vs_losses.wins} / ${data.wins_vs_losses.losses}`;

    // Update Asset Table
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

    // Update Parameters
    elements.avgWinPrice.textContent = `$${data.parameters.avg_win_price}`;
    elements.avgWinProb.textContent = `${(data.parameters.avg_win_prob * 100).toFixed(1)}%`;
    elements.avgLossPrice.textContent = `$${data.parameters.avg_loss_price}`;
    elements.avgLossProb.textContent = `${(data.parameters.avg_loss_prob * 100).toFixed(1)}%`;

    updateCharts(data.daily_pnl);
  }

  function updateCharts(dailyData) {
    const sortedDates = Object.keys(dailyData).sort();

    let cumulativePnl = 0;
    const pnlData = [];
    const winData = [];
    const lossData = [];

    for (const date of sortedDates) {
      cumulativePnl += dailyData[date].pnl;
      pnlData.push(cumulativePnl);
      winData.push(dailyData[date].wins);
      lossData.push(dailyData[date].losses);
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
        winData.push(null);
        lossData.push(null);
      }
    }

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
        options: commonOptions,
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
              label: "Выигрыши",
              data: winData,
              backgroundColor: "#2ecc71",
              maxBarThickness: 30,
            },
            {
              label: "Проигрыши",
              data: lossData,
              backgroundColor: "#e74c3c",
              maxBarThickness: 30,
            },
          ],
        },
        options: {
          ...commonOptions,
          scales: {
            x: {
              stacked: true,
              offset: true,
              ticks: { color: "rgba(255, 255, 255, 0.7)" },
              grid: { color: "rgba(255, 255, 255, 0.1)" },
            },
            y: {
              stacked: true,
              ticks: { color: "rgba(255, 255, 255, 0.7)" },
              grid: { color: "rgba(255, 255, 255, 0.1)" },
            }
          }
        },
      },
    );
  }

  if (elements.refreshBtn) {
    elements.refreshBtn.addEventListener("click", fetchStats);
  }

  // ----------------------------------------------------
  // Trading Settings Logic
  // ----------------------------------------------------
  let currentFlipThreshold = 0.70;

  const settingsElements = {
    apiKeyInput: document.getElementById("API_KEY"),
    minTimeLeft: document.getElementById("TRADE_MIN_TIME_LEFT_SEC"),
    maxTimeLeft: document.getElementById("TRADE_MAX_TIME_LEFT_SEC"),
    betSizingMode: document.getElementById("BET_SIZING_MODE"),
    maxBetSizeGroup: document.getElementById("max-bet-size-group"),
    maxBetSize: document.getElementById("MAX_BET_SIZE_USDC"),
    betSize: document.getElementById("TRADE_BET_SIZE_USDC"),
    noFlipThreshold: document.getElementById("TRADE_NO_FLIP_THRESHOLD"),
    deadZoneWidth: document.getElementById("DEAD_ZONE_WIDTH"),
    dailyLossLimit: document.getElementById("DAILY_LOSS_LIMIT_USDC"),
    tradingEnabled: document.getElementById("TRADING_ENABLED"),
    initialCapital: document.getElementById("INITIAL_CAPITAL"),
    minPrice: document.getElementById("TRADE_MIN_PRICE"),
    maxPrice: document.getElementById("TRADE_MAX_PRICE"),
    tradingModeRadios: document.querySelectorAll('input[name="trading_mode"]'),
    favoriteModeSettings: document.getElementById('favorite-mode-settings'),
    favoriteEntrySecInput: document.getElementById('FAVORITE_MODE_ENTRY_SEC'),
    tradingModeBadge: document.getElementById('trading-mode-badge'),
    pollIntervalInput: document.getElementById("LIVE_POLL_INTERVAL_SECONDS"),
    minEdge: document.getElementById("MIN_EDGE"),
    maxEdge: document.getElementById("MAX_EDGE"),
    favoriteThreshold: document.getElementById("FAVORITE_THRESHOLD"),
    tradeOnFlip: document.getElementById("TRADE_ON_FLIP"),
    flipThreshold: document.getElementById("FLIP_THRESHOLD"),
    noMaxPrice: document.getElementById("NO_MAX_PRICE"),
    noMinEdge: document.getElementById("NO_MIN_EDGE"),
    autoDeadZone: document.getElementById("AUTO_DEAD_ZONE"),
    autoDeadZoneWidth: document.getElementById("AUTO_DEAD_ZONE_WIDTH"),
  };

  function updateDeadZoneInfo() {
    if (!settingsElements.autoDeadZone) return;
    const isAuto = settingsElements.autoDeadZone.checked;
    const autoWidth = parseFloat(settingsElements.autoDeadZoneWidth.value) / 100 || 0.10;
    const manualWidth = parseFloat(settingsElements.deadZoneWidth.value) / 100 || 0.15;
    const noFlip = parseFloat(settingsElements.noFlipThreshold.value) / 100 || 0.45;
    
    // Toggle fields
    if (isAuto) {
      document.getElementById("auto-dead-zone-width-group").style.display = "block";
      document.getElementById("manual-dead-zone-width-group").style.display = "none";
      
      const baseFlip = currentFlipThreshold;
      const lower = Math.max(0, baseFlip - autoWidth / 2);
      const upper = Math.min(1, baseFlip + autoWidth / 2);
      document.getElementById("dead-zone-range-text").textContent = 
        `${Math.round(lower * 100)}% – ${Math.round(upper * 100)}% (Авто: YES < ${Math.round(lower * 100)}%, NO > ${Math.round(upper * 100)}%)`;
    } else {
      document.getElementById("auto-dead-zone-width-group").style.display = "none";
      document.getElementById("manual-dead-zone-width-group").style.display = "block";
      
      const lower = noFlip;
      const upper = noFlip + manualWidth;
      document.getElementById("dead-zone-range-text").textContent = 
        `${Math.round(lower * 100)}% – ${Math.round(upper * 100)}% (Ручной: YES < ${Math.round(lower * 100)}%, NO > ${Math.round(upper * 100)}%)`;
    }
  }

  if (settingsElements.autoDeadZone) {
    settingsElements.autoDeadZone.addEventListener("change", updateDeadZoneInfo);
  }
  if (settingsElements.autoDeadZoneWidth) {
    settingsElements.autoDeadZoneWidth.addEventListener("input", updateDeadZoneInfo);
  }
  if (settingsElements.deadZoneWidth) {
    settingsElements.deadZoneWidth.addEventListener("input", updateDeadZoneInfo);
  }
  if (settingsElements.noFlipThreshold) {
    settingsElements.noFlipThreshold.addEventListener("input", updateDeadZoneInfo);
  }
  if (settingsElements.flipThreshold) {
    settingsElements.flipThreshold.addEventListener("input", () => {
      currentFlipThreshold = parseFloat(settingsElements.flipThreshold.value) / 100 || 0.70;
      updateDeadZoneInfo();
    });
  }
  
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

      // Получаем dead_zone из инпута или API
      const deadZoneInput = document.getElementById("DEAD_ZONE_WIDTH");
      const deadZoneVal = deadZoneInput ? parseFloat(deadZoneInput.value) / 100 : g.dead_zone;
      const deadZonePct = Math.round(deadZoneVal * 100);

      // Текущее значение no_flip берем из инпута, если он загружен, иначе из API
      const noFlipInput = document.getElementById("TRADE_NO_FLIP_THRESHOLD");
      const currentNoFlipVal = noFlipInput ? parseFloat(noFlipInput.value) / 100 : g.current_no_flip;
      const currentNoFlipPct = Math.round(currentNoFlipVal * 100);
      
      const firstAsset = Object.keys(data.per_asset)[0];
      const recPct = firstAsset ? Math.round(data.per_asset[firstAsset].recommended_no_flip * 100) : Math.round(g.current_no_flip * 100);

      // Подсказка под полем no_flip
      const hint = document.getElementById("no-flip-hint");
      if (hint) {
        hint.innerHTML = `
            Текущее значение: <strong>${currentNoFlipPct}%</strong>
            &nbsp;(Ширина мёртвой зоны: <strong>${deadZonePct}%</strong>).
            ${firstAsset ? `Рекомендовано для ${firstAsset}: <strong style="color:#00ff88">${recPct}%</strong>` : ""}
        `;
      }

      // Кнопка "Применить рекомендованное"
      const btn = document.getElementById("btn-apply-recommended-no-flip");
      if (btn) {
        if (firstAsset) {
          btn.style.display = "inline-block";
          btn.onclick = () => {
            if (noFlipInput) {
              noFlipInput.value = recPct;
              loadRecommendedThresholds();
              if (hint) {
                hint.innerHTML += ' &nbsp;<span style="color:#00ff88">↑ применено</span>';
              }
            }
          };
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
    const isFavorite = mode === 'favorite';
    if (settingsElements.favoriteModeSettings) {
      settingsElements.favoriteModeSettings.style.display = isFavorite ? 'block' : 'none';
    }
    if (settingsElements.tradingModeBadge) {
      settingsElements.tradingModeBadge.textContent = isFavorite 
        ? '⚡ Режим: Pure Favorite' 
        : '🤖 Режим: ML';
      settingsElements.tradingModeBadge.className = `mode-badge mode-${mode}`;
    }
  }

  if (settingsElements.tradingModeRadios) {
    settingsElements.tradingModeRadios.forEach(radio => {
      radio.addEventListener('change', (e) => onTradingModeChange(e.target.value));
    });
  }

  async function loadSettings() {
    try {
      const res = await fetch(window.API_BASE + "/api/settings", {
        headers: { "X-API-Key": apiKey },
      });
      const data = await res.json();

      if (settingsElements.minTimeLeft && data.TRADE_MIN_TIME_LEFT_SEC)
        settingsElements.minTimeLeft.value = data.TRADE_MIN_TIME_LEFT_SEC;
      if (settingsElements.maxTimeLeft && data.TRADE_MAX_TIME_LEFT_SEC)
        settingsElements.maxTimeLeft.value = data.TRADE_MAX_TIME_LEFT_SEC;
      if (settingsElements.betSizingMode && data.BET_SIZING_MODE)
        settingsElements.betSizingMode.value = data.BET_SIZING_MODE;
      if (settingsElements.maxBetSize && data.MAX_BET_SIZE_USDC)
        settingsElements.maxBetSize.value = data.MAX_BET_SIZE_USDC;
      if (settingsElements.betSize && data.TRADE_BET_SIZE_USDC)
        settingsElements.betSize.value = data.TRADE_BET_SIZE_USDC;
      
      if (settingsElements.betSizingMode) {
        settingsElements.betSizingMode.dispatchEvent(new Event("change"));
      }
      if (settingsElements.noFlipThreshold && data.TRADE_NO_FLIP_THRESHOLD) {
        let val = parseFloat(data.TRADE_NO_FLIP_THRESHOLD);
        if (val > 1) val /= 100;
        settingsElements.noFlipThreshold.value = Math.round(val * 100);
      }
      if (settingsElements.deadZoneWidth && data.DEAD_ZONE_WIDTH) {
        let val = parseFloat(data.DEAD_ZONE_WIDTH);
        settingsElements.deadZoneWidth.value = Math.round(val * 100);
      }

      if (settingsElements.dailyLossLimit && data.DAILY_LOSS_LIMIT_USDC !== undefined) {
        settingsElements.dailyLossLimit.value = data.DAILY_LOSS_LIMIT_USDC;
      }
      if (settingsElements.tradingEnabled && data.TRADING_ENABLED) {
        settingsElements.tradingEnabled.checked =
          data.TRADING_ENABLED === "true";
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

      if (settingsElements.minEdge && data.MIN_EDGE !== undefined) {
        let val = parseFloat(data.MIN_EDGE);
        currentMinEdge = val;
        settingsElements.minEdge.value = (val * 100).toFixed(1);
      }
      if (settingsElements.maxEdge && data.MAX_EDGE !== undefined) {
        let val = parseFloat(data.MAX_EDGE);
        settingsElements.maxEdge.value = (val * 100).toFixed(1);
      }
      if (settingsElements.favoriteThreshold && data.FAVORITE_THRESHOLD !== undefined) {
        let val = parseFloat(data.FAVORITE_THRESHOLD);
        settingsElements.favoriteThreshold.value = val;
      }
      if (settingsElements.tradeOnFlip && data.TRADE_ON_FLIP) {
        settingsElements.tradeOnFlip.checked = data.TRADE_ON_FLIP === "true";
        updateOutsiderStrategyStatus();
      }
      if (settingsElements.flipThreshold && data.FLIP_THRESHOLD !== undefined) {
        let val = parseFloat(data.FLIP_THRESHOLD);
        settingsElements.flipThreshold.value = Math.round(val * 100);
        currentFlipThreshold = val;
      }
      if (settingsElements.noMaxPrice && data.NO_MAX_PRICE !== undefined) {
        settingsElements.noMaxPrice.value = data.NO_MAX_PRICE;
      }
      if (settingsElements.noMinEdge && data.NO_MIN_EDGE !== undefined) {
        let val = parseFloat(data.NO_MIN_EDGE);
        settingsElements.noMinEdge.value = (val * 100).toFixed(1);
      }
      if (settingsElements.autoDeadZone && data.AUTO_DEAD_ZONE) {
        settingsElements.autoDeadZone.checked = data.AUTO_DEAD_ZONE === "true";
      }
      if (settingsElements.autoDeadZoneWidth && data.AUTO_DEAD_ZONE_WIDTH !== undefined) {
        let val = parseFloat(data.AUTO_DEAD_ZONE_WIDTH);
        settingsElements.autoDeadZoneWidth.value = Math.round(val * 100);
      }
      updateDeadZoneInfo();
      if (data.TRADING_MODE) {
        const mode = data.TRADING_MODE;
        const radio = document.querySelector(`input[name="trading_mode"][value="${mode}"]`);
        if (radio) radio.checked = true;
        onTradingModeChange(mode);
      }
      if (settingsElements.favoriteEntrySecInput && data.FAVORITE_MODE_ENTRY_SEC) {
        settingsElements.favoriteEntrySecInput.value = data.FAVORITE_MODE_ENTRY_SEC;
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

      await loadRecommendedThresholds();
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

      const settingsToSave = {};
      if (settingsElements.minTimeLeft) settingsToSave.TRADE_MIN_TIME_LEFT_SEC = settingsElements.minTimeLeft.value;
      if (settingsElements.maxTimeLeft) settingsToSave.TRADE_MAX_TIME_LEFT_SEC = settingsElements.maxTimeLeft.value;
      if (settingsElements.betSizingMode) settingsToSave.BET_SIZING_MODE = settingsElements.betSizingMode.value;
      if (settingsElements.maxBetSize) settingsToSave.MAX_BET_SIZE_USDC = settingsElements.maxBetSize.value;
      if (settingsElements.betSize) settingsToSave.TRADE_BET_SIZE_USDC = settingsElements.betSize.value;
      if (settingsElements.noFlipThreshold) settingsToSave.TRADE_NO_FLIP_THRESHOLD = parseFloat(settingsElements.noFlipThreshold.value) / 100;
      if (settingsElements.deadZoneWidth) settingsToSave.DEAD_ZONE_WIDTH = parseFloat(settingsElements.deadZoneWidth.value) / 100;

      if (settingsElements.dailyLossLimit) settingsToSave.DAILY_LOSS_LIMIT_USDC = settingsElements.dailyLossLimit.value;
      if (settingsElements.tradingEnabled) settingsToSave.TRADING_ENABLED = settingsElements.tradingEnabled.checked ? "true" : "false";
      if (settingsElements.initialCapital) settingsToSave.INITIAL_CAPITAL = settingsElements.initialCapital.value;
      if (settingsElements.minPrice) settingsToSave.TRADE_MIN_PRICE = settingsElements.minPrice.value;
      if (settingsElements.maxPrice) settingsToSave.TRADE_MAX_PRICE = settingsElements.maxPrice.value;
      const activeMode = document.querySelector('input[name="trading_mode"]:checked')?.value || 'ml';
      settingsToSave.TRADING_MODE = activeMode;

      if (settingsElements.favoriteEntrySecInput) {
        settingsToSave.FAVORITE_MODE_ENTRY_SEC = settingsElements.favoriteEntrySecInput.value;
      }
      if (settingsElements.pollIntervalInput) settingsToSave.LIVE_POLL_INTERVAL_SECONDS = settingsElements.pollIntervalInput.value;
      if (settingsElements.minEdge) settingsToSave.MIN_EDGE = parseFloat(settingsElements.minEdge.value) / 100;
      if (settingsElements.maxEdge) settingsToSave.MAX_EDGE = parseFloat(settingsElements.maxEdge.value) / 100;
      if (settingsElements.favoriteThreshold) settingsToSave.FAVORITE_THRESHOLD = parseFloat(settingsElements.favoriteThreshold.value);
      if (settingsElements.tradeOnFlip) settingsToSave.TRADE_ON_FLIP = settingsElements.tradeOnFlip.checked ? "true" : "false";
      if (settingsElements.flipThreshold) settingsToSave.FLIP_THRESHOLD = parseFloat(settingsElements.flipThreshold.value) / 100;
      if (settingsElements.noMaxPrice) settingsToSave.NO_MAX_PRICE = parseFloat(settingsElements.noMaxPrice.value);
      if (settingsElements.noMinEdge) settingsToSave.NO_MIN_EDGE = parseFloat(settingsElements.noMinEdge.value) / 100;
      if (settingsElements.autoDeadZone) settingsToSave.AUTO_DEAD_ZONE = settingsElements.autoDeadZone.checked ? "true" : "false";
      if (settingsElements.autoDeadZoneWidth) settingsToSave.AUTO_DEAD_ZONE_WIDTH = parseFloat(settingsElements.autoDeadZoneWidth.value) / 100;
      settingsToSave.TRADE_ASSETS = tradeAssets;

      try {
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
          '<tr><td colspan="13" style="text-align:center; padding: 1rem;">Нет событий</td></tr>';
        renderPagination(currentPage, totalPages, data.total || 0);
        return;
      }

      const rows = [];

      data.items.forEach((log) => {
        const displayTime = log.updated_at ? log.updated_at : log.created_at;
        const timeStr = new Date(displayTime).toLocaleTimeString();
        const flipColor = log.predicted_flip_prob > 0.5 ? "#00ff88" : "#ff3366";
        let statusColor = "#8F9BB3"; // SKIPPED
        if (log.status === "SUCCESS") statusColor = "#00ff88";
        if (log.status === "FAILED") statusColor = "#ff3366";

        const reasonHtml =
          log.status === "SKIPPED"
            ? `<span style="color: #ffb020">${escapeHtml(log.error_msg)}</span>`
            : escapeHtml(log.error_msg || "-");
        const isPureFav = log.active_features === "PURE_FAVORITE";
        const modelStr = log.model_version 
          ? `v${log.model_version}` 
          : (isPureFav ? "PureFav" : (log.status === "SUCCESS" ? "legacy" : "-"));

        let pnlText = "-";
        let pnlColor = "var(--text-main)";
        if (log.status === "SUCCESS" && log.pnl !== null && log.pnl !== undefined) {
          const pnlVal = parseFloat(log.pnl);
          pnlText = (pnlVal >= 0 ? "+" : "") + pnlVal.toFixed(2) + " USDC";
          pnlColor = pnlVal >= 0 ? "#00ff88" : "#ff3366";
        }



        let betTypeHtml = `<span style="color: #00ff88; font-weight: 500;">Ставка по тренду</span>`;
        const isOutsider = (log.active_features && log.active_features.includes("outsider")) ||
                           (log.active_features && log.active_features.includes("OUTSIDER")) ||
                           (log.active_features &&
                            !log.active_features.includes("PURE_FAVORITE") &&
                            !log.active_features.includes("trend") &&
                            log.predicted_flip_prob !== null &&
                            log.predicted_flip_prob !== undefined &&
                            parseFloat(log.predicted_flip_prob) >= 0.5) ||
                           (log.error_msg && (log.error_msg.includes("TRADE_ON_FLIP") || log.error_msg.includes("Ожидается флип") || log.error_msg.includes("outsider")));
        if (isOutsider) {
          betTypeHtml = `<span style="color: #ffb020; font-weight: 500;">Ставка против тренда (Аутсайдер)</span>`;
        }

        let outcomeBadge = "";
        if (log.outcome_bought === "YES") {
          outcomeBadge = `<span style="color: #00ff88; font-size: 0.8em; margin-right: 6px; padding: 2px 4px; background: rgba(0,255,136,0.1); border-radius: 4px;">UP</span>`;
        } else if (log.outcome_bought === "NO") {
          outcomeBadge = `<span style="color: #ff3366; font-size: 0.8em; margin-right: 6px; padding: 2px 4px; background: rgba(255,51,102,0.1); border-radius: 4px;">DOWN</span>`;
        }

        const betText = log.amount_usdc > 0 ? `${outcomeBadge}$${parseFloat(log.amount_usdc).toFixed(2)}` : "-";

        const logDate = new Date(displayTime);
        const minutes = (logDate.getUTCMinutes() % 15) + 1;
        const intervalOffsetStr = `${String(minutes).padStart(2, '0')}:00`;

        rows.push(`
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 8px; color: var(--text-muted);">${intervalOffsetStr}</td>
                        <td style="padding: 8px; color: var(--text-muted);">${timeStr}</td>
                        <td style="padding: 8px;"><a href="#" class="market-link" data-market-id="${log.market_id}" data-asset="${escapeHtml(log.asset)}" style="color: var(--text-main); text-decoration: underline; cursor: pointer;">${escapeHtml(log.question)}</a></td>
                        <td style="padding: 8px; font-weight: bold;">${escapeHtml(log.asset)}</td>
                        <td style="padding: 8px; color: var(--poly-blue);">${modelStr}</td>
                        <td style="padding: 8px; color: ${statusColor};">${log.status}</td>
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
                        <td style="padding: 8px;">${reasonHtml}</td>
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

  async function loadActiveModels() {
    try {
      const res = await fetch(window.API_BASE + "/api/dashboard/status", {
        headers: { "X-API-Key": apiKey },
      });
      const data = await res.json();
      if (data.active_models) {
        const badge = document.getElementById("active-models-badge");
        if (badge) {
          const modelsText = Object.entries(data.active_models)
            .map(([asset, v]) => `${asset} v${v}`)
            .join(", ");
          badge.textContent = modelsText
            ? `[Активные модели: ${modelsText}]`
            : "[Нет активных моделей]";
        }
      }
    } catch (e) {
      console.error("Failed to load active models", e);
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

  const flipInput = document.getElementById("TRADE_FLIP_THRESHOLD");
  if (flipInput) {
    flipInput.addEventListener("input", () => {
      const val = parseFloat(flipInput.value) / 100;
      if (!isNaN(val)) loadRecommendedThresholds(val);
    });
  }

  const noFlipInput = document.getElementById("TRADE_NO_FLIP_THRESHOLD");
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

  // Initial fetch
  fetchStats();
  loadSettings();
  loadLogs();
  loadActiveModels();

  // Auto refresh every 5 min for stats, every 30 sec for logs (only if tab is active)
  if (window.statsIntervalId) clearInterval(window.statsIntervalId);
  if (window.logsIntervalId) clearInterval(window.logsIntervalId);

  window.statsIntervalId = setInterval(() => {
    if (document.hidden) return;
    fetchStats();
  }, 5 * 60 * 1000);
  
  window.logsIntervalId = setInterval(() => {
    if (document.hidden) return;
    loadLogs(currentPage);
  }, 30000);
});
