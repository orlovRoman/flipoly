document.addEventListener("DOMContentLoaded", () => {
  const ASSET_COLORS = {
    BTC: "#0072F5",
    ETH: "#00D395",
    SOL: "#9945FF",
    XRP: "#346AA9",
    DOGE: "#C2A633"
  };
  function getAssetColor(asset) {
    return ASSET_COLORS[asset.toUpperCase()] || "#8F9BB3";
  }

  // === Tab Switching Logic ===
  const navItems = document.querySelectorAll(".nav-item");
  const tabContents = document.querySelectorAll(".tab-content");

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      // Remove active class from all
      navItems.forEach((nav) => nav.classList.remove("active"));
      tabContents.forEach((tab) => tab.classList.remove("active"));

      // Add active class to clicked
      item.classList.add("active");
      const targetId = item.getAttribute("data-tab");
      if (targetId) {
        document.getElementById(targetId).classList.add("active");
      } else if (item.getAttribute("data-href")) {
        window.location.href = item.getAttribute("data-href");
      }
    });
  });

  // === API Key Management ===
  // Simple helper to get headers
  function getHeaders() {
    let apiKey = "test-key";
    try {
      apiKey = localStorage.getItem("polyflip_api_key") || "test-key";
    } catch (e) {
      console.warn("localStorage unavailable, using default key");
    }
    return {
      "Content-Type": "application/json",
      "X-API-Key": apiKey,
    };
  }

  // === URL Tab Handling ===
  const urlParams = new URLSearchParams(window.location.search);
  const initialTab = urlParams.get("tab");
  if (initialTab) {
    const targetNav = document.querySelector(
      `.nav-item[data-tab="${initialTab}"]`,
    );
    if (targetNav) {
      targetNav.click();
    }
  }

  // === Data Fetching & Rendering ===

  // Simple helper to prevent XSS in innerHTML
  const escapeHtml = (unsafe) => {
    return String(unsafe)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  };

  const translateFeatures = (featuresStr) => {
    if (!featuresStr) return "-";
    const mapping = {
      "time_left_min": "Время до конца",
      "mid_price": "Текущая цена",
      "spread": "Спред",
      "volume_5min": "Объем (5м)",
      "price_velocity": "Скорость цены",
      "hour_of_day": "Час суток"
    };
    return featuresStr.split(",")
      .map(f => mapping[f.trim()] || f.trim())
      .join(", ");
  };

  // 1. Fetch Summary
  async function loadSummary() {
    try {
      const res = await fetch(window.API_BASE + "/api/analytics/summary");
      const data = await res.json();

      document.getElementById("stat-markets").innerText =
        data.total_resolved_markets || 0;
      document.getElementById("stat-flips").innerText =
        (data.flip_percentage || 0) + "%";

      // Заполняем карточки активных моделей для всех активов
      document.querySelectorAll("[id^='stat-model-']").forEach((el) => {
        const asset = el.id.replace("stat-model-", "").toUpperCase();
        const model = data.active_models[asset] || data.active_models[asset.toLowerCase()];
        if (model) {
          const version = model.version !== undefined ? model.version : model;
          const accuracy = model.accuracy !== undefined ? ` (Acc: ${(model.accuracy * 100).toFixed(1)}%)` : '';
          el.innerText = `v${version}${accuracy}`;
        } else {
          el.innerText = "Нет модели";
        }
      });

      // Рендерим графики точности для всех доступных в истории активов
      if (data.model_history) {
        Object.keys(data.model_history).forEach((asset) => {
          renderAccuracyChart(data.model_history[asset], asset);
        });
      }
    } catch (e) {
      console.error("Failed to load summary", e);
    }
  }

  function renderAccuracyChart(historyData, asset) {
    const canvasId = `chart-model-accuracy-${asset.toLowerCase()}`;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let existingChart = Chart.getChart(canvasId);
    if (existingChart) existingChart.destroy();

    const labels = historyData.map((h) => `v${h.version}`);
    const data = historyData.map((h) => h.accuracy * 100);

    const color = getAssetColor(asset);
    createChart(
      ctx,
      labels,
      data,
      `Точность модели ${asset}`,
      color,
      "Версия модели",
    );
  }

  // 2. Fetch Probabilities and Render Charts
  let chartInstances = {};
  let chartDataStore = {};
  let chartsLoaded = false;

  async function loadCharts(force = false) {
    if (!force && !chartsLoaded) {
      return;
    }
    const btnPlaceholder = document.getElementById("btn-load-charts-placeholder");
    const btnLoad = document.getElementById("btn-load-charts");
    
    if (btnPlaceholder) {
      btnPlaceholder.innerText = "Загрузка...";
      btnPlaceholder.disabled = true;
    }
    if (btnLoad) {
      btnLoad.innerText = "Загрузка...";
      btnLoad.disabled = true;
    }

    try {
      const res = await fetch(window.API_BASE + "/api/analytics/probabilities");
      chartDataStore = await res.json();
      chartsLoaded = true;
      
      const placeholderEl = document.getElementById("charts-placeholder");
      if (placeholderEl) placeholderEl.style.display = "none";
      
      const selectorEl = document.getElementById("asset-selector");
      if (selectorEl) selectorEl.style.display = "block";
      
      if (btnLoad) {
        btnLoad.style.display = "block";
        btnLoad.innerText = "Обновить графики";
        btnLoad.disabled = false;
      }
      
      const wrapperEl = document.getElementById("flip-charts-wrapper");
      if (wrapperEl) wrapperEl.style.display = "grid";

      renderSelectedChart();
    } catch (e) {
      console.error("Failed to load charts", e);
      if (btnPlaceholder) {
        btnPlaceholder.innerText = "Ошибка. Повторить?";
        btnPlaceholder.disabled = false;
      }
      if (btnLoad) {
        btnLoad.innerText = "Ошибка. Повторить?";
        btnLoad.disabled = false;
      }
    }
  }

  // Hook up event listeners for chart loading
  const btnPlaceholder = document.getElementById("btn-load-charts-placeholder");
  if (btnPlaceholder) {
    btnPlaceholder.addEventListener("click", () => loadCharts(true));
  }
  const btnLoad = document.getElementById("btn-load-charts");
  if (btnLoad) {
    btnLoad.addEventListener("click", () => loadCharts(true));
  }

  function renderSelectedChart() {
    const selectorEl = document.getElementById("asset-selector");
    if (!selectorEl) return;
    const selectedAsset = selectorEl.value;
    const assetData = chartDataStore[selectedAsset] || {};
    const color = getAssetColor(selectedAsset);

    const chartConfigs = [
      {
        id: "chart-time",
        key: "time_left_min",
        xTitle: "Оставшееся время (минуты)",
      },
      { id: "chart-price", key: "mid_price", xTitle: "Цена токена (USD)" },
      { id: "chart-spread", key: "spread", xTitle: "Спред (USD)" },
      {
        id: "chart-volume",
        key: "volume_5min",
        xTitle: "Объем торгов за 5 мин (USDC)",
      },
      {
        id: "chart-velocity",
        key: "price_velocity",
        xTitle: "Скорость изменения цены за 5 мин",
      },
      { id: "chart-hour", key: "hour_of_day", xTitle: "Время суток (UTC)" },
    ];

    chartConfigs.forEach((cfg) => {
      const featureData = assetData[cfg.key] || {
        labels: [],
        probabilities: [],
      };
      const labels = featureData.labels;
      const points = featureData.probabilities.map((p) => p * 100);

      const canvas = document.getElementById(cfg.id);
      if (!canvas) return; // safety check
      const ctx = canvas.getContext("2d");

      let existingChart = Chart.getChart(cfg.id);
      if (existingChart) existingChart.destroy();

      chartInstances[cfg.id] = createChart(
        ctx,
        labels,
        points,
        `${selectedAsset} Флип %`,
        color,
        cfg.xTitle,
      );
    });
  }

  const assetSelector = document.getElementById("asset-selector");
  if (assetSelector) {
    assetSelector.addEventListener("change", renderSelectedChart);
  }

  function createChart(ctx, labels, data, labelText, color, xTitle) {
    return new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: labelText,
            data: data,
            borderColor: color,
            backgroundColor: color + "33", // 20% opacity
            borderWidth: 3,
            fill: true,
            tension: 0.4, // Smooth curves
            pointBackgroundColor: color,
            pointRadius: 4,
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (context) {
                return context.parsed.y.toFixed(1) + "%";
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: {
              color: "#8F9BB3",
              callback: function (value) {
                return value + "%";
              },
            },
          },
          x: {
            grid: { display: false },
            ticks: { color: "#8F9BB3", maxRotation: 45, minRotation: 45 },
            title: {
              display: true,
              text: xTitle,
              color: "#8F9BB3",
              font: { size: 13 },
            },
          },
        },
      },
    });
  }

  // 3. Fetch and Populate Settings
  async function loadSettings() {
    try {
      const res = await fetch(window.API_BASE + "/api/settings", {
        headers: getHeaders(),
      });
      const data = await res.json();

      if (data.ACTIVE_FEATURES) {
        const active = data.ACTIVE_FEATURES.split(",");
        document
          .querySelectorAll('#ml-features input[type="checkbox"]')
          .forEach((cb) => {
            cb.checked = active.includes(cb.value);
          });
      }
    } catch (e) {
      console.error("Failed to load settings", e);
    }
  }



  // === Button Handlers ===

  // Train Model
  document.querySelectorAll(".btn-train-asset").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      requestNotificationPermission();
      const asset = e.target.getAttribute("data-asset");
      const originalText = e.target.innerText;
      
      e.target.innerText = `Обучение ${asset}...`;
      e.target.disabled = true;

      try {
        const res = await fetch(window.API_BASE + `/api/analytics/train/${asset}`, {
          method: "POST",
          headers: getHeaders(),
        });
        if (res.ok) {
          alert(`Задание на обучение ${asset} отправлено в фон!`);
          pollTrainingStatus(e.target, asset, originalText);
        } else {
          alert("Ошибка запуска. Проверьте API Key.");
          e.target.innerText = originalText;
          e.target.disabled = false;
        }
      } catch (err) {
        console.error(err);
        alert("Network error.");
        e.target.innerText = originalText;
        e.target.disabled = false;
      }
    });
  });

  async function pollTrainingStatus(btnEl, asset, originalText) {
    try {
      const res = await fetch(window.API_BASE + `/api/analytics/train_status/${asset}`, {
        headers: getHeaders()
      });
      const data = await res.json();
      if (data.status === "running") {
        btnEl.innerText = "В процессе...";
        setTimeout(() => pollTrainingStatus(btnEl, asset, originalText), 2000);
      } else {
        btnEl.innerText = originalText;
        btnEl.disabled = false;
        
        // Системное push-уведомление
        const title = data.status === "success" ? `✅ Обучение ${asset} завершено` : `❌ Ошибка обучения ${asset}`;
        showNotification(title, data.message);
        
        alert(data.message);
        loadSummary();
        loadModelsHistory();
        loadParserStatus();
      }
    } catch(e) {
      btnEl.innerText = originalText;
      btnEl.disabled = false;
    }
  }

  // Save Settings
  const btnSaveSettings = document.getElementById("btn-save-ml-settings");
  if (btnSaveSettings) {
    btnSaveSettings.addEventListener("click", async (e) => {
      e.preventDefault();

      // Сбор фичей
      const activeFeatures = Array.from(
        document.querySelectorAll(
          '#ml-features input[type="checkbox"]:checked',
        ),
      )
        .map((cb) => cb.value)
        .join(",");

      const settingsToSave = {
        ACTIVE_FEATURES: activeFeatures,
      };

      let allOk = true;

      for (const [key, val] of Object.entries(settingsToSave)) {
        try {
          const res = await fetch(window.API_BASE + `/api/settings/${key}`, {
            method: "PUT",
            headers: getHeaders(),
            body: JSON.stringify({ value: String(val) }),
          });
          if (!res.ok) allOk = false;
        } catch (err) {
          allOk = false;
        }
      }

      if (allOk) {
        alert("Настройки успешно сохранены!");
      } else {
        alert("Ошибка при сохранении части настроек. Проверьте API Key.");
      }
    });
  }

  // 4. Fetch Parser Status
  async function loadParserStatus() {
    try {
      const res = await fetch(window.API_BASE + "/api/dashboard/status", {
        headers: getHeaders(),
      });
      const data = await res.json();

      // 4.1 Collector Card
      const collector = data.collector;
      if (collector) {
        const statusSpan = document.getElementById("cs-status");
        statusSpan.innerText = collector.status;
        statusSpan.style.color =
          collector.status === "success" ? "var(--poly-green)" : "#ff3366";

        document.getElementById("cs-run-at").innerText = new Date(
          collector.run_at,
        ).toLocaleString();
        document.getElementById("cs-duration").innerText =
          collector.duration_sec;
        document.getElementById("cs-found").innerText = collector.markets_found;
        document.getElementById("cs-saved").innerText = collector.markets_saved;

        if (collector.error_message) {
          document.getElementById("cs-error").innerText =
            "Error: " + collector.error_message;
          document.getElementById("cs-error").style.display = "block";
        } else {
          document.getElementById("cs-error").style.display = "none";
        }
      } else {
        document.getElementById("cs-status").innerText = "No data yet";
      }

      // 4.2 Dataset Table
      const dtBody = document.querySelector("#dataset-table tbody");
      dtBody.innerHTML = "";
      const dtRows = [];
      for (const [asset, counts] of Object.entries(data.dataset_summary)) {
        const hasModel = data.active_models && data.active_models[asset];
        const isTrading = data.trade_assets && data.trade_assets.includes(asset.toUpperCase());
        
        let statusBadge = "";
        if (hasModel && isTrading) {
          statusBadge = `<span style="font-size: 0.8rem; background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); color: #00ff88; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">💰 Торгуется</span>`;
        } else if (hasModel) {
          statusBadge = `<span style="font-size: 0.8rem; background: rgba(0, 114, 245, 0.1); border: 1px solid rgba(0, 114, 245, 0.3); color: var(--poly-blue); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">🤖 Обучена</span>`;
        } else {
          statusBadge = `<span style="font-size: 0.8rem; background: rgba(255, 176, 32, 0.1); border: 1px solid rgba(255, 176, 32, 0.3); color: #FFB020; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">📊 Сбор данных</span>`;
        }

        dtRows.push(`
                    <tr>
                        <td><strong>${escapeHtml(asset)}</strong></td>
                        <td>${statusBadge}</td>
                        <td style="color: var(--poly-green)">${counts.RESOLVED}</td>
                        <td style="color: #FFB020">${counts.PENDING}</td>
                    </tr>
                `);
      }
      dtBody.innerHTML = dtRows.length > 0 ? dtRows.join("") : `<tr><td colspan="4">Нет собранных данных</td></tr>`;

      // 4.3 Live Markets Table
      const ltBody = document.querySelector("#live-table tbody");
      ltBody.innerHTML = "";
      const ltRows = [];
      for (const lm of data.live_markets) {
        ltRows.push(`
                    <tr>
                        <td><strong>${escapeHtml(lm.asset)}</strong></td>
                        <td style="font-size: 0.8rem">${escapeHtml(lm.question)}</td>
                        <td>${lm.current_yes_price}</td>
                        <td>${lm.current_spread}</td>
                        <td>${lm.volume_5min}</td>
                        <td style="font-size: 0.8rem">${lm.end_time_est ? new Date(lm.end_time_est).toLocaleTimeString() : "N/A"}</td>
                    </tr>
                `);
      }
      ltBody.innerHTML = ltRows.length > 0 ? ltRows.join("") : `<tr><td colspan="6">Нет активных рынков</td></tr>`;
    } catch (e) {
      console.error("Failed to load parser status", e);
    }
  }

  // 5. Fetch Models History
  async function loadModelsHistory() {
    try {
      const [resModels, resPnl] = await Promise.all([
        fetch(window.API_BASE + "/api/analytics/models", { headers: getHeaders() }),
        fetch(window.API_BASE + "/api/dashboard/model_pnl", { headers: getHeaders() })
      ]);

      const data = await resModels.json();
      let pnlData = {};
      try {
        const pnlJson = await resPnl.json();
        pnlData = pnlJson.data || {};
      } catch (e) {
        console.error("Failed to parse model PnL", e);
      }

      const tbody = document.querySelector("#models-table tbody");
      if (!tbody) return;

      tbody.innerHTML = "";

      if (data.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color: var(--text-muted);">Нет сохраненных моделей</td></tr>`;
        return;
      }

      // Вычисляем модель с максимальной точностью для каждого актива
      const bestAccuracy = {};
      data.forEach((m) => {
        if (!bestAccuracy[m.asset] || m.accuracy > bestAccuracy[m.asset]) {
          bestAccuracy[m.asset] = m.accuracy;
        }
      });

      const rows = [];
      data.forEach((m) => {
        const isActive = m.is_active;
        const isBest = m.accuracy === bestAccuracy[m.asset];
        
        let statusHtml = isActive
          ? `<span style="color: var(--poly-green); font-weight: bold;">Активна</span>`
          : `<span style="color: #8F9BB3;">Архив</span>`;
          
        if (isBest) {
          statusHtml += ` <span class="status-indicator online" style="font-size: 0.75rem; padding: 0.1rem 0.5rem; margin-left: 5px; background: rgba(0, 114, 245, 0.1); color: var(--poly-blue); border: 1px solid rgba(0, 114, 245, 0.3);">Самая умная</span>`;
        }

        const actionHtml = isActive
          ? `<button class="btn btn-primary" disabled style="opacity: 0.5;">Текущая</button>`
          : `<button class="btn btn-primary btn-activate-model" data-asset="${m.asset}" data-version="${m.version}">Активировать</button>`;

        const baselineText = m.baseline != null ? (m.baseline * 100).toFixed(1) + "%" : "-";
        const accuracyText = m.accuracy != null ? (m.accuracy * 100).toFixed(1) + "%" : "-";
        
        let eceHtml = "-";
        if (m.ece != null) {
          if (m.ece < 0.03) {
            eceHtml = `<span style="color: var(--poly-green);">${m.ece.toFixed(4)} (Отлично)</span>`;
          } else if (m.ece < 0.07) {
            eceHtml = `<span style="color: #ff9f43;">${m.ece.toFixed(4)} (Нормально)</span>`;
          } else {
            eceHtml = `<span style="color: #ff3366;">${m.ece.toFixed(4)} (Плохо)</span>`;
          }
        }

        const pnlKey = `${m.asset}_v${m.version}`;
        const pnl = pnlData[pnlKey];

        let pnlHtml = '<td style="color: var(--text-muted);">—</td>';
        if (pnl && pnl.total_trades > 0) {
          const pnlVal = pnl.pnl;
          const color = pnlVal > 0 ? "var(--poly-green, #4ade80)" : pnlVal < 0 ? "#ff3366" : "var(--text-muted)";
          const sign = pnlVal > 0 ? "+" : "";
          const wr = pnl.win_rate !== null ? ` (${pnl.win_rate}% WR, ${pnl.total_trades} сд.)` : "";
          pnlHtml = `<td style="color:${color}; font-weight:600; white-space:nowrap;">
            ${sign}${pnlVal.toFixed(2)} USDC<span style="color:var(--text-muted);font-size:0.8rem;font-weight:normal;">${wr}</span>
          </td>`;
        } else if (pnl && pnl.total_trades === 0) {
          pnlHtml = '<td style="color: var(--text-muted); font-size:0.85rem;">Нет сделок</td>';
        }

        rows.push(`
                    <tr>
                        <td><strong>${escapeHtml(m.asset)}</strong></td>
                        <td>v${m.version}</td>
                        <td>${accuracyText}</td>
                        <td>${baselineText}</td>
                        <td>${eceHtml}</td>
                        <td style="font-size: 0.85rem; max-width: 220px; word-break: break-word; white-space: normal;">${escapeHtml(translateFeatures(m.features))}</td>
                        <td>${m.trained_at ? new Date(m.trained_at).toLocaleString() : "N/A"}</td>
                        ${pnlHtml}
                        <td>${statusHtml}</td>
                        <td>${actionHtml}</td>
                    </tr>
                `);
      });
      tbody.innerHTML = rows.join("");

      // Attach event listeners to activate buttons
      document.querySelectorAll(".btn-activate-model").forEach((btn) => {
        btn.addEventListener("click", async (e) => {
          const asset = e.target.getAttribute("data-asset");
          const version = e.target.getAttribute("data-version");
          if (confirm(`Сделать модель v${version} для ${asset} активной?`)) {
            e.target.disabled = true;
            e.target.innerText = "Активация...";
            try {
              const res = await fetch(
                window.API_BASE +
                  `/api/analytics/models/${asset}/activate/${version}`,
                {
                  method: "POST",
                  headers: getHeaders(),
                },
              );
              if (res.ok) {
                alert(`Модель v${version} для ${asset} успешно активирована!`);
                loadModelsHistory();
                loadSummary(); // update active models in summary
              } else {
                alert("Ошибка при активации модели.");
                e.target.disabled = false;
                e.target.innerText = "Активировать";
              }
            } catch (err) {
              alert("Ошибка сети.");
              e.target.disabled = false;
              e.target.innerText = "Активировать";
            }
          }
        });
      });
    } catch (e) {
      console.error("Failed to load models history", e);
    }
  }

  // Refresh Status
  const btnRefreshStatus = document.getElementById("btn-refresh-status");
  if (btnRefreshStatus) {
    btnRefreshStatus.addEventListener("click", () => {
      loadSummary();
      if (chartsLoaded) {
        loadCharts(true);
      }
      loadParserStatus();
    });
  }

  // 6. Verify Resolves handler
  const btnVerify = document.getElementById("btn-verify-resolves");
  if (btnVerify) {
    btnVerify.addEventListener("click", async () => {
      btnVerify.innerText = "Сверка...";
      btnVerify.disabled = true;
      
      const logPanel = document.getElementById("verify-resolves-log-panel");
      const logBox = document.getElementById("verify-resolves-log");
      
      if (logPanel) logPanel.style.display = "block";
      if (logBox) logBox.innerHTML = "Запуск сверки исходов последних 50 закрытых рынков с Polymarket...";
      
      try {
        const res = await fetch(window.API_BASE + "/api/dashboard/verify_resolves", {
          method: "POST",
          headers: getHeaders(),
        });
        
        if (res.ok) {
          const data = await res.json();
          btnVerify.innerText = "Сверить исходы рынков";
          btnVerify.disabled = false;
          
          if (data.results && data.results.length > 0) {
            const logLines = [];
            data.results.forEach((r) => {
              const color = r.status === "OK" ? "#00ff88" : "#ff3366";
              const questionEsc = escapeHtml(r.question);
              logLines.push(
                `<div style="margin-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">` +
                `[${escapeHtml(r.asset)}] <strong>${questionEsc}</strong><br/>` +
                `ID: <span style="color: var(--text-muted); font-size: 0.8rem;">${escapeHtml(r.market_id)}</span><br/>` +
                `БД: <strong style="color: var(--poly-blue)">${escapeHtml(r.db_outcome)}</strong> | ` +
                `API: <strong style="color: var(--poly-green)">${escapeHtml(r.api_outcome)}</strong> | ` +
                `Статус: <strong style="color: ${color}">${escapeHtml(r.status)}</strong>` +
                `</div>`
              );
            });
            logBox.innerHTML = logLines.join("");
          } else {
            logBox.innerHTML = `<span style="color: #ffb020">${escapeHtml(data.message || "Нет данных")}</span>`;
          }
        } else {
          alert("Ошибка при выполнении сверки.");
          btnVerify.innerText = "Сверить исходы рынков";
          btnVerify.disabled = false;
          if (logPanel) logPanel.style.display = "none";
        }
      } catch (e) {
        console.error(e);
        alert("Ошибка сети.");
        btnVerify.innerText = "Сверить исходы рынков";
        btnVerify.disabled = false;
        if (logPanel) logPanel.style.display = "none";
      }
    });
  }
 
  // === Init ===
  loadSummary();
  loadCharts();
  loadSettings();
  loadParserStatus();
  loadModelsHistory();
  
  // Auto-refresh parser status every 30 seconds (only if tab is active)
  setInterval(() => {
    if (document.hidden) return;
    loadParserStatus();
  }, 30000);
});
