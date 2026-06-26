document.addEventListener("DOMContentLoaded", () => {
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

  // 1. Fetch Summary
  async function loadSummary() {
    try {
      const res = await fetch(window.API_BASE + "/api/analytics/summary");
      const data = await res.json();

      document.getElementById("stat-markets").innerText =
        data.total_resolved_markets || 0;
      document.getElementById("stat-flips").innerText =
        (data.flip_percentage || 0) + "%";

      const btcModel = data.active_models["BTC"];
      if (btcModel) {
        document.getElementById("stat-model-btc").innerText =
          `v${btcModel.version} (Acc: ${btcModel.accuracy})`;
      }

      const ethModel = data.active_models["ETH"];
      if (ethModel) {
        document.getElementById("stat-model-eth").innerText =
          `v${ethModel.version} (Acc: ${ethModel.accuracy})`;
      }

      ["BTC", "ETH"].forEach((asset) => {
        if (data.model_history && data.model_history[asset]) {
          renderAccuracyChart(data.model_history[asset], asset);
        }
      });
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

    const color = asset === "BTC" ? "#0072F5" : "#00D395";
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

  async function loadCharts() {
    try {
      const res = await fetch(window.API_BASE + "/api/analytics/probabilities");
      chartDataStore = await res.json();
      renderSelectedChart();
    } catch (e) {
      console.error("Failed to load charts", e);
    }
  }

  function renderSelectedChart() {
    const selectorEl = document.getElementById("asset-selector");
    if (!selectorEl) return;
    const selectedAsset = selectorEl.value;
    const assetData = chartDataStore[selectedAsset] || {};
    const color = selectedAsset === "BTC" ? "#0072F5" : "#00D395";

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
  const btnTrain = document.getElementById("btn-train");
  if (btnTrain) {
    btnTrain.addEventListener("click", async () => {
      btnTrain.innerText = "Обучение...";
      btnTrain.disabled = true;

      try {
        const res = await fetch(window.API_BASE + "/api/analytics/train", {
          method: "POST",
          headers: getHeaders(),
        });
        if (res.ok) {
          alert("Задание на обучение отправлено в фон!");
          pollTrainingStatus();
        } else {
          alert("Ошибка запуска. Проверьте API Key.");
          btnTrain.innerText = "Запустить переобучение";
          btnTrain.disabled = false;
        }
      } catch (e) {
        console.error(e);
        alert("Network error.");
        btnTrain.innerText = "Запустить переобучение";
        btnTrain.disabled = false;
      }
    });

    async function pollTrainingStatus() {
      try {
        const res = await fetch(window.API_BASE + "/api/analytics/train_status", {
          headers: getHeaders()
        });
        const data = await res.json();
        if (data.status === "running") {
          btnTrain.innerText = "Обучение в процессе...";
          setTimeout(pollTrainingStatus, 2000);
        } else {
          btnTrain.innerText = "Запустить переобучение";
          btnTrain.disabled = false;
          alert(data.message);
          loadSummary();
          loadModelsHistory();
        }
      } catch(e) {
        btnTrain.innerText = "Запустить переобучение";
        btnTrain.disabled = false;
      }
    }
  }

  // Save Settings
  const btnSaveSettings = document.getElementById("btn-save-settings");
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
      for (const [asset, counts] of Object.entries(data.dataset_summary)) {
        dtBody.innerHTML += `
                    <tr>
                        <td><strong>${escapeHtml(asset)}</strong></td>
                        <td style="color: var(--poly-green)">${counts.RESOLVED}</td>
                        <td style="color: #FFB020">${counts.PENDING}</td>
                    </tr>
                `;
      }
      if (Object.keys(data.dataset_summary).length === 0) {
        dtBody.innerHTML = `<tr><td colspan="3">Нет собранных данных</td></tr>`;
      }

      // 4.3 Live Markets Table
      const ltBody = document.querySelector("#live-table tbody");
      ltBody.innerHTML = "";
      for (const lm of data.live_markets) {
        ltBody.innerHTML += `
                    <tr>
                        <td><strong>${escapeHtml(lm.asset)}</strong></td>
                        <td style="font-size: 0.8rem">${escapeHtml(lm.question)}</td>
                        <td>${lm.current_yes_price}</td>
                        <td>${lm.current_spread}</td>
                        <td>${lm.volume_5min}</td>
                        <td style="font-size: 0.8rem">${new Date(lm.end_time_est).toLocaleTimeString()}</td>
                    </tr>
                `;
      }
      if (data.live_markets.length === 0) {
        ltBody.innerHTML = `<tr><td colspan="6">Нет активных рынков</td></tr>`;
      }
    } catch (e) {
      console.error("Failed to load parser status", e);
    }
  }

  // 5. Fetch Models History
  async function loadModelsHistory() {
    try {
      const res = await fetch(window.API_BASE + "/api/analytics/models", {
        headers: getHeaders(),
      });
      const data = await res.json();

      const tbody = document.querySelector("#models-table tbody");
      if (!tbody) return;

      tbody.innerHTML = "";

      if (data.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8">Нет сохраненных моделей</td></tr>`;
        return;
      }

      // Вычисляем модель с максимальной точностью для каждого актива
      const bestAccuracy = {};
      data.forEach((m) => {
        if (!bestAccuracy[m.asset] || m.accuracy > bestAccuracy[m.asset]) {
          bestAccuracy[m.asset] = m.accuracy;
        }
      });

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

        const baselineText = m.baseline !== null && m.baseline !== undefined ? m.baseline : "-";

        tbody.innerHTML += `
                    <tr>
                        <td><strong>${escapeHtml(m.asset)}</strong></td>
                        <td>v${m.version}</td>
                        <td>${m.accuracy}</td>
                        <td>${baselineText}</td>
                        <td style="font-size: 0.85rem; max-width: 250px;">${escapeHtml(m.features)}</td>
                        <td>${m.trained_at ? new Date(m.trained_at).toLocaleString() : "N/A"}</td>
                        <td>${statusHtml}</td>
                        <td>${actionHtml}</td>
                    </tr>
                `;
      });

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
      loadCharts();
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
  
  // Auto-refresh parser status every 30 seconds
  setInterval(loadParserStatus, 30000);
});
