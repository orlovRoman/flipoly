document.addEventListener('DOMContentLoaded', () => {
    let apiKey = "test-key";
try {
    apiKey = localStorage.getItem("polyflip_api_key") || "test-key";
} catch (e) {
    console.warn("localStorage unavailable, using default key");
}
    
    const elements = {
        capital: document.getElementById('stat-capital'),
        pnl: document.getElementById('stat-pnl'),
        winrate: document.getElementById('stat-winrate'),
        wl: document.getElementById('stat-wl'),
        assetTable: document.querySelector('#asset-stats-table tbody'),
        avgWinPrice: document.getElementById('avg-win-price'),
        avgWinProb: document.getElementById('avg-win-prob'),
        avgLossPrice: document.getElementById('avg-loss-price'),
        avgLossProb: document.getElementById('avg-loss-prob'),
        refreshBtn: document.getElementById('btn-refresh-trading')
    };
    
    let pnlChart = null;
    let wlChart = null;
    
    async function fetchStats() {
        try {
            const response = await fetch(`${window.API_BASE}/api/trading/stats`, {
                headers: { 'X-API-Key': apiKey }
            });
            if (response.status === 401) {
                alert("Неверный API ключ. Введите его на вкладке 'Настройки' в основном дашборде.");
                return;
            }
            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error("Ошибка при загрузке данных:", error);
        }
    }
    
    function updateUI(data) {
        // Update KPIs
        elements.capital.textContent = `${data.capital.toFixed(2)} USDC`;
        elements.pnl.textContent = `${data.overall_pnl > 0 ? '+' : ''}${data.overall_pnl.toFixed(2)} USDC`;
        elements.pnl.style.color = data.overall_pnl >= 0 ? '#00ff88' : '#ff3366';
        
        elements.winrate.textContent = `${data.winrate}%`;
        elements.wl.textContent = `${data.wins_vs_losses.wins} / ${data.wins_vs_losses.losses}`;
        
        // Update Asset Table
        elements.assetTable.innerHTML = '';
        for (const [asset, stat] of Object.entries(data.assets)) {
            const winrate = stat.trades > 0 ? ((stat.wins / stat.trades) * 100).toFixed(1) : 0;
            const pnlColor = stat.pnl >= 0 ? '#00ff88' : '#ff3366';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${asset}</td>
                <td>${stat.trades}</td>
                <td>${winrate}%</td>
                <td style="color: ${pnlColor}">${stat.pnl > 0 ? '+' : ''}${stat.pnl.toFixed(2)}</td>
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
        
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true, labels: { color: 'white' } } },
            scales: {
                x: { ticks: { color: 'rgba(255, 255, 255, 0.7)' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
                y: { ticks: { color: 'rgba(255, 255, 255, 0.7)' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } }
            }
        };

        if (pnlChart) pnlChart.destroy();
        pnlChart = new Chart(document.getElementById('chart-daily-pnl').getContext('2d'), {
            type: 'line',
            data: {
                labels: sortedDates,
                datasets: [{
                    label: 'Кумулятивный PnL (USDC)',
                    data: pnlData,
                    borderColor: '#4facfe',
                    backgroundColor: 'rgba(79, 172, 254, 0.2)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: commonOptions
        });
        
        if (wlChart) wlChart.destroy();
        wlChart = new Chart(document.getElementById('chart-daily-wl').getContext('2d'), {
            type: 'bar',
            data: {
                labels: sortedDates,
                datasets: [
                    {
                        label: 'Выигрыши',
                        data: winData,
                        backgroundColor: '#00ff88'
                    },
                    {
                        label: 'Проигрыши',
                        data: lossData,
                        backgroundColor: '#ff3366'
                    }
                ]
            },
            options: commonOptions
        });
    }
    
    elements.refreshBtn.addEventListener('click', fetchStats);
    
    // Initial fetch
    fetchStats();
    
    // Auto refresh every 5 min
    setInterval(fetchStats, 5 * 60 * 1000);
});
