import re

with open('polyflip/templates/execution.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Remove renderFailedEntries
html = re.sub(r'\s*// Render Failed Entries\s*if \(typeof renderFailedEntries !== \'undefined\'\) \{\s*renderFailedEntries\(data\.failed_entries \|\| \[\]\);\s*\}', '', html)
html = re.sub(r'\s*function renderFailedEntries\(entries\) \{.*?\n\s*\}', '', html, flags=re.DOTALL)
html = re.sub(r'<div class="panel">\s*<h3>Ошибки входов \(ENTRY_FAILED\)</h3>.*?</div>', '', html, flags=re.DOTALL)

# 2. Update renderPositions
render_positions_new = '''
        function renderPositions(positions) {
            const tradable = positions.tradable || [];
            const resolved = positions.resolved || [];
            const archive = positions.archive || [];

            const tbodyTradable = document.querySelector('#tradable-positions-table tbody');
            if (!tradable.length) {
                tbodyTradable.innerHTML = '<tr><td colspan="10" style="text-align: center;">Нет торгуемых позиций</td></tr>';
            } else {
                tbodyTradable.innerHTML = tradable.map(p => {
                    const act = p.available_actions || {};
                    let actionHtml = '--';
                    if (act.close) {
                        actionHtml = `<button class="btn-action btn-danger" onclick="closePosition(${p.id})">Закрыть</button>`;
                    } else if (act.reconcile_resolution) {
                        actionHtml = `<button class="btn-action" style="background:#3182ce;" onclick="reconcileResolution(${p.id}, this)">Сверить результат</button>`;
                    }
                    return `
                    <tr>
                        <td>${escapeHtml(p.id)}</td>
                        <td>${p.created_at ? escapeHtml(p.created_at.substring(11, 19)) : '--'}</td>
                        <td>${escapeHtml(p.asset)} (${escapeHtml(p.market_id.substring(0, 8))}...)</td>
                        <td>${escapeHtml(p.outcome_bought)}</td>
                        <td>$${(p.entry_cost_usdc || p.amount_usdc || 0).toFixed(2)}</td>
                        <td>${(p.remaining_shares || 0).toFixed(2)}</td>
                        <td>$${(p.executed_price || 0).toFixed(3)}</td>
                        <td style="color: ${p.realized_pnl_usdc >= 0 ? '#48bb78' : '#e53e3e'};">${p.realized_pnl_usdc != null ? p.realized_pnl_usdc.toFixed(2) : '--'}</td>
                        <td><span class="status-badge status-live">${p.position_status}</span></td>
                        <td>${actionHtml}</td>
                    </tr>
                `;}).join('');
            }

            const tbodyResolved = document.querySelector('#resolved-markets-table tbody');
            if (!resolved.length) {
                tbodyResolved.innerHTML = '<tr><td colspan="10" style="text-align: center;">Нет завершённых рынков</td></tr>';
            } else {
                tbodyResolved.innerHTML = resolved.map(p => {
                    const act = p.available_actions || {};
                    let actions = [];
                    if (act.reconcile_resolution) {
                        actions.push(`<button class="btn-action" style="background:#3182ce; font-size:0.8rem; margin:2px;" onclick="reconcileResolution(${p.id}, this)">Сверить результат</button>`);
                    }
                    if (act.redeem) {
                        actions.push(`<button class="btn-action" style="background:#38a169; font-size:0.8rem; margin:2px;" onclick="redeemPosition(${p.id}, this)">Погасить</button>`);
                    }
                    if (act.reconcile_redemption) {
                        actions.push(`<button class="btn-action" style="background:#3182ce; font-size:0.8rem; margin:2px;" onclick="reconcileRedemption(${p.id}, this)">Проверить погашение</button>`);
                    }
                    return `
                    <tr>
                        <td>${escapeHtml(p.id)}</td>
                        <td>${p.created_at ? escapeHtml(p.created_at.substring(11, 19)) : '--'}</td>
                        <td>${escapeHtml(p.asset)} (${escapeHtml(p.market_id.substring(0, 8))}...)</td>
                        <td>${escapeHtml(p.outcome_bought)}</td>
                        <td>$${(p.entry_cost_usdc || p.amount_usdc || 0).toFixed(2)}</td>
                        <td>${(p.remaining_shares || 0).toFixed(2)}</td>
                        <td style="color: ${p.realized_pnl_usdc >= 0 ? '#48bb78' : '#e53e3e'};">${p.realized_pnl_usdc != null ? p.realized_pnl_usdc.toFixed(2) : '--'}</td>
                        <td><span class="status-badge status-shadow">${p.position_status}</span></td>
                        <td><span class="status-badge status-shadow">${p.redemption_status || 'UNKNOWN'}</span></td>
                        <td>${actions.join('')}</td>
                    </tr>
                `;}).join('');
            }

            const tbodyArchive = document.querySelector('#archive-table tbody');
            if (!archive.length) {
                tbodyArchive.innerHTML = '<tr><td colspan="8" style="text-align: center;">Архив пуст</td></tr>';
            } else {
                tbodyArchive.innerHTML = archive.map(p => `
                    <tr>
                        <td>${escapeHtml(p.id)}</td>
                        <td>${p.created_at ? escapeHtml(p.created_at.substring(11, 19)) : '--'}</td>
                        <td>${escapeHtml(p.asset)} (${escapeHtml(p.market_id.substring(0, 8))}...)</td>
                        <td>${escapeHtml(p.outcome_bought)}</td>
                        <td>$${(p.entry_cost_usdc || p.amount_usdc || 0).toFixed(2)}</td>
                        <td style="color: ${p.realized_pnl_usdc >= 0 ? '#48bb78' : '#e53e3e'};">${p.realized_pnl_usdc != null ? p.realized_pnl_usdc.toFixed(2) : '--'}</td>
                        <td><span class="status-badge status-shadow">${p.position_status}</span></td>
                    </tr>
                `).join('');
            }
        }
'''
html = re.sub(
    r'function renderPositions\(positions\) \{.*?const tbodyArchive = document\.querySelector\(\'#archive-table tbody\'\).*?\}\s*\}',
    render_positions_new.strip(),
    html, flags=re.DOTALL
)

# 3. Handle network error translation
html = html.replace('alert(`Ошибка отправки: ${e.message}`);', 'alert(`Сетевая ошибка: сервер недоступен (${e.message})`);')
html = html.replace('alert(`Ошибка сервера: ${e.message}`);', 'alert(`Ошибка сервера: сервер недоступен (${e.message})`);')
html = html.replace('alert("Ошибка сети");', 'alert("Сетевая ошибка: проверьте соединение");')
html = html.replace('catch (e) {\\n                console.error(e);\\n                alert("Ошибка сети или сервера");', 'catch (e) {\\n                console.error(e);\\n                alert(`Сетевая ошибка: ${e.message}`);')
html = html.replace('alert(`Network error: ${e.message}`);', 'alert(`Сетевая ошибка: ${e.message}`);')
html = html.replace('alert("Network error");', 'alert("Сетевая ошибка: проверьте соединение");')

with open('polyflip/templates/execution.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('done')
