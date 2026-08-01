import re

def modify_html():
    with open('polyflip/templates/execution.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update button visibility logic
    old_btn_logic = """                    // Button visibility
                    document.getElementById('btn-create-session').style.display = 'none';
                    if (s.status === 'ACTIVE') {
                        document.getElementById('btn-activate-session').style.display = 'none';
                        document.getElementById('btn-stop-session').style.display = 'inline-block';
                    } else {
                        document.getElementById('btn-activate-session').style.display = 'inline-block';
                        document.getElementById('btn-stop-session').style.display = 'none';
                    }
                    document.getElementById('btn-finish-session').style.display = 'inline-block';"""
    new_btn_logic = """                    // Button visibility
                    document.getElementById('btn-create-session').style.display = 'none';
                    
                    if (data.available_actions) {
                        document.getElementById('btn-check-readiness').style.display = data.available_actions.check_readiness ? 'inline-block' : 'none';
                        document.getElementById('btn-activate-session').style.display = data.available_actions.activate ? 'inline-block' : 'none';
                        document.getElementById('btn-stop-session').style.display = data.available_actions.stop ? 'inline-block' : 'none';
                        document.getElementById('btn-finish-session').style.display = data.available_actions.finish ? 'inline-block' : 'none';
                        document.getElementById('btn-close-all').style.display = data.available_actions.close_all ? 'inline-block' : 'none';
                    }"""
    content = content.replace(old_btn_logic, new_btn_logic)

    # hide close-all initially if no session
    old_no_sess_logic = """                    document.getElementById('btn-create-session').style.display = 'inline-block';
                    document.getElementById('btn-activate-session').style.display = 'none';
                    document.getElementById('btn-stop-session').style.display = 'none';
                    document.getElementById('btn-finish-session').style.display = 'none';"""
    new_no_sess_logic = """                    document.getElementById('btn-create-session').style.display = 'inline-block';
                    document.getElementById('btn-check-readiness').style.display = 'none';
                    document.getElementById('btn-activate-session').style.display = 'none';
                    document.getElementById('btn-stop-session').style.display = 'none';
                    document.getElementById('btn-finish-session').style.display = 'none';
                    document.getElementById('btn-close-all').style.display = 'none';"""
    content = content.replace(old_no_sess_logic, new_no_sess_logic)

    # 2. Add data.readiness handling
    readiness_handler = """
                if (data.readiness) {
                    updateChecklistUI(data.readiness);
                }"""
    # put it right after rendering positions
    content = content.replace("renderRequests(data.requests || []);", "renderRequests(data.requests || []);\n" + readiness_handler)

    # 3. Remove 5s setInterval
    interval_regex = re.compile(r'// Auto-refresh readiness.*?setInterval\([^)]+\);\s*}, 5000\);', re.DOTALL)
    content = interval_regex.sub('', content)

    # 4. Modify requests table to add reconcile button
    # First, remove old btn-reconcile from the header
    content = content.replace('<button id="btn-reconcile" class="btn-action" style="background: #3182ce;" onclick="reconcileActive()">Сверить с Polymarket</button>', '')
    
    # Then change renderRequests to include it
    old_row_actions = """                        ${r.can_mark_no_fill ? `
                            <button id="btn-no-fill-${r.id}" class="btn-action btn-danger" onclick="resolveNoFill('${r.id}')" style="padding: 4px 8px; font-size: 0.8rem;">
                                No Fill
                            </button>
                            <input type="checkbox" class="batch-no-fill-cb" value="${r.id}" style="margin-left: 5px;">
                        ` : (r.state === 'MANUAL_REVIEW_REQUIRED' ? `<span style="font-size: 0.85rem; color: #dd6b20;">${r.review_blockers && r.review_blockers.length ? escapeHtml(r.review_blockers.join(', ')) : 'Требуется сверка'}</span>` : '')}"""
    
    new_row_actions = """                        ${r.can_mark_no_fill ? `
                            <button id="btn-no-fill-${r.id}" class="btn-action btn-danger" onclick="resolveNoFill('${r.id}')" style="padding: 4px 8px; font-size: 0.8rem;">
                                No Fill
                            </button>
                            <input type="checkbox" class="batch-no-fill-cb" value="${r.id}" style="margin-left: 5px;">
                        ` : (r.state === 'MANUAL_REVIEW_REQUIRED' ? `<span style="font-size: 0.85rem; color: #dd6b20;">${r.review_blockers && r.review_blockers.length ? escapeHtml(r.review_blockers.join(', ')) : 'Требуется сверка'}</span>` : '')}
                        ${r.available_actions && r.available_actions.includes('RECONCILE_WITH_POLYMARKET') ? `
                            <button class="btn-action" style="background: #3182ce; padding: 4px 8px; font-size: 0.8rem; margin-top: 4px;" onclick="reconcileRequest('${r.id}')">
                                Сверить
                            </button>
                        ` : ''}"""
    
    content = content.replace(old_row_actions, new_row_actions)
    
    # Add reconcileRequest function
    new_reconcile_func = """
        async function reconcileRequest(reqId) {
            const res = await fetch(window.API_BASE + `/api/execution/requests/${reqId}/reconcile`, {
                method: 'POST', headers: getAuthHeaders()
            });
            if (res.ok) {
                alert("Заявка переведена в RECONCILING");
                loadLiveDashboard();
            } else {
                const err = await res.json();
                alert("Ошибка: " + (err.detail || JSON.stringify(err)));
            }
        }
    """
    
    content = content.replace("async function reconcileActive() {", new_reconcile_func + "\n        async function oldReconcileActive() {")
    
    # Update errors translation to use structured errors
    old_error_td = '<td style="color: #e53e3e; max-width: 200px; word-wrap: break-word;">${translateError(r.error_reason)}</td>'
    new_error_td = '<td style="color: #e53e3e; max-width: 200px; word-wrap: break-word;">${r.error_details && r.error_details.error_message_ru ? escapeHtml(r.error_details.error_message_ru) : (r.error_reason ? escapeHtml(r.error_reason) : \'--\')}</td>'
    content = content.replace(old_error_td, new_error_td)

    with open('polyflip/templates/execution.html', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    modify_html()
