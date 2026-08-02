import os
import re
import subprocess
import pytest
import shutil

@pytest.mark.skipif(not shutil.which("node"), reason="Node.js is not installed")
def test_javascript_syntax_in_templates():
    """
    Extracts all <script> blocks from HTML templates and checks their syntax using Node.js.
    """
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "polyflip", "templates")
    html_files = []
    
    for root, _, files in os.walk(templates_dir):
        for file in files:
            if file.endswith('.html'):
                html_files.append(os.path.join(root, file))
                
    assert len(html_files) > 0, "No HTML templates found"
    
    errors = []
    
    for html_file in html_files:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex to extract <script>...</script>
        # Note: this simple regex might not handle nested scripts or complex template tags perfectly,
        # but it works well for the current templates.
        scripts = re.findall(r'<script(?:[^>]*)>\s*(.*?)\s*</script>', content, re.IGNORECASE | re.DOTALL)
        
        for i, script in enumerate(scripts):
            # Skip empty scripts (e.g. script tags with only src attribute)
            if not script.strip():
                continue
            
            # Write to a temporary file
            temp_file = f"temp_syntax_check_{i}.js"
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(script)
                
                # Run node -c (check syntax without executing)
                res = subprocess.run(['node', '-c', temp_file], capture_output=True, text=True)
                
                if res.returncode != 0:
                    errors.append(f"Syntax error in {os.path.basename(html_file)} (script block #{i}):\n{res.stderr}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
    if errors:
        pytest.fail("\n\n".join(errors))
