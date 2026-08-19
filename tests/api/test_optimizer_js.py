import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


OPTIMIZER_JS = Path(__file__).resolve().parents[2] / "polyflip" / "static" / "js" / "optimizer.js"


def test_optimizer_fetches_use_api_key_headers_and_api_base():
    source = OPTIMIZER_JS.read_text(encoding="utf-8")
    fetch_count = len(re.findall(r"\bfetch\s*\(", source))

    assert fetch_count == source.count("headers: getAuthHeaders()")
    assert source.count("fetch(`${window.API_BASE}") == fetch_count
    assert "Authorization" not in source
    assert "Bearer" not in source
    assert 'localStorage.getItem("polyflip_api_key")' in source
    assert '"X-API-Key": apiKey' in source
    assert "Введите API key для AI Lab" in source


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js is not installed")
def test_optimizer_get_auth_headers_reads_and_prompts_for_api_key(tmp_path):
    source = OPTIMIZER_JS.read_text(encoding="utf-8")
    node_script = f"""
const vm = require("vm");
const source = {json.dumps(source)};
let promptCalls = 0;
const context = {{
  localStorage: {{
    value: "stored-key",
    getItem(key) {{ return key === "polyflip_api_key" ? this.value : null; }},
    setItem(key, value) {{ if (key === "polyflip_api_key") this.value = value; }},
  }},
  window: {{
    prompt(message) {{
      promptCalls += 1;
      if (message !== "Введите API key для AI Lab") throw new Error("wrong prompt");
      return "entered-key";
    }},
  }},
  document: {{ addEventListener() {{}} }},
  console: {{ warn() {{}}, error() {{}} }},
  setInterval() {{}},
  clearInterval() {{}},
}};
vm.createContext(context);
vm.runInContext(source, context);

const storedHeaders = context.getAuthHeaders();
if (JSON.stringify(storedHeaders) !== JSON.stringify({{"Content-Type":"application/json","X-API-Key":"stored-key"}})) throw new Error("stored key headers mismatch");
if (promptCalls !== 0) throw new Error("prompted despite stored key");

context.localStorage.value = "";
const promptedHeaders = context.getAuthHeaders();
if (promptedHeaders["X-API-Key"] !== "entered-key") throw new Error("prompted key missing");
if (context.localStorage.value !== "entered-key") throw new Error("prompted key was not persisted");
if (promptCalls !== 1) throw new Error("missing-key prompt count mismatch");
"""
    harness = tmp_path / "optimizer_auth_test.js"
    harness.write_text(node_script, encoding="utf-8")
    result = subprocess.run(
        ["node", str(harness)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
