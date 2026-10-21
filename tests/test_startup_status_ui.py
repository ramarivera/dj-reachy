"""Execute the packaged status/reconnect UI offline in Node, without a browser."""
import re
import json
import shutil
import subprocess
from pathlib import Path

import pytest


PAGE = Path(__file__).resolve().parents[1] / "src/dj_reachy_remix/static/index.html"


@pytest.mark.parametrize("prefix", ["", "/native/apps/dj-reachy-remix"])
def test_status_polling_never_reconnects_without_confirmed_click(prefix: str) -> None:
    """Keep polling read-only and require a confirmed click under either proxy root."""
    html = PAGE.read_text()
    assert 'id="voice-status"' in html and 'id="voice-reconnect"' in html
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for UI execution")
    harness = r'''
const vm = require('node:vm');
const source = process.argv[1], prefix = process.argv[2];
const requests = [], callbacks = {}, elements = {};
let confirm = false, poll;
const element = id => elements[id] ||= {hidden: false, disabled: false,
  addEventListener: (name, fn) => callbacks[id] = fn};
const context = vm.createContext({URL, console,
  document: {getElementById: element, querySelector: () => ({src: `https://fixture${prefix}/static/main.js`})},
  window: {addEventListener: (name, fn) => callbacks.loaded = fn, confirm: () => confirm},
  setInterval: fn => {poll = fn;},
  fetch: async (url, opts = {}) => {
    requests.push({path: new URL(url).pathname, method: opts.method || 'GET'});
    return {ok: true, json: async () => ({voice_state: 'offline', voice_error: 'voice_transport', voice_reconnect_available: true})};
  }
});
vm.runInContext(source, context);
(async () => {
  await callbacks.loaded();
  await poll();
  const message = elements['voice-status'].textContent;
  const visible = !elements['voice-reconnect'].hidden;
  await callbacks['voice-reconnect'](); // declined confirmation
  const before = requests.filter(r => r.method === 'POST').length;
  confirm = true;
  await callbacks['voice-reconnect']();
  console.log(JSON.stringify({requests, message, visible, before}));
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    result = subprocess.run([node, "-e", harness, scripts[0], prefix],
                            text=True, capture_output=True, timeout=10, check=True)
    data = json.loads(result.stdout)
    assert data["visible"] and data["before"] == 0
    assert "offline" in data["message"].lower()
    assert "transport" in data["message"].lower()
    assert "ready" not in data["message"].lower()
    posts = [row for row in data["requests"] if row["method"] == "POST"]
    assert posts == [{"path": f"{prefix}/api/dj-reachy-remix/voice/reconnect", "method": "POST"}]
