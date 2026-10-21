"""Execute the shipped dashboard JS without browsers, providers, or hardware."""

from __future__ import annotations
import json
import shutil
import subprocess
from typing import TypedDict
from pathlib import Path

import pytest


MAIN_JS = Path(__file__).resolve().parents[1] / "src/dj_reachy_remix/static/main.js"
NODE = shutil.which("node")

HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const page = process.argv[1];
const script = process.argv[2];
const elements = new Map();
const requests = [];
class Element {
  constructor() {
    this.value = ''; this.children = []; this.listeners = {}; this.classList = {toggle() {}, contains: () => false};
    this.style = {setProperty() {}}; this.offsetHeight = 0;  // toast heights (--h)
  }
  addEventListener(event, callback) { this.listeners[event] = callback; }
  append(...children) { children.forEach((child) => { child.parent = this; }); this.children.push(...children); }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); this.parent = null; }
  replaceChildren() { this.children = []; }
}
const element = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
const stop = new Element(); stop.dataset = {control: 'stop'};
const filename = 'space # café?.mp3';
const settings = {
  voice: {backend: 'subscription', model: 'fixture', voice: 'fixture', language: 'en'},
  music: {model_id: 'fixture', duration_ms: 10000, output_format: 'mp3', positive_styles: [], negative_styles: [], lyrics: '', language: 'en', force_instrumental: true},
  motion: {style: 'fixture', intensity: 0.5}
};
const context = vm.createContext({
  URL, console,
  setTimeout: () => 0,  // toast removal fallback; never needed within this fixture
  window: {location: new URL(page), confirm: () => true},
  document: {
    baseURI: page, currentScript: {src: script},
    getElementById: element,
    createElement: () => new Element(),
    querySelectorAll: () => [stop]
  },
  fetch: async (url, options) => {
    const resolved = new URL(url, page);
    requests.push({path: resolved.pathname, method: options.method, body: options.body});
    let data = {};
    if (resolved.pathname.endsWith('/settings')) data = settings;
    if (resolved.pathname.endsWith('/instructions')) data = {instructions: 'fixture'};
    if (resolved.pathname.endsWith('/status')) data = {billing: 'fixture', provider_access: 'unverified'};
    if (resolved.pathname.endsWith('/songs')) data = {songs: [{filename, title: 'fixture', size: 12}]};
    return {ok: true, json: async () => data};
  }
});
vm.runInContext(fs.readFileSync(process.argv[3], 'utf8'), context);
(async () => {
  // Flush the asynchronous startup, then exercise the real event callbacks.
  await new Promise(resolve => setImmediate(resolve));
  await vm.runInContext("save('motion', {style: 'fixture', intensity: 0.5})", context);
  await stop.listeners.click();
  const row = element('songs').children[0];
  const controls = row.children[2];
  await controls.children[0].listeners.click();
  const audio = new URL(row.children[3].src, page).pathname;
  await controls.children[1].listeners.click();
  console.log(JSON.stringify({requests, audio}));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


class RequestRow(TypedDict):
    """Validated request observation from the Node fixture."""

    path: str
    method: str
    body: str | None


class DashboardResult(TypedDict):
    """The two bounded outputs emitted by the JavaScript harness."""

    requests: list[RequestRow]
    audio: str


def execute_dashboard(page: str, script: str) -> DashboardResult:
    """Run real JS and browser-compatible URL resolution in a Node VM."""
    if NODE is None:
        pytest.skip("Node.js is required to execute dashboard regression tests")
    result = subprocess.run(
        [NODE, "-e", HARNESS, page, script, str(MAIN_JS)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    data = json.loads(result.stdout)
    assert isinstance(data, dict) and isinstance(data.get("requests"), list)
    rows: list[RequestRow] = []
    for entry in data["requests"]:
        assert isinstance(entry, dict)
        path, method, body = entry.get("path"), entry.get("method"), entry.get("body")
        assert isinstance(path, str) and isinstance(method, str)
        assert body is None or isinstance(body, str)
        rows.append({"path": path, "method": method, "body": body})
    audio = data.get("audio")
    assert isinstance(audio, str)
    return {"requests": rows, "audio": audio}


@pytest.mark.parametrize(
    "prefix,page_suffix",
    [
        ("", "/"),
        ("/native/apps/dj-reachy-remix", "/"),
        ("/native/apps/dj-reachy-remix", ""),
        ("/native/apps/dj-reachy-remix", "/?view=settings#music"),
    ],
)
def test_api_requests_stay_under_the_loaded_app(prefix: str, page_suffix: str) -> None:
    """Startup, saved settings, replay and stop use the proxy's app root."""
    result = execute_dashboard(
        f"https://desktop.example{prefix}{page_suffix}",
        f"https://desktop.example{prefix}/static/main.js?v=1",
    )
    actual = [(r["path"], r["method"]) for r in result["requests"] if "/api/" in r["path"]]
    assert actual == [
        (f"{prefix}/api/dj-reachy-remix/settings", "GET"),
        (f"{prefix}/api/dj-reachy-remix/instructions", "GET"),
        (f"{prefix}/api/dj-reachy-remix/status", "GET"),
        (f"{prefix}/api/dj-reachy-remix/settings", "PUT"),
        (f"{prefix}/api/dj-reachy-remix/music/stop", "POST"),
        (f"{prefix}/api/dj-reachy-remix/music/replay", "POST"),
    ]
    replay = next(r for r in result["requests"] if r["path"].endswith("/music/replay"))
    assert replay["body"] is not None
    assert json.loads(replay["body"]) == {"filename": "space # café?.mp3"}


@pytest.mark.parametrize("prefix", ["", "/native/apps/dj-reachy-remix"])
def test_song_listing_delete_and_audio_stay_under_the_loaded_app(prefix: str) -> None:
    """Media and deletion share the proxy root and preserve filename encoding."""
    result = execute_dashboard(
        f"https://desktop.example{prefix}/",
        f"https://desktop.example{prefix}/static/main.js",
    )
    encoded = "space%20%23%20caf%C3%A9%3F.mp3"
    song_requests = [(r["path"], r["method"]) for r in result["requests"] if "/songs" in r["path"]]
    assert song_requests == [
        (f"{prefix}/songs", "GET"),
        (f"{prefix}/songs/{encoded}", "DELETE"),
        (f"{prefix}/songs", "GET"),
    ]
    assert result["audio"] == f"{prefix}/songs/{encoded}"
