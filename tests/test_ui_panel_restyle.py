"""Offline checks for the restyled control panel: assets, motion policy and live states."""

from __future__ import annotations
import re
import json
import shutil
import subprocess
from typing import Any
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


STATIC = Path(__file__).resolve().parents[1] / "src/dj_reachy_remix/static"
NODE = shutil.which("node")


def test_panel_assets_are_served_by_the_settings_app(tmp_path: Path) -> None:
    """The app-manager page and its CSS/JS come from the packaged static directory."""
    from dj_reachy_remix.runtime_settings import register_settings_page

    app = FastAPI()
    register_settings_page(app, tmp_path / "settings.json")
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert 'href="/static/style.css"' in page.text and 'src="/static/main.js"' in page.text
        css = client.get("/static/style.css")
        assert css.status_code == 200 and "text/css" in css.headers["content-type"]
        js = client.get("/static/main.js")
        assert js.status_code == 200 and "javascript" in js.headers["content-type"]


@pytest.mark.parametrize("name", ["index.html", "style.css", "main.js"])
def test_panel_assets_work_offline(name: str) -> None:
    """No CDN scripts, remote stylesheets or web fonts: the robot LAN may be offline."""
    text = (STATIC / name).read_text(encoding="utf-8")
    assert re.search(r"(?:https?:)?//[A-Za-z0-9-]+\.[A-Za-z]", text) is None
    assert "@import" not in text and "@font-face" not in text


def test_reduced_motion_disables_nonessential_animation() -> None:
    """Users who ask for reduced motion get static equalizers, loaders and transitions."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    start = css.find("@media (prefers-reduced-motion: reduce)")
    assert start != -1
    block = css[start:]
    assert "animation-duration" in block and "transition-duration" in block
    assert "animation-iteration-count: 1" in block


def test_status_regions_are_announced() -> None:
    """Status text stays in polite live regions; visual toasts don't double-announce."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for region in ("status", "voice-status", "songs"):
        assert re.search(rf'id="{region}"[^>]*aria-live="polite"', html), region
    assert re.search(r'id="toasts"[^>]*aria-hidden="true"', html)
    assert 'id="deck"' in html and 'class="eq"' in html
    assert "prefers-reduced-motion" not in html  # motion policy lives in the stylesheet


HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
const requests = [];
const pending = [];
class El {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.listeners = {}; this.classes = new Set();
    this.textContent = ''; this.value = ''; this.checked = false; this.disabled = false; this.hidden = false;
    this.dataset = {}; this.parent = null;
    const self = this;
    this.classList = {
      toggle(name, force) {
        const on = force === undefined ? !self.classes.has(name) : Boolean(force);
        if (on) self.classes.add(name); else self.classes.delete(name);
        return on;
      },
      contains: (name) => self.classes.has(name),
    };
  }
  set className(value) { this.classes = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classes].join(' '); }
  addEventListener(event, callback) { (this.listeners[event] ||= []).push(callback); }
  fire(event, detail = {}) { return Promise.all((this.listeners[event] || []).map((fn) => fn(detail))); }
  append(...children) { for (const child of children) { child.remove(); child.parent = this; child.moves = (child.moves || 0) + 1; this.children.push(child); } }
  insertBefore(child, anchor) { child.remove(); child.parent = this; child.moves = (child.moves || 0) + 1; this.children.splice(this.children.indexOf(anchor), 0, child); }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); this.parent = null; }
  reportValidity() { return true; }
}
const element = (id) => { if (!elements.has(id)) elements.set(id, new El()); return elements.get(id); };
const controls = ['pause', 'resume', 'stop'].map((name) => { const b = new El('button'); b.dataset.control = name; return b; });
let library = ['b.mp3', 'a.mp3'];
let failStop = false, busyNext = false, failSettingsPut = false;
const titles = {}, sizes = {}, timers = [];
const settings = {
  voice: {backend: 'subscription', model: 'gpt-live-1-codex', voice: 'cove', language: 'auto'},
  music: {model_id: 'music_v2_5', duration_ms: 10000, output_format: 'mp3_48000_192', positive_styles: [],
          negative_styles: [], lyrics: '', language: 'auto', force_instrumental: false, finetune_id: null},
  motion: {style: 'smooth', intensity: 0.05},
};
const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r; }); return {promise, resolve}; };
const context = vm.createContext({
  URL, console,
  setTimeout: (fn, ms) => { timers.push({fn, ms}); return timers.length; },
  window: {confirm: () => true},
  document: {
    currentScript: {src: 'https://robot.local/static/main.js'},
    getElementById: element,
    createElement: (tag) => new El(tag),
    querySelectorAll: (selector) => selector === '[data-control]' ? controls : [],
  },
  fetch: async (url, options = {}) => {
    const path = new URL(url).pathname, method = options.method || 'GET';
    requests.push({path, method, body: options.body || null});
    let data = {}, ok = true;
    if (path.endsWith('/settings') && method === 'GET') data = settings;
    else if (path.endsWith('/settings') && failSettingsPut) { ok = false; data = {error: 'Settings could not be saved.'}; }
    else if (path.endsWith('/settings')) { const body = JSON.parse(options.body); Object.assign(settings[Object.keys(body)[0]], Object.values(body)[0]); data = settings; }
    else if (path.endsWith('/instructions')) data = {instructions: ''};
    else if (path.endsWith('/status')) data = {billing: 'ChatGPT subscription', provider_access: 'unverified', elevenlabs_key_configured: true};
    else if (path === '/songs') data = {songs: library.map((filename) => ({filename, title: titles[filename] || filename, size: sizes[filename] || '1.0 MB'}))};
    else if (path.startsWith('/songs/') && method === 'DELETE') { library = library.filter((f) => `/songs/${encodeURIComponent(f)}` !== path); data = {status: 'deleted'}; }
    else if (path.endsWith('/music/pause')) data = {status: 'paused'};
    else if (path.endsWith('/music/resume')) data = {status: 'resumed'};
    else if (path.endsWith('/music/stop')) { if (failStop) { ok = false; data = {error: 'Audio session is not running. Connect voice first.'}; } else data = {status: 'stopped', message: 'Local decoder stopped and queued audio cleared.'}; }
    else if ((path.endsWith('/music/generate') || path.endsWith('/music/replay')) && busyNext) { busyNext = false; data = {status: 'busy', message: 'A music job is already active. Stop it first.'}; }
    else if (path.endsWith('/music/generate') || path.endsWith('/music/replay')) { const d = deferred(); pending.push(d); data = await d.promise; }
    return {ok, status: ok ? 200 : 409, json: async () => data};
  },
});
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); };
const deckState = () => [...element('deck').classes].find((c) => c.startsWith('is-'));
const rowNames = () => element('songs').children.map((row) => row.children[0].textContent);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
(async () => {
  const out = {};
  await flush();
  out.loading = element('main').classes.has('is-loading');
  out.initialRows = rowNames();
  out.initialNew = element('songs').children.some((row) => row.classes.has('is-new'));
  out.intensitySlider = element('intensity').value;
  out.intensityReadout = element('intensity-value').textContent;
  out.styleHint = element('motion-style-hint').textContent;
  out.idle = deckState();

  element('paid-confirmation').checked = true;
  await element('paid-confirmation').fire('change');
  element('prompt').value = 'funk for a robot';
  const generating = element('generate').fire('click');
  await flush();
  out.generating = deckState();
  out.generatingText = `${element('deck-state').textContent} ${element('deck-detail').textContent}`;
  out.generateBusy = element('generate').classes.has('is-busy');
  await controls[0].fire('click'); await flush();
  out.paused = deckState();
  await controls[1].fire('click'); await flush();
  out.resumed = deckState();
  library = ['c.mp3', 'b.mp3', 'a.mp3'];
  pending.shift().resolve({status: 'completed'});
  await generating; await flush();
  out.afterGenerate = deckState();
  out.afterRows = rowNames();
  out.newRow = element('songs').children[0].classes.has('is-new');

  const victim = element('songs').children[1];
  await victim.children[2].children[1].fire('click'); await flush();
  out.leaving = victim.classes.has('is-leaving') && victim.parent !== null;
  await victim.fire('animationend', {animationName: 'track-out'});
  out.afterDelete = rowNames();

  // A stale replay must not reset the deck owned by a newer replay.
  const first = element('songs').children[0], second = element('songs').children[1];
  const staleReplay = first.children[2].children[0].fire('click'); await flush();
  await controls[2].fire('click'); await flush();
  out.stopped = deckState();
  const liveReplay = second.children[2].children[0].fire('click'); await flush();
  pending.shift().resolve({status: 'stopped'});
  await staleReplay; await flush();
  out.afterStale = deckState();
  pending.shift().resolve({status: 'completed'});
  await liveReplay; await flush();
  out.afterLive = deckState();

  failStop = true;
  await controls[2].fire('click'); await flush();
  const toasts = element('toasts').children;
  out.errorToast = toasts[toasts.length - 1].classes.has('is-error');
  out.errorText = toasts[toasts.length - 1].textContent;
  const before = toasts.length;
  await toasts[0].fire('animationend', {animationName: 'toast-out-still'});  // reduced-motion exit
  out.toastRemoved = element('toasts').children.length === before - 1;

  element('intensity').value = '100';
  await element('intensity').fire('input');
  out.maxReadout = element('intensity-value').textContent;
  await element('motion-reset').fire('click'); await flush();
  out.reset = {style: element('motion-style').value, slider: element('intensity').value};
  out.motionPuts = requests.filter((r) => r.method === 'PUT' && r.path.endsWith('/settings')).map((r) => JSON.parse(r.body));

  // A busy rejection keeps the running job's deck state and ownership token.
  const running = element('songs').children[0].children[2].children[0].fire('click'); await flush();
  const runningDetail = element('deck-detail').textContent;
  busyNext = true;
  await element('songs').children[1].children[2].children[0].fire('click'); await flush();
  out.busyReplay = {state: deckState(), kept: element('deck-detail').textContent === runningDetail};
  busyNext = true; element('paid-confirmation').checked = true; await element('paid-confirmation').fire('change');
  element('prompt').value = 'funk for a robot';
  await element('generate').fire('click'); await flush();
  out.busyGenerate = {state: deckState(), kept: element('deck-detail').textContent === runningDetail};
  pending.shift().resolve({status: 'completed'}); await running; await flush();
  out.afterRunning = deckState();

  // One refresh that renames a kept track, removes one and adds one.
  const kept = element('songs').children[1], keptAudio = kept.children[3], goner = element('songs').children[0];
  const movesBefore = kept.moves;
  titles['a.mp3'] = 'Renamed groove'; sizes['a.mp3'] = '2.5 MB';
  library = ['d.mp3', 'a.mp3'];
  await element('refresh').fire('click'); await flush();
  out.renamed = {same: kept.parent === element('songs'), title: kept.children[0].textContent,
                 meta: kept.children[1].textContent, audio: kept.children[3] === keptAudio, moved: kept.moves !== movesBefore};
  out.combined = {gonerLeaving: goner.classes.has('is-leaving') && goner.parent !== null,
                  added: element('songs').children.some((r) => r.children[0].textContent === 'd.mp3' && r.classes.has('is-new')),
                  live: element('songs').children.filter((r) => !r.classes.has('is-leaving')).map((r) => r.children[0].textContent)};

  // A failed reset restores the user's previous motion controls.
  failSettingsPut = true;
  element('motion-style').value = 'smooth'; await element('motion-style').fire('change');
  element('intensity').value = '20'; await element('intensity').fire('input');
  await element('motion-reset').fire('click'); await flush();
  out.failedReset = {style: element('motion-style').value, slider: element('intensity').value,
                     readout: element('intensity-value').textContent, hint: element('motion-style-hint').textContent};
  failSettingsPut = false;

  // Every toast has a removal timer even if animation events never fire.
  const toastCount = element('toasts').children.length;
  out.toastCount = toastCount;
  out.toastTimers = timers.filter((t) => t.ms >= 5000).length >= toastCount && toastCount > 0;
  timers.splice(0).forEach((t) => t.fn());
  out.toastsAfterTimers = element('toasts').children.length;
  console.log(JSON.stringify(out));
})().catch((error) => { console.error(error); process.exitCode = 1; });
"""


def run_panel() -> dict[str, Any]:
    """Execute the shipped main.js against a fake DOM and loopback-free fake fetch."""
    if NODE is None:
        pytest.skip("Node.js is required to execute dashboard regression tests")
    result = subprocess.run([NODE, "-e", HARNESS, str(STATIC / "main.js")],
                            check=True, capture_output=True, text=True, timeout=10)
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    return data


def test_deck_follows_generation_and_playback_controls() -> None:
    """Generating, paused, playing and idle reflect the real request lifecycle."""
    out = run_panel()
    assert out["loading"] is False
    assert out["idle"] == "is-idle"
    assert out["generating"] == "is-generating" and out["generateBusy"] is True
    assert out["paused"] == "is-paused"
    assert out["resumed"] == "is-playing"
    assert out["afterGenerate"] == "is-idle"


def test_stale_replay_cannot_reset_a_newer_deck_owner() -> None:
    """A finished old request leaves the newer replay's playing state alone."""
    out = run_panel()
    assert out["stopped"] == "is-idle"
    assert out["afterStale"] == "is-playing"
    assert out["afterLive"] == "is-idle"


def test_library_animates_inserts_and_removals() -> None:
    """Rows are keyed by filename: new tracks enter, deleted tracks leave before removal."""
    out = run_panel()
    assert out["initialRows"] == ["b.mp3", "a.mp3"] and out["initialNew"] is False
    assert out["afterRows"] == ["c.mp3", "b.mp3", "a.mp3"] and out["newRow"] is True
    assert out["leaving"] is True
    assert out["afterDelete"] == ["c.mp3", "a.mp3"]


def test_errors_raise_an_error_toast() -> None:
    """Failed requests surface their message visually as an error."""
    out = run_panel()
    assert out["errorToast"] is True
    assert "Connect voice first" in out["errorText"]
    assert out["toastRemoved"] is True


def test_motion_intensity_uses_the_original_human_scale() -> None:
    """The slider shows upstream's 0-100 scale while the API keeps 0-0.25 floats."""
    out = run_panel()
    assert out["intensitySlider"] == "20"  # stored 0.05
    assert "25%" in out["intensityReadout"]  # a quarter of the original 0.20
    assert "half" in out["styleHint"].lower()  # smooth preset
    assert "125%" in out["maxReadout"]
    assert out["reset"] == {"style": "groove", "slider": "80"}
    assert out["motionPuts"][-1] == {"motion": {"style": "groove", "intensity": 0.2}}


def test_busy_start_keeps_the_running_deck() -> None:
    """A busy replay or generation is a rejected start, not a new deck owner."""
    out = run_panel()
    assert out["busyReplay"] == {"state": "is-playing", "kept": True}
    assert out["busyGenerate"] == {"state": "is-playing", "kept": True}
    assert out["afterRunning"] == "is-idle"


def test_generating_copy_covers_composing_and_playback() -> None:
    """The generate request stays open through playback, so the copy says both."""
    out = run_panel()
    assert "playing" in out["generatingText"].lower()
    assert "until the track ends" in out["generatingText"]


def test_refresh_updates_kept_rows_and_animates_combined_changes() -> None:
    """Reused rows take fresh metadata in place; adds and removals both animate."""
    out = run_panel()
    assert out["renamed"] == {"same": True, "title": "Renamed groove", "meta": "a.mp3, 2.5 MB",
                              "audio": True, "moved": False}
    assert out["combined"]["gonerLeaving"] is True and out["combined"]["added"] is True
    assert out["combined"]["live"] == ["d.mp3", "Renamed groove"]


def test_failed_motion_reset_restores_previous_controls() -> None:
    """A rejected reset leaves the slider, style and readouts on the saved-by-user values."""
    out = run_panel()
    assert out["failedReset"]["style"] == "smooth" and out["failedReset"]["slider"] == "20"
    assert "25%" in out["failedReset"]["readout"] and "half" in out["failedReset"]["hint"].lower()


def test_toasts_are_removed_without_animation_events() -> None:
    """A timer fallback clears toasts when animationend never fires."""
    out = run_panel()
    assert out["toastTimers"] is True
    assert out["toastsAfterTimers"] == 0


def test_only_the_newest_three_toasts_stay_in_the_dom() -> None:
    """Hidden toasts never animate out, so older ones reappeared later (PR #11 review)."""
    out = run_panel()
    assert 0 < out["toastCount"] <= 3
