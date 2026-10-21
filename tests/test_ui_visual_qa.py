"""Regressions from the headless visual QA pass (#14): failures, toasts, row motion and layout rules."""

from __future__ import annotations
import re
import json
import shutil
import subprocess
from typing import Any
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / "src/dj_reachy_remix/static"
NODE = shutil.which("node")

# A fake DOM whose offsetHeight is 0 while an entry/exit animation holds an
# element at its first keyframe, so measuring at the wrong moment shows up.
HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const HEIGHTS = {track: 117, toast: 48};
class El {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.listeners = {}; this.classes = new Set(); this.props = {};
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
    this.style = {setProperty: (k, v) => { self.props[k] = v; }, removeProperty: (k) => { delete self.props[k]; }};
  }
  set className(value) { this.classes = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classes].join(' '); }
  get offsetHeight() {
    const base = this.classes.has('track') ? HEIGHTS.track : this.classes.has('toast') ? HEIGHTS.toast : 20;
    const held = ['is-new', 'is-leaving'].some((c) => this.classes.has(c))
      || (this.classes.has('toast') && !this.classes.has('is-measuring') && !this.classes.has('is-repeat'));
    return held ? 0 : base;
  }
  addEventListener(event, callback) { (this.listeners[event] ||= []).push(callback); }
  fire(event, detail = {}) { return Promise.all((this.listeners[event] || []).map((fn) => fn(detail))); }
  append(...children) { for (const child of children) { child.remove(); child.parent = this; this.children.push(child); } }
  insertBefore(child, anchor) { child.remove(); child.parent = this; this.children.splice(this.children.indexOf(anchor), 0, child); }
  replaceChildren(...children) { for (const child of this.children) child.parent = null; this.children = []; this.append(...children); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); this.parent = null; }
  reportValidity() { return true; }
}
const SETTINGS = {
  voice: {backend: 'subscription', model: 'gpt-live-1-codex', voice: 'cove', language: 'auto'},
  music: {model_id: 'music_v2_5', duration_ms: 10000, output_format: 'mp3_48000_192', positive_styles: [],
          negative_styles: [], lyrics: '', language: 'auto', force_instrumental: false, finetune_id: null},
  motion: {style: 'groove', intensity: 0.2},
};
const flush = async () => { for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r)); };
function world(mode = {}) {
  const elements = new Map(), requests = [], timers = [], pending = [], held = [];
  let library = ['b.mp3', 'a.mp3'];
  const element = (id) => { if (!elements.has(id)) elements.set(id, new El()); return elements.get(id); };
  const controls = ['pause', 'resume', 'stop'].map((name) => { const b = new El('button'); b.dataset.control = name; return b; });
  const reply = (data, ok = true, status = 200) => ({ok, status, json: async () => data});
  const proxyPage = (status) => ({ok: false, status, json: async () => {
    throw new SyntaxError(`Unexpected token '<', "<html><bod"... is not valid JSON`); }});
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
      requests.push(`${method} ${path}`);
      if (path.endsWith('/settings') && method === 'GET') {
        if (mode.settings === 'proxy') return proxyPage(502);
        if (mode.settings === 'down') throw new TypeError('Failed to fetch');
        return reply(SETTINGS);
      }
      if (path.endsWith('/instructions')) return reply({instructions: ''});
      if (path.endsWith('/status')) return reply({billing: 'fixture', provider_access: 'unverified', elevenlabs_key_configured: true});
      if (path === '/songs') {
        if (mode.songs === 'down') throw new TypeError('Failed to fetch');
        const snapshot = reply({songs: library.map((filename) => ({filename, title: filename, size: '1.0 MB'}))});
        if (mode.holdSongs) { let resolve; const p = new Promise((r) => { resolve = r; }); held.push(resolve); await p; }
        return snapshot;
      }
      if (path.endsWith('/music/pause')) return reply({status: 'not_playing'});
      if (path.endsWith('/music/stop')) return reply({status: 'stopped'});
      if (path.endsWith('/music/replay')) { let resolve; const p = new Promise((r) => { resolve = r; }); pending.push(resolve); return reply(await p); }
      return reply({});
    },
  });
  vm.runInContext(source, context);
  return {element, requests, timers, pending, held, controls, mode, setLibrary: (names) => { library = names; }};
}
const toastTexts = (w) => w.element('toasts').children.map((t) => t.textContent);
const chips = (w) => w.element('connection').children.map((c) => c.textContent);
const scenarios = {
  async settingsProxyError() {
    const w = world({settings: 'proxy'}); await flush();
    return {toasts: toastTexts(w), chips: chips(w), songsRequested: w.requests.includes('GET /songs'),
            rows: w.element('songs').children.length, loading: w.element('main').classes.has('is-loading')};
  },
  async settingsDown() {
    const w = world({settings: 'down'}); await flush();
    return {toasts: toastTexts(w), songsRequested: w.requests.includes('GET /songs')};
  },
  async songsDown() {
    const w = world({songs: 'down'}); await flush();
    const out = {toasts: toastTexts(w), chips: chips(w), rows: w.element('songs').children.length,
                 errorShown: w.element('songs-error').hidden === false};
    w.mode.songs = 'ok';
    await w.element('refresh').fire('click'); await flush();
    out.recovered = {rows: w.element('songs').children.length, errorHidden: w.element('songs-error').hidden === true};
    return out;
  },
  async toastRepeat() {
    const w = world(); await flush();
    for (let i = 0; i < 5; i++) { await w.controls[0].fire('click'); await flush(); }
    const stack = w.element('toasts').children;
    const out = {count: stack.length, texts: toastTexts(w), repeat: stack[stack.length - 1].classes.has('is-repeat'),
                 height: stack[stack.length - 1].props['--h']};
    await w.controls[2].fire('click'); await flush();
    out.afterOther = toastTexts(w);
    const first = w.element('toasts').children[0];
    await first.fire('click');
    out.dismiss = {leaving: first.classes.has('is-leaving'), stillThere: first.parent !== null, height: first.props['--h']};
    await first.fire('animationend', {animationName: 'toast-out'});
    out.dismiss.removed = first.parent === null;
    return out;
  },
  async rowMotion() {
    const w = world(); await flush();
    const goner = w.element('songs').children[1];
    w.setLibrary(['c.mp3', 'b.mp3']);
    await w.element('refresh').fire('click'); await flush();
    const fresh = w.element('songs').children[0];
    return {fresh: {isNew: fresh.classes.has('is-new'), height: fresh.props['--h']},
            goner: {leaving: goner.classes.has('is-leaving'), height: goner.props['--h']}};
  },
  async staleRefresh() {
    const w = world(); await flush();
    w.mode.holdSongs = true;
    w.setLibrary(['old.mp3']);
    w.element('refresh').fire('click'); await flush();
    w.setLibrary(['n1.mp3', 'n2.mp3', 'n3.mp3']);
    w.element('refresh').fire('click'); await flush();
    w.held[1](); await flush();
    w.held[0](); await flush();
    return {requests: w.held.length, rows: w.element('songs').children.filter((r) => !r.classes.has('is-leaving')).length};
  },
  async deckSettles() {
    const w = world(); await flush();
    const deck = () => w.element('deck').className;
    const out = {boot: deck()};
    const play = w.element('songs').children[0].children[2].children[0];
    const replay = play.fire('click'); await flush();
    out.playing = deck();
    await w.controls[2].fire('click'); await flush();
    out.stopped = deck();
    w.pending.shift()({status: 'stopped'}); await replay; await flush();
    return out;
  },
};
(async () => {
  const out = {};
  for (const [name, run] of Object.entries(scenarios)) out[name] = await run();
  console.log(JSON.stringify(out));
})().catch((error) => { console.error(error); process.exitCode = 1; });
"""


@pytest.fixture(scope="module")
def panel() -> dict[str, Any]:
    """Run every scenario once against the shipped main.js."""
    if NODE is None:
        pytest.skip("Node.js is required to execute dashboard regression tests")
    result = subprocess.run([NODE, "-e", HARNESS, str(STATIC / "main.js")],
                            check=True, capture_output=True, text=True, timeout=20)
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    return data


def css_block(css: str, header: str) -> str:
    """Return the brace-matched body that follows a selector or at-rule header."""
    start = css.find(header)
    assert start != -1, header
    depth, i = 0, css.index("{", start)
    for j in range(i, len(css)):
        depth += {"{": 1, "}": -1}.get(css[j], 0)
        if depth == 0:
            return css[i + 1:j]
    raise AssertionError(f"unterminated block {header}")


def test_proxy_error_pages_read_as_http_errors(panel: dict[str, Any]) -> None:
    """A 502 HTML page from a proxy says HTTP 502, not a JSON parser error."""
    out = panel["settingsProxyError"]
    assert any("HTTP 502" in text for text in out["toasts"]), out["toasts"]
    assert not any("Unexpected token" in text or "JSON" in text for text in out["toasts"])
    assert out["chips"] == ["Settings couldn't load. Check the local app server."]
    assert out["loading"] is False


def test_network_failures_read_as_unreachable_server(panel: dict[str, Any]) -> None:
    """A dropped connection names the app server instead of the browser's 'Failed to fetch'."""
    texts = panel["settingsDown"]["toasts"] + panel["songsDown"]["toasts"]
    assert texts and not any("Failed to fetch" in text for text in texts)
    assert any("app server" in text for text in texts)


def test_library_loads_even_when_settings_fail(panel: dict[str, Any]) -> None:
    """Settings and the library are independent, so a settings failure doesn't strand the skeleton."""
    out = panel["settingsProxyError"]
    assert out["songsRequested"] is True and out["rows"] == 2
    assert panel["settingsDown"]["songsRequested"] is True


def test_failed_library_load_replaces_the_skeleton(panel: dict[str, Any]) -> None:
    """A failed first load shows a retry hint, keeps status chips and recovers on Refresh."""
    out = panel["songsDown"]
    assert out["rows"] == 0 and out["errorShown"] is True
    assert "Settings couldn't load. Check the local app server." not in out["chips"]
    assert out["recovered"] == {"rows": 2, "errorHidden": True}


def test_repeated_toast_replaces_the_newest_instead_of_stacking(panel: dict[str, Any]) -> None:
    """Five identical messages keep one toast; a different one stacks normally."""
    out = panel["toastRepeat"]
    assert out["count"] == 1 and out["repeat"] is True
    assert out["texts"] == ["Nothing is playing right now."]
    assert out["height"] == "48px"
    assert len(out["afterOther"]) == 2


def test_dismissed_toast_collapses_before_removal(panel: dict[str, Any]) -> None:
    """Clicking a toast animates it out from its measured height; animationend removes it."""
    assert panel["toastRepeat"]["dismiss"] == {"leaving": True, "stillThere": True, "height": "48px", "removed": True}


def test_rows_are_measured_before_they_grow_or_collapse(panel: dict[str, Any]) -> None:
    """New and leaving rows carry their real height so the panels below glide instead of jumping."""
    out = panel["rowMotion"]
    assert out["fresh"] == {"isNew": True, "height": "117px"}
    assert out["goner"] == {"leaving": True, "height": "117px"}


def test_equalizer_wakes_once_then_settles(panel: dict[str, Any]) -> None:
    """Only the first load replays the wake-up; returning to idle after a track settles in place."""
    out = panel["deckSettles"]
    assert "has-played" not in out["boot"].split()
    assert "is-playing" in out["playing"].split() and "has-played" in out["playing"].split()
    assert out["stopped"].split()[:2] == ["deck", "is-idle"] and "has-played" in out["stopped"].split()


def test_height_animations_use_the_measured_height() -> None:
    """Entry and exit keyframes interpolate to and from --h, not auto."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for name in ("track-in", "track-out", "toast-in", "toast-out"):
        assert "var(--h)" in css_block(css, f"@keyframes {name} "), name
    assert "animation: none" in css_block(css, ".deck.has-played .eq i")


def test_reduced_motion_covers_dismissed_toasts() -> None:
    """A clicked toast still leaves under reduced motion, as a fade."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    block = css_block(css, "@media (prefers-reduced-motion: reduce)")
    assert re.search(r"\.toast\.is-leaving\s*\{[^}]*toast-out-still[^}]*!important", block)


def test_deck_stacks_before_its_readout_is_squeezed() -> None:
    """Below 1220px the transport moves under the screen so state copy keeps a readable width."""
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    block = css_block(css, "@media (max-width: 1220px)")
    assert re.search(r'\.deck\s*\{[^}]*grid-template-areas:\s*"screen"\s*"transport"\s*"note"', block)
    assert re.search(r"\.deck-detail\s*\{[^}]*min-height", css)


def test_skeleton_rows_mirror_track_rows() -> None:
    """Skeletons use the track grid (title, meta, actions, audio), so rows land without a jump."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    skeletons = re.findall(r'<div class="track-skeleton"[^>]*>((?:<i></i>)+)</div>', html)
    assert len(skeletons) == 3 and all(s.count("<i></i>") == 4 for s in skeletons)
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert '"title actions" "meta actions" "audio audio"' in css_block(css, ".track-skeleton {")
    assert 'id="songs-error"' in html


def test_an_older_library_response_cannot_replace_a_newer_one(panel: dict[str, Any]) -> None:
    """Overlapping refreshes apply only the newest request, whatever order they finish in."""
    assert panel["staleRefresh"] == {"requests": 2, "rows": 3}
