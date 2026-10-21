"use strict";
const $ = (id) => document.getElementById(id);
// Resolve relative to the loaded script, preserving a Desktop proxy prefix.
const appBase = new URL("../", document.currentScript.src);
const api = new URL("api/dj-reachy-remix", appBase).href;
const songs = new URL("songs", appBase).href;
let settings;

// Feedback. #status is the screen-reader announcement; toasts are its visual
// twin and are removed by their CSS exit animation, so no timers are needed.
function toast(text, error) {
  const item = document.createElement("div");
  item.className = error ? "toast is-error" : "toast";
  item.textContent = text;
  item.addEventListener("animationend", (event) => { if (event.animationName.startsWith("toast-out")) item.remove(); });
  item.addEventListener("click", () => item.remove());
  $("toasts").append(item);
}
function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
  toast(text, error);
}
async function request(path, method = "GET", body) {
  const response = await fetch(path, { method, headers: { "Content-Type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed, HTTP ${response.status}`);
  return data;
}
const resultText = {
  paused: "Paused.", resumed: "Resumed.", stopped: "Stopped.", completed: "Track finished.",
  not_playing: "Nothing is playing right now.",
};
function showResult(result) {
  status(result.message || result.error || resultText[result.status] || result.status || "Saved.", Boolean(result.error));
}

// The deck shows what this panel last asked for. Generation and replay requests
// stay open until the track ends or stops, so the open request is the source of
// truth. A token stops an older request from resetting a newer one's state.
const deckTitles = { idle: "Ready", generating: "Generating", playing: "Playing", paused: "Paused", stopping: "Stopping" };
let deckOwner = 0;
function deck(state, detail) {
  $("deck").className = `deck is-${state}`;
  $("deck-state").textContent = deckTitles[state];
  $("deck-detail").textContent = detail;
}
function finished(token, detail) {
  if (token === deckOwner) deck("idle", detail);
}
const controlStates = {
  paused: ["paused", "Playback is held. Resume picks up where it stopped."],
  resumed: ["playing", "Back on. Pause and stop stay available."],
  stopped: ["idle", "Local playback stopped and queued audio cleared. Provider generation may continue."],
  cancellation_requested: ["stopping", "Local result will be discarded. Provider work may continue."],
};
function busy(button, on) {
  if (!button) return;
  button.disabled = on;
  button.classList.toggle("is-busy", on);
}

function styles(id) { return $(id).value.split(",").map((s) => s.trim()).filter(Boolean); }
function musicSettings() {
  return { model_id: $("music-model").value, duration_ms: Math.round(Number($("duration").value) * 1000),
    output_format: $("output-format").value, positive_styles: styles("positive-styles"),
    negative_styles: styles("negative-styles"), lyrics: $("lyrics").value,
    language: $("lyric-language").value || "auto", force_instrumental: $("instrumental").checked,
    finetune_id: $("finetune").value.trim() || null };
}
async function save(group, values) {
  settings = await request(`${api}/settings`, "PUT", { [group]: values });
  status(group === "voice" ? "Conversation settings saved. Disconnect and reconnect voice to apply them. No provider access has been verified." : "Settings saved and applied.");
}
function mode() {
  const plan = $("music-mode").value === "plan";
  $("plan-field").hidden = !plan; $("prompt-field").hidden = plan;
  $("instrumental").disabled = plan;
  if (plan) $("instrumental").checked = false;
}
function attachForm(id, action) {
  $(id).addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.submitter;
    busy(button, true);
    try { await action(); } catch (error) { status(error.message, true); }
    finally { busy(button, false); }
  });
}

// Motion keeps the API's 0-0.25 float but shows upstream DJ Reachy's 0-100
// slider (value / 400), where 80 is the original 0.20 default.
const ORIGINAL_INTENSITY = 0.2;
const styleHints = {
  groove: "The original DJ Reachy dance.",
  bounce: "Bigger head and antenna moves.",
  classic: "Slightly calmer than groove.",
  smooth: "About half the motion of groove.",
};
const toSlider = (intensity) => String(Math.round(intensity * 400));
function showIntensity() {
  const slider = Number($("intensity").value);
  const share = Math.round((slider / 400 / ORIGINAL_INTENSITY) * 100);
  $("intensity").style?.setProperty("--pos", String(slider)); // fader fill, cosmetic only
  $("intensity-value").textContent = slider === 80
    ? "80 of 100, the original DJ Reachy level"
    : `${slider} of 100, ${share}% of the original level`;
}
function showStyle() { $("motion-style-hint").textContent = styleHints[$("motion-style").value] || ""; }
function motionSettings() { return { style: $("motion-style").value, intensity: Number($("intensity").value) / 400 }; }

function chip(text, tone) {
  const item = document.createElement("span");
  item.className = `chip is-${tone}`;
  item.textContent = text;
  return item;
}
async function refreshStatus() {
  const state = await request(`${api}/status`);
  const key = Boolean(state.elevenlabs_key_configured);
  $("connection").replaceChildren();
  $("connection").append(
    chip(`${state.billing} selected`, "info"),
    chip(`Provider access ${state.provider_access}`, state.provider_access === "verified" ? "ok" : "muted"),
    chip(`Music key ${key ? "configured" : "missing"}`, key ? "ok" : "warn"));
  $("voice-link").hidden = !state.browser_voice_available;
}

// Rows are keyed by filename so a refresh keeps existing rows (and any preview
// in progress), animates new rows in and lets deleted rows play an exit first.
const rows = new Map();
let listed = false;
function replay(song, row, button) {
  return async () => {
    const token = ++deckOwner;
    busy(button, true); row.classList.toggle("is-playing", true);
    deck("playing", `${song.title}, from your library. No regeneration.`);
    status("Replaying saved track. Stop remains available.");
    try { const result = await request(`${api}/music/replay`, "POST", { filename: song.filename }); showResult(result); finished(token, "Track finished or stopped.");
    } catch (error) { status(error.message, true); finished(token, "Replay didn't start. Check the message and try again.");
    } finally { busy(button, false); row.classList.toggle("is-playing", false); }
  };
}
function trackRow(song) {
  const row = document.createElement("article"); row.className = "track";
  const title = document.createElement("h3"); title.textContent = song.title;
  const filename = document.createElement("p"); filename.className = "track-meta";
  filename.textContent = `${song.filename}, ${song.size}`;
  const controls = document.createElement("div"); controls.className = "actions";
  const play = document.createElement("button"); play.textContent = "Play on Reachy"; play.className = "small";
  play.addEventListener("click", replay(song, row, play));
  const remove = document.createElement("button"); remove.textContent = "Delete"; remove.className = "small danger";
  remove.addEventListener("click", async () => {
    if (!window.confirm(`Delete saved track ${song.filename}?`)) return;
    busy(remove, true);
    try { await request(`${songs}/${encodeURIComponent(song.filename)}`, "DELETE"); await refreshSongs(); status("Saved track deleted."); }
    catch (error) { status(error.message, true); }
    finally { busy(remove, false); }
  });
  const audio = document.createElement("audio"); audio.controls = true; audio.preload = "none";
  audio.src = `${songs}/${encodeURIComponent(song.filename)}`;
  row.addEventListener("animationend", (event) => {
    if (event.animationName === "track-out") row.remove();
    if (event.animationName === "track-in") row.classList.toggle("is-new", false);
  });
  controls.append(play, remove); row.append(title, filename, controls, audio);
  return row;
}
async function refreshSongs() {
  const data = await request(songs);
  const target = $("songs");
  const present = new Set(data.songs.map((song) => song.filename));
  for (const [name, row] of rows) {
    if (present.has(name)) continue;
    rows.delete(name);
    row.inert = true;
    row.classList.toggle("is-leaving", true);
  }
  let inserted = false;
  const next = data.songs.map((song) => {
    let row = rows.get(song.filename);
    if (!row) {
      row = trackRow(song); rows.set(song.filename, row); inserted = true;
      row.classList.toggle("is-new", listed);
    }
    return row;
  });
  if (!listed || inserted) { target.replaceChildren(); target.append(...next); }
  listed = true;
  $("songs-empty").hidden = data.songs.length > 0;
}

attachForm("voice-form", async () => {
  await save("voice", { backend: $("backend").value, model: $("voice-model").value,
    voice: $("voice-name").value, language: $("conversation-language").value });
  await refreshStatus();
});
attachForm("music-form", () => save("music", musicSettings()));
attachForm("motion-form", () => save("motion", motionSettings()));
attachForm("instructions-form", async () => {
  await request(`${api}/instructions`, "PUT", { instructions: $("instructions").value });
  status("Instructions saved. Disconnect and reconnect voice to reload them.");
});
attachForm("credential-form", async () => {
  try {
    await request(`${api}/credentials`, "POST", { provider: $("credential-provider").value, key: $("credential").value });
    status("Private key saved. Backend selection hasn't changed. Reconnect voice if you changed its key.");
    await refreshStatus();
  } finally { $("credential").value = ""; }
});
$("backend").addEventListener("change", () => {
  const subscription = $("backend").value === "subscription";
  $("voice-model").value = subscription ? "gpt-live-1-codex" : "gpt-realtime";
  $("voice-name").value = subscription ? "cove" : "cedar";
});
$("music-mode").addEventListener("change", mode);
$("intensity").addEventListener("input", showIntensity);
$("motion-style").addEventListener("change", showStyle);
$("motion-reset").addEventListener("click", async () => {
  $("motion-style").value = "groove"; $("intensity").value = toSlider(ORIGINAL_INTENSITY);
  showStyle(); showIntensity();
  busy($("motion-reset"), true);
  try {
    settings = await request(`${api}/settings`, "PUT", { motion: { style: "groove", intensity: ORIGINAL_INTENSITY } });
    status("Motion reset to the original DJ Reachy groove and level.");
  } catch (error) { status(error.message, true); }
  finally { busy($("motion-reset"), false); }
});
$("paid-confirmation").addEventListener("change", () => { $("generate").disabled = !$("paid-confirmation").checked; });
$("generate").addEventListener("click", async () => {
  if (!$("paid-confirmation").checked || !$("music-form").reportValidity()) return;
  const music = musicSettings();
  const { duration_ms, ...options } = music;
  options.music_length_ms = duration_ms;
  let token = 0;
  try {
    if ($("music-mode").value === "plan") options.composition_plan = JSON.parse($("plan").value);
    else options.prompt = $("prompt").value;
    $("generate").disabled = true; $("paid-confirmation").checked = false;
    $("generate").classList.toggle("is-busy", true); $("generate").textContent = "Generating...";
    token = ++deckOwner;
    deck("generating", "Eleven Music is composing. The track plays on Reachy as soon as it arrives.");
    status("Generating and playing the requested track. Pause and stop remain available.");
    const result = await request(`${api}/music/generate`, "POST", options);
    showResult(result);
    finished(token, result.status === "completed" ? "Track finished. It's saved in your library." : "Generation ended. Check the latest message.");
    await refreshSongs();
  } catch (error) {
    status(error.message, true);
    if (token) finished(token, "Generation didn't finish. Check the message and try again.");
  } finally {
    $("generate").classList.toggle("is-busy", false); $("generate").textContent = "Generate and play new track";
    $("generate").disabled = !$("paid-confirmation").checked;
  }
});
for (const button of document.querySelectorAll("[data-control]")) {
  button.addEventListener("click", async () => {
    button.classList.toggle("is-busy", true);
    try {
      const result = await request(`${api}/music/${button.dataset.control}`, "POST", {});
      showResult(result);
      const next = controlStates[result.status];
      if (next) {
        if (next[0] === "idle") deckOwner += 1; // the open request no longer owns the deck
        deck(next[0], next[1]);
      }
    } catch (error) { status(error.message, true); }
    finally { button.classList.toggle("is-busy", false); }
  });
}
$("refresh").addEventListener("click", async () => {
  busy($("refresh"), true);
  try { await refreshSongs(); } catch (error) { status(error.message, true); }
  finally { busy($("refresh"), false); }
});
(async () => {
  try {
    settings = await request(`${api}/settings`);
    const voice = settings.voice, music = settings.music, motion = settings.motion;
    $("backend").value = voice.backend; $("voice-model").value = voice.model; $("voice-name").value = voice.voice;
    $("conversation-language").value = voice.language;
    $("music-model").value = music.model_id; $("duration").value = music.duration_ms / 1000;
    $("output-format").value = music.output_format; $("positive-styles").value = music.positive_styles.join(", ");
    $("negative-styles").value = music.negative_styles.join(", "); $("lyrics").value = music.lyrics;
    $("lyric-language").value = music.language; $("instrumental").checked = music.force_instrumental;
    $("finetune").value = music.finetune_id || ""; $("motion-style").value = motion.style; $("intensity").value = toSlider(motion.intensity);
    showStyle(); showIntensity();
    $("instructions").value = (await request(`${api}/instructions`)).instructions;
    mode(); deck("idle", "Generate a new track or play one from your library.");
    $("main").classList.toggle("is-loading", false);
    await Promise.all([refreshStatus(), refreshSongs()]);
  } catch (error) {
    $("main").classList.toggle("is-loading", false);
    $("connection").replaceChildren();
    $("connection").append(chip("Settings couldn't load. Check the local app server.", "error"));
    status(error.message, true);
  }
})();
