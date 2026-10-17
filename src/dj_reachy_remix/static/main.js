"use strict";
const $ = (id) => document.getElementById(id);
// Resolve relative to the loaded script, preserving a Desktop proxy prefix.
const appBase = new URL("../", document.currentScript.src);
const api = new URL("api/dj-reachy-remix", appBase).href;
const songs = new URL("songs", appBase).href;
let settings;
function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
async function request(path, method = "GET", body) {
  const response = await fetch(path, { method, headers: { "Content-Type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed, HTTP ${response.status}`);
  return data;
}
function showResult(result) {
  status(result.message || result.error || result.status || "Saved.", Boolean(result.error));
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
    if (button) button.disabled = true;
    try { await action(); } catch (error) { status(error.message, true); }
    finally { if (button) button.disabled = false; }
  });
}
async function refreshStatus() {
  const state = await request(`${api}/status`);
  $("connection").textContent = `${state.billing} selected. Provider access ${state.provider_access}. Music key ${state.elevenlabs_key_configured ? "configured" : "missing"}.`;
  $("voice-link").hidden = !state.browser_voice_available;
}
async function refreshSongs() {
  const data = await request(songs);
  const target = $("songs"); target.replaceChildren();
  if (!data.songs.length) { target.textContent = "No saved tracks yet."; return; }
  for (const song of data.songs) {
    const row = document.createElement("article"); row.className = "track";
    const title = document.createElement("h3"); title.textContent = song.title;
    const filename = document.createElement("p"); filename.className = "hint";
    filename.textContent = `${song.filename} · ${song.size}`;
    const controls = document.createElement("div"); controls.className = "actions";
    const replay = document.createElement("button"); replay.textContent = "Play on Reachy";
    replay.addEventListener("click", async () => {
      replay.disabled = true; status("Replaying saved track. Stop remains available.");
      try { showResult(await request(`${api}/music/replay`, "POST", { filename: song.filename })); }
      catch (error) { status(error.message, true); } finally { replay.disabled = false; }
    });
    const remove = document.createElement("button"); remove.textContent = "Delete"; remove.className = "stop";
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Delete saved track ${song.filename}?`)) return;
      try { await request(`${songs}/${encodeURIComponent(song.filename)}`, "DELETE"); await refreshSongs(); status("Saved track deleted."); }
      catch (error) { status(error.message, true); }
    });
    const audio = document.createElement("audio"); audio.controls = true; audio.preload = "none";
    audio.src = `${songs}/${encodeURIComponent(song.filename)}`;
    controls.append(replay, remove); row.append(title, filename, controls, audio); target.append(row);
  }
}
attachForm("voice-form", async () => {
  await save("voice", { backend: $("backend").value, model: $("voice-model").value,
    voice: $("voice-name").value, language: $("conversation-language").value });
  await refreshStatus();
});
attachForm("music-form", () => save("music", musicSettings()));
attachForm("motion-form", () => save("motion", { style: $("motion-style").value, intensity: Number($("intensity").value) }));
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
$("paid-confirmation").addEventListener("change", () => { $("generate").disabled = !$("paid-confirmation").checked; });
$("generate").addEventListener("click", async () => {
  if (!$("paid-confirmation").checked || !$("music-form").reportValidity()) return;
  const music = musicSettings();
  const { duration_ms, ...options } = music;
  options.music_length_ms = duration_ms;
  try {
    if ($("music-mode").value === "plan") options.composition_plan = JSON.parse($("plan").value);
    else options.prompt = $("prompt").value;
    $("generate").disabled = true; $("paid-confirmation").checked = false;
    status("Generating and playing the requested track. Pause and stop remain available.");
    showResult(await request(`${api}/music/generate`, "POST", options));
    await refreshSongs();
  } catch (error) { status(error.message, true); }
  finally { $("generate").disabled = !$("paid-confirmation").checked; }
});
for (const button of document.querySelectorAll("[data-control]")) {
  button.addEventListener("click", async () => {
    try { showResult(await request(`${api}/music/${button.dataset.control}`, "POST", {})); }
    catch (error) { status(error.message, true); }
  });
}
$("refresh").addEventListener("click", () => refreshSongs().catch((error) => status(error.message, true)));
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
    $("finetune").value = music.finetune_id || ""; $("motion-style").value = motion.style; $("intensity").value = motion.intensity;
    $("instructions").value = (await request(`${api}/instructions`)).instructions;
    mode(); await Promise.all([refreshStatus(), refreshSongs()]);
  } catch (error) { $("connection").textContent = "Settings couldn't load. Check the local app server."; status(error.message, true); }
})();
