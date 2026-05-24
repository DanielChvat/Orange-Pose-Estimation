const initial = JSON.parse(document.getElementById("pipelineInitial").textContent);

const stateText = document.getElementById("stateText");
const errorText = document.getElementById("errorText");
const logText = document.getElementById("logText");
const stageGraph = document.getElementById("stageGraph");
const progressDesc = document.getElementById("progressDesc");
const progressCount = document.getElementById("progressCount");
const progressInner = document.getElementById("progressInner");
const overallPill = document.getElementById("overallPill");
const videoInput = document.getElementById("videoInput");
const existingVideo = document.getElementById("existingVideo");
const videoInfo = document.getElementById("videoInfo");
const runButton = document.getElementById("runButton");
const stopButton = document.getElementById("stopButton");
const runForm = document.getElementById("runForm");
const addObject = document.getElementById("addObject");
const objectList = document.getElementById("objectList");
const promptTest = document.getElementById("promptTest");
const testFrameSlider = document.getElementById("testFrameSlider");
const testFrameIndex = document.getElementById("testFrameIndex");
const testFramePreview = document.getElementById("testFramePreview");
const testSamButton = document.getElementById("testSamButton");
const testSamStatus = document.getElementById("testSamStatus");
const settingsButton = document.getElementById("settingsButton");
const settingsDialog = document.getElementById("settingsDialog");
const closeSettings = document.getElementById("closeSettings");
const iterationsInput = document.getElementById("iterationsInput");
const samFpsInput = document.getElementById("samFpsInput");
const reconstructionFpsInput = document.getElementById("reconstructionFpsInput");
const fullImagePassInput = document.getElementById("fullImagePassInput");
const detailPassInput = document.getElementById("detailPassInput");
const tileContextMarginInput = document.getElementById("tileContextMarginInput");
const iterationsHidden = document.getElementById("iterationsHidden");
const samFpsHidden = document.getElementById("samFpsHidden");
const reconstructionFpsHidden = document.getElementById("reconstructionFpsHidden");
const fullImagePassHidden = document.getElementById("fullImagePassHidden");
const detailPassHidden = document.getElementById("detailPassHidden");
const tileContextMarginHidden = document.getElementById("tileContextMarginHidden");
const samEstimate = document.getElementById("samEstimate");
const logTabButton = document.getElementById("logTabButton");
const samTabButton = document.getElementById("samTabButton");
const logPanel = document.getElementById("logPanel");
const samPanel = document.getElementById("samPanel");
const samBasePreview = document.getElementById("samBasePreview");
const samPreview = document.getElementById("samPreview");
const samSummary = document.getElementById("samSummary");
const samBins = document.getElementById("samBins");
const samSourceSelect = document.getElementById("samSourceSelect");
const samFrameSlider = document.getElementById("samFrameSlider");
const samFrameIndex = document.getElementById("samFrameIndex");
const samFrameLabel = document.getElementById("samFrameLabel");
const samPrevFrame = document.getElementById("samPrevFrame");
const samNextFrame = document.getElementById("samNextFrame");
const samLiveFrame = document.getElementById("samLiveFrame");

let inspectedVideo = null;
let testFramePreviewTimer = null;
let testFramePreviewSerial = 0;
let testFramePreviewAbort = null;
let samDisplaySource = "pipeline";
let samManualFrame = false;
let samSelectedFrame = null;
let samFrames = [];
let samRenderSerial = 0;
let lastSamPreviewUrl = "";
let lastSamPreviewKey = "";
let lastSamBinsKey = "";
let currentSteps = initial.state.steps || [];

const STEP_DETAILS = {
  "Cleaning stale outputs": "Removes stale artifacts from previous runs and prepares output folders.",
  "SAM masks": "Runs prompt-conditioned SAM on sampled video frames and writes per-frame mask files.",
  "COLMAP dataset for 3DGS": "Extracts reconstruction frames, estimates camera geometry, and prepares the 3DGS scene.",
  "3DGS": "Optimizes the Gaussian scene representation from the COLMAP cameras and input frames.",
  "Voting SAM evidence onto Gaussians": "Projects masks into the Gaussian scene and accumulates object ownership evidence.",
  "Assigning IDs to foreground splat blobs": "Splits the extracted foreground Gaussians into spatially connected blobs and assigns one object ID per blob.",
  "Fitting OBB + inscribed sphere per object": "For each detected object, runs PCA to get an oriented bounding box and fits an inscribed sphere that hugs the Gaussians.",
  "Building browser viewer": "Exports the in-browser 3DGS viewer with togglable layers + the fitted spheres overlay.",
};

function setText(element, value) {
  element.textContent = value == null ? "" : String(value);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function syncSettingsHidden() {
  iterationsHidden.value = iterationsInput.value;
  samFpsHidden.value = samFpsInput.value;
  reconstructionFpsHidden.value = reconstructionFpsInput.value;
  fullImagePassHidden.value = fullImagePassInput.checked ? "true" : "false";
  detailPassHidden.value = detailPassInput.checked ? "true" : "false";
  tileContextMarginHidden.value = tileContextMarginInput.value || "-1";
}

function settingsFromInitial() {
  const settings = initial.settings || {};
  iterationsInput.value = settings.iterations || 7000;
  samFpsInput.value = settings.sam_fps || 4;
  reconstructionFpsInput.value = settings.reconstruction_fps ?? 6;
  fullImagePassInput.checked = settings.full_image_pass !== false;
  detailPassInput.checked = settings.detail_pass !== false;
  tileContextMarginInput.value = Number.isFinite(Number(settings.tile_context_margin))
    ? settings.tile_context_margin
    : -1;
  syncSettingsHidden();
}

function addObjectRow(value = "") {
  const row = document.createElement("div");
  row.className = "objectRow";
  const input = document.createElement("input");
  input.type = "text";
  input.name = "prompts";
  input.placeholder = "object prompt, e.g. orange fruit";
  input.value = value;
  input.required = true;

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "secondary";
  remove.textContent = "Remove";
  remove.onclick = () => {
    if (objectList.children.length > 1) row.remove();
  };

  row.append(input, remove);
  objectList.append(row);
}

function currentPrompts() {
  return [...objectList.querySelectorAll('input[name="prompts"]')]
    .map((input) => input.value.trim())
    .filter(Boolean);
}

function setRunningDisabled(running) {
  runButton.disabled = running;
  videoInput.disabled = running;
  addObject.disabled = running;
  testSamButton.disabled = running || !Boolean(existingVideo.value);
  testFrameSlider.disabled = running || !Boolean(existingVideo.value);
  testFrameIndex.disabled = running || !Boolean(existingVideo.value);
  stopButton.disabled = !running;
  objectList.querySelectorAll("input, button").forEach((element) => {
    element.disabled = running;
  });
}

function describeVideo(info) {
  const duration = info.duration ? `${info.duration.toFixed(1)}s` : "unknown duration";
  const fps = info.fps ? `${info.fps.toFixed(2)} fps` : "unknown fps";
  return `${info.name} - ${info.width}x${info.height} - ${info.frame_count} frames - ${fps} - ${duration}`;
}

function updateSamEstimate() {
  syncSettingsHidden();
  if (!inspectedVideo || !inspectedVideo.frame_count || !inspectedVideo.fps) {
    samEstimate.textContent = "Choose a video to estimate how many frames SAM will process.";
    return;
  }
  const targetFps = Math.max(0.001, Number(samFpsInput.value || 4));
  const stride = Math.max(1, Math.round(inspectedVideo.fps / targetFps));
  const sampled = Math.ceil(inspectedVideo.frame_count / stride);
  samEstimate.textContent = `${sampled} / ${inspectedVideo.frame_count} frames will be passed to SAM (every ${stride} source frame${stride === 1 ? "" : "s"}).`;
}

function setTestFrameControls(info) {
  const maxFrame = Math.max(0, Number(info.frame_count || 1) - 1);
  promptTest.hidden = false;
  testFrameSlider.max = String(maxFrame);
  testFrameIndex.max = String(maxFrame);
  const mid = Math.floor(maxFrame / 2);
  testFrameSlider.value = String(mid);
  testFrameIndex.value = String(mid);
  updateTestFramePreview(true);
}

function selectedTestFrame() {
  const maxFrame = Number(testFrameSlider.max || 0);
  const value = clamp(Number(testFrameIndex.value || testFrameSlider.value || 0), 0, maxFrame);
  testFrameSlider.value = String(value);
  testFrameIndex.value = String(value);
  return value;
}

async function fetchTestFramePreview(frameIndex, serial) {
  if (!existingVideo.value) return;
  if (testFramePreviewAbort) testFramePreviewAbort.abort();
  testFramePreviewAbort = new AbortController();
  const url = `/video-frame?path=${encodeURIComponent(existingVideo.value)}&frame=${frameIndex}&v=${Date.now()}`;
  try {
    const response = await fetch(url, { cache: "no-store", signal: testFramePreviewAbort.signal });
    if (!response.ok) throw new Error(response.statusText);
    const blob = await response.blob();
    if (serial !== testFramePreviewSerial) return;
    const objectUrl = URL.createObjectURL(blob);
    const previous = testFramePreview.dataset.objectUrl;
    testFramePreview.onload = () => {
      if (previous) URL.revokeObjectURL(previous);
    };
    testFramePreview.dataset.objectUrl = objectUrl;
    testFramePreview.src = objectUrl;
  } catch (err) {
    if (err.name !== "AbortError") console.warn("Frame preview failed", err);
  }
}

function updateTestFramePreview(immediate = false) {
  if (!existingVideo.value) return;
  const value = selectedTestFrame();
  const serial = ++testFramePreviewSerial;
  window.clearTimeout(testFramePreviewTimer);
  const run = () => fetchTestFramePreview(value, serial);
  if (immediate) run();
  else testFramePreviewTimer = window.setTimeout(run, 60);
}

function formDataWithSettings() {
  syncSettingsHidden();
  const data = new FormData();
  data.set("existing_video", existingVideo.value || "");
  data.set("iterations", iterationsHidden.value);
  data.set("sam_fps", samFpsHidden.value);
  data.set("reconstruction_fps", reconstructionFpsHidden.value);
  data.set("full_image_pass", fullImagePassHidden.value);
  data.set("detail_pass", detailPassHidden.value);
  data.set("tile_context_margin", tileContextMarginHidden.value);
  for (const prompt of currentPrompts()) data.append("prompts", prompt);
  return data;
}

function stageTitle(step) {
  return String(step || "").replace(/^\[\d+\/\d+\]\s*/, "");
}

function stageNumber(step) {
  const match = String(step || "").match(/^\[(\d+)\/\d+\]/);
  return match ? Number(match[1]) : null;
}

function activeStepIndex(state) {
  if (state.current_step === "[DONE]" || state.returncode === 0) return currentSteps.length;
  const current = String(state.current_step || "");
  if (!current) return -1;

  const currentNumber = stageNumber(current);
  if (currentNumber !== null) {
    const numericIndex = currentSteps.findIndex((step) => stageNumber(step) === currentNumber);
    if (numericIndex >= 0) return numericIndex;
  }

  const currentTitle = stageTitle(current);
  const titleIndex = currentSteps.findIndex((step) => stageTitle(step) === currentTitle);
  if (titleIndex >= 0) return titleIndex;

  return currentSteps.findIndex((step) => current.startsWith(step) || current.includes(stageTitle(step)));
}

function stageStatus(state, idx, active) {
  if (active < 0) return "Queued";
  if (state.returncode && idx === active) return "Failed";
  if (idx < active || state.returncode === 0) return "Done";
  if (idx === active && state.running) return "Working";
  if (idx === active && state.current_step) return "Paused";
  return "Queued";
}

function stageProgressText(state, idx, active) {
  const progress = state.progress || {};
  if (idx !== active || !progress.total) return "";
  return `${progress.current || 0} / ${progress.total} ${progress.unit || ""}`.trim();
}

function renderStageGraph(state) {
  currentSteps = state.steps || currentSteps || [];
  const active = activeStepIndex(state);
  stageGraph.replaceChildren();
  currentSteps.forEach((step, idx) => {
    const nameText = stageTitle(step);
    const status = stageStatus(state, idx, active);
    const card = document.createElement("div");
    card.className = "stageStep";
    card.tabIndex = 0;
    card.classList.add(status.toLowerCase());
    if (idx === active) card.classList.add("active");

    const marker = document.createElement("span");
    marker.className = "stageMarker";

    const body = document.createElement("div");
    body.className = "stageBody";
    const name = document.createElement("div");
    name.className = "stageName";
    name.textContent = nameText;
    const detail = document.createElement("div");
    detail.className = "stageDetail";
    detail.textContent = STEP_DETAILS[nameText] || "Pipeline stage";
    body.append(name, detail);

    const badge = document.createElement("span");
    badge.className = "stageBadge";
    badge.textContent = status;
    card.append(marker, body, badge);

    const popover = document.createElement("aside");
    popover.className = "stagePopover";
    const progressText = stageProgressText(state, idx, active);
    const nextStep = currentSteps[idx + 1] ? stageTitle(currentSteps[idx + 1]) : "Complete";
    popover.innerHTML = `
      <span>${idx === active ? "Current step" : "Pipeline step"}</span>
      <strong>${nameText}</strong>
      <em>${status}</em>
      <p>${STEP_DETAILS[nameText] || "Pipeline stage"}</p>
      ${progressText ? `<div class="popoverProgress"><small>Stage progress</small><b>${progressText}</b></div>` : ""}
      <small>Next: ${nextStep}</small>
    `;
    card.append(popover);
    stageGraph.append(card);
  });
}

function renderProgress(progress) {
  if (!progress) {
    progressDesc.textContent = "No active progress";
    progressCount.textContent = "";
    overallPill.textContent = "Overall";
    progressInner.style.width = "0%";
    return;
  }
  const current = Number(progress.current || 0);
  const total = Number(progress.total || 0);
  const percent = total > 0 ? clamp((current / total) * 100, 0, 100) : 0;
  progressDesc.textContent = progress.stage || "Working";
  progressCount.textContent = total > 0 ? `${current} / ${total} ${progress.unit || ""}` : "";
  overallPill.textContent = total > 0 ? `${current} / ${total} ${progress.unit || ""}` : "Working";
  progressInner.style.width = `${percent}%`;
}

function setActivePanel(name) {
  const samActive = name === "sam";
  samPanel.hidden = !samActive;
  logPanel.hidden = samActive;
  samTabButton.classList.toggle("active", samActive);
  logTabButton.classList.toggle("active", !samActive);
}

function samFrameIndexOf(frame) {
  return samFrames.findIndex((item) => item.frame === frame);
}

function setSamFrameByIndex(index) {
  if (!samFrames.length) return;
  const next = samFrames[clamp(index, 0, samFrames.length - 1)];
  samManualFrame = true;
  samSelectedFrame = next.frame;
  pollSamStatus();
}

function setSamControls(payload) {
  samFrames = payload.frames || payload.recent_frames || [];
  const maxIndex = Math.max(0, samFrames.length - 1);
  samFrameSlider.max = String(maxIndex);
  samFrameIndex.max = String(maxIndex);

  if (!samManualFrame || !samSelectedFrame) {
    samSelectedFrame = payload.latest_frame || (samFrames.length ? samFrames[maxIndex].frame : null);
  }

  let idx = samFrameIndexOf(samSelectedFrame);
  if (idx < 0 && samFrames.length) {
    idx = maxIndex;
    samSelectedFrame = samFrames[idx].frame;
  }
  idx = Math.max(0, idx);
  samFrameSlider.value = String(idx);
  samFrameIndex.value = String(idx);
  samFrameSlider.disabled = samFrames.length < 2;
  samFrameIndex.disabled = samFrames.length < 2;
  samPrevFrame.disabled = samFrames.length < 2 || idx <= 0;
  samNextFrame.disabled = samFrames.length < 2 || idx >= maxIndex;
  samFrameLabel.textContent = samSelectedFrame
    ? `${samSelectedFrame} (${idx + 1} / ${samFrames.length || 1})`
    : "No frames yet";
}

function stableRecordSignature(records) {
  return (records || []).map((record) => [
    record.prompt || "",
    record.local_id || 0,
    Math.round(Number(record.area || 0)),
    Number(record.score || 0).toFixed(3),
    (record.bbox || []).map((value) => Math.round(Number(value || 0))).join(":"),
  ].join("|")).join(";");
}

function hashString(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function stablePayloadKey(payload) {
  const raw = [
    payload.source || "",
    payload.latest_frame || "",
    stableRecordSignature(payload.latest_objects || []),
  ].join("::");
  return hashString(raw);
}

function preloadSamPreview(baseUrl, overlayUrl, key, onLoaded = null) {
  if (key === lastSamPreviewKey && samPreview.getAttribute("src") && samBasePreview.getAttribute("src")) {
    return false;
  }
  const serial = ++samRenderSerial;
  const baseProbe = new Image();
  const overlayProbe = new Image();
  let baseLoaded = false;
  let overlayLoaded = false;
  const commit = () => {
    if (!baseLoaded || !overlayLoaded) return;
    if (serial !== samRenderSerial) return;
    lastSamPreviewKey = key;
    lastSamPreviewUrl = overlayUrl;
    samBasePreview.src = baseUrl;
    samPreview.src = overlayUrl;
    onLoaded?.();
  };
  baseProbe.onload = () => {
    baseLoaded = true;
    commit();
  };
  overlayProbe.onload = () => {
    overlayLoaded = true;
    commit();
  };
  const fail = () => {
    if (serial !== samRenderSerial) return;
    if (!lastSamPreviewUrl) {
      samBasePreview.removeAttribute("src");
      samPreview.removeAttribute("src");
    }
  };
  baseProbe.onerror = fail;
  overlayProbe.onerror = fail;
  baseProbe.src = baseUrl;
  overlayProbe.src = overlayUrl;
  return true;
}

function recordsByPrompt(records) {
  const grouped = new Map();
  for (const record of records || []) {
    const prompt = record.prompt || "object";
    if (!grouped.has(prompt)) grouped.set(prompt, []);
    grouped.get(prompt).push(record);
  }
  return grouped;
}

function renderSamBins(payload) {
  const key = stablePayloadKey(payload);
  if (key === lastSamBinsKey) return;
  lastSamBinsKey = key;
  const grouped = recordsByPrompt(payload.latest_objects || []);
  const prompts = payload.prompts && payload.prompts.length
    ? payload.prompts
    : [...grouped.keys()].sort();
  samBins.replaceChildren();

  for (const prompt of prompts) {
    const records = grouped.get(prompt) || [];
    const bin = document.createElement("section");
    bin.className = "classBin";
    const header = document.createElement("div");
    header.className = "classHeader";
    const title = document.createElement("strong");
    title.textContent = prompt;
    const count = document.createElement("span");
    const total = (payload.total_counts || {})[prompt] || 0;
    count.textContent = `${records.length} in frame - ${total} total`;
    header.append(title, count);
    bin.append(header);

    if (!records.length) {
      const empty = document.createElement("div");
      empty.className = "emptyBin";
      empty.textContent = "No mask for this class in the selected frame.";
      bin.append(empty);
    } else {
      const grid = document.createElement("div");
      grid.className = "cutoutGrid";
      for (const record of records) {
        const card = document.createElement("div");
        card.className = "cutoutCard";
        const img = document.createElement("img");
        img.alt = `${prompt} cutout`;
        img.loading = "lazy";
        img.src = `/sam-cutout?source=${encodeURIComponent(payload.source)}&frame=${encodeURIComponent(payload.latest_frame)}&id=${record.local_id}&sig=${encodeURIComponent(key)}`;
        const meta = document.createElement("div");
        meta.className = "cutoutMeta";
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        const color = record.color || [160, 160, 160];
        swatch.style.background = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        const score = document.createElement("span");
        score.textContent = typeof record.score === "number" ? record.score.toFixed(2) : "mask";
        meta.append(swatch, score);
        card.append(img, meta);
        grid.append(card);
      }
      bin.append(grid);
    }
    samBins.append(bin);
  }
}

async function renderSamState(payload) {
  setSamControls(payload);
  const frame = payload.latest_frame;
  if (!frame) {
    samSummary.textContent = "SAM output will appear once masks are being written.";
    samBasePreview.removeAttribute("src");
    samPreview.removeAttribute("src");
    lastSamPreviewUrl = "";
    lastSamPreviewKey = "";
    renderSamBins(payload);
    return;
  }
  samSummary.textContent = `${payload.source} masks - ${payload.processed_frames || 0} frame${payload.processed_frames === 1 ? "" : "s"} available`;
  const key = stablePayloadKey(payload);
  const query = `source=${encodeURIComponent(payload.source)}&frame=${encodeURIComponent(frame)}&sig=${encodeURIComponent(key)}`;
  const baseUrl = `/sam-frame-preview?${query}`;
  const overlayUrl = `/sam-preview?${query}`;
  const loadingNewPreview = preloadSamPreview(baseUrl, overlayUrl, key, () => renderSamBins(payload));
  if (!loadingNewPreview) renderSamBins(payload);
}

async function pollSamStatus() {
  const params = new URLSearchParams({ source: samDisplaySource });
  if (samManualFrame && samSelectedFrame) params.set("frame", samSelectedFrame);
  try {
    const response = await fetch(`/sam-status.json?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    await renderSamState(payload);
  } catch (err) {
    console.warn("SAM polling failed", err);
  }
}

async function pollStatus() {
  try {
    const response = await fetch("/status.json", { cache: "no-store" });
    const state = await response.json();
    stateText.textContent = state.label;
    errorText.textContent = state.error || "";
    logText.textContent = state.log || "";
    logText.scrollTop = logText.scrollHeight;
    renderStageGraph(state);
    renderProgress(state.progress);
    setRunningDisabled(Boolean(state.running));
  } catch (err) {
    console.warn("Status polling failed", err);
  }
}

settingsFromInitial();
for (const prompt of initial.prompts && initial.prompts.length ? initial.prompts : ["orange fruit"]) {
  addObjectRow(prompt);
}
addObject.onclick = () => addObjectRow("");
if (initial.state.last_video) {
  existingVideo.value = initial.state.last_video;
  videoInfo.textContent = `Selected: ${initial.state.last_video.split("/").pop()}`;
}
setRunningDisabled(Boolean(initial.state.running));
renderStageGraph(initial.state);
renderProgress(initial.state.progress);
setText(stateText, initial.state.label || "Idle");
setText(errorText, initial.state.error || "");
setText(logText, initial.state.log || "");

logTabButton.onclick = () => setActivePanel("log");
samTabButton.onclick = () => {
  setActivePanel("sam");
  pollSamStatus();
};
settingsButton.onclick = () => settingsDialog.showModal();
closeSettings.onclick = () => settingsDialog.close();

iterationsInput.addEventListener("input", updateSamEstimate);
samFpsInput.addEventListener("input", updateSamEstimate);
reconstructionFpsInput.addEventListener("input", syncSettingsHidden);
fullImagePassInput.addEventListener("change", syncSettingsHidden);
detailPassInput.addEventListener("change", syncSettingsHidden);
tileContextMarginInput.addEventListener("input", syncSettingsHidden);
updateSamEstimate();

videoInput.addEventListener("change", async () => {
  if (!videoInput.files.length) return;
  const data = new FormData();
  data.append("video", videoInput.files[0]);
  videoInfo.textContent = "Inspecting video...";
  runButton.disabled = true;
  testSamButton.disabled = true;
  try {
    const response = await fetch("/inspect-video", { method: "POST", body: data });
    if (!response.ok) throw new Error(await response.text());
    inspectedVideo = await response.json();
    existingVideo.value = inspectedVideo.path;
    videoInfo.textContent = describeVideo(inspectedVideo);
    setTestFrameControls(inspectedVideo);
    updateSamEstimate();
    runButton.disabled = false;
    testSamButton.disabled = false;
  } catch (err) {
    videoInfo.textContent = `Video inspection failed: ${err.message}`;
    existingVideo.value = "";
  }
});

testFrameSlider.addEventListener("input", () => {
  testFrameIndex.value = testFrameSlider.value;
  updateTestFramePreview(false);
});
testFrameIndex.addEventListener("input", () => {
  testFrameSlider.value = testFrameIndex.value;
  updateTestFramePreview(false);
});

testSamButton.onclick = async () => {
  if (!currentPrompts().length) {
    testSamStatus.textContent = "Add at least one prompt first.";
    return;
  }
  const data = formDataWithSettings();
  data.set("frame_index", String(selectedTestFrame()));
  testSamButton.disabled = true;
  testSamStatus.textContent = "Running SAM on selected frame...";
  try {
    const response = await fetch("/test-sam", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.log || "Prompt test failed");
    testSamStatus.textContent = "Prompt test complete.";
    samDisplaySource = "test";
    samSourceSelect.value = "test";
    samManualFrame = false;
    lastSamPreviewKey = "";
    lastSamBinsKey = "";
    setActivePanel("sam");
    await renderSamState(payload);
  } catch (err) {
    testSamStatus.textContent = `Prompt test failed: ${String(err.message || err).slice(0, 180)}`;
  } finally {
    testSamButton.disabled = false;
  }
};

runForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!existingVideo.value && !videoInput.files.length) {
    videoInfo.textContent = "Choose a video first.";
    return;
  }
  const data = formDataWithSettings();
  if (videoInput.files.length) data.set("video", videoInput.files[0]);
  runButton.disabled = true;
  try {
    const response = await fetch("/run", { method: "POST", body: data });
    if (!response.ok) throw new Error(await response.text());
    samDisplaySource = "pipeline";
    samSourceSelect.value = "pipeline";
    samManualFrame = false;
    lastSamPreviewKey = "";
    lastSamBinsKey = "";
    setActivePanel("sam");
    await pollStatus();
  } catch (err) {
    errorText.textContent = `Could not start pipeline: ${err.message}`;
  }
});

stopButton.onclick = async () => {
  stopButton.disabled = true;
  try {
    await fetch("/stop", { method: "POST" });
  } finally {
    await pollStatus();
  }
};

samSourceSelect.addEventListener("change", () => {
  samDisplaySource = samSourceSelect.value;
  samManualFrame = false;
  samSelectedFrame = null;
  lastSamPreviewKey = "";
  lastSamBinsKey = "";
  pollSamStatus();
});
samFrameSlider.addEventListener("input", () => setSamFrameByIndex(Number(samFrameSlider.value || 0)));
samFrameIndex.addEventListener("input", () => setSamFrameByIndex(Number(samFrameIndex.value || 0)));
samPrevFrame.onclick = () => setSamFrameByIndex(Number(samFrameSlider.value || 0) - 1);
samNextFrame.onclick = () => setSamFrameByIndex(Number(samFrameSlider.value || 0) + 1);
samLiveFrame.onclick = () => {
  samManualFrame = false;
  samSelectedFrame = null;
  lastSamPreviewKey = "";
  lastSamBinsKey = "";
  pollSamStatus();
};

setInterval(pollStatus, 2000);
setInterval(pollSamStatus, 1200);
pollStatus();
pollSamStatus();
