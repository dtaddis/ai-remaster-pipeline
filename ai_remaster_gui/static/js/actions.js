async function postJson(path, payload) {
  return await api(path, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

async function redrawWithState(nextState, snap, forceSignature = false) {
  state = nextState || await api(stateUrl());
  pruneSelected();
  draw(false);
  if (forceSignature) lastRenderSignature = renderSignature();
  restoreScrollState(snap);
}

async function scrubShot(manifest, index, time) {
  const snap = captureScrollState();
  const result = await postJson('/api/shot-scrub', { manifest, index, time });
  if (!result.ok) return alert(result.error || 'Could not update shot frame');

  await redrawWithState(null, snap);
}

async function saveShotPrompt(manifest, index, prompt) {
  const result = await postJson('/api/shot-prompt', { manifest, index, prompt });
  if (!result.ok) return alert(result.error || 'Could not save prompt');

  state = await api(stateUrl());
}

async function saveOutpaintChunk(index) {
  const snap = captureScrollState();
  const payload = outpaintChunkForm(index);
  const result = await postJson('/api/outpaint-chunk', payload);
  if (!result.ok) return alert(result.error || 'Could not save chunk');

  if (result.state) {
    state = result.state;
    draw(false);
    lastRenderSignature = renderSignature();
    lastOutpaintVisualSignature = outpaintVisualSignature();
    restoreScrollState(snap);
    return;
  }

  await redrawWithState(result.state, snap);
}

function outpaintChunkForm(index) {
  const customCheckbox = document.getElementById(`chunkCustom_${index}`);
  return {
    index,
    seed: document.getElementById(`chunkSeed_${index}`).value,
    custom_length: !!(customCheckbox && customCheckbox.checked),
    custom_seconds: outpaintChunkCustomSeconds(index),
    prompt_suffix: document.getElementById(`chunkPrompt_${index}`).value,
    negative_suffix: document.getElementById(`chunkNegative_${index}`).value,
  };
}

function openImageModal(src, title) {
  let modal = document.getElementById('imageModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'imageModal';
    modal.className = 'image-modal hidden';
    modal.innerHTML = `
      <div class="image-modal-backdrop" onclick="closeImageModal()"></div>
      <div class="image-modal-panel">
        <div class="image-modal-heading">
          <strong id="imageModalTitle"></strong>
          <button type="button" onclick="closeImageModal()" aria-label="Close image preview">Close</button>
        </div>
        <img id="imageModalImg" alt="">
      </div>
    `;
    document.body.appendChild(modal);
  }

  document.getElementById('imageModalTitle').textContent = title || 'Image preview';
  document.getElementById('imageModalImg').src = src;
  modal.classList.remove('hidden');
}

function closeImageModal() {
  const modal = document.getElementById('imageModal');
  if (!modal) return;
  modal.classList.add('hidden');
  const img = document.getElementById('imageModalImg');
  if (img) img.removeAttribute('src');
}

function outpaintChunkCustomSeconds(index) {
  const checkbox = document.getElementById(`chunkCustom_${index}`);
  const slider = document.getElementById(`chunkFrames_${index}`);
  if (!(checkbox && checkbox.checked && slider)) return '';
  const fps = Math.max(1, Number(slider.dataset.fps || 24));
  return (Math.max(1, Number(slider.value) || 1) / fps).toFixed(6);
}

function releaseChunkMedia(index) {
  const card = document.querySelectorAll('.chunk-card')[index];
  if (!card) return;

  card.querySelectorAll('video').forEach(video => {
    try {
      video.pause();
      video.removeAttribute('src');
      video.load();
    } catch {}
  });
}

async function regenerateOutpaintChunk(index) {
  releaseChunkMedia(index);

  const result = await postJson('/api/outpaint-chunk-regenerate', outpaintChunkForm(index));
  if (!result.ok) return alert(result.error || result.message || 'Could not regenerate chunk');

  state = result.state;
  draw(false);
  setTimeout(() => refresh(true), 500);
}

async function saveShotEnabled(manifest, index, enabled) {
  const snap = captureScrollState();
  const result = await postJson('/api/shot-enabled', { manifest, index, enabled });
  if (!result.ok) return alert(result.error || 'Could not save shot setting');

  await redrawWithState(null, snap);
}

async function mergeShot(manifest, index) {
  if (!confirm('Merge this shot with the next one and use the same reference?')) return;

  const snap = captureScrollState();
  const result = await postJson('/api/shot-merge', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not merge shots');

  await redrawWithState(result.state, snap, true);
}

async function splitShot(manifest, index) {
  if (!confirm('Split this shot at its midpoint? Existing generated references for the two halves will be cleared.')) return;

  const snap = captureScrollState();
  const result = await postJson('/api/shot-split', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not split shot');

  await redrawWithState(result.state, snap, true);
}

async function setShotBoundary(manifest, index, edge, time) {
  const snap = captureScrollState();
  const result = await postJson('/api/shot-boundary', { manifest, index, edge, time });
  if (!result.ok) return alert(result.error || 'Could not update shot boundary');

  await redrawWithState(result.state, snap, true);
}

async function saveShotFade(manifest, index, enabled, crossfade_seconds) {
  const snap = captureScrollState();
  const result = await postJson('/api/shot-fade', { manifest, index, enabled, crossfade_seconds });
  if (!result.ok) return alert(result.error || 'Could not update fade transition');

  await redrawWithState(result.state, snap, true);
}

function nudgeShotBoundary(manifest, index, edge, frames) {
  const rows = (state.shot_views && state.shot_views.shots) || [];
  const row = rows[index];
  if (!row) return;

  const frameCount = Number(row.end_frame) - Number(row.start_frame) + 1;
  const duration = Math.max(0.001, Number(row.end) - Number(row.start));
  const fps = Math.max(1, frameCount / duration);
  const base = edge === 'start' ? Number(row.start) : Number(row.end);
  setShotBoundary(manifest, index, edge, base + (Number(frames) || 0) / fps);
}

const previewTimers = {};

function updateShotPreview(manifest, index, time, imgId, labelId) {
  document.getElementById(labelId).textContent = formatSeconds(time);
  clearTimeout(previewTimers[imgId]);

  previewTimers[imgId] = setTimeout(async () => {
    const query = '?manifest=' + encodeURIComponent(manifest)
      + '&index=' + index
      + '&time=' + encodeURIComponent(time);
    const result = await api('/api/shot-preview' + query);
    const img = document.getElementById(imgId);
    if (result.ok && result.path && img) img.src = media(result.path) + '&t=' + Date.now();
  }, 180);
}

async function regenerateReference(manifest, index) {
  const snap = captureScrollState();
  const result = await postJson('/api/reference-regenerate', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not regenerate reference');

  await redrawWithState(result.state, snap);
  setTimeout(refresh, 1000);
}

async function deleteReference(manifest, index) {
  if (!confirm('Delete this color reference? It will be regenerated next time you run Reference Generation.')) return;

  const snap = captureScrollState();
  const result = await postJson('/api/reference-delete', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not delete reference');

  await redrawWithState(result.state, snap);
}

async function chooseCustomReference(manifest, index) {
  const snap = captureScrollState();
  const result = await postJson('/api/reference-custom', { manifest, index });
  if (!result.ok) return alert(result.error || 'Could not install custom reference');
  if (!result.selected) return;

  await redrawWithState(result.state, snap);
}

async function exportMedia(path) {
  const result = await postJson('/api/export-media', { path });
  if (!result.ok) return alert(result.error || 'Could not save media file');
  if (result.saved) alert('Saved:\n' + result.saved);
}

const DEFAULT_ANCHOR_PROMPT = 'Fill the black outpaint margins with a natural continuation of this black-and-white film frame. Preserve the centre/original frame area, composition, lighting, paper, clothing, and background. If hands or fingers extend into the new margins, make them anatomically natural with five fingers and normal joints. Do not colorize. Do not add text, captions, logos, or unrelated new objects.';

// ── Guide frame list actions ─────────────────────────────────────────────────

async function autoSaveGuideFrame(chunkIndex, guideIndex) {
  const slider = document.getElementById(`gfSlider_${chunkIndex}_${guideIndex}`);
  const strEl = document.getElementById(`gfStrength_${chunkIndex}_${guideIndex}`);
  if (!slider) return;
  const frame_idx = Number(slider.value);
  const strength = strEl ? Number(strEl.value) : 0.7;
  await postJson('/api/guide-frame-save', { chunk_index: chunkIndex, guide_index: guideIndex, frame_idx, strength });
}

async function guideFrameRedraw(result, snap) {
  if (!result.ok) return;
  if (result.state) {
    state = result.state;
    draw(false);
    lastRenderSignature = renderSignature();
    lastOutpaintVisualSignature = outpaintVisualSignature();
    restoreScrollState(snap);
  }
}

async function addGuideFrame(chunkIndex) {
  const snap = captureScrollState();
  const result = await postJson('/api/guide-frame-add', { chunk_index: chunkIndex });
  if (!result.ok) return alert(result.error || 'Could not add guide frame');
  await guideFrameRedraw(result, snap);
}

async function removeGuideFrame(chunkIndex, guideIndex) {
  const snap = captureScrollState();
  const result = await postJson('/api/guide-frame-remove', { chunk_index: chunkIndex, guide_index: guideIndex });
  if (!result.ok) return alert(result.error || 'Could not remove guide frame');
  await guideFrameRedraw(result, snap);
}

async function saveGuideFrameSettings(chunkIndex, guideIndex) {
  const snap = captureScrollState();
  const slider = document.getElementById(`gfSlider_${chunkIndex}_${guideIndex}`);
  const strEl = document.getElementById(`gfStrength_${chunkIndex}_${guideIndex}`);
  const frame_idx = slider ? Number(slider.value) : 0;
  const strength = strEl ? Number(strEl.value) : 0.7;
  const result = await postJson('/api/guide-frame-save', { chunk_index: chunkIndex, guide_index: guideIndex, frame_idx, strength });
  if (!result.ok) return alert(result.error || 'Could not save guide frame');
  await guideFrameRedraw(result, snap);
}

async function uploadGuideFrameImage(chunkIndex, guideIndex) {
  await autoSaveGuideFrame(chunkIndex, guideIndex);
  const snap = captureScrollState();
  const result = await postJson('/api/guide-frame-upload', { chunk_index: chunkIndex, guide_index: guideIndex });
  if (!result.ok) return alert(result.error || 'Could not upload guide frame image');
  if (!result.selected) return;
  await guideFrameRedraw(result, snap);
}

async function clearGuideFrameImage(chunkIndex, guideIndex) {
  const snap = captureScrollState();
  const result = await postJson('/api/guide-frame-clear', { chunk_index: chunkIndex, guide_index: guideIndex });
  if (!result.ok) return alert(result.error || 'Could not clear guide frame image');
  await guideFrameRedraw(result, snap);
}

function openGuideFrameGenerateModal(chunkIndex, guideIndex, frameIdx) {
  let modal = document.getElementById('guideFrameGenerateModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'guideFrameGenerateModal';
    modal.className = 'image-modal hidden';
    modal.innerHTML = `
      <div class="image-modal-backdrop" onclick="closeGuideFrameGenerateModal()"></div>
      <div class="prompt-modal-panel">
        <div class="image-modal-heading">
          <strong>Generate Guide Frame Image</strong>
          <button type="button" onclick="closeGuideFrameGenerateModal()">Close</button>
        </div>
        <p class="shot-empty">Qwen will colorize the source frame at the selected position.</p>
        <label>Qwen edit prompt</label>
        <textarea id="guideFrameGeneratePrompt"></textarea>
        <div class="actions">
          <button class="primary" type="button" onclick="submitGuideFrameGenerate()">Generate</button>
          <button type="button" onclick="closeGuideFrameGenerateModal()">Cancel</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
  }
  modal.dataset.chunkIndex = String(chunkIndex);
  modal.dataset.guideIndex = String(guideIndex);
  modal.dataset.frameIdx = String(frameIdx);
  // Pick up current slider value in case the user changed it without saving
  const slider = document.getElementById(`gfSlider_${chunkIndex}_${guideIndex}`);
  if (slider) modal.dataset.frameIdx = slider.value;
  document.getElementById('guideFrameGeneratePrompt').value = DEFAULT_ANCHOR_PROMPT;
  modal.classList.remove('hidden');
}

function closeGuideFrameGenerateModal() {
  const modal = document.getElementById('guideFrameGenerateModal');
  if (modal) modal.classList.add('hidden');
}

async function submitGuideFrameGenerate() {
  const modal = document.getElementById('guideFrameGenerateModal');
  if (!modal) return;
  const chunkIndex = Number(modal.dataset.chunkIndex || 0);
  const guideIndex = Number(modal.dataset.guideIndex || 0);
  const frameIdx = Number(modal.dataset.frameIdx || 0);
  const prompt = document.getElementById('guideFrameGeneratePrompt')?.value || DEFAULT_ANCHOR_PROMPT;
  closeGuideFrameGenerateModal();

  // Save current frame_idx and strength before generating.
  await postJson('/api/guide-frame-save', { chunk_index: chunkIndex, guide_index: guideIndex, frame_idx: frameIdx, strength: Number(document.getElementById(`gfStrength_${chunkIndex}_${guideIndex}`)?.value || 0.7) });

  const result = await postJson('/api/guide-frame-generate', { chunk_index: chunkIndex, guide_index: guideIndex, frame_idx: frameIdx, prompt });
  if (!result.ok) return alert(result.error || result.message || 'Could not generate guide frame image');
  if (result.state) {
    state = result.state;
    draw(false);
    lastRenderSignature = renderSignature();
    lastOutpaintVisualSignature = outpaintVisualSignature();
  }
}

async function saveStage(key, redraw = false) {
  const snap = captureScrollState();
  await postJson('/api/settings', { stage: key, values: formValues() });

  state = await api(stateUrl());
  pruneSelected();

  if (redraw) {
    draw(false);
    restoreScrollState(snap);
  }

  showCommand(key);
}

function formValues() {
  const values = {};
  document.querySelectorAll('[data-field]').forEach(el => {
    values[el.dataset.field] = el.type === 'checkbox' ? String(el.checked) : el.value;
  });
  return values;
}

async function saveGlobal() {
  await postJson('/api/settings', {
    stage: 'global',
    values: { source: document.getElementById('globalSource').value },
  });

  selected = {};
  state = await api(stateUrl());
  pruneSelected();
  if (!availableTabs().includes(active)) active = 'global';
  drawTabs();
  draw();
  lastRenderSignature = renderSignature();
}

async function saveGlobalPipelineOptions() {
  const snap = captureScrollState();
  await postJson('/api/settings', {
    stage: 'global',
    values: {
      expand_outpaint: String(document.getElementById('globalExpandOutpaint').checked),
      colorize: String(document.getElementById('globalColorize').checked),
    },
  });

  state = await api(stateUrl());
  pruneSelected();
  if (!availableTabs().includes(active)) active = 'global';
  drawTabs();
  draw(false);
  restoreScrollState(snap);
}

async function saveGlobalSection() {
  await postJson('/api/settings', {
    stage: 'global',
    values: {
      section_start: document.getElementById('sectionStart')?.value || '0',
      section_end: document.getElementById('sectionEnd')?.value || '',
    },
  });
  state = await api(stateUrl());
  pruneSelected();
  updateOverviewDynamicStatus();
  lastRenderSignature = renderSignature();
}

async function autoCropOutpaint() {
  const slider = document.getElementById('aspectPreviewTime');
  const time = slider ? slider.value : '0';
  const result = await api('/api/outpaint-auto-crop?time=' + encodeURIComponent(time));
  if (!result.ok) return alert(result.error || 'Auto Crop failed');

  state = result.state || state;
  ['crop_left', 'crop_right', 'crop_top', 'crop_bottom'].forEach(key => {
    const el = document.querySelector(`[data-field="${key}"]`);
    const value = result[key] ?? settings('outpaint')[key] ?? '0';
    if (el) {
      el.value = value;
      const label = document.getElementById(`${key}Value`);
      if (label) label.textContent = value;
    }
  });

  const img = document.getElementById('aspectPreviewImg');
  if (result.preview && img) img.src = media(result.preview) + '&t=' + Date.now();
  showCommand('outpaint');
  lastRenderSignature = renderSignature();
  lastOutpaintVisualSignature = outpaintVisualSignature();
}

async function nudgeSectionBoundary(edge, frames) {
  const id = edge === 'start' ? 'sectionStart' : 'sectionEnd';
  const labelId = edge === 'start' ? 'sectionStartLabel' : 'sectionEndLabel';
  const slider = document.getElementById(id);
  if (!slider) return;
  const fps = Math.max(1, Number(slider.dataset.fps || 24));
  const step = 1 / fps;
  const min = Number(slider.min || 0);
  const max = Number(slider.max || 0);
  const current = Number(slider.value || 0);
  slider.value = Math.min(max, Math.max(min, current + frames * step)).toFixed(6);
  const label = document.getElementById(labelId);
  if (label) label.textContent = formatSeconds(slider.value);
  await saveGlobalSection();
}

async function markSourceSection(edge) {
  const video = document.getElementById('sourceSectionVideo');
  const target = document.getElementById(edge === 'start' ? 'sectionStart' : 'sectionEnd');
  if (!video || !target) return;

  target.value = Math.max(0, video.currentTime || 0).toFixed(3);
  const label = document.getElementById(edge === 'start' ? 'sectionStartLabel' : 'sectionEndLabel');
  if (label) label.textContent = formatSeconds(target.value);

  await saveGlobalSection();
}

async function browseGlobalSource() {
  const el = document.getElementById('globalSource');
  const result = await postJson('/api/browse-global-source', { current: el.value });
  if (!result.ok) return alert(result.error || 'Browse failed');

  if (!result.path) return await refresh(true);

  selected = {};
  state = result.state;
  pruneSelected();
  draw();
  lastRenderSignature = renderSignature();
}

async function clearOverview() {
  if (!confirm('Clear the selected source material from the UI? Generated files are left on disk.')) return;

  selected = {};
  const result = await postJson('/api/overview-clear', {});
  if (!result.ok) return alert(result.error || 'Could not clear overview');

  state = result.state;
  pruneSelected();
  active = 'global';
  drawTabs();
  draw();
  lastRenderSignature = renderSignature();
}

async function saveProject() {
  const result = await postJson('/api/project-save', { save_as: false });
  if (!result.ok) return alert(result.error || 'Could not save project');
  if (result.path) alert('Saved ARP project:\n' + result.path);
  if (result.state) {
    state = result.state;
    lastRenderSignature = renderSignature();
  }
}

async function saveProjectAs() {
  const result = await postJson('/api/project-save', { save_as: true });
  if (!result.ok) return alert(result.error || 'Could not save project');
  if (result.path) alert('Saved ARP project:\n' + result.path);
  if (result.state) {
    state = result.state;
    lastRenderSignature = renderSignature();
  }
}

async function loadProject() {
  const result = await postJson('/api/project-load', {});
  if (!result.ok) return alert(result.error || 'Could not load project');
  if (!result.path) return;

  selected = {};
  state = result.state;
  pruneSelected();
  active = 'global';
  drawTabs();
  draw();
  lastRenderSignature = renderSignature();
}

async function browseField(stageKey, fieldKey, kind) {
  const el = document.querySelector(`[data-field="${fieldKey}"]`);
  const result = await postJson('/api/browse', { kind, current: el.value });
  if (!result.ok) return alert(result.error || 'Browse failed');

  if (result.path) {
    el.value = result.path;
    await saveStage(stageKey);
  }
}

async function showCommand(key) {
  const result = await api('/api/command?stage=' + encodeURIComponent(key));
  const el = document.getElementById('cmd');
  if (el) el.textContent = result.command.join(' ');
}

async function confirmOverwrite(key) {
  const force = settings(key).force === 'true';
  if (!force && key !== 'shots') return true;

  const result = await api('/api/existing-outputs?stage=' + encodeURIComponent(key));
  if (!result.paths || !result.paths.length) return true;

  const reason = force ? 'Regenerate is enabled' : 'Shot Detection rewrites its manifest';
  return confirm(reason + ' and these output paths already exist:\n\n' + result.paths.join('\n') + '\n\nOverwrite them?');
}

async function runStage(key) {
  if (key === 'recomp') releaseFinalOutputVideos();
  await saveStage(key);
  if (!(await confirmOverwrite(key))) return;

  const result = await postJson('/api/run', { stage: key });
  if (!result.ok) alert(result.message);
  setTimeout(() => refresh(true), 500);
}

function releaseFinalOutputVideos() {
  const output = ((state.expected_outputs && state.expected_outputs.output) || [])[0]
    || settings('recomp').output
    || '';
  if (!output) return;

  const encoded = encodeURIComponent(output);
  document.querySelectorAll('video').forEach(video => {
    const src = video.getAttribute('src') || '';
    if (!src.includes(encoded) && !src.includes(output)) return;
    try {
      video.pause();
      video.removeAttribute('src');
      video.load();
    } catch {}
  });
}

async function runAll() {
  for (const st of state.stages) {
    if (st.key === 'output') continue;
    if (!(await confirmOverwrite(st.key))) return;
  }

  const result = await postJson('/api/run', { all: true });
  if (!result.ok) alert(result.message);
  setTimeout(() => refresh(true), 500);
}

async function stopRun() {
  await postJson('/api/stop', {});
  refresh(true);
}

async function deleteCacheFile(path) {
  if (!confirm('Delete this cached file?\n\n' + path + '\n\nThis cannot be undone.')) return;

  const result = await postJson('/api/cache-delete', { path });
  if (!result.ok) return alert(result.error || 'Could not delete cached file');

  state = result.state;
  drawCache();
  lastRenderSignature = renderSignature();
}

async function clearCacheCategory(category, title) {
  const message = 'Clear every cached file in "' + title + '"?\n\n'
    + 'This removes generated intermediate files in that category and cannot be undone.';
  if (!confirm(message)) return;

  const result = await postJson('/api/cache-delete', { category });
  if (!result.ok) return alert(result.error || 'Could not clear cache category');

  state = result.state;
  drawCache();
  lastRenderSignature = renderSignature();
}

async function clearAllCache() {
  const message = 'Clear ALL ARP cached/intermediate files?\n\n'
    + 'This deletes generated previews, outpaint chunks, prepared videos, references, colorized intermediates, and manifests. '
    + 'Source videos, installed tools, and downloaded models are left alone.\n\n'
    + 'This cannot be undone.';
  if (!confirm(message)) return;

  const result = await postJson('/api/cache-delete', { all: true });
  if (!result.ok) return alert(result.error || 'Could not clear cache');

  state = result.state;
  drawCache();
  lastRenderSignature = renderSignature();
}
