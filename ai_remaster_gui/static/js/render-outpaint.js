function outpaintChunkCards() {
  const chunks = state.outpaint_chunks || {};
  const rows = chunks.rows || [];

  if (!rows.length) {
    const message = chunks.error || 'Choose source material to preview outpaint chunks.';
    return `<p class="shot-empty">${esc(message)}</p>`;
  }

  return `<div class="chunk-list">${rows.map(outpaintChunkCard).join('')}</div>`;
}

function outpaintChunkCard(row) {
  const idx = row.index;
  return `
    <article class="chunk-card">
      ${outpaintChunkSummary(row)}
      <div class="chunk-frame-rows">
        <div class="chunk-frame-row">
          <label>Original frames</label>
          ${chunkStillStrip(row, 'source', false)}
        </div>
        <div class="chunk-frame-row">
          ${outpaintChunkGuide(row)}
        </div>
        <div class="chunk-frame-row">
          <label>Outpainted frames</label>
          <div id="chunkRawFrames_${idx}" data-raw-signature="${esc(outpaintRawSignature(row))}">
            ${outpaintRawFramesHtml(row)}
          </div>
        </div>
      </div>
      ${outpaintChunkPrompt(row)}
    </article>
  `;
}

function outpaintRawSignature(row) {
  return `${row.raw_path || ''}|${row.raw_exists ? '1' : '0'}|${row.raw_mtime || 0}|${row.raw_start_preview || ''}|${row.raw_middle_preview || ''}|${row.raw_end_preview || ''}`;
}

function outpaintRawFramesHtml(row) {
  return row.raw_exists ? chunkStillStrip(row, 'raw', false) : missingChunkStillStrip('Outpainted chunk not present');
}

function outpaintChunkSummary(row) {
  const idx = row.index;
  const fps = Math.max(1, Number(row.fps || 24));
  const projectFrames = Math.max(1, Math.round(Number(settings('outpaint').chunk_seconds || 20) * fps));
  const frameCount = Math.max(1, Number(row.custom_seconds ? Math.round(Number(row.custom_seconds) * fps) : row.length_frames || projectFrames));
  const defaultFrames = Math.max(1, Math.min(Number(row.max_length_frames || projectFrames), projectFrames));
  const maxFrames = Math.max(frameCount, Number(row.max_length_frames || frameCount));
  const custom = !!row.custom_seconds;
  const autoStartGuide = idx > 0 && String(row.auto_start_guide || '').toLowerCase() !== 'false';
  const autoStartGuideDisabled = idx === 0 ? 'disabled' : '';

  return `
    <div>
      <div class="shot-number">Chunk ${idx + 1}</div>
      <div class="shot-time">${esc(row.start_label)} to ${esc(row.end_label)}</div>
      <p class="shot-time">Frames ${esc(row.start_frame)}-${esc(row.end_frame)}</p>
      <label><input id="chunkCustom_${idx}" type="checkbox" ${custom ? 'checked' : ''} onchange="toggleChunkLength(${idx})"> Custom length</label>
      <label>Length: <span id="chunkFramesLabel_${idx}">${chunkLengthLabel(frameCount, fps)}</span></label>
      <input id="chunkFrames_${idx}" data-fps="${fps}" data-default-frames="${defaultFrames}" type="range" min="1" max="${maxFrames}" step="1" value="${frameCount}" ${custom ? '' : 'disabled'} oninput="updateChunkLengthLabel(${idx})">
      <input id="chunkFramesInput_${idx}" class="frame-input compact" type="number" min="1" max="${maxFrames}" step="1" value="${frameCount}" ${custom ? '' : 'disabled'} onchange="setChunkLengthFrames(${idx},this.value)">
      ${chunkOffsetControls(row)}
      <label>Seed</label>
      <input id="chunkSeed_${idx}" type="number" value="${esc(row.seed || '42')}">
      <label><input id="chunkAutoStartGuide_${idx}" type="checkbox" ${autoStartGuide ? 'checked' : ''} ${autoStartGuideDisabled} onchange="syncAutoStartGuideVisibility(${idx})"> Use previous chunk as start guide</label>
      <div class="shot-tools">
        <button type="button" onclick="saveOutpaintChunk(${idx})">Save</button>
        <button type="button" data-outpaint-disable-running="true" onclick="regenerateOutpaintChunk(${idx})" ${state.running ? 'disabled' : ''}>Regenerate Chunk</button>
      </div>
    </div>
  `;
}

function outpaintChunkGuide(row) {
  const idx = row.index;
  const guides = row.guides || [];
  const lengthFrames = Math.max(1, Number(row.length_frames || 1));
  const fps = Math.max(1, Number(row.fps || 24));
  const autoStartGuideAvailable = !!row.auto_start_guide_available;
  const autoStartGuideActive = autoStartGuideAvailable && String(row.auto_start_guide || '').toLowerCase() !== 'false';

  // Warn about duplicate effective positions, mirroring resolve_guide_coords in
  // outpaint_video.py: negatives resolve from the chunk's latent count, and positions
  // 1 above a multiple of 8 shift down 1 at render (IC-LoRA coordinate clash).
  const resolvedIdxs = guides.map(g => resolveGuideCoord(Number(g.frame_idx), lengthFrames));
  const dupIdxs = new Set(resolvedIdxs.filter((v, i, a) => a.indexOf(v) !== i));
  const guideLabelOffset = autoStartGuideActive ? 1 : 0;

  const guideCards = guides.map((g, gi) => guideFrameCard(
    idx,
    gi,
    g,
    lengthFrames,
    fps,
    dupIdxs,
    guideLabelOffset,
    autoStartGuideActive && guideUsesStartPosition(g, lengthFrames)
  )).join('');

  const empty = guides.length === 0
    ? `<p class="shot-empty guide-empty${autoStartGuideActive ? ' hidden' : ''}">No guide frames set. Add one below to steer LTX at specific points in the chunk.</p>`
    : '';

  const tooMany = guides.length > Math.floor(lengthFrames / 8) + 1
    ? `<p class="shot-time guide-warn">⚠ More guides than 8-frame positions in this chunk — duplicate positions are skipped at render.</p>`
    : '';

  return `
    <div class="chunk-guide-section">
      ${empty}${tooMany}
      <div id="guideFrameList_${idx}">
        ${autoStartGuideAvailable ? autoStartGuideCard(row, autoStartGuideActive) : ''}
        ${guideCards}
      </div>
      <div class="shot-tools guide-add-row">
        <button type="button" data-outpaint-disable-running="true"
          onclick="addGuideFrame(${idx})" ${state.running ? 'disabled' : ''}>+ Add Guide Frame</button>
      </div>
      <p class="shot-time chunk-guide-hint">💡 frame_idx 0 = chunk start (i2v), −1 = last frame (FLF2V), multiples of 8 in between. Positions 1 above a multiple of 8 shift down 1 at render; other non-multiples sit between LTX's 8-frame grid and pin less strongly.</p>
    </div>
  `;
}

function resolveGuideCoord(fi, lengthFrames) {
  // Mirrors resolve_guide_coords in scripts/outpaint_video.py.
  let coord = fi;
  if (fi < 0) {
    const latentCount = Math.floor((Math.max(1, lengthFrames) - 1) / 8) + 1;
    coord = Math.max((latentCount - 1) * 8 + 1 + fi, 0);
  }
  if (coord > 0 && coord % 8 === 1) coord -= 1;
  return coord;
}

function guideUsesStartPosition(g, lengthFrames) {
  return resolveGuideCoord(Number(g.frame_idx), lengthFrames) === 0;
}

function guideFrameLabel(fi, maxFrame) {
  if (fi === 0) return '0 — chunk start (i2v)';
  if (fi >= maxFrame) return `${fi} — last frame (FLF2V)`;
  if (fi % 8 === 1) return `${fi} → renders at ${fi - 1}`;
  if (fi % 8 !== 0) return `${fi} — off the 8-frame grid (weaker)`;
  return `${fi}`;
}

function autoStartGuideCard(row, visible) {
  const idx = Number(row.index);
  const sourceIndex = Number(row.auto_start_guide_source_index);
  const maxFrame = Math.max(0, Number(row.length_frames || 1) - 1);
  const title = `Last outpainted frame from Chunk ${sourceIndex + 1}`;
  return `
    <div id="autoStartGuide_${idx}" class="chunk-guide guide-frame-card auto-start-guide-card${visible ? '' : ' hidden'}">
      <div>
        <label>Guide 1 - Use previous chunk as start guide</label>
        <figure class="still-figure has-anchor">
          <img id="autoStartGuideThumb_${idx}" class="pending-thumb" alt=""
            data-outpaint-thumb="true"
            data-chunk-index="${sourceIndex}"
            data-thumb-kind="raw"
            data-thumb-position="end"
            title="${esc(title)}">
          <span class="anchor-badge">Auto</span>
          <figcaption>Previous chunk end frame</figcaption>
        </figure>
      </div>
      <div>
        <label>Frame: ${esc(guideFrameLabel(0, maxFrame))}</label>
        <p class="shot-time">Uses the last rendered frame from Chunk ${sourceIndex + 1}.</p>
      </div>
    </div>
  `;
}

function guideFrameCard(chunkIdx, guideIdx, g, lengthFrames, fps, dupIdxs, guideLabelOffset = 0, overridden = false) {
  const maxFrame = Math.max(0, lengthFrames - 1);
  // Normalise stored frame_idx: negative values → equivalent positive index
  const rawFi = Number(g.frame_idx);
  const fi = rawFi < 0 ? Math.max(0, maxFrame + rawFi + 1) : Math.min(rawFi, maxFrame);
  const strength = Number(g.strength || 0.7);
  const hasImage = !!g.image_exists;
  const isDup = dupIdxs.has(resolveGuideCoord(rawFi, lengthFrames));
  const outOfRange = rawFi >= lengthFrames || rawFi < -lengthFrames;
  const baseGuideNumber = guideIdx + 1;
  const shownGuideNumber = baseGuideNumber + guideLabelOffset;

  const thumbSrc = hasImage
    ? media(g.image) + (g.image_mtime ? '&t=' + g.image_mtime : '')
    : (g.source_preview ? media(g.source_preview) : '');
  const thumbTitle = hasImage ? 'Current guide image' : 'Source frame at this position';
  const thumbHtml = thumbSrc
    ? `<img id="gfThumb_${chunkIdx}_${guideIdx}" src="${esc(thumbSrc)}" alt="" title="${esc(thumbTitle)}" onclick="openGuideEditor(${chunkIdx},${guideIdx})">`
    : `<img id="gfThumb_${chunkIdx}_${guideIdx}" class="pending-thumb" alt="" title="Source preview not ready">`;

  return `
    <div class="chunk-guide guide-frame-card${isDup || outOfRange ? ' guide-frame-dup' : ''}${overridden ? ' guide-frame-overridden' : ''}" data-start-position-guide="${guideUsesStartPosition(g, lengthFrames) ? 'true' : 'false'}">
      <div>
        <label><span class="guide-label-text" data-base-guide-number="${baseGuideNumber}">Guide ${shownGuideNumber}</span>${overridden ? '<span class="guide-override-note"> - overridden by Guide 1</span>' : '<span class="guide-override-note hidden"> - overridden by Guide 1</span>'}${isDup ? ' ⚠ duplicate position' : ''}${outOfRange ? ' ⚠ beyond chunk end — skipped at render' : ''}</label>
        <figure class="still-figure ${hasImage ? 'has-anchor' : ''}">
          ${thumbHtml}
          ${hasImage ? `<span class="anchor-badge">Guide</span>` : ''}
          <figcaption id="gfThumbCaption_${chunkIdx}_${guideIdx}">${esc(hasImage ? 'Guide image set' : 'Source preview')}</figcaption>
        </figure>
      </div>
      <div>
        <label>Frame: <span id="gfLabel_${chunkIdx}_${guideIdx}">${esc(guideFrameLabel(fi, maxFrame))}</span></label>
        <input id="gfSlider_${chunkIdx}_${guideIdx}" type="range"
          min="0" max="${maxFrame}" step="1" value="${fi}"
          oninput="onGuideFrameSlide(${chunkIdx},${guideIdx},this.value,${maxFrame})">
        <div class="shot-tools">
          <button type="button" onclick="nudgeGuideFrame(${chunkIdx},${guideIdx},-8)">−8</button>
          <input id="gfInput_${chunkIdx}_${guideIdx}" class="frame-input" type="number"
            min="0" max="${maxFrame}" step="1" value="${fi}"
            onchange="setGuideFrameIdx(${chunkIdx},${guideIdx},this.value,${maxFrame})">
          <button type="button" onclick="nudgeGuideFrame(${chunkIdx},${guideIdx},8)">+8</button>
        </div>
        <label>Strength: <span id="gfStrLabel_${chunkIdx}_${guideIdx}">${strength.toFixed(2)}</span></label>
        <input id="gfStrength_${chunkIdx}_${guideIdx}" type="range" min="0" max="1" step="0.01" value="${strength}"
          oninput="document.getElementById('gfStrLabel_${chunkIdx}_${guideIdx}').textContent=parseFloat(this.value).toFixed(2)">
        <div class="shot-tools">
          <button type="button" data-outpaint-disable-running="true"
            onclick="saveGuideFrameSettings(${chunkIdx},${guideIdx})" ${state.running ? 'disabled' : ''}>Save</button>
          <button type="button" data-outpaint-disable-running="true"
            onclick="uploadGuideFrameImage(${chunkIdx},${guideIdx})" ${state.running ? 'disabled' : ''}>Upload Image</button>
          <button type="button" data-outpaint-disable-running="true"
            onclick="openGuideFrameGenerateModal(${chunkIdx},${guideIdx},${fi})" ${state.running ? 'disabled' : ''}>Generate</button>
          <button type="button"
            onclick="clearGuideFrameImage(${chunkIdx},${guideIdx})" ${!hasImage ? 'disabled' : ''}>Clear Image</button>
          <button type="button" class="warn"
            onclick="removeGuideFrame(${chunkIdx},${guideIdx})">Remove</button>
        </div>
      </div>
    </div>
  `;
}

function syncAutoStartGuideVisibility(index) {
  const checkbox = document.getElementById(`chunkAutoStartGuide_${index}`);
  const autoCard = document.getElementById(`autoStartGuide_${index}`);
  const visible = !!(checkbox && checkbox.checked && autoCard);
  if (autoCard) autoCard.classList.toggle('hidden', !visible);

  document.querySelectorAll(`#guideFrameList_${index} .guide-label-text`).forEach(label => {
    const base = Number(label.dataset.baseGuideNumber || 0);
    if (base > 0) label.textContent = `Guide ${base + (visible ? 1 : 0)}`;
  });
  document.querySelectorAll(`#guideFrameList_${index} .guide-frame-card[data-start-position-guide="true"]`).forEach(card => {
    card.classList.toggle('guide-frame-overridden', visible);
    const note = card.querySelector('.guide-override-note');
    if (note) note.classList.toggle('hidden', !visible);
  });

  const empty = document.querySelector(`#guideFrameList_${index}`)?.parentElement?.querySelector('.guide-empty');
  if (empty) empty.classList.toggle('hidden', visible);
}

function updateGuideFrameStartOverride(chunkIdx, guideIdx, frameIdx, maxFrame) {
  const img = document.getElementById(`gfThumb_${chunkIdx}_${guideIdx}`);
  const card = img ? img.closest('.guide-frame-card') : null;
  if (!card) return;
  const usesStartPosition = resolveGuideCoord(Number(frameIdx), Number(maxFrame) + 1) === 0;
  const autoCard = document.getElementById(`autoStartGuide_${chunkIdx}`);
  const autoVisible = !!(autoCard && !autoCard.classList.contains('hidden'));
  card.dataset.startPositionGuide = usesStartPosition ? 'true' : 'false';
  card.classList.toggle('guide-frame-overridden', autoVisible && usesStartPosition);
  const note = card.querySelector('.guide-override-note');
  if (note) note.classList.toggle('hidden', !(autoVisible && usesStartPosition));
}

const _gfPreviewTimers = {};

function onGuideFrameSlide(chunkIdx, guideIdx, value, maxFrame) {
  const fi = Math.max(0, Math.min(maxFrame, Math.round(Number(value))));
  const label = document.getElementById(`gfLabel_${chunkIdx}_${guideIdx}`);
  const input = document.getElementById(`gfInput_${chunkIdx}_${guideIdx}`);
  if (label) label.textContent = guideFrameLabel(fi, maxFrame);
  if (input) input.value = fi;
  updateGuideFrameStartOverride(chunkIdx, guideIdx, fi, maxFrame);
  // Update generate modal frame_idx live
  const modal = document.getElementById('guideFrameGenerateModal');
  if (modal && Number(modal.dataset.chunkIndex) === chunkIdx && Number(modal.dataset.guideIndex) === guideIdx) {
    modal.dataset.frameIdx = fi;
  }
  // Debounce the source preview fetch and auto-save
  const key = `${chunkIdx}_${guideIdx}`;
  clearTimeout(_gfPreviewTimers[key]);
  _gfPreviewTimers[key] = setTimeout(() => {
    fetchGuideFramePreview(chunkIdx, guideIdx, fi);
    autoSaveGuideFrame(chunkIdx, guideIdx);
  }, 400);
}

async function fetchGuideFramePreview(chunkIdx, guideIdx, fi) {
  const img = document.getElementById(`gfThumb_${chunkIdx}_${guideIdx}`);
  const cap = document.getElementById(`gfThumbCaption_${chunkIdx}_${guideIdx}`);
  if (!img) return;
  // Only update if no guide image is set (it would be the authoritative thumbnail)
  if (img.closest('.has-anchor')) return;
  const result = await api(`/api/outpaint-guide-preview?chunk_index=${chunkIdx}&frame_idx=${fi}`);
  if (result && result.preview) {
    img.classList.remove('pending-thumb');
    img.src = media(result.preview) + '&t=' + Date.now();
    img.onclick = () => openGuideEditor(chunkIdx, guideIdx, result.preview);
    if (cap) cap.textContent = 'Source preview';
  }
}

function setGuideFrameIdx(chunkIdx, guideIdx, value, maxFrame) {
  const fi = Math.max(0, Math.min(maxFrame, Math.round(Number(value))));
  const slider = document.getElementById(`gfSlider_${chunkIdx}_${guideIdx}`);
  if (slider) slider.value = fi;
  onGuideFrameSlide(chunkIdx, guideIdx, fi, maxFrame);
}

function nudgeGuideFrame(chunkIdx, guideIdx, delta) {
  const slider = document.getElementById(`gfSlider_${chunkIdx}_${guideIdx}`);
  if (!slider) return;
  setGuideFrameIdx(chunkIdx, guideIdx, Number(slider.value) + delta, Number(slider.max));
}

function updateOutpaintGuidePreviews() {
  if (active !== 'outpaint') return;
  const rows = (state.outpaint_chunks && state.outpaint_chunks.rows) || [];
  for (const row of rows) {
    for (const guide of (row.guides || [])) {
      updateGuideFrameCard(row.index, guide);
    }
  }
  hydratePendingGuidePreviews();
}

function updateGuideFrameCard(chunkIdx, guide) {
  const guideIdx = Number(guide.guide_index);
  if (!Number.isFinite(guideIdx)) return;
  const img = document.getElementById(`gfThumb_${chunkIdx}_${guideIdx}`);
  if (!img) return;
  const card = img.closest('.guide-frame-card');
  const figure = img.closest('.still-figure');
  const caption = document.getElementById(`gfThumbCaption_${chunkIdx}_${guideIdx}`);
  const hasImage = !!guide.image_exists && !!guide.image;

  if (hasImage) {
    const src = media(guide.image) + (guide.image_mtime ? '&t=' + guide.image_mtime : '');
    if (img.getAttribute('src') !== src) img.src = src;
    img.classList.remove('pending-thumb');
    img.dataset.guideLoaded = 'true';
    delete img.dataset.guideLoading;
    img.title = 'Current guide image';
    img.onclick = () => openGuideEditor(chunkIdx, guideIdx);
    if (figure) {
      figure.classList.add('has-anchor');
      if (!figure.querySelector('.anchor-badge')) {
        figure.insertAdjacentHTML('beforeend', '<span class="anchor-badge">Guide</span>');
      }
    }
    if (caption) caption.textContent = 'Guide image set';
    if (card) {
      card.querySelectorAll('button').forEach(button => {
        if ((button.getAttribute('onclick') || '').startsWith('clearGuideFrameImage')) {
          button.disabled = false;
        }
      });
    }
    return;
  }

  if (guide.source_preview && img.classList.contains('pending-thumb')) {
    const src = media(guide.source_preview);
    if (img.getAttribute('src') !== src) img.src = src;
    img.classList.remove('pending-thumb');
    img.onclick = () => openGuideEditor(chunkIdx, guideIdx, guide.source_preview);
    if (caption) caption.textContent = 'Source preview';
  }
}

function updateOutpaintRawPreviews() {
  if (active !== 'outpaint') return;
  const rows = (state.outpaint_chunks && state.outpaint_chunks.rows) || [];
  for (const row of rows) {
    const container = document.getElementById(`chunkRawFrames_${row.index}`);
    if (!container) continue;
    const signature = outpaintRawSignature(row);
    if (container.dataset.rawSignature === signature) continue;
    container.innerHTML = outpaintRawFramesHtml(row);
    container.dataset.rawSignature = signature;
  }
  hydratePendingOutpaintPreviews();
  hydratePendingGuidePreviews();
}

function updateOutpaintRuntimeControls() {
  if (active !== 'outpaint') return;
  const sp = stageProgress('outpaint');
  const progress = document.getElementById('outpaintProgress');
  if (progress) {
    const percent = Math.max(0, Math.min(100, Number(sp.percent) || 0));
    const label = progress.querySelector('[data-progress-label]');
    const value = progress.querySelector('[data-progress-percent]');
    const bar = progress.querySelector('progress');
    if (label) label.textContent = sp.label || 'Waiting';
    if (value) value.textContent = `${percent}%`;
    if (bar) bar.value = percent;
  }

  document.querySelectorAll('[data-outpaint-disable-running]').forEach(button => {
    button.disabled = !!state.running;
  });
  document.querySelectorAll('[data-outpaint-enable-running]').forEach(button => {
    button.disabled = !state.running;
  });
}

function outpaintChunkPrompt(row) {
  const idx = row.index;

  return `
    <div>
      <label>Prompt suffix</label>
      <textarea id="chunkPrompt_${idx}" placeholder="Optional direction for this chunk">${esc(row.prompt_suffix || '')}</textarea>
      <label>Negative suffix</label>
      <textarea id="chunkNegative_${idx}" placeholder="Optional things to avoid in this chunk">${esc(row.negative_suffix || '')}</textarea>
      <p class="shot-time">Use these to nudge LTX away from odd extra objects, warped geometry, hands, or missing details.</p>
    </div>
  `;
}

function toggleChunkLength(index) {
  const checkbox = document.getElementById(`chunkCustom_${index}`);
  const slider = document.getElementById(`chunkFrames_${index}`);
  const input = document.getElementById(`chunkFramesInput_${index}`);
  const buttons = slider ? slider.parentElement.querySelectorAll('.shot-tools button') : [];
  const enabled = !!(checkbox && checkbox.checked);
  if (slider && !enabled) {
    slider.value = Math.max(Number(slider.min || 1), Math.min(Number(slider.max || 1), Math.round(Number(slider.dataset.defaultFrames || slider.value) || 1)));
    updateChunkLengthLabel(index);
  }
  if (slider) slider.disabled = !enabled;
  if (input) input.disabled = !enabled;
  buttons.forEach(button => { button.disabled = !enabled; });
}

function updateChunkLengthLabel(index) {
  const slider = document.getElementById(`chunkFrames_${index}`);
  const label = document.getElementById(`chunkFramesLabel_${index}`);
  if (!slider || !label) return;
  label.textContent = chunkLengthLabel(Number(slider.value), Number(slider.dataset.fps || 24));
  const input = document.getElementById(`chunkFramesInput_${index}`);
  if (input) input.value = slider.value;
}

function setChunkLengthFrames(index, value) {
  const slider = document.getElementById(`chunkFrames_${index}`);
  if (!slider) return;
  const next = Math.max(Number(slider.min || 1), Math.min(Number(slider.max || value), Math.round(Number(value) || 1)));
  slider.value = next;
  updateChunkLengthLabel(index);
}

function nudgeChunkLength(index, delta) {
  const slider = document.getElementById(`chunkFrames_${index}`);
  if (!slider || slider.disabled) return;
  setChunkLengthFrames(index, Number(slider.value) + Number(delta || 0));
}

function chunkLengthLabel(frames, fps) {
  const safeFrames = Math.max(1, Math.round(Number(frames) || 1));
  const safeFps = Math.max(1, Number(fps) || 24);
  return `${safeFrames} frames (${(safeFrames / safeFps).toFixed(3)}s)`;
}


function chunkStillStrip(row, prefix, canAnchor) {
  const frames = [
    [prefix + '_start_preview', 'Start', 'start'],
    [prefix + '_middle_preview', 'Middle', 'middle'],
    [prefix + '_end_preview', 'End', 'end'],
  ];
  return `
    <div class="chunk-stills">
      ${frames.map(([key, label, position]) => row[key]
        ? chunkStillFigure(row, row[key], label, position, canAnchor)
        : pendingChunkStillFigure(row, prefix, label, position)
      ).join('')}
    </div>
  `;
}

function chunkOffsetControls(row) {
  const idx = row.index;
  const custom = String(row.offset_mode || 'inherit').toLowerCase() === 'custom';
  const disabled = custom ? '' : 'disabled';
  return `
    <div class="chunk-offsets">
      <label class="chunk-offset-mode"><input id="chunkOffsetOverride_${idx}" type="checkbox" ${custom ? 'checked' : ''} onchange="toggleChunkOffsetOverride(${idx})"> Override whole-video offset</label>
      <label>Offset X</label>
      ${chunkOffsetControl(idx, 'x', row.offset_x || '0', disabled)}
      <label>Offset Y</label>
      ${chunkOffsetControl(idx, 'y', row.offset_y || '0', disabled)}
    </div>
  `;
}

function chunkOffsetControl(idx, axis, value, disabled = '') {
  return `
    <input id="chunkOffset_${axis}_${idx}" class="pixel-input chunk-offset-input" type="number" step="1" value="${esc(value)}" ${disabled}>
  `;
}

function toggleChunkOffsetOverride(idx) {
  const enabled = !!document.getElementById(`chunkOffsetOverride_${idx}`)?.checked;
  ['x', 'y'].forEach(axis => {
    const input = document.getElementById(`chunkOffset_${axis}_${idx}`);
    if (input) input.disabled = !enabled;
  });
}

function chunkStillFigure(row, path, label, position, canAnchor) {
  const shownPath = path;
  const cacheBust = shownPath.includes('_raw_') && row.raw_mtime ? '&t=' + row.raw_mtime : '';
  const src = media(shownPath) + cacheBust;
  const title = `${label} frame`;
  return `
    <figure class="still-figure">
      <img src="${src}" alt="" onclick="openImageModal(${jsArg(src)},${jsArg(title)})">
      <div class="still-actions">
        <button type="button" onclick="event.stopPropagation(); exportMedia(${jsArg(shownPath)})" title="Save this frame">&#128190;</button>
      </div>
      <figcaption>${esc(label)}</figcaption>
    </figure>
  `;
}

function pendingChunkStillFigure(row, prefix, label, position) {
  const idx = row.index;
  const kind = prefix === 'raw' ? 'raw' : 'source';
  if (kind === 'raw' && !row.raw_exists) {
    return missingImage(label + ': Outpainted chunk not present');
  }
  return `
    <figure class="still-figure">
      <img
        id="chunkThumb_${kind}_${idx}_${position}"
        class="pending-thumb"
        alt=""
        data-outpaint-thumb="true"
        data-chunk-index="${idx}"
        data-thumb-kind="${kind}"
        data-thumb-position="${esc(position)}"
        title="Preview loading"
      >
      <figcaption>${esc(label)}</figcaption>
    </figure>
  `;
}

function missingChunkStillStrip(text) {
  return `
    <div class="chunk-stills">
      ${['Start', 'Middle', 'End'].map(label => `
        <figure>
          ${missingImage(`${label}: ${text}`)}
          <figcaption>${esc(label)}</figcaption>
        </figure>
      `).join('')}
    </div>
  `;
}

const outpaintThumbQueue = [];
let outpaintThumbActive = 0;

function hydratePendingOutpaintPreviews() {
  if (active !== 'outpaint') return;
  document.querySelectorAll('img[data-outpaint-thumb]:not([data-thumb-loading]):not([data-thumb-loaded])').forEach(img => {
    img.dataset.thumbLoading = 'true';
    outpaintThumbQueue.push(img);
  });
  pumpOutpaintThumbQueue();
}

function pumpOutpaintThumbQueue() {
  while (outpaintThumbActive < 2 && outpaintThumbQueue.length) {
    const img = outpaintThumbQueue.shift();
    if (!img || !img.isConnected) continue;
    outpaintThumbActive += 1;
    loadOutpaintThumb(img).finally(() => {
      outpaintThumbActive = Math.max(0, outpaintThumbActive - 1);
      pumpOutpaintThumbQueue();
    });
  }
}

async function loadOutpaintThumb(img) {
  const chunkIndex = img.dataset.chunkIndex || '0';
  const kind = img.dataset.thumbKind || 'source';
  const position = img.dataset.thumbPosition || 'middle';
  const result = await api(
    `/api/outpaint-chunk-preview?chunk_index=${encodeURIComponent(chunkIndex)}`
    + `&kind=${encodeURIComponent(kind)}&position=${encodeURIComponent(position)}`
  );
  if (!img.isConnected) return;
  if (result && result.preview) {
    const src = media(result.preview) + '&t=' + Date.now();
    img.classList.remove('pending-thumb');
    img.src = src;
    img.onclick = () => openImageModal(src, `${position[0].toUpperCase()}${position.slice(1)} frame`);
    if (!img.parentElement.querySelector('.still-actions')) {
      img.insertAdjacentHTML('afterend', `
        <div class="still-actions">
          <button type="button" onclick="event.stopPropagation(); exportMedia(${jsArg(result.preview)})" title="Save this frame">&#128190;</button>
        </div>
      `);
    }
    img.dataset.thumbLoaded = 'true';
    img.removeAttribute('title');
  } else {
    delete img.dataset.thumbLoading;
  }
}

function hydratePendingGuidePreviews() {
  if (active !== 'outpaint') return;
  document.querySelectorAll('img[id^="gfThumb_"].pending-thumb:not([data-guide-loading]):not([data-guide-loaded])').forEach(img => {
    const match = img.id.match(/^gfThumb_(\d+)_(\d+)$/);
    if (!match || img.closest('.has-anchor')) return;
    const slider = document.getElementById(`gfSlider_${match[1]}_${match[2]}`);
    img.dataset.guideLoading = 'true';
    fetchGuideFramePreview(Number(match[1]), Number(match[2]), Number(slider ? slider.value : 0))
      .then(() => {
        if (img.isConnected && img.src) img.dataset.guideLoaded = 'true';
      })
      .finally(() => {
        if (img.isConnected && !img.dataset.guideLoaded) delete img.dataset.guideLoading;
      });
  });
}

function drawOutpaint(st, s, expected, sp) {
  const offsetKeys = new Set(['offset_x', 'offset_y']);
  const mainFields = st.fields.filter(f => !f[0].startsWith('crop_') && !offsetKeys.has(f[0]));
  const offsetFields = st.fields.filter(f => offsetKeys.has(f[0]));
  const cropFields = st.fields.filter(f => f[0].startsWith('crop_'));

  document.getElementById('app').innerHTML = `
    <div class="editor-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        <div id="outpaintProgress">${progressHtml(sp.percent, sp.label)}</div>
        ${mainFields.map(f => fieldHtml(st, f)).join('')}
        ${outpaintOverlapWarning(s)}
        ${stageCheckboxes(s)}
        <div class="actions">
          <button class="primary" data-outpaint-disable-running="true" onclick="runStage('outpaint')" ${state.running ? 'disabled' : ''}>Run Outpainting</button>
          <button class="warn" data-outpaint-enable-running="true" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card preview compact outpaint-composition">
        ${aspectPreviewHtml(st)}
        <div class="outpaint-composition-controls">
          <section class="composition-control-group">
            <div class="crop-head">
              <div>
                <h3>Source Crop</h3>
                <p class="shot-empty">Crop away black borders before ARP expands the frame.</p>
              </div>
              <button type="button" onclick="autoCropOutpaint()">Auto Crop</button>
            </div>
            <div class="editor-controls composition-crop-fields">
              ${cropFields.map(f => `<div>${fieldHtml(st, f)}</div>`).join('')}
            </div>
          </section>
          <section class="composition-control-group">
            <h3>Whole-video offset</h3>
            <p class="shot-empty">Shift the cropped source inside the expanded frame. Chunks inherit these values unless overridden below.</p>
            <div class="editor-controls composition-offset-fields">
              ${offsetFields.map(f => `<div>${fieldHtml(st, f)}</div>`).join('')}
            </div>
          </section>
        </div>
      </section>
    </div>
    <section class="card chunk-section">
      <h2>Outpaint Chunks</h2>
      <p class="shot-empty">Chunks are the fixed video segments sent to LTX. They are separate from shot detection and can be regenerated individually.</p>
      ${outpaintChunkCards()}
    </section>
  `;

  bindStageFields('outpaint');
  showCommand('outpaint');
  hydratePendingOutpaintPreviews();
  hydratePendingGuidePreviews();
}
