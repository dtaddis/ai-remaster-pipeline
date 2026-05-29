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
  const maxFrames = Math.max(frameCount, Number(row.max_length_frames || frameCount));
  const custom = !!row.custom_seconds;

  return `
    <div>
      <div class="shot-number">Chunk ${idx + 1}</div>
      <div class="shot-time">${esc(row.start_label)} to ${esc(row.end_label)}</div>
      <p class="shot-time">Frames ${esc(row.start_frame)}-${esc(row.end_frame)}</p>
      <label><input id="chunkCustom_${idx}" type="checkbox" ${custom ? 'checked' : ''} onchange="toggleChunkLength(${idx})"> Custom length</label>
      <label>Length: <span id="chunkFramesLabel_${idx}">${chunkLengthLabel(frameCount, fps)}</span></label>
      <input id="chunkFrames_${idx}" data-fps="${fps}" type="range" min="1" max="${maxFrames}" step="1" value="${frameCount}" ${custom ? '' : 'disabled'} oninput="updateChunkLengthLabel(${idx})">
      <div class="shot-tools">
        <button type="button" onclick="nudgeChunkLength(${idx},-1)" ${custom ? '' : 'disabled'}>-1 frame</button>
        <input id="chunkFramesInput_${idx}" class="frame-input" type="number" min="1" max="${maxFrames}" step="1" value="${frameCount}" ${custom ? '' : 'disabled'} onchange="setChunkLengthFrames(${idx},this.value)">
        <button type="button" onclick="nudgeChunkLength(${idx},1)" ${custom ? '' : 'disabled'}>+1 frame</button>
      </div>
      <label>Seed</label>
      <input id="chunkSeed_${idx}" type="number" value="${esc(row.seed || '42')}">
      <div class="shot-tools">
        <button type="button" onclick="saveOutpaintChunk(${idx})">Save</button>
        <button type="button" data-outpaint-disable-running="true" onclick="regenerateOutpaintChunk(${idx})" ${state.running ? 'disabled' : ''}>Regenerate Chunk</button>
      </div>
    </div>
  `;
}

function outpaintChunkGuide(row) {
  const idx = row.index;
  const guidePath = row.guide_exists ? row.guide_image : row.guide_frame_preview;
  const guideTitle = row.guide_exists ? 'Current guide frame' : 'Chunk start frame (no guide set)';
  const src = guidePath ? media(guidePath) + (row.guide_exists && row.guide_mtime ? '&t=' + row.guide_mtime : '') : '';
  const guideStatus = outpaintGuideStatus(row);

  return `
    <div class="chunk-guide">
      <div>
        <label>Guide frame</label>
        ${guidePath ? `
          <figure id="chunkGuideFigure_${idx}" class="still-figure ${row.guide_exists ? 'has-anchor' : ''}">
            <img id="chunkGuideImg_${idx}" src="${src}" alt="" onclick="openImageModal(this.src,${jsArg(guideTitle)})">
            <span id="chunkGuideBadge_${idx}" class="anchor-badge ${row.guide_exists ? '' : 'hidden'}">Guide</span>
            <figcaption id="chunkGuideCaption_${idx}">${esc(guideTitle)}</figcaption>
          </figure>
        ` : missingImage('Chunk start frame not present')}
      </div>
      <div>
        <p id="chunkGuideStatus_${idx}" class="shot-time">${esc(guideStatus)}</p>
        <p class="shot-time">Applied at the start of the chunk via LTX i2v conditioning. If no guide is set, the last frame of the previous chunk is used automatically for continuity.</p>
        <div class="shot-tools">
          <button type="button" onclick="chooseOutpaintAnchor(${idx})">Upload Guide</button>
          <button type="button" data-outpaint-disable-running="true" onclick="openAnchorPromptModal(${idx})" ${state.running ? 'disabled' : ''}>Generate Guide</button>
          <button type="button" onclick="clearOutpaintAnchor(${idx})" ${row.guide_image ? '' : 'disabled'}>Clear</button>
        </div>
      </div>
    </div>
  `;
}

function outpaintGuideStatus(row) {
  return row.guide_exists
    ? 'Guide frame set. LTX will target this appearance at the start of the chunk.'
    : 'No guide frame set. The previous chunk\'s last frame will be used automatically, or LTX will generate freely for the first chunk.';
}

function updateOutpaintGuidePreviews() {
  if (active !== 'outpaint') return;
  const rows = (state.outpaint_chunks && state.outpaint_chunks.rows) || [];
  for (const row of rows) {
    const idx = row.index;
    const img = document.getElementById(`chunkGuideImg_${idx}`);
    if (!img) continue;
    const guidePath = row.guide_exists ? row.guide_image : row.guide_frame_preview;
    if (!guidePath) continue;
    const title = row.guide_exists ? 'Current guide frame' : 'Chunk start frame (no guide set)';
    const src = media(guidePath) + (row.guide_exists && row.guide_mtime ? '&t=' + row.guide_mtime : '');
    if (img.getAttribute('src') !== src) {
      img.setAttribute('src', src);
    }
    img.onclick = () => openImageModal(img.src, title);

    const figure = document.getElementById(`chunkGuideFigure_${idx}`);
    if (figure) figure.classList.toggle('has-anchor', !!row.guide_exists);
    const badge = document.getElementById(`chunkGuideBadge_${idx}`);
    if (badge) badge.classList.toggle('hidden', !row.guide_exists);
    const caption = document.getElementById(`chunkGuideCaption_${idx}`);
    if (caption) caption.textContent = title;
    const status = document.getElementById(`chunkGuideStatus_${idx}`);
    if (status) status.textContent = outpaintGuideStatus(row);
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
  if (slider) slider.disabled = !(checkbox && checkbox.checked);
  if (input) input.disabled = !(checkbox && checkbox.checked);
  buttons.forEach(button => { button.disabled = !(checkbox && checkbox.checked); });
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
      ${frames.map(([key, label, position]) => row[key] ? chunkStillFigure(row, row[key], label, position, canAnchor) : missingImage(label + ' frame not present')).join('')}
    </div>
  `;
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

function drawOutpaint(st, s, expected, sp) {
  const mainFields = st.fields.filter(f => !f[0].startsWith('crop_'));
  const cropFields = st.fields.filter(f => f[0].startsWith('crop_'));

  document.getElementById('app').innerHTML = `
    <div class="editor-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        <div id="outpaintProgress">${progressHtml(sp.percent, sp.label)}</div>
        ${mainFields.map(f => fieldHtml(st, f)).join('')}
        ${outpaintOverlapWarning(s)}
        <h3>Source Crop</h3>
        <p class="shot-empty">Crop away black borders before ARP expands the frame.</p>
        <div class="editor-controls">
          ${cropFields.map(f => `<div>${fieldHtml(st, f)}</div>`).join('')}
        </div>
        ${stageCheckboxes(s)}
        <div class="actions">
          <button class="primary" data-outpaint-disable-running="true" onclick="runStage('outpaint')" ${state.running ? 'disabled' : ''}>Run Outpainting</button>
          <button class="warn" data-outpaint-enable-running="true" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card preview compact">${aspectPreviewHtml(st)}</section>
    </div>
    <section class="card chunk-section">
      <h2>Outpaint Chunks</h2>
      <p class="shot-empty">Chunks are the fixed video segments sent to LTX. They are separate from shot detection and can be regenerated individually.</p>
      ${outpaintChunkCards()}
    </section>
    <section class="card" style="margin-top:16px">${runLogHtml()}</section>
  `;

  bindStageFields('outpaint');
  showCommand('outpaint');
}
