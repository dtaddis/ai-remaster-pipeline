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
          ${chunkStillStrip(row, 'source')}
        </div>
        <div class="chunk-frame-row">
          <label>Outpainted frames</label>
          ${row.raw_exists ? chunkStillStrip(row, 'raw') : missingImage('Outpainted chunk not present')}
        </div>
      </div>
      ${outpaintChunkPrompt(row)}
    </article>
  `;
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
        <button type="button" onclick="regenerateOutpaintChunk(${idx})" ${state.running ? 'disabled' : ''}>Regenerate Chunk</button>
      </div>
    </div>
  `;
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

function chunkStillStrip(row, prefix) {
  const frames = [
    [prefix + '_start_preview', 'Start', 'start'],
    [prefix + '_middle_preview', 'Middle', 'middle'],
    [prefix + '_end_preview', 'End', 'end'],
  ];
  return `
    <div class="chunk-stills">
      ${frames.map(([key, label, position]) => row[key] ? chunkStillFigure(row, row[key], label, position) : missingImage(label + ' frame not present')).join('')}
    </div>
  `;
}

function chunkStillFigure(row, path, label, position) {
  const src = media(path);
  const activeAnchor = row.anchor_image && row.anchor_position === position;
  return `
    <figure class="still-figure ${activeAnchor ? 'has-anchor' : ''}">
      <img src="${src}" alt="" onclick="openImageModal(${jsArg(src)},${jsArg(label + ' frame')})">
      <div class="still-actions">
        <button type="button" onclick="event.stopPropagation(); exportMedia(${jsArg(path)})" title="Save this frame">&#128190;</button>
        <button type="button" onclick="event.stopPropagation(); chooseOutpaintAnchor(${row.index},${jsArg(position)})" title="Upload anchor frame for this chunk">&#9875;</button>
      </div>
      ${activeAnchor ? '<span class="anchor-badge">Anchor</span>' : ''}
      <figcaption>${esc(label)}</figcaption>
    </figure>
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
        ${progressHtml(sp.percent, sp.label)}
        ${mainFields.map(f => fieldHtml(st, f)).join('')}
        ${outpaintOverlapWarning(s)}
        <h3>Source Crop</h3>
        <p class="shot-empty">Crop away black borders before ARP expands the frame.</p>
        <div class="editor-controls">
          ${cropFields.map(f => `<div>${fieldHtml(st, f)}</div>`).join('')}
        </div>
        ${stageCheckboxes(s)}
        <div class="actions">
          <button class="primary" onclick="runStage('outpaint')" ${state.running ? 'disabled' : ''}>Run Outpainting</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
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
