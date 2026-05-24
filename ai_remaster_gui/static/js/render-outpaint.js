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
  const source = (state.settings.global && state.settings.global.source) || '';
  const start = Number(row.start || 0).toFixed(3);
  const end = Number(row.end || 0).toFixed(3);

  return `
    <article class="chunk-card">
      ${outpaintChunkSummary(row)}
      <div>
        <label>Original chunk</label>
        ${source ? `<video src="${mediaClip(source, start, end, 'outpaint_src_' + idx)}" controls preload="metadata"></video>` : missingImage('Video not present')}
      </div>
      <div>
        <label>Outpainted chunk</label>
        ${row.raw_exists ? `<video src="${media(row.raw_path)}&t=${row.raw_mtime}" controls preload="metadata"></video>` : missingImage('Outpainted chunk not present')}
      </div>
      ${outpaintChunkPrompt(row)}
    </article>
  `;
}

function outpaintChunkSummary(row) {
  const idx = row.index;

  return `
    <div>
      <div class="shot-number">Chunk ${idx + 1}</div>
      <div class="shot-time">${esc(row.start_label)} to ${esc(row.end_label)}</div>
      <p class="shot-time">Frames ${esc(row.start_frame)}-${esc(row.end_frame)}</p>
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
      <p class="shot-time">Use this to nudge LTX away from odd extra objects, warped geometry, or missing details.</p>
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
