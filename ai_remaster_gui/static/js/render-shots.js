function drawShots() {
  drawShotStage({
    key: 'shots',
    heading: 'Shots',
    runLabel: 'Run Shot Detection',
    outputLimit: null,
  });
}

function drawReferences() {
  drawShotStage({
    key: 'references',
    heading: 'References',
    runLabel: 'Run Reference Generation',
    outputLimit: 8,
    afterRender: wireReferenceTimeControls,
  });
}

function drawColour() {
  drawShotStage({
    key: 'colour',
    heading: 'Shot Segments',
    runLabel: 'Run Colorization',
    outputLimit: null,
  });
}

function outputExists(stageKey, path) {
  if (!path) return false;
  const needle = normalizeStatePath(path);
  const outputs = (state.existing_outputs && state.existing_outputs[stageKey]) || [];
  return outputs.some(existing => {
    const current = normalizeStatePath(existing);
    return current === needle || current.endsWith('/' + needle) || needle.endsWith('/' + current);
  });
}

function normalizeStatePath(path) {
  return String(path || '').replace(/\\/g, '/').replace(/^\.\//, '').toLowerCase();
}

function drawShotStage({ key, heading, runLabel, outputLimit, afterRender }) {
  const st = stage(key);
  const s = settings(key);
  const expected = (state.expected_outputs && state.expected_outputs[key]) || [];
  const sp = stageProgress(key);
  const visibleFields = shotStageVisibleFields(st);

  document.getElementById('app').innerHTML = `
    <div class="shot-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        ${progressHtml(sp.percent, sp.label)}
        ${key === 'colour' ? colorizationMethodWarning(s) : ''}
        ${key === 'shots' ? shotDetectionInputStatus(s) : ''}
        ${visibleFields.map(f => fieldHtml(st, f)).join('')}
        ${shotOutputList(expected, outputLimit)}
        ${stageCheckboxes(s)}
        <div class="actions">
          <button class="primary" onclick="runStage('${key}')" ${state.running ? 'disabled' : ''}>${runLabel}</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card">
        <h2>${heading}</h2>
        ${shotCards(key)}
      </section>
    </div>
    <section class="card" style="margin-top:16px">${runLogHtml()}</section>
  `;

  bindStageFields(key);
  if (afterRender) afterRender();
  showCommand(key);
}

function shotStageVisibleFields(st) {
  if (st.key === 'shots') return st.fields.filter(field => field[0] !== 'outpainted_video');
  return st.fields;
}

function shotDetectionInputStatus(s) {
  const source = s.outpainted_video || '';
  if (!source) {
    return '<div class="inline-warning">Complete the previous step first, or choose source material on the Overview tab.</div>';
  }
  return `
    <div class="source-info">
      <div>
        <span>Input video</span>
        <strong>${esc(source)}</strong>
      </div>
    </div>
  `;
}

function colorizationMethodWarning(s) {
  if (!['colormnet', 'both'].includes(s.method)) return '';
  return '<div class="inline-warning">ColorMNet is available for comparison, but its custom node reports a CC BY-NC-SA 4.0 license. Use it only for non-commercial work unless you have separate rights.</div>';
}

function shotCards(mode) {
  const view = state.shot_views || {};
  const rows = view[mode] || [];
  const manifest = view[mode + '_manifest'] || '';

  if (!rows.length) {
    return '<p class="shot-empty">No shot manifest yet. Run Shot Detection first.</p>';
  }

  return `<div class="shot-list">${rows.map(row => shotCard(mode, manifest, row)).join('')}</div>`;
}

function shotCard(mode, manifest, row) {
  const context = shotCardContext(mode, manifest, row);

  if (mode === 'shots') return shotBoundaryCard(context);
  if (mode === 'colour') return colourSegmentCard(context);
  return referenceCard(context);
}

function shotCardContext(mode, manifest, row) {
  const idx = row.index;
  const src = row.source_reference || '';
  const col = row.color_reference || '';
  const srcReady = src && row.source_reference_mtime;
  const colReady = col && row.color_reference_mtime;

  return {
    mode,
    manifest,
    row,
    idx,
    enabled: String(row.enabled || 'true').toLowerCase() !== 'false',
    sourceUrl: srcReady ? media(src) + '&t=' + (row.source_reference_mtime || 0) : '',
    colorUrl: colReady ? media(col) + '&t=' + (row.color_reference_mtime || 0) : '',
    sourceReady: srcReady,
    colorReady: colReady,
  };
}

function shotSummary({ manifest, row, idx, enabled }, extra = '') {
  return `
    <div>
      <div class="shot-number">Shot ${idx + 1}</div>
      <div class="shot-time">${esc(row.start_label)} to ${esc(row.end_label)}</div>
      <label>
        <input type="checkbox" ${enabled ? 'checked' : ''} onchange="saveShotEnabled('${esc(manifest)}',${idx},this.checked)">
        Use shot
      </label>
      ${extra}
    </div>
  `;
}

function shotBoundaryCard(context) {
  const { manifest, row, idx } = context;
  const mergeButton = row.can_merge_next
    ? `<button type="button" onclick="mergeShot('${esc(manifest)}',${idx})">Merge Next</button>`
    : '';
  const splitButton = row.can_split
    ? `<button type="button" onclick="splitShot('${esc(manifest)}',${idx})">Split</button>`
    : '';

  return `
    <article class="shot-card">
      ${shotSummary(context, `<div class="shot-tools">${mergeButton}${splitButton}</div>`)}
      ${boundaryFrameCard(context, 'start')}
      <div>
        <label>Middle</label>
        ${row.middle_preview ? `<img src="${media(row.middle_preview)}" alt="">` : missingImage('Image not present')}
      </div>
      ${boundaryFrameCard(context, 'end')}
    </article>
  `;
}

function boundaryFrameCard({ manifest, row, idx }, edge) {
  const isStart = edge === 'start';
  const frame = isStart ? row.start_frame : row.end_frame;
  const preview = isStart ? row.start_preview : row.end_preview;
  const value = isStart ? row.start : row.end;
  const min = isStart ? Math.max(0, Number(row.start) - 1) : row.start;
  const max = isStart ? row.end : Number(row.end) + 1;
  const disabled = isStart && idx === 0 ? 'disabled' : '';
  const label = isStart ? 'Start' : 'End';

  return `
    <div>
      <label>${label} frame ${frame ?? ''}</label>
      ${preview ? `<img src="${media(preview)}" alt="">` : missingImage('Image not present')}
      <input
        type="range"
        min="${min}"
        max="${max}"
        step="0.041"
        value="${value}"
        ${disabled}
        onchange="setShotBoundary('${esc(manifest)}',${idx},'${edge}',this.value)"
      >
      <div class="shot-tools">
        <button type="button" ${disabled} onclick="nudgeShotBoundary('${esc(manifest)}',${idx},'${edge}',-1)">-1 frame</button>
        <button type="button" ${disabled} onclick="nudgeShotBoundary('${esc(manifest)}',${idx},'${edge}',1)">+1 frame</button>
      </div>
    </div>
  `;
}

function colourSegmentCard(context) {
  const { row, idx, enabled, colorReady, colorUrl } = context;
  const start = Math.max(0, Number(row.start) || 0).toFixed(3);
  const end = Math.max(0, Number(row.end) || 0).toFixed(3);
  const expectedVideos = (state.expected_outputs && state.expected_outputs.colour) || [];
  const candidateVideos = [
    row.colorized_video,
    settings('recomp').colorized_video,
    ...expectedVideos,
  ].filter(Boolean);
  const colourVideo = candidateVideos.find(path => outputExists('colour', path)) || '';
  const method = settings('colour').method || 'deepexemplar';
  const status = enabled ? (colorReady ? `Ready for ${colorizationLabel(method)}` : 'Missing color reference') : 'Disabled in manifest';

  return `
    <article class="shot-card">
      ${shotSummary(context, `<p class="shot-empty">${status}</p>`)}
      <div>
        <label>Color reference</label>
        ${colorReady ? `<img src="${colorUrl}" alt="">` : missingImage('Image not present')}
      </div>
      <div>
        <label>Colorized shot video</label>
        ${colourVideo ? `<video src="${mediaClip(colourVideo, start, end, 'colour_' + idx)}" controls preload="metadata"></video>` : missingImage('Video not present')}
      </div>
      <div>
        <label>Segment</label>
        <p class="shot-time">${esc(colorizationLabel(method))} uses this reference for the selected shot range.</p>
      </div>
    </article>
  `;
}

function colorizationLabel(method) {
  if (method === 'colormnet') return 'ColorMNet';
  if (method === 'both') return 'Deep Exemplar and ColorMNet';
  return 'Deep Exemplar';
}

function referenceCard(context) {
  const { manifest, row, idx, sourceReady, sourceUrl, colorReady, colorUrl } = context;
  const slider = `shotSlider_references_${idx}`;
  const label = `shotLabel_references_${idx}`;
  const img = `shotImg_references_${idx}`;

  return `
    <article class="shot-card">
      ${shotSummary(context, referenceTimeControl(manifest, row, idx, slider, label, img))}
      <div>
        <label>B&W screenshot</label>
        ${sourceReady ? `<img id="${img}" src="${sourceUrl}" alt="">` : missingImage('Image not present')}
      </div>
      <div>
        <label>Qwen color reference</label>
        ${colorReady ? colorReferenceThumb(manifest, idx, colorUrl) : missingImage('Image not present')}
      </div>
      <div>${referencePromptTools(context)}</div>
    </article>
  `;
}

function referenceTimeControl(manifest, row, idx, slider, label, img) {
  return `
    <label>Reference time</label>
    <input
      id="${slider}"
      type="range"
      min="${row.start}"
      max="${row.end}"
      step="0.041"
      value="${row.selected_time}"
      disabled
      oninput="updateShotPreview('${esc(manifest)}',${idx},this.value,'${img}','${label}')"
    >
    <div class="shot-time" id="${label}">${esc(row.selected_label)}</div>
  `;
}

function colorReferenceThumb(manifest, idx, colorUrl) {
  return `
    <div class="thumb-wrap">
      <img src="${colorUrl}" alt="">
      <button class="icon-button" type="button" title="Delete color reference" onclick="deleteReference('${esc(manifest)}',${idx})">&#128465;</button>
    </div>
  `;
}

function referencePromptTools({ manifest, row, idx }) {
  const regenerating = state.running_reference
    && state.running_reference.index === idx
    && state.running_reference.manifest === manifest;
  const rp = stageProgress('references');

  return `
    <label>Shot prompt</label>
    <textarea data-shot-prompt="${idx}" onblur="saveShotPrompt('${esc(manifest)}',${idx},this.value)" placeholder="Optional extra direction for this shot">${esc(row.prompt || '')}</textarea>
    <div class="shot-tools">
      <button type="button" onclick="chooseCustomReference('${esc(manifest)}',${idx})">
        Use Custom Image
      </button>
      <button type="button" onclick="regenerateReference('${esc(manifest)}',${idx})" ${state.running ? 'disabled' : ''}>
        ${regenerating ? 'Regenerating...' : 'Regenerate Reference'}
      </button>
      ${regenerating ? '<span class="spinner" aria-label="In progress"></span>' : ''}
    </div>
    ${regenerating ? referenceProgress(rp) : ''}
  `;
}

function referenceProgress(progress) {
  const percent = Math.max(5, Math.min(100, Number(progress.percent) || 5));
  return `
    <div class="mini-progress">
      <div>${esc(progress.label || 'Regenerating reference')}</div>
      <progress value="${percent}" max="100"></progress>
    </div>
  `;
}

function missingImage(text) {
  return `
    <div class="missing-image" role="img" aria-label="${esc(text)}">
      <div class="missing-icon">[ ]</div>
      <div>${esc(text)}</div>
    </div>
  `;
}

function wireReferenceTimeControls() {
  const manifest = (state.shot_views && state.shot_views.references_manifest) || '';
  const rows = (state.shot_views && state.shot_views.references) || [];

  for (const row of rows) {
    const slider = document.getElementById(`shotSlider_references_${row.index}`);
    if (!slider) continue;

    slider.disabled = false;
    slider.min = row.start;
    slider.max = row.end;

    let tools = slider.parentElement.querySelector('.reference-time-tools');
    if (!tools) {
      tools = document.createElement('div');
      tools.className = 'shot-tools reference-time-tools';
      slider.parentElement.appendChild(tools);
    }

    tools.innerHTML = `
      <button type="button" onclick="scrubShot('${esc(manifest)}',${row.index},document.getElementById('shotSlider_references_${row.index}').value)">
        Use Frame
      </button>
    `;
  }
}

function formatSeconds(value) {
  const total = Math.max(0, Number(value) || 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = (total % 60).toFixed(3).padStart(6, '0');
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s}`;
}

function parseDuration(value) {
  const text = String(value || '').trim();
  if (!text) return 0;

  const parts = text.split(':').map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];

  const n = Number(text);
  return Number.isFinite(n) ? n : 0;
}

let aspectPreviewTimer = null;

function updateAspectPreview(time) {
  const label = document.getElementById('aspectPreviewLabel');
  if (label) label.textContent = formatSeconds(time);

  clearTimeout(aspectPreviewTimer);
  aspectPreviewTimer = setTimeout(async () => {
    const r = await api('/api/aspect-preview?time=' + encodeURIComponent(time));
    const img = document.getElementById('aspectPreviewImg');
    if (r.ok && r.path && img) img.src = media(r.path) + '&t=' + Date.now();
  }, 160);
}
