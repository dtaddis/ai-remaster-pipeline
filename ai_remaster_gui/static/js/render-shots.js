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
  const visibleFields = shotStageVisibleFields(st, s);

  document.getElementById('app').innerHTML = `
    <div class="shot-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        ${progressHtml(sp.percent, sp.label)}
        ${key === 'colour' ? colorizationMethodWarning(s) : ''}
        ${key === 'shots' ? shotDetectionInputStatus(s) : ''}
        ${visibleFields.map(f => fieldHtml(st, f)).join('')}
        ${key === 'references' ? referenceGenerationOptionsHtml(s) : ''}
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
  `;

  bindStageFields(key);
  if (afterRender) afterRender();
  showCommand(key);
}

function shotStageVisibleFields(st, s = {}) {
  if (st.key === 'shots') return st.fields.filter(field => field[0] !== 'outpainted_video');
  if (st.key === 'references') return st.fields.filter(field => field[0] !== 'manifest');
  if (st.key === 'colour') {
    const common = new Set(['manifest', 'method', 'processing_height', 'crf']);
    const cloud = new Set(['openai_image_model', 'openai_previous_frames', 'openai_image_size', 'openai_image_quality', 'openai_prompt']);
    const deep = new Set(['frame_propagate', 'use_half_resolution', 'use_torch_compile', 'use_sage_attention']);
    const legacyColorMNet = new Set(['use_torch_compile', 'colormnet_memory_mode', 'colormnet_feature_encoder', 'colormnet_text_guidance', 'colormnet_text_guidance_weight']);
    const method = s.method || 'deepexemplar';
    if (method === 'openai') return st.fields.filter(field => common.has(field[0]) || cloud.has(field[0]));
    if (method === 'cmnet2') return st.fields.filter(field => common.has(field[0]));
    if (method === 'colormnet') return st.fields.filter(field => common.has(field[0]) || legacyColorMNet.has(field[0]));
    if (method === 'deepexemplar') return st.fields.filter(field => common.has(field[0]) || deep.has(field[0]));
    return st.fields.filter(field => !cloud.has(field[0]));
  }
  return st.fields;
}

function referenceGenerationOptionsHtml(s) {
  if ((s.method || 'qwen') !== 'openai') return '';
  return `
    <div class="checks">
      <label>
        <input data-field="openai_send_references" type="checkbox" ${s.openai_send_references === 'true' ? 'checked' : ''}>
        Also send (3) previous images as references
      </label>
    </div>
  `;
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
  if (s.method === 'openai') {
    return '<div class="inline-warning">OpenAI Cloud makes one paid image-edit request per video frame. Completed frames are cached for safe resume. The existing colour is retained as evidence, unlike the compulsory grayscale input used by Deep Exemplar and ColorMNet.</div>';
  }
  if (s.method === 'cmnet2') {
    return '<div class="inline-warning">CMNET2 preloads every approved reference frame for each shot into permanent memory. It inherits ColorMNet\'s non-commercial CC BY-NC-SA 4.0 terms.</div>';
  }
  if (!['colormnet', 'both'].includes(s.method)) return '';
  return '<div class="inline-warning">ColorMNet uses Reference 1 only. Its custom node reports a CC BY-NC-SA 4.0 license; use it only for non-commercial work unless you have separate rights.</div>';
}

function shotCards(mode) {
  const view = state.shot_views || {};
  const rows = view[mode] || [];
  const manifest = view[mode + '_manifest'] || '';

  if (!rows.length) {
    return '<p class="shot-empty">No shot manifest yet. Run Shot Detection first.</p>';
  }

  return `<div class="shot-list">${rows.map(row => shotListEntry(mode, manifest, row)).join('')}</div>`;
}

function shotListEntry(mode, manifest, row) {
  return shotCard(mode, manifest, row) + shotTransitionControl(mode, manifest, row);
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
    ? `<button type="button" onclick="mergeShot(${jsArg(manifest)},${idx},this)">Merge Next</button>`
    : '';
  const splitButton = row.can_split
    ? `<button type="button" onclick="splitShot(${jsArg(manifest)},${idx},this)">Split</button>`
    : '';

  return `
    <article class="shot-card" data-shot-card-mode="shots" data-shot-card-index="${idx}">
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

function shotTransitionControl(mode, manifest, row) {
  if (mode !== 'shots' || !row.can_fade_next) return '';
  const checked = String(row.fade_to_next || '').toLowerCase() === 'true';
  const value = row.crossfade_seconds || '1.0';
  return `
    <div class="shot-transition" data-shot-transition-mode="${mode}" data-shot-transition-index="${row.index}">
      <span>Between shot ${row.index + 1} and ${row.index + 2}</span>
      <label>
        <input
          type="checkbox"
          id="fade_${row.index}"
          ${checked ? 'checked' : ''}
          onchange="saveShotFade('${esc(manifest)}',${row.index},this.checked,document.getElementById('crossfade_${row.index}').value)"
        >
        Fading transition
      </label>
      <label class="compact-field">
        Crossfade seconds
        <input
          id="crossfade_${row.index}"
          type="number"
          min="0.041"
          step="0.041"
          value="${esc(value)}"
          onchange="saveShotFade('${esc(manifest)}',${row.index},document.getElementById('fade_${row.index}')?.checked ?? ${checked},this.value)"
        >
      </label>
    </div>
  `;
}

function boundaryFrameCard({ manifest, row, idx }, edge) {
  const isStart = edge === 'start';
  const frame = isStart ? Number(row.start_frame || 0) : Number(row.end_boundary_frame || (Number(row.end_frame || 0) + 1));
  const displayFrame = isStart ? frame : Math.max(0, frame - 1);
  const preview = isStart ? row.start_preview : row.end_preview;
  const min = isStart ? Math.max(0, Number(row.previous_start_frame ?? (Number(row.start_frame || 0) - 1)) + 1) : Number(row.start_frame || 0) + 1;
  const max = isStart ? Number(row.end_boundary_frame || (Number(row.end_frame || 0) + 1)) - 1 : Number(row.next_end_boundary_frame ?? (frame + 1)) - 1;
  const disabled = isStart && idx === 0 ? 'disabled' : '';
  const label = isStart ? 'Start' : 'End';
  const fps = Math.max(1, Number(row.fps || 24));
  const imgId = `shotBoundaryImg_${edge}_${idx}`;
  const labelId = `shotBoundaryLabel_${edge}_${idx}`;
  const previewOffsetFrames = isStart ? 0 : -1;

  return `
    <div>
      <label id="${labelId}">${label} frame ${displayFrame}</label>
      ${preview ? `<img id="${imgId}" src="${media(preview)}" alt="">` : missingImage('Image not present')}
      <input
        type="range"
        min="${min}"
        max="${max}"
        step="1"
        value="${frame}"
        ${disabled}
        data-edge="${edge}"
        data-fps="${fps}"
        data-preview-offset-frames="${previewOffsetFrames}"
        oninput="updateShotBoundaryPreview('${esc(manifest)}',${idx},this.value,'${imgId}','${labelId}',this.dataset)"
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
    <article class="shot-card" data-shot-card-mode="colour" data-shot-card-index="${idx}">
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
  if (method === 'openai') return 'OpenAI Cloud';
  if (method === 'cmnet2') return 'CMNET2';
  if (method === 'colormnet') return 'ColorMNet';
  if (method === 'both') return 'Deep Exemplar and ColorMNet';
  return 'Deep Exemplar';
}

function referenceCard(context) {
  const { manifest, row, idx } = context;
  const items = row.reference_items || [];
  const primaryTime = Number(row.selected_time || row.start || 0);
  const primaryFrame = Number(row.selected_frame || row.start_frame || 0);
  return `
    <article class="shot-card reference-shot-card" data-shot-card-mode="references" data-shot-card-index="${idx}">
      ${shotSummary(context, `<p class="shot-empty">${items.length} reference frame${items.length === 1 ? '' : 's'}</p>`)}
      <div class="reference-shot-workspace">
        <div class="reference-use-note">
          <strong>Reference frames</strong>
          <span>CMNET2 and OpenAI Cloud use every reference frame. ColorMNet and Deep Exemplar use Reference 1 only.</span>
        </div>
        <div class="reference-anchor-grid">
          ${items.map(item => referenceAnchorCard(manifest, row, idx, item)).join('')}
        </div>
        <div class="shot-tools reference-add-row">
          <button type="button" onclick="addReferenceAtCurrentFrame('${esc(manifest)}',${idx},${primaryTime},${primaryFrame})">+ Add Reference</button>
          <span class="shot-time">Add a reference, then scrub its frame slider and press Use Frame.</span>
        </div>
        <div class="reference-shot-prompt">${referencePromptTools(context)}</div>
      </div>
    </article>
  `;
}

function referenceAnchorCard(manifest, row, idx, item) {
  const referenceIndex = Number(item.reference_index || 0);
  const sourceReady = item.source_reference && item.source_reference_mtime;
  const colorReady = item.color_reference && item.color_reference_mtime;
  const sourceUrl = sourceReady ? media(item.source_reference) + '&t=' + item.source_reference_mtime : '';
  const colorUrl = colorReady ? media(item.color_reference) + '&t=' + item.color_reference_mtime : '';
  const fps = Math.max(1, Number(row.fps || 24));
  const frame = Math.max(Number(row.start_frame || 0), Math.min(Number(row.end_frame || 0), Number(item.selected_frame || row.start_frame || 0)));
  const slider = `referenceSlider_${idx}_${referenceIndex}`;
  const label = `referenceLabel_${idx}_${referenceIndex}`;
  const image = `referenceImage_${idx}_${referenceIndex}`;
  const regenerating = state.running_reference
    && state.running_reference.index === idx
    && state.running_reference.manifest === manifest;
  return `
    <section class="reference-anchor-card">
      <div class="reference-anchor-heading">
        <strong>Reference ${referenceIndex + 1}</strong>
        <span class="shot-time" id="${label}">Frame ${frame} · ${esc(item.selected_label || '')}</span>
      </div>
      <div class="reference-anchor-images">
        <div>
          <label>Source frame</label>
          ${sourceReady ? `<div class="thumb-wrap"><img id="${image}" src="${sourceUrl}" alt="B&W keyframe" onclick="openImageModal(this.src,${jsArg('B&W keyframe')})"><button class="icon-button" type="button" title="Save source frame" onclick="exportMedia('${esc(item.source_reference)}')">&#128190;</button></div>` : `<img id="${image}" alt="Scrub to preview this source frame">`}
        </div>
        <div>
          <label>Colour reference</label>
          ${colorReady ? (referenceIndex === 0
            ? colorReferenceThumb(manifest, idx, colorUrl, row, 0)
            : `<div class="thumb-wrap"><img src="${colorUrl}" alt="Colour keyframe" onclick="openImageModal(this.src,${jsArg('Colour keyframe')})"><button class="icon-button" type="button" title="Delete colour reference" onclick="deleteReference('${esc(manifest)}',${idx},${referenceIndex})">&#128465;</button></div>`)
            : missingImage('Colour reference missing')}
        </div>
      </div>
      <label>Reference frame</label>
      <input
        id="${slider}"
        type="range"
        min="${Number(row.start_frame || 0)}"
        max="${Number(row.end_frame || 0)}"
        step="1"
        value="${frame}"
        data-reference-anchor-slider="true"
        data-shot-index="${idx}"
        data-reference-index="${referenceIndex}"
        data-fps="${fps}"
        oninput="this.dataset.referenceTimeDirty='true';updateReferenceAnchorPreview('${esc(manifest)}',${idx},${referenceIndex},this.value,'${image}','${label}',${fps})"
      >
      <div class="shot-tools">
        <button type="button" onclick="useReferenceAnchorFrame('${esc(manifest)}',${idx},${referenceIndex},document.getElementById('${slider}').value,${fps})">Use Frame</button>
        <button type="button" onclick="chooseCustomReference('${esc(manifest)}',${idx},${referenceIndex})">Use Custom Image</button>
        <button type="button" onclick="regenerateReference('${esc(manifest)}',${idx},${referenceIndex})" ${state.running ? 'disabled' : ''}>${regenerating ? 'Generating...' : 'Generate Reference'}</button>
        ${colorReady ? `<button type="button" onclick="deleteReference('${esc(manifest)}',${idx},${referenceIndex})">Delete Colour</button>` : ''}
        ${referenceIndex > 0 ? `<button type="button" class="danger" onclick="removeAdditionalReference('${esc(manifest)}',${idx},${referenceIndex})">Remove Reference</button>` : ''}
      </div>
    </section>
  `;
}

function refreshShotRows(mode, indices) {
  const view = state.shot_views || {};
  const rows = view[mode] || [];
  const manifest = view[mode + '_manifest'] || '';
  const unique = [...new Set(indices)]
    .filter(index => Number.isInteger(index) && index >= 0 && index < rows.length)
    .sort((a, b) => a - b);

  for (const index of unique) {
    const row = rows[index];
    const card = document.querySelector(`[data-shot-card-mode="${mode}"][data-shot-card-index="${index}"]`);
    if (card) {
      card.outerHTML = shotCard(mode, manifest, row);
    }

    const transition = document.querySelector(`[data-shot-transition-mode="${mode}"][data-shot-transition-index="${index}"]`);
    const transitionHtml = shotTransitionControl(mode, manifest, row);
    if (transition && transitionHtml) {
      transition.outerHTML = transitionHtml;
    } else if (transition) {
      transition.remove();
    } else if (transitionHtml) {
      const updatedCard = document.querySelector(`[data-shot-card-mode="${mode}"][data-shot-card-index="${index}"]`);
      if (updatedCard) updatedCard.insertAdjacentHTML('afterend', transitionHtml);
    }
  }

  if (mode === 'references') wireReferenceTimeControls();
}

function updateReferencesDynamicStatus() {
  const sp = stageProgress('references');
  const progressEl = document.querySelector('.shot-page > section:first-child .phase-progress');
  if (progressEl) progressEl.outerHTML = progressHtml(sp.percent, sp.label);

  const dirtyRows = new Set(
    [...document.querySelectorAll('[data-reference-anchor-slider][data-reference-time-dirty="true"]')]
      .map(el => Number(el.dataset.shotIndex))
      .filter(Number.isInteger)
  );
  const rows = ((state.shot_views && state.shot_views.references) || [])
    .map(row => row.index)
    .filter(index => !dirtyRows.has(index));
  refreshShotRows('references', rows);
}

function colorReferenceThumb(manifest, idx, colorUrl, row, referenceIndex = 0) {
  return `
    <div class="thumb-wrap">
      <img src="${colorUrl}" alt="" onclick="openReferenceEditor('${esc(manifest)}',${idx})" title="Open advanced reference editor">
      ${row.color_reference_edited ? '<span class="edit-badge">Edited</span>' : ''}
      <button class="icon-button" type="button" title="Delete color reference" onclick="deleteReference('${esc(manifest)}',${idx},${referenceIndex})">&#128465;</button>
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
        ${regenerating ? 'Generating...' : 'Generate Reference'}
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
  document.querySelectorAll('[data-reference-anchor-slider]').forEach(slider => { slider.disabled = false; });
}

function wireColourShotVideos() {
  if (active !== 'colour') return;
  document.querySelectorAll('.shot-card video').forEach((video, index) => {
    try {
      const url = new URL(video.getAttribute('src'), window.location.href);
      const hash = url.hash || '';
      if (!hash.startsWith('#t=')) return;
      const parts = hash.slice(3).split(',');
      if (parts.length < 2) return;
      video.src = mediaClip(url.searchParams.get('path') || '', parts[0], parts[1], 'colour_' + index);
      video.removeAttribute('data-cued');
    } catch {
      // Leave the original source in place if a card has an invalid media URL.
    }
  });
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
