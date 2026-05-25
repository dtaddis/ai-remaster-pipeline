function drawRecomp() {
  const st = stage('recomp');
  const s = settings('recomp');
  const expected = (state.expected_outputs && state.expected_outputs.recomp) || [];
  const sp = stageProgress('recomp');

  document.getElementById('app').innerHTML = `
    <div class="editor-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        ${progressHtml(sp.percent, sp.label)}
        ${recompLayerSummary(s)}
        ${recompPathFields(st)}
        <h3>Blend Parameters</h3>
        ${recompControlFields(st)}
        ${shotOutputList(expected, null)}
        ${stageCheckboxes(s)}
        <div class="actions">
          <button class="primary" onclick="runStage('recomp')" ${state.running ? 'disabled' : ''}>Run Recomposition</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card editor-viewer">
        <h2>Live Composite Preview</h2>
        ${liveCompositeHtml(s)}
        ${layerPreviewHtml(s)}
        ${recompTimelineHtml()}
      </section>
    </div>
    <section class="card" style="margin-top:16px">${runLogHtml()}</section>
  `;

  bindStageFields('recomp');
  wireEditorVideo();
  showCommand('recomp');
}

function recompLayerSummary(s) {
  return `
    <div class="layer-grid">
      ${recompLayerItem('Top layer - Color blend', s.colorized_video, 'Colorized video not set')}
      ${recompLayerItem('Middle layer', s.source, 'Original source not set')}
      ${recompLayerItem('Bottom layer', s.outpainted_video, 'Outpainted video not set')}
    </div>
  `;
}

function recompLayerItem(label, path, fallback) {
  return `
    <div class="layer-item">
      <span>${esc(label)}</span>
      <div class="layer-file-row">
        <strong>${esc(path || fallback)}</strong>
        <button
          class="icon-button inline"
          type="button"
          title="Save this layer as..."
          onclick="exportMedia(${jsArg(path)})"
          ${path ? '' : 'disabled'}
        >&#128190;</button>
      </div>
    </div>
  `;
}

function recompPathFields(st) {
  return ['outpainted_video', 'source', 'colorization_method', 'colorized_video']
    .map(key => fieldHtml(st, st.fields.find(f => f[0] === key)))
    .join('');
}

function recompControlFields(st) {
  const controls = ['feather_pixels', 'saturation', 'temperature', 'color_opacity', 'encoder'];
  return `
    <div class="editor-controls">
      ${controls.map(key => `<div>${fieldHtml(st, st.fields.find(f => f[0] === key))}</div>`).join('')}
    </div>
  `;
}

function recompLayerToggles() {
  return `
    <div class="checks layer-toggles">
      <label><input type="checkbox" id="showLayerColor" checked onchange="updateRecompPreview()">Color</label>
      <label><input type="checkbox" id="showLayerOriginal" checked onchange="updateRecompPreview()">Original</label>
      <label><input type="checkbox" id="showLayerOutpaint" checked onchange="updateRecompPreview()">Outpainted</label>
    </div>
  `;
}

function recompTimelineHtml() {
  return `
    <div class="timeline">
      <input id="recompScrub" type="range" min="0" max="1000" value="0" oninput="scrubEditorVideo(this.value)">
      <p class="shot-empty">Use the checkboxes below to inspect the contribution of each layer.</p>
      ${recompLayerToggles()}
    </div>
  `;
}

function drawOutput() {
  const selection = state.output_selection || {};
  const path = selection.path || '';

  document.getElementById('app').innerHTML = `
    <section class="card editor-viewer">
      <h2>Output</h2>
      ${path ? outputVideoHtml(path, selection) : '<p class="shot-empty">Run Recomposition to create a composited render.</p>'}
    </section>
  `;
}

function drawUpscale() {
  const st = stage('upscale');
  const s = settings('upscale');
  const expected = (state.expected_outputs && state.expected_outputs.upscale) || [];
  const sp = stageProgress('upscale');

  document.getElementById('app').innerHTML = `
    <div class="editor-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        ${progressHtml(sp.percent, sp.label)}
        ${upscaleDependencyNote(s)}
        ${upscaleFields(st)}
        ${shotOutputList(expected, null)}
        ${stageCheckboxes(s)}
        <div class="actions">
          <button type="button" onclick="generateUpscalePreview()" ${state.running ? 'disabled' : ''}>Generate Preview</button>
          <button class="primary" onclick="runStage('upscale')" ${state.running ? 'disabled' : ''}>Run Upscaling</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card editor-viewer">
        <h2>Upscale Preview</h2>
        ${upscalePreviewHtml(s, expected[0] || s.output || '')}
      </section>
    </div>
    <section class="card" style="margin-top:16px">${runLogHtml()}</section>
  `;

  bindStageFields('upscale');
  wireUpscaleComparison();
  showCommand('upscale');
}

function upscaleFields(st) {
  return ['input_video', 'method', 'scale', 'output', 'preview_seconds']
    .map(key => fieldHtml(st, st.fields.find(f => f[0] === key)))
    .join('');
}

function upscaleDependencyNote(s) {
  return `
    <div class="inline-warning">
      RealBasicVSR runs after Recomposition. ARP will try the current Python environment first; if the repo or checkpoint is missing, the run log will tell you what to install.
    </div>
  `;
}

function upscalePreviewHtml(s, outputPath) {
  const input = s.input_video || settings('recomp').output || '';
  const seconds = Math.max(1, Number(s.preview_seconds || 6));
  const preview = state.upscale_preview || {};
  const compareOutput = preview.exists ? preview.output : (preview.full_output_exists ? (preview.full_output || outputPath) : '');
  const sourceClip = input ? mediaClip(input, 0, seconds, 'upscale_compare_input_' + seconds) : '';
  const outputClip = compareOutput
    ? (preview.exists ? media(compareOutput) : mediaClip(compareOutput, 0, seconds, 'upscale_compare_output_' + seconds))
    : '';
  if (!input) return missingImage('Input video not present');
  if (!compareOutput) return missingImage('Generate a preview or run upscaling to compare output');

  return `
    <div class="comparison-player" style="--split:50%">
      <video id="upscaleBefore" src="${sourceClip}" controls preload="metadata"></video>
      <div class="comparison-after">
        <video id="upscaleAfter" src="${outputClip}" muted preload="metadata"></video>
      </div>
      <div class="comparison-divider"></div>
      <div class="comparison-label before">Before</div>
      <div class="comparison-label after">After</div>
    </div>
    <div class="comparison-controls">
      <input id="upscaleSplit" type="range" min="0" max="100" value="50" oninput="setUpscaleSplit(this.value)">
      <div class="comparison-meta">
        <span>${preview.exists ? 'Preview output' : 'Upscaled output sample'}</span>
        <strong>${esc(compareOutput)}</strong>
      </div>
    </div>
    <p class="shot-empty">The comparison uses the first ${esc(seconds)} seconds. Drag the split to inspect before and after on the same frame.</p>
  `;
}

function setUpscaleSplit(value) {
  const player = document.querySelector('.comparison-player');
  if (!player) return;
  const split = Math.max(0, Math.min(100, Number(value) || 0));
  player.style.setProperty('--split', split + '%');
}

function wireUpscaleComparison() {
  const before = document.getElementById('upscaleBefore');
  const after = document.getElementById('upscaleAfter');
  if (!(before && after)) return;

  const syncAfter = force => {
    if (!after.readyState) return;
    const tolerance = force ? 0.02 : 0.16;
    if (Math.abs((after.currentTime || 0) - (before.currentTime || 0)) <= tolerance) return;
    try {
      after.currentTime = before.currentTime || 0;
    } catch {}
  };

  before.addEventListener('loadedmetadata', () => syncAfter(true));
  before.addEventListener('play', () => {
    syncAfter(true);
    after.play().catch(() => {});
  });
  before.addEventListener('pause', () => after.pause());
  before.addEventListener('seeking', () => syncAfter(true));
  before.addEventListener('timeupdate', () => syncAfter(false));
  before.addEventListener('ratechange', () => {
    after.playbackRate = before.playbackRate;
  });
}

function outputVideoHtml(path, selection = {}) {
  return `
    <div class="source-info output-choice">
      <div><span>Selected output</span><strong>${esc(selection.label || 'Output')}</strong></div>
      <div><span>Preference</span><strong>${selection.kind === 'upscaled' ? 'Upscaled output found' : 'Using composited render'}</strong></div>
    </div>
    <video src="${media(path)}" controls preload="metadata"></video>
    <h3>${esc(selection.label || 'Output')}</h3>
    <ul class="output-list"><li>${esc(path)}</li></ul>
  `;
}

function liveCompositeHtml(s) {
  if (!s.outpainted_video && !s.source && !s.colorized_video) {
    return '<p class="shot-empty">Run the earlier phases to preview the live composite.</p>';
  }

  return `
    <div class="live-composite">
      ${s.outpainted_video ? `<video id="recompVideo" class="sync-layer-video live-outpaint" src="${media(s.outpainted_video)}" controls preload="metadata"></video>` : ''}
      ${s.source ? `<video class="sync-layer-video live-original" src="${media(s.source)}" muted preload="metadata" style="${originalLayerStyle(s)}"></video>` : ''}
      ${s.colorized_video ? `<video class="sync-layer-video live-color" src="${media(s.colorized_video)}" muted preload="metadata" style="${colorLayerStyle(s)}"></video>` : ''}
    </div>
  `;
}

function layerPreviewHtml(s) {
  return `
    <div class="layer-preview-grid">
      <div><label>Outpainted</label>${layerVideo(s.outpainted_video, 'layer-outpaint')}</div>
      <div><label>Original, feathered</label>${layerVideo(s.source, 'layer-original', originalFeatherStyle(s))}</div>
      <div><label>Color</label>${layerVideo(s.colorized_video, 'layer-colour', colorLayerStyle(s))}</div>
    </div>
  `;
}

function layerVideo(path, cls, style = '') {
  if (!path) return missingImage('Video not present');
  return `<video class="sync-layer-video ${cls}" src="${media(path)}" muted preload="metadata" style="${style}"></video>`;
}

function colorLayerStyle(s) {
  const saturation = Math.max(0, Number(s.saturation || 1));
  const temp = Number(s.temperature || 0);
  const opacity = Math.max(0, Math.min(1, Number(s.color_opacity || 1)));
  const hue = temp === 0 ? 0 : (temp > 0 ? -10 : 10) * Math.min(3, Math.abs(temp) * 30);
  return `filter:saturate(${saturation});opacity:${opacity};hue-rotate(${hue}deg)`;
}

function originalFeatherStyle(s) {
  const feather = Math.max(1, Number(s.feather_pixels || 80));
  const edge = Math.max(2, Math.min(45, feather / 8));
  return `-webkit-mask-image:linear-gradient(90deg,transparent 0,#000 ${edge}%,#000 ${100 - edge}%,transparent 100%);mask-image:linear-gradient(90deg,transparent 0,#000 ${edge}%,#000 ${100 - edge}%,transparent 100%)`;
}

function originalLayerStyle(s) {
  return `${originalLayerBoxStyle()};${originalFeatherStyle(s)};object-fit:fill`;
}

function originalLayerBoxStyle() {
  const sourceAspect = sourceAspectRatio();
  const targetAspect = targetAspectRatio();
  if (!sourceAspect || !targetAspect) return '';

  if (sourceAspect <= targetAspect) {
    const width = Math.max(1, Math.min(100, (sourceAspect / targetAspect) * 100));
    const left = (100 - width) / 2;
    return `width:${width}%;height:100%;left:${left}%;right:auto;top:0;bottom:auto`;
  }

  const height = Math.max(1, Math.min(100, (targetAspect / sourceAspect) * 100));
  const top = (100 - height) / 2;
  return `width:100%;height:${height}%;top:${top}%;bottom:auto;left:0;right:auto`;
}

function sourceAspectRatio() {
  const text = (state.source_info && state.source_info.aspect) || '';
  const value = Number(String(text).split(':')[0]);
  return Number.isFinite(value) && value > 0 ? value : 4 / 3;
}

function targetAspectRatio() {
  const value = settings('outpaint').target_aspect || '16:9';
  const parts = String(value).split(':').map(Number);
  if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) return parts[0] / parts[1];
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : 16 / 9;
}

function wireEditorVideo() {
  const mainVideo = document.getElementById('recompVideo');
  const scrubber = document.getElementById('recompScrub');
  const layers = [...document.querySelectorAll('.sync-layer-video')].filter(item => item !== mainVideo);

  if (!mainVideo || !scrubber) return;

  const syncLayers = force => {
    const tolerance = force ? 0.02 : 0.18;
    for (const item of layers) {
      if (!item.readyState) continue;
      if (Math.abs((item.currentTime || 0) - (mainVideo.currentTime || 0)) <= tolerance) continue;
      try {
        item.currentTime = mainVideo.currentTime || 0;
      } catch {}
    }
  };

  mainVideo.addEventListener('loadedmetadata', () => {
    scrubber.value = 0;
    syncLayers(true);
  });
  mainVideo.addEventListener('play', () => {
    syncLayers(true);
    layers.forEach(item => item.play().catch(() => {}));
  });
  mainVideo.addEventListener('pause', () => layers.forEach(item => item.pause()));
  mainVideo.addEventListener('seeking', () => syncLayers(true));
  mainVideo.addEventListener('timeupdate', () => {
    if (mainVideo.duration && !scrubber.matches(':active')) {
      scrubber.value = Math.round((mainVideo.currentTime / mainVideo.duration) * 1000);
    }
    syncLayers(false);
  });
  mainVideo.addEventListener('ratechange', () => {
    layers.forEach(item => {
      item.playbackRate = mainVideo.playbackRate;
    });
  });
  updateRecompPreview();
}

function scrubEditorVideo(value) {
  const mainVideo = document.getElementById('recompVideo');
  if (!mainVideo || !mainVideo.duration) return;

  mainVideo.currentTime = ((Number(value) || 0) / 1000) * mainVideo.duration;
  document.querySelectorAll('.sync-layer-video').forEach(item => {
    try {
      item.currentTime = mainVideo.currentTime;
    } catch {}
  });
}

function updateRecompPreview() {
  const showOutpaint = document.getElementById('showLayerOutpaint')?.checked ?? true;
  const showOriginal = document.getElementById('showLayerOriginal')?.checked ?? true;
  const showColor = document.getElementById('showLayerColor')?.checked ?? true;
  document.querySelectorAll('.live-outpaint,.layer-outpaint').forEach(el => { el.style.visibility = showOutpaint ? 'visible' : 'hidden'; });
  document.querySelectorAll('.live-original,.layer-original').forEach(el => { el.style.visibility = showOriginal ? 'visible' : 'hidden'; });
  document.querySelectorAll('.live-color,.layer-colour').forEach(el => { el.style.visibility = showColor ? 'visible' : 'hidden'; });
}
