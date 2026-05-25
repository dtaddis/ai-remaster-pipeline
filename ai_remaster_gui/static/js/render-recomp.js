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
  const expected = (state.expected_outputs && state.expected_outputs.output) || [];
  const path = expected[0] || settings('recomp').output || '';

  document.getElementById('app').innerHTML = `
    <section class="card editor-viewer">
      <h2>Output</h2>
      ${path ? outputVideoHtml(path) : '<p class="shot-empty">Run Recomposition to create the final movie.</p>'}
    </section>
  `;
}

function outputVideoHtml(path) {
  return `
    <video src="${media(path)}" controls preload="metadata"></video>
    <h3>Final output</h3>
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
