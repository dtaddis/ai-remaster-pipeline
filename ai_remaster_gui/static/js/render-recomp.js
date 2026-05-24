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
      <div class="layer-item">
        <span>Top layer - Color blend</span>
        <strong>${esc(s.colorized_video || 'Colorized video not set')}</strong>
      </div>
      <div class="layer-item">
        <span>Middle layer</span>
        <strong>${esc(s.source || 'Original source not set')}</strong>
      </div>
      <div class="layer-item">
        <span>Bottom layer</span>
        <strong>${esc(s.outpainted_video || 'Outpainted video not set')}</strong>
      </div>
    </div>
  `;
}

function recompPathFields(st) {
  return ['outpainted_video', 'source', 'colorized_video']
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

function recompTimelineHtml() {
  return `
    <div class="timeline">
      <input id="recompScrub" type="range" min="0" max="1000" value="0" oninput="scrubEditorVideo(this.value)">
      ${timelineTrack('Color', 'track-colour')}
      ${timelineTrack('Original', 'track-original')}
      ${timelineTrack('Outpainted', 'track-outpaint')}
    </div>
  `;
}

function timelineTrack(name, cls) {
  return `
    <div class="track">
      <div class="track-name">${name}</div>
      <div class="track-bar ${cls}"></div>
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
      ${s.source ? `<video class="sync-layer-video live-original" src="${media(s.source)}" muted preload="metadata"></video>` : ''}
      ${s.colorized_video ? `<video class="sync-layer-video live-color" src="${media(s.colorized_video)}" muted preload="metadata"></video>` : ''}
    </div>
  `;
}

function layerPreviewHtml(s) {
  return `
    <div class="layer-preview-grid">
      <div><label>Outpainted</label>${layerVideo(s.outpainted_video, 'layer-outpaint')}</div>
      <div><label>Original, feathered</label>${layerVideo(s.source, 'layer-original')}</div>
      <div><label>Color</label>${layerVideo(s.colorized_video, 'layer-colour')}</div>
    </div>
  `;
}

function layerVideo(path, cls) {
  if (!path) return missingImage('Video not present');
  return `<video class="sync-layer-video ${cls}" src="${media(path)}" muted preload="metadata"></video>`;
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
