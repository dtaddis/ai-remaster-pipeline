let recompMaskFrame = 0;
let recompColorFrame = 0;

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
  const outpaintFallback = s.outpainted_video ? 'Outpainted video not set' : 'N/A - colourisation-only workflow';
  return `
    <div class="layer-grid">
      ${recompLayerItem('Top layer - Color blend', s.colorized_video, 'Colorized video not set')}
      ${recompLayerItem('Middle layer', s.source, 'Original source not set')}
      ${recompLayerItem('Bottom layer', s.outpainted_video, outpaintFallback)}
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
  const hasOutpainted = !!settings('recomp').outpainted_video;
  const controls = [
    ...(hasOutpainted ? ['feather_pixels'] : []),
    'saturation',
    'temperature',
    'color_opacity',
    'encoder',
  ];
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
  const selected = state.output_selection || {};

  document.getElementById('app').innerHTML = `
    <section class="card editor-viewer">
      <h2>Output</h2>
      ${path ? outputVideoHtml(path, selected) : '<p class="shot-empty">Run a selected workflow step to create an output movie.</p>'}
    </section>
  `;
}

function outputVideoHtml(path, selected = {}) {
  const label = selected.label || 'Selected output';
  return `
    <video src="${media(path)}" controls preload="metadata"></video>
    <h3>${esc(label)}</h3>
    <ul class="output-list"><li>${esc(path)}</li></ul>
  `;
}

function drawUpscale() {
  const st = stage('upscale');
  const s = settings('upscale');
  const expected = (state.expected_outputs && state.expected_outputs.upscale) || [];
  const preview = state.upscale_preview || {};
  const sp = stageProgress('upscale');

  document.getElementById('app').innerHTML = `
    <div class="editor-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        ${progressHtml(sp.percent, sp.label)}
        ${upscaleInputSummary(s)}
        ${upscaleMainFields(st)}
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
        <h2>${esc(preview.title || 'Upscale Preview')}</h2>
        ${upscaleComparisonHtml(s, preview)}
      </section>
    </div>
    <section class="card" style="margin-top:16px">${runLogHtml()}</section>
  `;

  bindStageFields('upscale');
  bindUpscaleComparison();
  showCommand('upscale');
}

function upscaleMainFields(st) {
  const fieldKeys = [
    'flashvsr_model', 'flashvsr_mode', 'flashvsr_scale',
    'flashvsr_pre_downscale',
    'flashvsr_tiled_dit', 'flashvsr_tile_size', 'flashvsr_tile_overlap',
    'flashvsr_vae_tile_multiplier',
    'flashvsr_local_range', 'flashvsr_sparse_ratio', 'flashvsr_kv_ratio',
    'flashvsr_color_fix', 'flashvsr_tiled_vae', 'flashvsr_unload_dit',
    'flashvsr_seed',
  ];
  fieldKeys.push('target_width', 'target_height', 'chunk_seconds', 'overlap_frames', 'preview_seconds');
  return fieldKeys
    .map(key => fieldHtml(st, st.fields.find(f => f[0] === key)))
    .join('');
}

function upscaleInputSummary(s) {
  const source = (state.upscale_preview && state.upscale_preview.source) || s.input_video || '';
  const label = (state.output_selection && state.output_selection.kind || '').startsWith('recomposed')
    ? 'Recomposition output'
    : 'Selected source or source section';
  return `
    <div class="layer-item">
      <span>Input source</span>
      <strong>${esc(source || 'Choose source material on the Overview page')}</strong>
      <p class="shot-empty">${esc(label)} is selected automatically from the active workflow.</p>
    </div>
  `;
}

function upscaleComparisonHtml(s, preview) {
  const before = preview.source || s.input_video || '';
  const after = preview.exists === 'true' ? preview.output : '';
  if (!before) return '<p class="shot-empty">Choose a source on the Overview page, then enable Upscale.</p>';
  if (!after) {
    return `
      <video src="${media(before)}" controls preload="metadata"></video>
      <p class="shot-empty">Generate a preview to compare FlashVSR output against the input.</p>
    `;
  }
  return `
    <div class="comparison-player">
      <video class="compare-before" src="${media(before)}" controls preload="metadata"></video>
      <video class="compare-after" src="${media(after)}" muted preload="metadata"></video>
      <div class="compare-after-mask" style="width:50%"></div>
      <div class="compare-handle" style="left:50%"></div>
    </div>
    <input id="upscaleCompareSlider" class="compare-slider" type="range" min="0" max="100" value="50" aria-label="Before after split">
    <div class="source-info">
      <div><span>Before</span><strong>${esc(before)}</strong></div>
      <div><span>After</span><strong>${esc(after)}</strong></div>
    </div>
  `;
}

function bindUpscaleComparison() {
  const before = document.querySelector('.compare-before');
  const after = document.querySelector('.compare-after');
  const slider = document.getElementById('upscaleCompareSlider');
  const mask = document.querySelector('.compare-after-mask');
  const handle = document.querySelector('.compare-handle');
  if (!before || !after || !slider || !mask || !handle) return;

  const setSplit = () => {
    after.style.clipPath = `inset(0 ${100 - Number(slider.value)}% 0 0)`;
    mask.style.width = slider.value + '%';
    handle.style.left = slider.value + '%';
  };
  const sync = force => {
    if (!after.readyState) return;
    const tolerance = force ? 0.02 : 0.18;
    if (Math.abs((after.currentTime || 0) - (before.currentTime || 0)) > tolerance) {
      try { after.currentTime = before.currentTime || 0; } catch {}
    }
  };

  slider.addEventListener('input', setSplit);
  before.addEventListener('play', () => { sync(true); after.play().catch(() => {}); });
  before.addEventListener('pause', () => after.pause());
  before.addEventListener('seeking', () => sync(true));
  before.addEventListener('timeupdate', () => sync(false));
  before.addEventListener('ratechange', () => { after.playbackRate = before.playbackRate; });
  setSplit();
}

function liveCompositeHtml(s) {
  if (!s.outpainted_video && !s.source && !s.colorized_video) {
    return '<p class="shot-empty">Run the earlier phases to preview the live composite.</p>';
  }

  return `
    <div class="live-composite">
      ${s.outpainted_video ? `<video id="recompVideo" class="sync-layer-video live-outpaint" src="${media(s.outpainted_video)}" controls preload="metadata"></video>` : ''}
      ${originalLiveLayerHtml(s, !s.outpainted_video)}
      ${s.colorized_video ? `<video class="sync-layer-video color-preview-source" src="${media(s.colorized_video)}" muted preload="metadata" style="visibility:hidden;pointer-events:none"></video><canvas class="live-color live-color-accurate"></canvas>` : ''}
    </div>
  `;
}

function originalLiveLayerHtml(s, asBase = false) {
  if (!s.source) return '';
  const id = asBase ? ' id="recompVideo"' : '';
  const controls = asBase ? ' controls' : '';
  const cls = asBase ? 'live-outpaint live-original' : 'live-original';
  const style = asBase ? 'object-fit:contain' : originalLayerStyle(s);
  if (!sourceBlackTransparent()) {
    return `<video${id} class="sync-layer-video ${cls}" src="${media(s.source)}"${asBase ? '' : ' muted'}${controls} preload="metadata" style="${style}"></video>`;
  }
  if (asBase) return `<video${id} class="sync-layer-video ${cls}" src="${media(s.source)}" controls preload="metadata" style="${style}"></video>`;
  return `
    <video class="sync-layer-video source-mask-video" src="${media(s.source)}" muted preload="metadata"></video>
    <canvas class="live-original source-mask-canvas" style="${originalLayerStyle(s)}"></canvas>
  `;
}

function layerPreviewHtml(s) {
  return `
    <div class="layer-preview-grid">
      <div><label>Outpainted</label>${s.outpainted_video ? layerVideo(s.outpainted_video, 'layer-outpaint') : missingImage('N/A for colourisation-only')}</div>
      <div><label>Original, feathered</label>${originalLayerPreviewHtml(s)}</div>
      <div><label>Color</label>${layerVideo(s.colorized_video, 'layer-colour', colorLayerStyle(s))}</div>
    </div>
  `;
}

function originalLayerPreviewHtml(s) {
  if (!s.source) return missingImage('Video not present');
  if (!s.outpainted_video) return layerVideo(s.source, 'layer-original');
  if (!sourceBlackTransparent()) {
    return layerVideo(s.source, 'layer-original', originalFeatherStyle(s));
  }
  return `<canvas class="layer-original source-mask-canvas" style="${originalFeatherStyle(s)}"></canvas>`;
}

function layerVideo(path, cls, style = '') {
  if (!path) return missingImage('Video not present');
  return `<video class="sync-layer-video ${cls}" src="${media(path)}" muted preload="metadata" style="${style}"></video>`;
}

function sourceBlackTransparent() {
  return settings('outpaint').outpaint_all_black_regions === 'true';
}

function colorLayerStyle(s) {
  const rawSaturation = Math.max(0, Number(s.saturation || 100));
  const saturation = rawSaturation > 4 ? rawSaturation / 100 : rawSaturation;
  const temp = Number(s.temperature || 6500);
  const rawOpacity = Math.max(0, Number(s.color_opacity || 100));
  const opacity = Math.max(0, Math.min(1, rawOpacity > 1 ? rawOpacity / 100 : rawOpacity));
  const hue = Math.max(-18, Math.min(18, (6500 - temp) / 180));
  return `filter:saturate(${saturation}) hue-rotate(${hue}deg);opacity:${opacity}`;
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

  setupSourceBlackPreview();
  setupAccurateColorPreview();
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

function setupAccurateColorPreview() {
  if (recompColorFrame) {
    cancelAnimationFrame(recompColorFrame);
    recompColorFrame = 0;
  }
  const canvas = document.querySelector('.live-color-accurate');
  const color = document.querySelector('.color-preview-source');
  const main = document.getElementById('recompVideo');
  if (!canvas || !color || !main) return;

  const scratch = document.createElement('canvas');
  const scratchCtx = scratch.getContext('2d', { willReadFrequently: true });
  const draw = () => {
    const activeCanvas = document.querySelector('.live-color-accurate');
    const activeColor = document.querySelector('.color-preview-source');
    const activeMain = document.getElementById('recompVideo');
    if (!activeCanvas || !activeColor || !activeMain) {
      recompColorFrame = 0;
      return;
    }
    drawAccurateColorComposite(activeCanvas, activeMain, activeColor, scratch, scratchCtx);
    recompColorFrame = requestAnimationFrame(draw);
  };

  draw();
}

function drawAccurateColorComposite(canvas, mainVideo, colorVideo, scratch, scratchCtx) {
  if (!mainVideo.videoWidth || !mainVideo.videoHeight || mainVideo.readyState < 2 || colorVideo.readyState < 2) return;
  const scale = Math.min(1, 960 / mainVideo.videoWidth);
  const width = Math.max(2, Math.round(mainVideo.videoWidth * scale));
  const height = Math.max(2, Math.round(mainVideo.videoHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  if (scratch.width !== width || scratch.height !== height) {
    scratch.width = width;
    scratch.height = height;
  }

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.clearRect(0, 0, width, height);
  if (document.getElementById('showLayerOutpaint')?.checked ?? true) {
    ctx.drawImage(mainVideo, 0, 0, width, height);
  } else {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);
  }

  const sourceVideo = document.querySelector('.source-mask-video') || document.querySelector('.live-original:not(.live-outpaint)');
  const showOriginal = document.getElementById('showLayerOriginal')?.checked ?? true;
  if (sourceVideo && showOriginal && sourceVideo.readyState >= 2) {
    drawSourceLayerToContext(ctx, sourceVideo, width, height);
  }

  const base = ctx.getImageData(0, 0, width, height);
  scratchCtx.clearRect(0, 0, width, height);
  scratchCtx.drawImage(colorVideo, 0, 0, width, height);
  const color = scratchCtx.getImageData(0, 0, width, height);
  applyColorComposite(base.data, color.data, settings('recomp'));
  ctx.putImageData(base, 0, 0);
}

function drawSourceLayerToContext(ctx, sourceVideo, width, height) {
  const sourceAspect = sourceVideo.videoWidth / Math.max(1, sourceVideo.videoHeight);
  const drawH = height;
  const drawW = Math.max(2, Math.round(drawH * sourceAspect));
  const x = Math.round((width - drawW) / 2);
  const y = 0;
  const temp = document.createElement('canvas');
  temp.width = width;
  temp.height = height;
  const tctx = temp.getContext('2d', { willReadFrequently: true });
  tctx.drawImage(sourceVideo, x, y, drawW, drawH);
  const image = tctx.getImageData(0, 0, width, height);
  const data = image.data;
  const feather = Math.max(1, Number(settings('recomp').feather_pixels || 80));
  const threshold = sourceBlackTransparent() ? 24 : -1;
  for (let i = 0; i < data.length; i += 4) {
    const pixel = i / 4;
    const px = pixel % width;
    const edge = Math.min(px - x, x + drawW - px);
    let alpha = edge <= 0 ? 0 : Math.min(255, 255 * edge / feather);
    if (threshold >= 0 && Math.max(data[i], data[i + 1], data[i + 2]) <= threshold) alpha = 0;
    data[i + 3] = Math.min(data[i + 3], alpha);
  }
  tctx.putImageData(image, 0, 0);
  ctx.drawImage(temp, 0, 0);
}

function applyColorComposite(base, color, s) {
  const saturation = normalizedPercent(s.saturation || 100, 1);
  const opacity = Math.max(0, Math.min(1, normalizedPercent(s.color_opacity || 100, 1)));
  const [redBalance, blueBalance] = kelvinBalance(s.temperature || 6500);
  for (let i = 0; i < base.length; i += 4) {
    const br = base[i];
    const bg = base[i + 1];
    const bb = base[i + 2];
    let cr = color[i];
    let cg = color[i + 1];
    let cb = color[i + 2];
    [cr, cg, cb] = adjustSaturationRgb(cr, cg, cb, saturation);
    const shadow = 1 - Math.max(cr, cg, cb) / 255;
    cr = clampByte(cr + redBalance * shadow * 255);
    cb = clampByte(cb + blueBalance * shadow * 255);
    cr = br * (1 - opacity) + cr * opacity;
    cg = bg * (1 - opacity) + cg * opacity;
    cb = bb * (1 - opacity) + cb * opacity;
    const y = 0.299 * br + 0.587 * bg + 0.114 * bb;
    const u = -0.168736 * cr - 0.331264 * cg + 0.5 * cb + 128;
    const v = 0.5 * cr - 0.418688 * cg - 0.081312 * cb + 128;
    base[i] = clampByte(y + 1.402 * (v - 128));
    base[i + 1] = clampByte(y - 0.344136 * (u - 128) - 0.714136 * (v - 128));
    base[i + 2] = clampByte(y + 1.772 * (u - 128));
    base[i + 3] = 255;
  }
}

function normalizedPercent(value, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return number > 4 ? number / 100 : number;
}

function kelvinBalance(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return [0, 0];
  if (Math.abs(number) <= 20) return [Math.max(number, 0), Math.max(-number, 0)];
  const delta = Math.max(-4000, Math.min(4000, number - 6500));
  const strength = Math.abs(delta) / 4000 * 0.12;
  return delta < 0 ? [strength, 0] : [0, strength];
}

function adjustSaturationRgb(r, g, b, saturation) {
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  return [
    clampByte(y + (r - y) * saturation),
    clampByte(y + (g - y) * saturation),
    clampByte(y + (b - y) * saturation),
  ];
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function setupSourceBlackPreview() {
  if (recompMaskFrame) {
    cancelAnimationFrame(recompMaskFrame);
    recompMaskFrame = 0;
  }
  if (!sourceBlackTransparent()) return;

  const source = document.querySelector('.source-mask-video');
  const canvases = [...document.querySelectorAll('.source-mask-canvas')];
  if (!source || !canvases.length) return;

  const draw = () => {
    const activeSource = document.querySelector('.source-mask-video');
    const activeCanvases = [...document.querySelectorAll('.source-mask-canvas')];
    if (!activeSource || !activeCanvases.length) {
      recompMaskFrame = 0;
      return;
    }
    drawSourceBlackMask(activeSource, activeCanvases);
    recompMaskFrame = requestAnimationFrame(draw);
  };

  source.addEventListener('loadeddata', () => drawSourceBlackMask(source, canvases), { once: true });
  draw();
}

function drawSourceBlackMask(source, canvases) {
  if (!source.videoWidth || !source.videoHeight || source.readyState < 2) return;
  const threshold = 24;
  const shrink = 2;
  const scale = Math.min(1, 960 / source.videoWidth);
  const width = Math.max(2, Math.round(source.videoWidth * scale));
  const height = Math.max(2, Math.round(source.videoHeight * scale));
  for (const canvas of canvases) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = image.data;
    const blackAt = (x, y) => {
      const clampedX = Math.max(0, Math.min(canvas.width - 1, x));
      const clampedY = Math.max(0, Math.min(canvas.height - 1, y));
      const index = (clampedY * canvas.width + clampedX) * 4;
      return Math.max(data[index], data[index + 1], data[index + 2]) <= threshold;
    };
    for (let i = 0; i < data.length; i += 4) {
      const pixel = i / 4;
      const x = pixel % canvas.width;
      const y = Math.floor(pixel / canvas.width);
      if (
        blackAt(x, y) ||
        blackAt(x - shrink, y) ||
        blackAt(x + shrink, y) ||
        blackAt(x, y - shrink) ||
        blackAt(x, y + shrink) ||
        blackAt(x - shrink, y - shrink) ||
        blackAt(x + shrink, y - shrink) ||
        blackAt(x - shrink, y + shrink) ||
        blackAt(x + shrink, y + shrink)
      ) {
        data[i + 3] = 0;
      }
    }
    ctx.putImageData(image, 0, 0);
  }
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
