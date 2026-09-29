// Per-frame outpaint mask editor: single-frame patches (light leaks, edge lettering, dirt,
// scratches) painted on top of the global custom mask. Each frame's patch is saved on its
// own when you step away from it, so there is no separate Save step.

const FRAME_MASK_BLOCK = 24;
// Amber, so frame patches never read as the coral global mask.
const FRAME_MASK_COLOUR = [255, 176, 32];

const frameMaskEditor = {
  frame: 0,
  frameCount: 0,
  fps: 24,
  previews: new Map(),    // frame -> preview image path
  masks: new Map(),       // frame -> PNG data URL ('' when the frame has no patch)
  loadingBlocks: new Map(),
  masked: new Set(),
  dirty: false,
  drawing: false,
  lastPoint: null,
  subtract: false,
  brushSize: 18,
  onion: true,
  undo: [],
  token: 0,
  busy: Promise.resolve(),
};

function frameMaskState() {
  return state.frame_outpaint_masks || {};
}

function frameMaskSummaryText() {
  const info = frameMaskState();
  const count = (info.frames || []).length;
  const patched = count ? `${count} frame${count === 1 ? '' : 's'} patched.` : 'No frames patched yet.';
  const model = info.supported ? '' : ' Frame masks are used by the LTX outpaint models only (not Wan VACE or H3).';
  return `${patched}${model}`;
}

function frameMaskSummaryHtml() {
  const info = frameMaskState();
  const count = (info.frames || []).length;
  return `
    <div class="outpaint-mask-summary">
      <div>
        <h3>Frame Masks</h3>
        <p class="shot-empty">Patch single frames for ${outpaintModelName()} to inpaint: light leaks, film edge lettering, dirt and scratches. Step through the clip frame by frame; each patch is added to the custom mask on that frame only and feathered the same way in Recomposition.</p>
        <p id="frameMaskSummaryText" class="shot-empty">${esc(frameMaskSummaryText())}</p>
      </div>
      <div class="actions">
        <button id="frameMaskEditButton" type="button" onclick="openFrameMaskEditor()">${count ? 'Edit' : 'Paint'} Frame Masks</button>
        <button id="frameMaskClearButton" type="button" class="warn" onclick="clearAllFrameMasks()" ${count ? '' : 'disabled'}>Clear All</button>
      </div>
    </div>
  `;
}

function syncFrameMaskSummary() {
  const info = frameMaskState();
  const count = (info.frames || []).length;
  const text = document.getElementById('frameMaskSummaryText');
  if (text) text.textContent = frameMaskSummaryText();
  const edit = document.getElementById('frameMaskEditButton');
  if (edit) edit.textContent = `${count ? 'Edit' : 'Paint'} Frame Masks`;
  const clear = document.getElementById('frameMaskClearButton');
  if (clear) clear.disabled = !count;
}

function ensureFrameMaskModal() {
  if (document.getElementById('frameMaskModal')) return;
  const modal = document.createElement('div');
  modal.id = 'frameMaskModal';
  modal.className = 'image-modal hidden';
  modal.innerHTML = `
    <div class="image-modal-backdrop" onclick="closeFrameMaskEditor()"></div>
    <div class="outpaint-mask-panel frame-mask-panel">
      <div class="image-modal-heading">
        <strong>Frame Masks</strong>
        <button type="button" onclick="closeFrameMaskEditor()">Close</button>
      </div>
      <div class="outpaint-mask-layout">
        <div>
          <div class="outpaint-mask-canvas-wrap frame-mask-canvas-wrap">
            <canvas id="frameMaskImageCanvas"></canvas>
            <canvas id="frameMaskGlobalCanvas"></canvas>
            <canvas id="frameMaskOnionCanvas"></canvas>
            <canvas id="frameMaskPaintCanvas"></canvas>
          </div>
          <div class="frame-mask-transport">
            <button type="button" title="Previous patched frame" onclick="stepToMaskedFrame(-1)">&#x23EE;</button>
            <button type="button" title="Back 10 frames (Shift+Left)" onclick="stepFrameMask(-10)">-10</button>
            <button type="button" title="Previous frame (Left)" onclick="stepFrameMask(-1)">&#x25C0;</button>
            <input id="frameMaskNumber" type="number" min="0" value="0" onchange="goToFrameMask(Number(this.value))">
            <span id="frameMaskTotal" class="shot-empty"></span>
            <button type="button" title="Next frame (Right)" onclick="stepFrameMask(1)">&#x25B6;</button>
            <button type="button" title="Forward 10 frames (Shift+Right)" onclick="stepFrameMask(10)">+10</button>
            <button type="button" title="Next patched frame" onclick="stepToMaskedFrame(1)">&#x23ED;</button>
            <span id="frameMaskTime" class="shot-empty"></span>
          </div>
          <canvas id="frameMaskTimeline" class="frame-mask-timeline" title="Patched frames. Click to jump."></canvas>
        </div>
        <aside>
          <div class="reference-tool-grid">
            <button id="frameMaskAdd" class="reference-tool-button active" type="button" title="Paint patch (B)" onclick="setFrameMaskTool(false)">${referenceIcon('brush-add')}</button>
            <button id="frameMaskSubtract" class="reference-tool-button" type="button" title="Erase patch (E)" onclick="setFrameMaskTool(true)">${referenceIcon('brush-subtract')}</button>
            <button class="reference-tool-button" type="button" title="Clear this frame (Delete)" onclick="clearCurrentFrameMask()">${referenceIcon('clear')}</button>
          </div>
          <label>Brush size <span id="frameMaskBrushValue">${frameMaskEditor.brushSize}</span></label>
          <input id="frameMaskBrushSize" type="range" min="2" max="120" value="${frameMaskEditor.brushSize}" oninput="setFrameMaskBrush(Number(this.value))">
          <div class="actions frame-mask-actions">
            <button type="button" title="Replace this frame's patch with the previous frame's (C)" onclick="copyPreviousFrameMask()">Copy Previous Frame</button>
            <button type="button" title="Undo (Ctrl+Z)" onclick="undoFrameMask()">Undo</button>
          </div>
          <label class="frame-mask-check"><input id="frameMaskOnion" type="checkbox" checked onchange="setFrameMaskOnion(this.checked)"> Show previous frame's patch</label>
          <p class="shot-empty">Amber is this frame's patch. The cyan outline is the previous frame's, and grey is the global custom mask, which already applies to every frame.</p>
          <p class="shot-empty">Keys: Left/Right step a frame (Shift: 10), C copies the previous frame, B/E brush/erase, [ and ] brush size, Delete clears the frame, Ctrl+Z undoes.</p>
          <div id="frameMaskStatus" class="shot-empty"></div>
        </aside>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  const wrap = modal.querySelector('.frame-mask-canvas-wrap');
  wrap.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    pushFrameMaskUndo();
    frameMaskEditor.drawing = true;
    frameMaskEditor.lastPoint = null;
    drawFrameMaskPoint(event);
    wrap.setPointerCapture?.(event.pointerId);
  });
  wrap.addEventListener('pointermove', event => {
    if (frameMaskEditor.drawing && (event.buttons & 1)) drawFrameMaskPoint(event);
  });
  window.addEventListener('pointerup', () => {
    frameMaskEditor.drawing = false;
    frameMaskEditor.lastPoint = null;
  });
  modal.querySelector('#frameMaskTimeline').addEventListener('click', event => {
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - rect.left) / Math.max(1, rect.width);
    goToFrameMask(Math.round(fraction * Math.max(0, frameMaskEditor.frameCount - 1)));
  });
  document.addEventListener('keydown', onFrameMaskKey);
}

function frameMaskModalOpen() {
  const modal = document.getElementById('frameMaskModal');
  return !!modal && !modal.classList.contains('hidden');
}

function onFrameMaskKey(event) {
  if (!frameMaskModalOpen()) return;
  if (event.target && event.target.id === 'frameMaskNumber') return;
  const key = event.key;
  let handled = true;
  if (key === 'ArrowRight') stepFrameMask(event.shiftKey ? 10 : 1);
  else if (key === 'ArrowLeft') stepFrameMask(event.shiftKey ? -10 : -1);
  else if ((key === 'z' || key === 'Z') && (event.ctrlKey || event.metaKey)) undoFrameMask();
  else if (key === 'c' || key === 'C') copyPreviousFrameMask();
  else if (key === 'b' || key === 'B') setFrameMaskTool(false);
  else if (key === 'e' || key === 'E') setFrameMaskTool(true);
  else if (key === '[') setFrameMaskBrush(frameMaskEditor.brushSize - 2);
  else if (key === ']') setFrameMaskBrush(frameMaskEditor.brushSize + 2);
  else if (key === 'Delete' || key === 'Backspace') clearCurrentFrameMask();
  else if (key === 'Escape') closeFrameMaskEditor();
  else handled = false;
  if (handled) event.preventDefault();
}

async function openFrameMaskEditor() {
  ensureFrameMaskModal();
  const editor = frameMaskEditor;
  editor.token += 1;
  editor.previews.clear();
  editor.masks.clear();
  editor.loadingBlocks.clear();
  editor.masked = new Set(frameMaskState().frames || []);
  editor.dirty = false;
  editor.undo = [];
  setFrameMaskStatus('Loading frames...');
  document.getElementById('frameMaskModal').classList.remove('hidden');
  setFrameMaskTool(false);
  const first = Math.max(0, Math.floor(editor.frame / FRAME_MASK_BLOCK) * FRAME_MASK_BLOCK);
  const block = await loadFrameMaskBlock(first);
  if (!block) return;
  await showFrameMask(Math.min(editor.frame, Math.max(0, editor.frameCount - 1)));
  drawGlobalMaskLayer();
}

async function closeFrameMaskEditor() {
  if (!frameMaskModalOpen()) return;
  await saveCurrentFrameMask();
  document.getElementById('frameMaskModal')?.classList.add('hidden');
  syncFrameMaskSummary();
  framePatchPreview.cache.clear();
  syncFramePatchPreview(framePatchPreview.frame);
}

async function clearAllFrameMasks() {
  const count = (frameMaskState().frames || []).length;
  if (!count || !confirm(`Clear the patches on all ${count} frame${count === 1 ? '' : 's'}? This cannot be undone.`)) return;
  const result = await postJson('/api/outpaint-frame-mask-clear', {});
  if (!result.ok) return alert(result.error || 'Could not clear frame masks.');
  state = result.state || state;
  frameMaskEditor.masks.clear();
  frameMaskEditor.masked.clear();
  syncFrameMaskSummary();
  framePatchPreview.cache.clear();
  syncFramePatchPreview(framePatchPreview.frame);
}

function setFrameMaskStatus(text) {
  const status = document.getElementById('frameMaskStatus');
  if (status) status.textContent = text;
}

function setFrameMaskTool(subtract) {
  frameMaskEditor.subtract = !!subtract;
  document.getElementById('frameMaskAdd')?.classList.toggle('active', !subtract);
  document.getElementById('frameMaskSubtract')?.classList.toggle('active', !!subtract);
}

function setFrameMaskBrush(size) {
  frameMaskEditor.brushSize = Math.max(2, Math.min(120, Math.round(size)));
  const range = document.getElementById('frameMaskBrushSize');
  if (range) range.value = frameMaskEditor.brushSize;
  const label = document.getElementById('frameMaskBrushValue');
  if (label) label.textContent = frameMaskEditor.brushSize;
}

function setFrameMaskOnion(enabled) {
  frameMaskEditor.onion = !!enabled;
  drawOnionLayer();
}

async function loadFrameMaskBlock(first) {
  const editor = frameMaskEditor;
  const token = editor.token;
  if (editor.loadingBlocks.has(first)) return await editor.loadingBlocks.get(first);
  const request = (async () => {
    const result = await api(`/api/outpaint-frame-mask-frames?first=${first}&count=${FRAME_MASK_BLOCK}`);
    if (token !== editor.token) return null;
    if (!result.ok) {
      setFrameMaskStatus(result.error || 'Could not load frames.');
      return null;
    }
    editor.frameCount = result.frame_count || editor.frameCount;
    editor.fps = result.fps || editor.fps;
    (result.previews || []).forEach((path, index) => editor.previews.set(result.first + index, path));
    return result;
  })();
  editor.loadingBlocks.set(first, request);
  const result = await request;
  if (!result) editor.loadingBlocks.delete(first);
  return result;
}

async function framePreviewPath(frame) {
  const editor = frameMaskEditor;
  if (!editor.previews.has(frame)) await loadFrameMaskBlock(Math.floor(frame / FRAME_MASK_BLOCK) * FRAME_MASK_BLOCK);
  // Warm the next block while the user is still near the end of this one.
  const nextBlock = (Math.floor(frame / FRAME_MASK_BLOCK) + 1) * FRAME_MASK_BLOCK;
  if (frame + 6 >= nextBlock && nextBlock < editor.frameCount && !editor.previews.has(nextBlock)) loadFrameMaskBlock(nextBlock);
  return editor.previews.get(frame) || '';
}

async function frameMaskData(frame) {
  const editor = frameMaskEditor;
  if (frame < 0) return '';
  if (editor.masks.has(frame)) return editor.masks.get(frame);
  if (!editor.masked.has(frame)) return '';
  const result = await api(`/api/outpaint-frame-mask?frame=${frame}`);
  const image = result.ok ? (result.image || '') : '';
  editor.masks.set(frame, image);
  return image;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

// Render a stored patch (the server's 0/255 PNG, or the editor's own coral overlay) as a
// coloured overlay or as a bare outline, scaled to the editor canvas.
async function paintMaskInto(canvas, src, { alpha = 156, outline = false } = {}) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!src) return;
  const image = await loadImage(src);
  // Nearest-neighbour, so a scaled-down binary mask doesn't grow a smoothed fringe.
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = pixels.data;
  const selected = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    selected[p] = data[i + 3] >= 16 && Math.max(data[i], data[i + 1], data[i + 2]) >= 16 ? 1 : 0;
  }
  const w = canvas.width;
  for (let p = 0, i = 0; p < selected.length; p += 1, i += 4) {
    let on = selected[p];
    if (outline && on) {
      const x = p % w;
      const edge = x === 0 || x === w - 1 || !selected[p - 1] || !selected[p + 1] || !selected[p - w] || !selected[p + w];
      on = edge ? 1 : 0;
    }
    data[i] = outline ? 70 : FRAME_MASK_COLOUR[0];
    data[i + 1] = outline ? 214 : FRAME_MASK_COLOUR[1];
    data[i + 2] = outline ? 255 : FRAME_MASK_COLOUR[2];
    data[i + 3] = on ? (outline ? 230 : alpha) : 0;
  }
  ctx.putImageData(pixels, 0, 0);
}

function frameMaskCanvases() {
  return ['frameMaskImageCanvas', 'frameMaskGlobalCanvas', 'frameMaskOnionCanvas', 'frameMaskPaintCanvas']
    .map(id => document.getElementById(id));
}

async function showFrameMask(frame) {
  const editor = frameMaskEditor;
  const token = editor.token;
  frame = Math.max(0, Math.min(frame, Math.max(0, editor.frameCount - 1)));
  editor.frame = frame;
  updateFrameMaskTransport();
  const path = await framePreviewPath(frame);
  if (token !== editor.token || editor.frame !== frame) return;
  if (!path) {
    setFrameMaskStatus(`Frame ${frame} could not be extracted.`);
    return;
  }
  const image = await loadImage(media(path));
  if (token !== editor.token || editor.frame !== frame) return;
  const [imageCanvas, globalCanvas, onionCanvas, paintCanvas] = frameMaskCanvases();
  const resized = imageCanvas.width !== image.naturalWidth || imageCanvas.height !== image.naturalHeight;
  for (const canvas of [imageCanvas, globalCanvas, onionCanvas, paintCanvas]) {
    if (resized) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
    }
    canvas.style.aspectRatio = `${image.naturalWidth}/${image.naturalHeight}`;
  }
  imageCanvas.getContext('2d').drawImage(image, 0, 0);
  if (resized) drawGlobalMaskLayer();
  await paintMaskInto(paintCanvas, await frameMaskData(frame));
  if (editor.frame !== frame) return;
  await drawOnionLayer();
  editor.dirty = false;
  editor.undo = [];
  setFrameMaskStatus(editor.masked.has(frame) ? `Frame ${frame} has a patch.` : `Frame ${frame}: no patch.`);
}

async function drawOnionLayer() {
  const canvas = document.getElementById('frameMaskOnionCanvas');
  if (!canvas) return;
  const frame = frameMaskEditor.frame;
  const src = frameMaskEditor.onion ? await frameMaskData(frame - 1) : '';
  if (frameMaskEditor.frame !== frame) return;
  await paintMaskInto(canvas, src, { outline: true });
}

async function drawGlobalMaskLayer() {
  const canvas = document.getElementById('frameMaskGlobalCanvas');
  if (!canvas) return;
  const mask = state.custom_outpaint_mask || {};
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!mask.exists || !mask.path) return;
  const image = await loadImage(media(mask.path) + '&t=' + (mask.mtime || Date.now()));
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < pixels.data.length; i += 4) {
    const on = pixels.data[i] >= 16;
    pixels.data[i] = 150;
    pixels.data[i + 1] = 160;
    pixels.data[i + 2] = 168;
    pixels.data[i + 3] = on ? 110 : 0;
  }
  ctx.putImageData(pixels, 0, 0);
}

function updateFrameMaskTransport() {
  const editor = frameMaskEditor;
  const number = document.getElementById('frameMaskNumber');
  if (number) {
    number.value = editor.frame;
    number.max = Math.max(0, editor.frameCount - 1);
  }
  const total = document.getElementById('frameMaskTotal');
  if (total) total.textContent = `/ ${Math.max(0, editor.frameCount - 1)}`;
  const time = document.getElementById('frameMaskTime');
  if (time) time.textContent = formatSeconds((editor.frame / (editor.fps || 24)).toFixed(3));
  drawFrameMaskTimeline();
}

function drawFrameMaskTimeline() {
  const canvas = document.getElementById('frameMaskTimeline');
  if (!canvas) return;
  const editor = frameMaskEditor;
  const width = Math.max(1, Math.round(canvas.clientWidth || 600));
  const height = 22;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b1115';
  ctx.fillRect(0, 0, width, height);
  const span = Math.max(1, editor.frameCount - 1);
  ctx.fillStyle = `rgb(${FRAME_MASK_COLOUR.join(',')})`;
  for (const frame of editor.masked) {
    const x = Math.round(frame / span * (width - 2));
    ctx.fillRect(x, 3, Math.max(2, Math.ceil(width / Math.max(1, editor.frameCount))), height - 6);
  }
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(Math.round(editor.frame / span * (width - 2)), 0, 2, height);
}

function drawFrameMaskPoint(event) {
  const canvas = document.getElementById('frameMaskPaintCanvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const point = {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height,
  };
  const ctx = canvas.getContext('2d');
  const last = frameMaskEditor.lastPoint || point;
  const colour = `rgba(${FRAME_MASK_COLOUR.join(',')},.62)`;
  ctx.save();
  ctx.globalCompositeOperation = frameMaskEditor.subtract ? 'destination-out' : 'source-over';
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = frameMaskEditor.brushSize;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(point.x, point.y, frameMaskEditor.brushSize / 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  frameMaskEditor.lastPoint = point;
  markFrameMaskDirty();
}

function markFrameMaskDirty() {
  frameMaskEditor.dirty = true;
  setFrameMaskStatus(`Frame ${frameMaskEditor.frame}: unsaved changes (saved when you leave the frame).`);
}

function pushFrameMaskUndo() {
  const canvas = document.getElementById('frameMaskPaintCanvas');
  if (!canvas || !canvas.width) return;
  frameMaskEditor.undo.push(canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height));
  if (frameMaskEditor.undo.length > 30) frameMaskEditor.undo.shift();
}

function undoFrameMask() {
  const snapshot = frameMaskEditor.undo.pop();
  const canvas = document.getElementById('frameMaskPaintCanvas');
  if (!snapshot || !canvas) return;
  canvas.getContext('2d').putImageData(snapshot, 0, 0);
  markFrameMaskDirty();
}

function clearCurrentFrameMask() {
  const canvas = document.getElementById('frameMaskPaintCanvas');
  if (!canvas) return;
  pushFrameMaskUndo();
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  markFrameMaskDirty();
}

async function copyPreviousFrameMask() {
  const editor = frameMaskEditor;
  if (editor.frame <= 0) return;
  const frame = editor.frame;
  const src = await frameMaskData(frame - 1);
  if (editor.frame !== frame) return;
  if (!src) {
    setFrameMaskStatus(`Frame ${frame - 1} has no patch to copy.`);
    return;
  }
  pushFrameMaskUndo();
  await paintMaskInto(document.getElementById('frameMaskPaintCanvas'), src);
  markFrameMaskDirty();
}

function paintCanvasHasMask(canvas) {
  const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) if (data[i] >= 16) return true;
  return false;
}

// Saves are chained so a fast run of arrow presses writes each frame once, in order.
function saveCurrentFrameMask() {
  const editor = frameMaskEditor;
  if (!editor.dirty) return editor.busy;
  const canvas = document.getElementById('frameMaskPaintCanvas');
  const frame = editor.frame;
  const image = canvas && paintCanvasHasMask(canvas) ? canvas.toDataURL('image/png') : '';
  editor.dirty = false;
  editor.masks.set(frame, image);
  if (image) editor.masked.add(frame); else editor.masked.delete(frame);
  drawFrameMaskTimeline();
  editor.busy = editor.busy.then(async () => {
    const result = await postJson('/api/outpaint-frame-mask-save', { frame, image });
    if (!result.ok) {
      setFrameMaskStatus(`Frame ${frame} was not saved: ${result.error || 'unknown error'}`);
      return;
    }
    state.frame_outpaint_masks = result.frame_outpaint_masks || state.frame_outpaint_masks;
    syncFrameMaskSummary();
  });
  return editor.busy;
}

async function goToFrameMask(frame) {
  const editor = frameMaskEditor;
  if (!Number.isFinite(frame)) return;
  const target = Math.max(0, Math.min(Math.round(frame), Math.max(0, editor.frameCount - 1)));
  saveCurrentFrameMask();
  await showFrameMask(target);
}

function stepFrameMask(delta) {
  return goToFrameMask(frameMaskEditor.frame + delta);
}

function stepToMaskedFrame(direction) {
  const editor = frameMaskEditor;
  const frames = [...editor.masked].sort((a, b) => a - b);
  const target = direction > 0
    ? frames.find(frame => frame > editor.frame)
    : frames.reverse().find(frame => frame < editor.frame);
  if (target !== undefined) goToFrameMask(target);
}

// Target Preview overlay: the patch on the frame the preview is showing, in amber over the
// coral global mask. The server reports which frame it showed, so the two always agree.
const framePatchPreview = { frame: 0, cache: new Map(), token: 0 };

async function framePatchImage(frame) {
  const key = `${frame}:${frameMaskState().mtime || 0}`;
  if (!framePatchPreview.cache.has(key)) {
    const result = await api(`/api/outpaint-frame-mask?frame=${frame}`);
    framePatchPreview.cache.set(key, result.ok ? (result.image || '') : '');
  }
  return framePatchPreview.cache.get(key);
}

async function syncFramePatchPreview(frame) {
  framePatchPreview.frame = Number(frame) || 0;
  const token = ++framePatchPreview.token;
  const holder = document.querySelector('.aspect-preview-frame');
  const base = document.getElementById('aspectPreviewImg');
  let canvas = document.getElementById('outpaintFramePatchOverlay');
  let badge = document.querySelector('.outpaint-frame-patch-badge');
  const patched = (frameMaskState().frames || []).includes(framePatchPreview.frame);
  const src = holder && base && patched ? await framePatchImage(framePatchPreview.frame) : '';
  if (token !== framePatchPreview.token) return;
  if (!src) {
    canvas?.remove();
    badge?.remove();
    return;
  }
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = 'outpaintFramePatchOverlay';
    canvas.setAttribute('aria-label', 'Frame patch overlay');
    holder.appendChild(canvas);
  }
  if (!badge) {
    badge = document.createElement('span');
    badge.className = 'outpaint-frame-patch-badge';
    holder.appendChild(badge);
  }
  badge.textContent = `Frame patch ${framePatchPreview.frame}`;
  const paint = async () => {
    if (token !== framePatchPreview.token) return;
    canvas.width = Math.max(1, base.naturalWidth);
    canvas.height = Math.max(1, base.naturalHeight);
    await paintMaskInto(canvas, src, { alpha: 190 });
  };
  if (!base.complete) base.addEventListener('load', paint, { once: true });
  else await paint();
}
