const SOURCE_INFO_LABELS = {
  resolution: 'Resolution',
  aspect: 'Aspect',
  duration: 'Duration',
  frame_rate: 'Frame rate',
  frames: 'Frames',
  video_codec: 'Video codec',
  pixel_format: 'Pixel format',
  colour: 'Color',
  audio: 'Audio',
  container: 'Container',
  overall_bitrate: 'Overall bitrate',
  video_bitrate: 'Video bitrate',
  size: 'File size',
  codec_note: 'Note',
};

const SOURCE_INFO_KEYS = [
  'resolution',
  'aspect',
  'duration',
  'frame_rate',
  'frames',
  'video_codec',
  'pixel_format',
  'colour',
  'audio',
  'container',
  'overall_bitrate',
  'video_bitrate',
  'size',
  'codec_note',
];

function sourceInfoHtml(info) {
  const items = SOURCE_INFO_KEYS
    .filter(key => info[key])
    .map(key => `
      <div>
        <span>${SOURCE_INFO_LABELS[key] || key}</span>
        <strong>${esc(info[key])}</strong>
      </div>
    `)
    .join('');

  return items ? `<div class="source-info">${items}</div>` : '';
}

// Help text shown under stage fields. Keys are "<stage>.<field>" with a bare
// "<field>" fallback for keys that are unique across stages.
const FIELD_DESCRIPTIONS = {
  'colour.processing_height':
    'Downscale frames before they are sent to ComfyUI. Source keeps the original resolution; 1080p is a practical first stop for 4K material.',
  'colour.use_half_resolution':
    'DeepExemplar node option. Keep this separate from Processing scale: Processing scale changes the input video size, while this asks the node to work internally at half resolution.',
  'recomp.temperature':
    'Color temperature in Kelvin. 6500K is neutral; lower values warm the colour layer, higher values cool it.',
  'recomp.saturation':
    'Color layer saturation as a percentage before chroma is blended onto the luminance of the base composite.',
  'recomp.color_opacity':
    'How strongly the colorized chroma layer contributes, as a percentage.',
  'recomp.feather_pixels':
    'Only used when an outpainted video is present. It softens the original source edge over the generated sides.',
  'upscale.flashvsr_mode':
    'tiny = fastest, but its distilled decoder can smear fine motion such as lips. ' +
    'full = real VAE decoder with the best fidelity for faces and small movements, slowest. ' +
    'tiny-long = tiny with lower VRAM use on long clips.',
  'upscale.flashvsr_scale':
    'How far the model upscales before the final resize to the target size. 2 stays closest to the source; 3-4 invent more detail (and hallucinate more).',
  'upscale.flashvsr_pre_downscale':
    'Downscale the input to target size divided by FlashVSR scale before processing. Much faster, but asks FlashVSR to reconstruct more detail.',
  'upscale.flashvsr_tiled_dit':
    'Processes the frame as small tiles to save VRAM. Tiles only see their own patch, so small faces can lose identity ' +
    '(the model invents a plausible face). Untick for full-frame context if VRAM allows - the biggest lever against changed faces.',
  'upscale.flashvsr_tile_size':
    'Tile edge in pixels when tiled diffusion is on (multiples of 32, max 1024). Larger tiles give faces more surrounding context at the cost of VRAM. Try 512 if full-frame does not fit.',
  'upscale.flashvsr_tile_overlap':
    'Feathered overlap between tiles that hides seams. Raise it if you can see tile borders.',
  'upscale.flashvsr_vae_tile_multiplier':
    'Multiplies full-model VAE decode tile size. Higher is faster if VRAM allows; try 2 before disabling tiled decode.',
  'upscale.flashvsr_local_range':
    'Temporal attention window. 11 = more stable but can freeze small motion such as mouths; 9 = sharper, livelier detail with slightly more shimmer.',
  'upscale.flashvsr_sparse_ratio':
    'Sparse attention density, 1.5 to 2.0. 2.0 = most stable output; 1.5 = faster.',
  'upscale.flashvsr_kv_ratio':
    'Attention memory budget, 1.0 to 3.0. 3.0 = highest quality; lower it to save VRAM, e.g. when turning tiled diffusion off.',
  'upscale.flashvsr_color_fix':
    'Wavelet transform that matches output colors back to the source. Leave on to prevent color drift on colorized footage.',
  'upscale.flashvsr_tiled_vae':
    'Decode the output in tiles to reduce VRAM at some speed cost. Negligible quality impact.',
  'upscale.flashvsr_unload_dit':
    'Unload the diffusion model before decoding to lower peak VRAM. Slower; only needed if decoding runs out of memory.',
  'upscale.flashvsr_seed':
    'Changes the detail the model invents. If a face renders wrong, re-rolling the seed (with Regenerate) often fixes it.',
  'upscale.chunk_seconds':
    'The clip is upscaled in chunks of roughly this many seconds; each chunk restarts the model’s temporal stream. 0 sends the whole clip at once.',
  'upscale.overlap_frames':
    'Warm-up frames repeated before each chunk and trimmed afterwards. Raise to 16-24 if chunk starts look unstable or faces flip identity mid-scene.',
};

function fieldDescription(stageKey, key) {
  return FIELD_DESCRIPTIONS[`${stageKey}.${key}`] || FIELD_DESCRIPTIONS[key] || '';
}

function fieldHelpHtml(help) {
  return help ? `<small class="field-help">${esc(help)}</small>` : '';
}

function fieldHtml(st, field) {
  const [key, label, kind, defaultValue] = field;
  const value = settings(st.key)[key] ?? defaultValue ?? '';
  const help = fieldDescription(st.key, key);

  if (kind.startsWith('select:')) return selectFieldHtml(key, label, kind, value) + fieldHelpHtml(help);
  if (kind.startsWith('range:')) return rangeFieldHtml(key, label, kind, value) + fieldHelpHtml(help);
  if (kind === 'checkbox') return checkboxFieldHtml(key, label, value, help);

  const input = `
    <input data-field="${key}" data-kind="${kind}" type="${kind === 'number' ? 'number' : 'text'}" step="any" value="${esc(value)}">
  `;

  if (['file', 'folder', 'save'].includes(kind)) {
    return `
      <label>${label}</label>
      <div class="field-row">
        ${input}
        <button type="button" onclick="browseField('${st.key}','${key}','${kind}')">Browse</button>
      </div>
      ${fieldHelpHtml(help)}
    `;
  }

  return `<label>${label}</label>${input}${fieldHelpHtml(help)}`;
}

function selectFieldHtml(key, label, kind, value) {
  const options = kind.slice(7).split('|')
    .map(option => `<option value="${esc(option)}" ${value === option ? 'selected' : ''}>${esc(selectOptionLabel(key, option))}</option>`)
    .join('');
  return `<label>${label}</label><select data-field="${key}">${options}</select>`;
}

function selectOptionLabel(key, option) {
  if (key === 'method' && option === 'qwen') return 'Qwen 2511 (local)';
  if (key === 'method' && option === 'openai') return 'OpenAI API (cloud)';
  if (key === 'target_height' && option === 'source') {
    const resolution = (state.source_info && state.source_info.resolution) || '';
    const match = String(resolution).match(/x(\d+)/i);
    return match ? `Source height (${match[1]}p)` : 'Source height';
  }
  if (key === 'target_height' && /^\d+$/.test(option)) return `${option}p`;
  if (key === 'processing_height' && option === 'source') return 'Original / source';
  if (key === 'processing_height' && /^\d+$/.test(option)) return `${option}p max height`;
  return option;
}

const RANGE_FIELD_UNITS = {
  feather_pixels: ' px',
  saturation: '%',
  temperature: ' K',
  color_opacity: '%',
};

const RANGE_NUDGE_FIELDS = new Set(['crop_left', 'crop_right', 'crop_top', 'crop_bottom', 'feather_pixels', 'saturation', 'temperature', 'color_opacity']);

function rangeDisplayValue(key, value) {
  const number = Number(value);
  const shown = Number.isFinite(number) && Number.isInteger(number) ? String(number) : String(value);
  return shown + (RANGE_FIELD_UNITS[key] || '');
}

function rangeFieldHtml(key, label, kind, value) {
  const [min, max, step] = kind.slice(6).split('|');
  const hasNudge = RANGE_NUDGE_FIELDS.has(key);
  const controls = hasNudge ? `
    <div class="pixel-nudge-row">
      <button type="button" onclick="nudgeRangeField('${key}',-1)">-1</button>
      <input
        id="${key}Input"
        class="pixel-input"
        type="number"
        min="${esc(min)}"
        max="${esc(max)}"
        step="${esc(step || '1')}"
        value="${esc(value)}"
        onchange="setRangeFieldValue('${key}',this.value,true)"
      >
      <button type="button" onclick="nudgeRangeField('${key}',1)">+1</button>
    </div>
  ` : '';
  return `
    <label>${label}: <span id="${key}Value">${esc(rangeDisplayValue(key, value))}</span></label>
    <input
      id="${key}Range"
      data-field="${key}"
      data-kind="${kind}"
      type="range"
      min="${esc(min)}"
      max="${esc(max)}"
      step="${esc(step || '1')}"
      value="${esc(value)}"
      oninput="setRangeFieldValue('${key}',this.value,false)"
    >
    ${controls}
  `;
}

const CHECKBOX_DESCRIPTIONS = {
  seed_qwen_guides:
    'Use this if LTX does not outpaint the source material (it hands back the black bars). ' +
    'Before each chunk renders, a guide frame is generated at every detected shot change with ' +
    'Qwen Image Edit ("Replace the black bars.") and fed to LTX as the anchor for that shot, so ' +
    'it extends from a filled frame instead of copying the bars. Slower, but reliable on stubborn clips.',
  outpaint_all_black_regions:
    "Don't expand the canvas, just paint over all pure black areas. Use this when the region to be extended changes, e.g. you have mixed-size footage in your clip.",
};

function checkboxFieldHtml(key, label, value, help = '') {
  const description = CHECKBOX_DESCRIPTIONS[key];
  if (description) {
    return `
      <label class="checkbox-feature" title="${esc(description)}">
        <input data-field="${key}" data-kind="checkbox" type="checkbox" ${value === 'true' ? 'checked' : ''}>
        <span class="checkbox-feature-text">
          <strong>${esc(label)}</strong>
        </span>
      </label>
    `;
  }
  if (help) {
    return `
      <label class="checkbox-described">
        <input data-field="${key}" data-kind="checkbox" type="checkbox" ${value === 'true' ? 'checked' : ''}>
        <span class="checkbox-described-text">
          <strong>${esc(label)}</strong>
          <small>${esc(help)}</small>
        </span>
      </label>
    `;
  }
  return `
    <label class="checkbox-field">
      <input data-field="${key}" data-kind="checkbox" type="checkbox" ${value === 'true' ? 'checked' : ''}>
      ${esc(label)}
    </label>
  `;
}

function setRangeFieldValue(key, value, save = false) {
  const range = document.getElementById(`${key}Range`);
  if (!range) return;
  const min = Number(range.min || 0);
  const max = Number(range.max || value);
  const step = Number(range.step || 1);
  let next = Number(value);
  if (!Number.isFinite(next)) next = Number(range.value || 0);
  next = Math.max(min, Math.min(max, Math.round(next / step) * step));
  range.value = String(next);
  const label = document.getElementById(`${key}Value`);
  if (label) label.textContent = rangeDisplayValue(key, range.value);
  const input = document.getElementById(`${key}Input`);
  if (input) input.value = range.value;
  if (save) range.dispatchEvent(new Event('change', { bubbles: true }));
}

function nudgeRangeField(key, delta) {
  const range = document.getElementById(`${key}Range`);
  if (!range) return;
  const step = Number(range.step || 1);
  setRangeFieldValue(key, Number(range.value || 0) + Number(delta || 0) * step, true);
}

function aspectPreviewHtml(st) {
  if (st.key !== 'outpaint') return '';

  const img = state.aspect_preview;
  const outputs = (state.expected_outputs && state.expected_outputs.outpaint) || [];
  const range = aspectPreviewRange();

  return `
    <h3>Target Preview</h3>
    <div class="aspect-preview-frame">
      ${img ? `<img id="aspectPreviewImg" src="${media(img)}" alt="Target aspect preview">` : '<p>Choose source material on the Overview tab to preview the target frame.</p>'}
    </div>
    ${range.duration ? aspectPreviewSlider(range) : ''}
    ${shotOutputList(outputs, null)}
  `;
}

function aspectPreviewRange() {
  const sourceDuration = parseDuration((state.source_info && state.source_info.duration) || '0');
  const section = state.source_section || {};
  const start = Number(section.enabled ? section.start : 0) || 0;
  const end = Number(section.enabled ? section.end : sourceDuration) || sourceDuration;
  return {
    start,
    end: Math.max(start, end),
    value: section.enabled ? start : Math.min(10, sourceDuration),
    duration: Math.max(0, end - start),
  };
}

function aspectPreviewSlider(range) {
  return `
    <label>Preview time: <span id="aspectPreviewLabel">${formatSeconds(range.value)}</span></label>
    <input id="aspectPreviewTime" type="range" min="${range.start}" max="${range.end}" step="0.041" value="${range.value}" oninput="updateAspectPreview(this.value)">
  `;
}

function outpaintOverlapWarning(s) {
  const warnings = [];
  if (!String(s.prompt || '').toLowerCase().includes('outpaint')) {
    warnings.push('The global Outpainting prompt does not contain "outpaint". The LTX IC-LoRA usually needs that word to activate.');
  }
  const overlap = Number(s.overlap_frames ?? 8);
  const chunkSeconds = Number(s.chunk_seconds ?? 20);
  if (Number.isFinite(overlap) && overlap < 8) {
    warnings.push('Overlap below 8 frames can cause held-frame seams if LTX returns short chunks. 8 or 9 frames is recommended.');
  }
  if (Number.isFinite(chunkSeconds) && chunkSeconds > 0 && chunkSeconds < 10) {
    warnings.push('Short chunks create many separate LTX jobs and can make outpainting dramatically slower. Use around 20 seconds unless a shot needs special handling.');
  }
  if (!warnings.length) return '';

  return `<div class="inline-warning">${warnings.map(esc).join('<br>')}</div>`;
}

function shotOutputList(paths, limit) {
  if (!paths.length) return '';

  const shown = limit ? paths.slice(0, limit) : paths;
  const items = shown.map(path => `<li>${esc(path)}</li>`).join('');
  const remainder = limit && paths.length > limit ? `<li>${paths.length - limit} more...</li>` : '';

  return `<h3>Output Path</h3><ul class="output-list">${items}${remainder}</ul>`;
}

function fileRow(st, file) {
  const thumb = file.preview ? `<img class="file-thumb" src="${media(file.preview)}" alt="">` : '';
  const emptyClass = thumb ? '' : 'no-thumb';

  return `
    <div class="file ${emptyClass}" onclick="selected['${st.key}']='${esc(file.path)}';draw()">
      ${thumb}
      <div class="file-path">${esc(file.path)}</div>
    </div>
  `;
}

function drawStage(st) {
  const s = settings(st.key);
  const selectedFile = selected[st.key];
  const expected = (state.expected_outputs && state.expected_outputs[st.key]) || [];
  const sp = stageProgress(st.key);

  if (st.key === 'outpaint') return drawOutpaint(st, s, expected, sp);

  document.getElementById('app').innerHTML = `
    <div class="grid">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        ${progressHtml(sp.percent, sp.label)}
        ${st.fields.map(f => fieldHtml(st, f)).join('')}
        ${st.key === 'audio' ? audioStemLinksHtml() : ''}
        ${shotOutputList(expected, null)}
        ${stageCheckboxes(s)}
        <div class="actions">
          <button class="primary" onclick="runStage('${st.key}')" ${state.running ? 'disabled' : ''}>Run ${st.title}</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card files">
        <h3>Intermediate Files</h3>
        ${st.files.map(f => fileRow(st, f)).join('') || '<p>No files yet.</p>'}
      </section>
      <section class="card preview">
        ${aspectPreviewHtml(st)}
        <h3>${selectedFile ? esc(selectedFile) : 'Preview'}</h3>
        ${preview(selectedFile)}
      </section>
    </div>
    <section class="card" style="margin-top:16px">${runLogHtml()}</section>
  `;

  bindStageFields(st.key);
  showCommand(st.key);
}

function audioStemLinksHtml() {
  const stems = state.audio_stems || [];
  if (!stems.length) return '';
  return `
    <h3>Audio Stems</h3>
    <div class="audio-stems">
      ${stems.map(audioStemItem).join('')}
    </div>
  `;
}

function audioStemItem(stem) {
  const exists = Boolean(stem.exists);
  const path = stem.path || '';
  const size = Number(stem.size || 0);
  const sizeLabel = size ? ` (${formatBytes(size)})` : '';
  const controls = exists ? `
    <audio controls preload="none" src="${media(path)}"></audio>
    <div class="audio-stem-actions">
      <a class="button-like" href="${media(path)}" download>Download WAV</a>
      <button class="icon-button inline" type="button" title="Save this stem as..." onclick="exportMedia(${jsArg(path)})">&#128190;</button>
    </div>
  ` : '<p>Not generated yet.</p>';
  return `
    <div class="layer-item audio-stem-item">
      <span>${esc(stem.label || 'Audio stem')}${esc(sizeLabel)}</span>
      <strong>${esc(path)}</strong>
      ${controls}
    </div>
  `;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`;
}

function stageCheckboxes(s) {
  return `
    <div class="checks">
      <label><input data-field="force" type="checkbox" ${s.force === 'true' ? 'checked' : ''}>Regenerate</label>
      <label><input data-field="dry_run" type="checkbox" ${s.dry_run === 'true' ? 'checked' : ''}>Dry run</label>
    </div>
  `;
}

function bindStageFields(key) {
  document.querySelectorAll('[data-field]').forEach(el => {
    el.addEventListener('change', () => saveStage(key, true));
  });
}

function stageProgress(key) {
  return ((state.phase_progress && state.phase_progress.stages) || []).find(p => p.key === key)
    || { percent: 0, label: 'Waiting' };
}

function stageProgressByTitle(title) {
  return ((state.phase_progress && state.phase_progress.stages) || []).find(p => p.stage === title)
    || { percent: 0, label: 'Waiting' };
}

function progressHtml(percent, label) {
  const p = Math.max(0, Math.min(100, Number(percent) || 0));
  return `
    <div class="phase-progress">
      <div><span data-progress-label>${esc(label || 'Waiting')}</span><span data-progress-percent>${p}%</span></div>
      <progress value="${p}" max="100"></progress>
    </div>
  `;
}

function scrollableElements() {
  return [...document.querySelectorAll('.files, pre.log')];
}

function scrollElementKey(el, index) {
  if (el.id) return '#' + el.id;
  if (el.classList.contains('files')) return 'files:' + index;
  if (el.classList.contains('log')) return 'log:' + index;
  return 'scroll:' + index;
}

function captureScrollState() {
  const entries = scrollableElements().map((el, index) => ({
    key: scrollElementKey(el, index),
    top: el.scrollTop,
    left: el.scrollLeft,
    atBottom: el.classList.contains('log') && el.scrollHeight - el.clientHeight - el.scrollTop < 28,
  }));

  return { windowX: window.scrollX, windowY: window.scrollY, entries };
}

function restoreScrollState(snap) {
  if (!snap) return;

  const apply = () => {
    const byKey = new Map(snap.entries.map(item => [item.key, item]));
    scrollableElements().forEach((el, index) => {
      const saved = byKey.get(scrollElementKey(el, index));
      if (!saved) return;
      el.scrollTop = saved.atBottom ? el.scrollHeight : saved.top;
      el.scrollLeft = saved.left;
    });
    window.scrollTo(snap.windowX || 0, snap.windowY || 0);
  };

  apply();
  setTimeout(apply, 80);
}

function isEditingField() {
  const el = document.activeElement;
  return !!(el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName));
}

function runLogHtml() {
  return `
    <div class="log-heading">
      <h3>Run Log</h3>
      <button type="button" onclick="copyRunLog()">Copy Log</button>
    </div>
    <pre class="log" data-run-log="true">${logHtml(state.log)}</pre>
  `;
}

function logHtml(text) {
  return String(text || '')
    .split('\n')
    .map(line => `<span class="${logClass(line)}">${esc(line)}</span>`)
    .join('\n');
}

function logClass(line) {
  const lower = String(line).toLowerCase();
  if (lower.includes('polling temporarily failed')) return 'log-warn';
  // Lines explicitly labelled "Warning:"/"Notice:" stay yellow even if they contain words
  // like "failed" — check this before the error pattern below.
  if (/^\s*(warning|notice):/.test(lower)) return 'log-warn';
  if (/traceback|runtimeerror|exception|error|failed|refused|exit code [1-9]|filenotfound|permissionerror/.test(lower)) return 'log-error';
  if (/warning|skipping|timed out/.test(lower)) return 'log-warn';
  if (/ready|reuse|wrote|finished with exit code 0|started/.test(lower)) return 'log-ok';
  return '';
}

async function copyRunLog() {
  const text = state.log || '';

  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement('textarea');
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }
}

function preview(path) {
  if (!path) return '<p>Select an image, video, manifest, workflow, or log file.</p>';

  const ext = path.split('.').pop().toLowerCase();
  if (['png', 'jpg', 'jpeg', 'webp', 'tif', 'tiff'].includes(ext)) return `<img src="${media(path)}">`;
  if (['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v'].includes(ext)) return `<video src="${media(path)}" controls></video>`;

  return `
    <pre id="textPreview">Text preview opens via the browser media endpoint.</pre>
    <p><a href="${media(path)}" target="_blank">Open file</a></p>
  `;
}
