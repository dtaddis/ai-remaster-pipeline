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

function fieldHtml(st, field) {
  const [key, label, kind, defaultValue] = field;
  const value = settings(st.key)[key] ?? defaultValue ?? '';

  if (kind.startsWith('select:')) return selectFieldHtml(key, label, kind, value);
  if (kind.startsWith('range:')) return rangeFieldHtml(key, label, kind, value);

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
    `;
  }

  return `<label>${label}</label>${input}`;
}

function selectFieldHtml(key, label, kind, value) {
  const options = kind.slice(7).split('|')
    .map(option => `<option ${value === option ? 'selected' : ''}>${option}</option>`)
    .join('');
  return `<label>${label}</label><select data-field="${key}">${options}</select>`;
}

function rangeFieldHtml(key, label, kind, value) {
  const [min, max, step] = kind.slice(6).split('|');
  return `
    <label>${label}: <span id="${key}Value">${esc(value)}</span></label>
    <input
      data-field="${key}"
      data-kind="${kind}"
      type="range"
      min="${esc(min)}"
      max="${esc(max)}"
      step="${esc(step || '1')}"
      value="${esc(value)}"
      oninput="document.getElementById('${key}Value').textContent=this.value"
    >
  `;
}

function aspectPreviewHtml(st) {
  if (st.key !== 'outpaint') return '';

  const img = state.aspect_preview;
  const outputs = (state.expected_outputs && state.expected_outputs.outpaint) || [];
  const range = aspectPreviewRange();

  return `
    <h3>Target Preview</h3>
    ${img ? `<img id="aspectPreviewImg" src="${media(img)}" alt="Target aspect preview">` : '<p>Choose source material on the Overview tab to preview the target frame.</p>'}
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
    <input type="range" min="${range.start}" max="${range.end}" step="0.041" value="${range.value}" oninput="updateAspectPreview(this.value)">
  `;
}

function outpaintOverlapWarning(s) {
  const value = Number(s.overlap_frames ?? 8);
  if (!Number.isFinite(value) || value >= 8) return '';

  return '<div class="inline-warning">Overlap below 8 frames can cause held-frame seams if LTX returns short chunks. 8 or 9 frames is recommended.</div>';
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
      <div><span>${esc(label || 'Waiting')}</span><span>${p}%</span></div>
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
      el.scrollTop = saved.top;
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
  if (/traceback|runtimeerror|exception|error|failed|refused|exit code [1-9]|filenotfound/.test(lower)) return 'log-error';
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
