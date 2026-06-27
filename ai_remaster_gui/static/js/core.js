let state = null;
let active = 'global';
let selected = {};
let lastRenderSignature = '';
let lastOutpaintVisualSignature = '';
let lastSeenLogCount = null;
let quitting = false;

const media = path => '/media?path=' + encodeURIComponent(path);
const mediaClip = (path, start, end, key) => (
  '/media?path=' + encodeURIComponent(path)
  + '&clip_start=' + encodeURIComponent(start)
  + '&clip_end=' + encodeURIComponent(end)
  + '&clip_key=' + encodeURIComponent(key || '')
);
const stateUrl = (options = {}) => {
  const params = new URLSearchParams({ active: active || '' });
  if (options.shotPreviews) params.set('shot_previews', options.shotPreviews);
  return '/api/state?' + params.toString();
};

async function api(path, opts = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  return await response.json();
}

async function refresh(force = false) {
  if (quitting) return;
  const snap = captureScrollState();
  const editing = isEditingField();
  const mediaActive = hasMediaOnPage();

  const appHasContent = !!document.getElementById('app')?.children.length;
  const useCachedShotPreviews = !force && active === 'shots' && appHasContent;
  state = await api(stateUrl({ shotPreviews: useCachedShotPreviews ? 'cached' : 'generate' }));
  pruneSelected();
  if (!availableTabs().includes(active)) active = 'global';
  notifyNewLogErrors();

  document.getElementById('root').textContent = state.root + (
    state.running ? '  |  Running: ' + state.running_stage : ''
  );
  const version = document.getElementById('version');
  if (version) version.textContent = state.version || '';
  updateSystemStatus();

  const sig = renderSignature();
  const currentOutpaintVisualSignature = outpaintVisualSignature();
  const outpaintVisualChanged = active === 'outpaint'
    && currentOutpaintVisualSignature !== lastOutpaintVisualSignature;
  if (!force && active === 'global' && document.getElementById('app')?.children.length) {
    updateOverviewDynamicStatus();
    updateRunLogs();
    lastRenderSignature = sig;
    lastOutpaintVisualSignature = currentOutpaintVisualSignature;
    return;
  }

  if (!force && active === 'outpaint' && document.getElementById('app')?.children.length) {
    updateOutpaintGuidePreviews();
    updateOutpaintRawPreviews();
    updateRunLogs();
    updateOutpaintRuntimeControls();
    lastRenderSignature = sig;
    lastOutpaintVisualSignature = currentOutpaintVisualSignature;
    return;
  }

  if (!force && !editing && active === 'references' && document.getElementById('app')?.children.length) {
    updateReferencesDynamicStatus();
    updateRunLogs();
    lastRenderSignature = sig;
    lastOutpaintVisualSignature = currentOutpaintVisualSignature;
    return;
  }

  if (!force && (editing || (shouldPreserveInteractiveDom(mediaActive) && !outpaintVisualChanged) || sig === lastRenderSignature)) {
    if (active === 'global') updateOverviewDynamicStatus();
    updateOutpaintGuidePreviews();
    updateOutpaintRawPreviews();
    updateRunLogs();
    if (active === 'outpaint') updateOutpaintRuntimeControls();
    lastRenderSignature = sig;
    lastOutpaintVisualSignature = currentOutpaintVisualSignature;
    return;
  }

  drawTabs();
  draw(false);
  wireColourShotVideos();
  lastRenderSignature = sig;
  lastOutpaintVisualSignature = currentOutpaintVisualSignature;
  restoreScrollState(snap);
}

function renderSignature() {
  if (!state) return '';

  return JSON.stringify({
    active,
    stages: state.stages,
    settings: state.settings,
    expected_outputs: state.expected_outputs,
    existing_outputs: state.existing_outputs,
    upscale_preview: state.upscale_preview,
    output_selection: state.output_selection,
    source_previews: state.source_previews,
    source_info: state.source_info,
    source_monochrome: state.source_monochrome,
    aspect_preview: state.aspect_preview,
    shot_views: state.shot_views,
    outpaint_chunks: state.outpaint_chunks,
    audio_stems: state.audio_stems,
    cache: state.cache,
    progress: state.progress,
    phase_progress: state.phase_progress,
    running: state.running,
    running_stage: state.running_stage,
    running_reference: state.running_reference,
  });
}

function updateSystemStatus() {
  const el = document.getElementById('systemStatus');
  if (!el) return;
  const status = (state && state.system_status) || {};
  const cuda = status.cuda || {};
  const cpu = status.cpu || {};
  const gpu = status.gpu || {};
  const ram = status.ram || {};
  const vram = status.vram || {};
  const cudaTone = cuda.available ? (cuda.warning ? 'warn' : 'ok') : 'bad';
  const badges = [
    statusBadge('CUDA', cuda.capability || (cuda.available ? 'Yes' : 'No'), cudaTone, statusDetail(cuda)),
    utilizationBadge(cpu, 'CPU'),
    utilizationBadge(gpu, 'GPU'),
    memoryBadge(ram),
    memoryBadge(vram),
  ];
  if (cuda.warning) badges.push(statusBadge('FlashVSR', 'Unsupported', 'warn', cuda.warning));
  el.innerHTML = badges.join('');
  updateStickyOffsets();
}

function statusBadge(name, value, tone, title) {
  return `<span class="status-badge ${tone}" title="${esc(title || '')}"><span>${esc(name)}</span><strong>${esc(value)}</strong></span>`;
}

function memoryBadge(item) {
  const text = item && item.label ? String(item.label) : 'Unknown';
  const match = text.match(/^(\S+)\s+(.+)$/);
  const percent = Number(item && item.percent);
  const tone = Number.isFinite(percent) ? (percent >= 90 ? 'bad' : percent >= 75 ? 'warn' : 'ok') : 'unknown';
  return statusBadge(match ? match[1] : 'Memory', match ? match[2] : text, tone, statusDetail(item || {}));
}

function utilizationBadge(item, fallbackName) {
  const percent = Number(item && item.percent);
  const tone = Number.isFinite(percent) ? (percent >= 95 ? 'bad' : percent >= 80 ? 'warn' : 'ok') : 'unknown';
  return statusBadge((item && item.name) || fallbackName, Number.isFinite(percent) ? `${Math.round(percent)}%` : 'Unknown', tone, statusDetail(item || {}));
}

function statusDetail(item) {
  const details = [];
  if (item.detail) details.push(String(item.detail));
  if (item.torch) details.push('torch ' + item.torch);
  if (item.cuda) details.push('CUDA ' + item.cuda);
  if (item.warning) details.push(String(item.warning));
  return details.join(' | ');
}

function updateStickyOffsets() {
  const header = document.querySelector('header');
  if (!header) return;
  document.documentElement.style.setProperty('--header-height', `${Math.ceil(header.getBoundingClientRect().height)}px`);
}

window.addEventListener('resize', updateStickyOffsets);

function updateOverviewDynamicStatus() {
  const analysis = state.source_analysis || {};
  const analysisEl = document.getElementById('overviewSourceAnalysis');
  if (analysisEl) {
    analysisEl.outerHTML = sourceAnalysisHtml(analysis);
  }

  const toneEl = document.getElementById('overviewSourceTone');
  if (toneEl) {
    const source = (state.settings.global || {}).source || '';
    toneEl.textContent = source && !analysis.ready
      ? (analysis.message || 'Analyzing source material')
      : (state.source_monochrome ? 'Source looks black and white' : 'Source appears to contain colour');
  }

  const filmstripEl = document.getElementById('overviewFilmstrip');
  if (filmstripEl) filmstripEl.innerHTML = overviewFilmstripInner();

  const infoEl = document.getElementById('overviewSourceInfo');
  if (infoEl) infoEl.innerHTML = sourceInfoHtml(state.source_info || {});
  updateOverviewSectionControlsFromState();

  const progressEl = document.getElementById('overviewPipelineProgress');
  if (progressEl) {
    const progress = (state.phase_progress && state.phase_progress.global) || { percent: 0, label: 'Waiting' };
    progressEl.innerHTML = progressHtml(progress.percent, progress.label);
  }

  const tableEl = document.getElementById('overviewProgressTable');
  if (tableEl) tableEl.innerHTML = overviewProgressTable();
}

function updateOverviewSectionControlsFromState() {
  const start = document.getElementById('sectionStart');
  const end = document.getElementById('sectionEnd');
  if (!start || !end) return;
  if ([start, end].includes(document.activeElement)) return;

  const global = (state.settings && state.settings.global) || {};
  const duration = parseDuration((state.source_info && state.source_info.duration) || '0');
  const startValue = Number(global.section_start || 0);
  const endValue = Number(global.section_end || duration || 0);
  const sliderMax = Math.max(duration, startValue, endValue, 1);
  const fpsText = (state.source_info && (state.source_info.fps || state.source_info.frame_rate)) || '';
  const fps = Math.max(1, Number(String(fpsText).split(' ')[0]) || 24);
  const step = (1 / fps).toFixed(6);

  start.max = String(sliderMax);
  end.max = String(sliderMax);
  start.step = step;
  end.step = step;
  start.dataset.fps = String(fps);
  end.dataset.fps = String(fps);
  start.value = String(Math.min(sliderMax, Math.max(0, startValue)));
  end.value = String(Math.min(sliderMax, Math.max(0, endValue)));

  const startLabel = document.getElementById('sectionStartLabel');
  const endLabel = document.getElementById('sectionEndLabel');
  if (startLabel) startLabel.textContent = formatSeconds(start.value);
  if (endLabel) endLabel.textContent = endValue ? formatSeconds(end.value) : 'End';
}

function hasMediaOnPage() {
  return ['outpaint', 'colour', 'recomp', 'audio', 'upscale', 'output'].includes(active)
    ? document.querySelectorAll('video').length > 0
    : false;
}

function shouldPreserveInteractiveDom(mediaActive) {
  const app = document.getElementById('app');
  if (!app || !app.children.length) return false;
  if (active === 'outpaint') return true;
  if (!mediaActive) return false;

  // Normal polling must not recreate video elements while the user is inspecting
  // chunk, shot, or recomposition previews. A manual Refresh still redraws.
  return ['outpaint', 'colour', 'recomp', 'audio', 'upscale', 'output'].includes(active);
}

function outpaintVisualSignature() {
  if (!state || !state.outpaint_chunks) return '';
  return JSON.stringify((state.outpaint_chunks.rows || []).map(row => ({
    index: row.index,
    raw_exists: row.raw_exists,
    raw_mtime: row.raw_mtime,
    anchor_exists: row.anchor_exists,
    anchor_mtime: row.anchor_mtime,
    anchor_seconds: row.anchor_seconds,
    anchor_frame_preview: row.anchor_frame_preview,
    raw_start_preview: row.raw_start_preview,
    raw_middle_preview: row.raw_middle_preview,
    raw_end_preview: row.raw_end_preview,
    guides_sig: (row.guides || []).map(g => `${g.frame_idx}:${g.image_mtime}`).join(','),
  })));
}

function updateRunLogs() {
  document.querySelectorAll('[data-run-log]').forEach(el => {
    const html = logHtml(state.log);
    if (el.innerHTML === html) return;
    const shouldFollow = isNearLogBottom(el);
    el.innerHTML = html;
    if (shouldFollow) el.scrollTop = el.scrollHeight;
  });
}

function notifyNewLogErrors() {
  const count = Number(state && state.log_count);
  const lines = String((state && state.log) || '').split('\n');
  if (!Number.isFinite(count)) return;
  if (lastSeenLogCount === null) {
    lastSeenLogCount = count;
    return;
  }
  if (count <= lastSeenLogCount) return;

  const firstLineNumber = Math.max(0, count - lines.length);
  const startOffset = Math.max(0, lastSeenLogCount - firstLineNumber);
  const newLines = lines.slice(startOffset);
  lastSeenLogCount = count;

  const errorIndex = newLines.findIndex(isLogErrorLine);
  if (errorIndex === -1) return;

  const excerptStart = Math.max(0, errorIndex - 8);
  const excerptEnd = Math.min(newLines.length, errorIndex + 18);
  const excerpt = newLines.slice(excerptStart, excerptEnd).join('\n').trim();
  showErrorPopup(excerpt || newLines[errorIndex]);
}

function isLogErrorLine(line) {
  const lower = String(line || '').toLowerCase();
  if (lower.includes('polling temporarily failed')) return false;
  // Lines a tool explicitly labels "Warning:"/"Notice:" are non-fatal by intent, even when
  // their text contains words like "failed" (e.g. the outpaint black-margin notice). Don't
  // raise the error popup for them.
  if (/^\s*(warning|notice):/.test(lower)) return false;
  return /traceback|runtimeerror|exception|error|failed|refused|exit code [1-9]|filenotfound|permissionerror/.test(lower);
}

function showErrorPopup(excerpt) {
  let modal = document.getElementById('logErrorModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'logErrorModal';
    modal.className = 'image-modal hidden';
    modal.innerHTML = `
      <div class="image-modal-backdrop" onclick="closeErrorPopup()"></div>
      <div class="prompt-modal-panel error-modal-panel">
        <div class="image-modal-heading">
          <strong>ARP Noticed An Error</strong>
          <button type="button" onclick="closeErrorPopup()">Close</button>
        </div>
        <p class="shot-empty">A new error appeared in the run log. The stage may need attention before continuing.</p>
        <pre id="logErrorExcerpt" class="log error-popup-log"></pre>
        <div class="actions">
          <button class="primary" type="button" onclick="copyRunLog()">Copy Log</button>
          <button type="button" onclick="closeErrorPopup()">Dismiss</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
  }
  const excerptEl = document.getElementById('logErrorExcerpt');
  if (excerptEl) excerptEl.textContent = excerpt;
  modal.classList.remove('hidden');
}

function closeErrorPopup() {
  const modal = document.getElementById('logErrorModal');
  if (modal) modal.classList.add('hidden');
}

function isNearLogBottom(el) {
  return el.scrollHeight - el.clientHeight - el.scrollTop < 28;
}

function availableTabs() {
  const tabs = ['global'];
  for (const st of state.stages) {
    if (st.key === 'output') tabs.push('cache');
    tabs.push(st.key);
  }
  tabs.push('settings');
  return tabs;
}

function drawTabs() {
  const names = { global: 'Overview', cache: 'Cached Files', settings: 'Settings' };
  const tabs = availableTabs().map(tabButton).join('');
  document.getElementById('tabs').innerHTML = tabs;

  function tabButton(tab) {
    const label = names[tab] || stage(tab).title;
    const activeClass = active === tab ? 'active' : '';

    return `
      <button class="tab ${activeClass}" onclick="selectTab('${tab}')">
        ${label}
      </button>
    `;
  }
}

async function selectTab(tab) {
  active = tab;
  drawTabs();
  if (tab === 'outpaint') showOutpaintLoadingShell();
  else if (['shots', 'references', 'colour'].includes(tab)) showShotLoadingShell(tab);
  state = await api(stateUrl());
  pruneSelected();
  drawTabs();
  draw(false);
  wireColourShotVideos();
  lastRenderSignature = renderSignature();
  lastOutpaintVisualSignature = outpaintVisualSignature();
}

function showOutpaintLoadingShell() {
  const app = document.getElementById('app');
  if (!app) return;
  app.innerHTML = `
    <section class="card outpaint-loading">
      <span class="spinner"></span>
      <div>
        <h2>Preparing Outpainting</h2>
        <p class="shot-empty">Checking chunk ranges and cached previews...</p>
      </div>
    </section>
  `;
}

function showShotLoadingShell(tab) {
  const app = document.getElementById('app');
  if (!app) return;
  const copy = {
    shots: ['Loading Shot Detection', 'Reading the shot manifest and rendering boundary thumbnails...'],
    references: ['Loading References', 'Reading the shot manifest and reference thumbnails...'],
    colour: ['Loading Shot Segments', 'Reading the shot manifest and colour previews...'],
  }[tab] || ['Loading', 'Reading the shot manifest...'];
  app.innerHTML = `
    <section class="card outpaint-loading">
      <span class="spinner"></span>
      <div>
        <h2>${esc(copy[0])}</h2>
        <p class="shot-empty">${esc(copy[1])}</p>
      </div>
    </section>
  `;
}

function stage(key) {
  return state.stages.find(s => s.key === key);
}

function settings(key) {
  return state.settings[key] || {};
}

function pruneSelected() {
  if (!state || !state.stages) return;

  for (const st of state.stages) {
    if (selected[st.key] && !st.files.some(f => f.path === selected[st.key])) {
      delete selected[st.key];
    }
  }
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

function jsArg(value) {
  return esc(JSON.stringify(String(value ?? '')));
}
