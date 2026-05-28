let state = null;
let active = 'global';
let selected = {};
let lastRenderSignature = '';
let lastOutpaintVisualSignature = '';
let lastSeenLogCount = null;

const media = path => '/media?path=' + encodeURIComponent(path);
const mediaClip = (path, start, end, key) => (
  '/media?path=' + encodeURIComponent(path)
  + '&clip_start=' + encodeURIComponent(start)
  + '&clip_end=' + encodeURIComponent(end)
  + '&clip_key=' + encodeURIComponent(key || '')
);

async function api(path, opts = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  return await response.json();
}

async function refresh(force = false) {
  const snap = captureScrollState();
  const editing = isEditingField();
  const mediaActive = hasMediaOnPage();

  state = await api('/api/state');
  pruneSelected();
  if (!availableTabs().includes(active)) active = 'global';
  notifyNewLogErrors();

  document.getElementById('root').textContent = state.root + (
    state.running ? '  |  Running: ' + state.running_stage : ''
  );
  const version = document.getElementById('version');
  if (version) version.textContent = state.version || '';

  const sig = renderSignature();
  const outpaintVisualChanged = active === 'outpaint'
    && outpaintVisualSignature() !== lastOutpaintVisualSignature;
  if (!force && (editing || (shouldPreserveInteractiveDom(mediaActive) && !outpaintVisualChanged) || sig === lastRenderSignature)) {
    updateOutpaintGuidePreviews();
    updateRunLogs();
    return;
  }

  drawTabs();
  draw(false);
  wireColourShotVideos();
  lastRenderSignature = sig;
  lastOutpaintVisualSignature = outpaintVisualSignature();
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
    source_previews: state.source_previews,
    source_info: state.source_info,
    source_monochrome: state.source_monochrome,
    aspect_preview: state.aspect_preview,
    shot_views: state.shot_views,
    outpaint_chunks: state.outpaint_chunks,
    cache: state.cache,
    progress: state.progress,
    phase_progress: state.phase_progress,
    running: state.running,
    running_stage: state.running_stage,
    running_reference: state.running_reference,
  });
}

function hasMediaOnPage() {
  return ['outpaint', 'colour', 'recomp', 'output'].includes(active)
    ? document.querySelectorAll('video').length > 0
    : false;
}

function shouldPreserveInteractiveDom(mediaActive) {
  const app = document.getElementById('app');
  if (!app || !app.children.length) return false;
  if (['global', 'outpaint'].includes(active)) return true;
  if (!mediaActive) return false;

  // Normal polling must not recreate video elements while the user is inspecting
  // chunk, shot, or recomposition previews. A manual Refresh still redraws.
  return ['outpaint', 'colour', 'recomp', 'output'].includes(active);
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

function selectTab(tab) {
  active = tab;
  drawTabs();
  draw(false);
  wireColourShotVideos();
  lastRenderSignature = renderSignature();
  lastOutpaintVisualSignature = outpaintVisualSignature();
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
