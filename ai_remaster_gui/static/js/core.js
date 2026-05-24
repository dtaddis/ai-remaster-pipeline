let state = null;
let active = 'global';
let selected = {};
let lastRenderSignature = '';

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

  document.getElementById('root').textContent = state.root + (
    state.running ? '  |  Running: ' + state.running_stage : ''
  );

  const sig = renderSignature();
  if (!force && (editing || shouldPreserveMediaDom(mediaActive) || sig === lastRenderSignature)) {
    updateRunLogs();
    return;
  }

  drawTabs();
  draw(false);
  wireColourShotVideos();
  lastRenderSignature = sig;
  restoreScrollState(snap);
}

function renderSignature() {
  if (!state) return '';

  return JSON.stringify({
    active,
    stages: state.stages,
    settings: state.settings,
    expected_outputs: state.expected_outputs,
    source_previews: state.source_previews,
    source_info: state.source_info,
    source_monochrome: state.source_monochrome,
    aspect_preview: state.aspect_preview,
    shot_views: state.shot_views,
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

function shouldPreserveMediaDom(mediaActive) {
  if (!mediaActive) return false;

  // Normal polling must not recreate video elements while the user is inspecting
  // chunk, shot, or recomposition previews. A manual Refresh still redraws.
  return ['outpaint', 'colour', 'recomp', 'output'].includes(active);
}

function updateRunLogs() {
  document.querySelectorAll('[data-run-log]').forEach(el => {
    const html = logHtml(state.log);
    if (el.innerHTML !== html) el.innerHTML = html;
  });
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
  draw();
  wireColourShotVideos();
  lastRenderSignature = renderSignature();
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
