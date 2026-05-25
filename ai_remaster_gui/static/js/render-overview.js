function drawGlobal() {
  const global = state.settings.global || {};
  const source = global.source || '';
  const colorize = global.colorize !== 'false';
  const sourceTone = state.source_monochrome
    ? 'Source looks black and white'
    : 'Source appears to contain colour';
  const progress = (state.phase_progress && state.phase_progress.global) || { percent: 0, label: 'Waiting' };

  document.getElementById('app').innerHTML = `
    <section class="card">
      <div class="global-top">
        <div>
          <p class="hero">AI Remaster Pipeline</p>
          <p>Choose the source material, then run or inspect each stage.</p>
        </div>
        <div class="actions compact-actions">
          <button type="button" onclick="saveProject()">Save</button>
          <button type="button" onclick="saveProjectAs()">Save As...</button>
          <button type="button" onclick="loadProject()">Load Project</button>
          <button type="button" onclick="clearOverview()">Clear</button>
        </div>
      </div>
      <img class="hero-logo" src="/media?path=assets/branding/arp-logo-wide.png" alt="ARP - AI Remaster Pipeline">
      ${overviewSourcePicker(source)}
      ${overviewSectionPicker(global, source)}
      <div class="checks">
        <label><input id="globalColorize" type="checkbox" ${colorize ? 'checked' : ''}>Colorize</label>
        <span class="shot-time">${esc(sourceTone)}</span>
      </div>
      ${overviewFilmstrip()}
      ${sourceInfoHtml(state.source_info || {})}
      ${progressHtml(progress.percent, progress.label)}
      ${overviewActions()}
      ${overviewProgressTable()}
      ${runLogHtml()}
    </section>
  `;

  document.getElementById('globalSource').addEventListener('change', saveGlobal);
  document.getElementById('globalColorize').addEventListener('change', saveGlobalColorize);
  bindOverviewSectionControls();
}

function overviewSourcePicker(source) {
  return `
    <label>Source material</label>
    <div class="field-row">
      <input id="globalSource" value="${esc(source)}">
      <button type="button" onclick="browseGlobalSource()">Browse</button>
    </div>
  `;
}

function overviewFilmstrip() {
  const thumbs = (state.source_previews || [])
    .map(path => `<img src="${media(path)}" alt="">`)
    .join('');
  return thumbs ? `<div class="filmstrip">${thumbs}</div>` : '';
}

function overviewSectionPicker(global, source) {
  const start = Number(global.section_start || 0);
  const duration = parseDuration((state.source_info && state.source_info.duration) || '0');
  const end = Number(global.section_end || duration || 0);
  return `
    <div class="section-picker">
      ${source ? `<video id="sourceSectionVideo" class="section-video" src="${media(source)}" controls preload="metadata"></video>` : ''}
      <div class="editor-controls">
        <div>
          <label>Start: <span id="sectionStartLabel">${formatSeconds(start)}</span></label>
          <input id="sectionStart" type="range" min="0" max="${Math.max(duration, end, 1)}" step="0.041" value="${start}">
        </div>
        <div>
          <label>End: <span id="sectionEndLabel">${end ? formatSeconds(end) : 'End'}</span></label>
          <input id="sectionEnd" type="range" min="0" max="${Math.max(duration, end, 1)}" step="0.041" value="${end || duration || 0}">
        </div>
      </div>
      <div class="shot-tools">
        <button type="button" onclick="markSourceSection('start')">Mark Start</button>
        <button type="button" onclick="markSourceSection('end')">Mark End</button>
      </div>
    </div>
  `;
}

function bindOverviewSectionControls() {
  const start = document.getElementById('sectionStart');
  const end = document.getElementById('sectionEnd');
  if (start) start.addEventListener('input', () => document.getElementById('sectionStartLabel').textContent = formatSeconds(start.value));
  if (end) end.addEventListener('input', () => document.getElementById('sectionEndLabel').textContent = formatSeconds(end.value));
  if (start) start.addEventListener('change', saveGlobalSection);
  if (end) end.addEventListener('change', saveGlobalSection);
}

function overviewActions() {
  return `
    <div class="actions">
      <button class="primary" onclick="runAll()">Run Whole Remaster</button>
      <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
    </div>
  `;
}

function overviewProgressTable() {
  const rows = state.progress.map(progressRow => {
    const sp = stageProgressByTitle(progressRow.stage);
    return `
      <tr>
        <td>${progressRow.stage}</td>
        <td class="status-${progressRow.status.toLowerCase()}">${progressRow.status}</td>
        <td>${progressHtml(sp.percent, sp.label)}</td>
        <td>${esc(progressRow.latest)}</td>
      </tr>
    `;
  }).join('');

  return `
    <table>
      <tr><th>Stage</th><th>Status</th><th>Progress</th><th>Latest output</th></tr>
      ${rows}
    </table>
  `;
}
