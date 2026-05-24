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
        <button type="button" onclick="clearOverview()">Clear</button>
      </div>
      <img class="hero-logo" src="/media?path=assets/branding/arp-logo-wide.png" alt="ARP - AI Remaster Pipeline">
      ${overviewSourcePicker(source)}
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
