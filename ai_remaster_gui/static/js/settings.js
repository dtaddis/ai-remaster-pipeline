function drawSettings() {
  const refs = settings('references');
  const outpaint = settings('outpaint');
  const colour = settings('colour');
  const recomp = settings('recomp');
  const upscale = settings('upscale');

  document.getElementById('app').innerHTML = `
    <section class="card">
      <h2>Settings</h2>
      ${comfySettingsHtml()}
      ${qwenSettingsHtml(refs)}
      ${upscaleSettingsHtml(upscale)}
      ${pipelineDefaultsHtml(outpaint, colour, recomp)}
      ${logFileSettingsHtml()}
    </section>
  `;
}

function comfySettingsHtml() {
  return `
    <h3>ComfyUI</h3>
    <div class="row">
      <input id="comfyUrl" value="http://127.0.0.1:8188">
      <button onclick="loadComfy()">Refresh Queue</button>
    </div>
    <pre class="log" id="queue"></pre>
  `;
}

function qwenSettingsHtml(refs) {
  return `
    <h3>Qwen Reference Generation</h3>
    <label>Workflow</label>
    <input value="${esc(refs.workflow || '')}" readonly>
    <label>Model backend</label>
    <input value="${esc(refs.model_backend || 'gguf')}" readonly>
    <label>GGUF model</label>
    <input value="${esc(refs.gguf_model || 'qwen-image-edit-2511-Q4_K_M.gguf')}" readonly>
    <label>Prompt</label>
    <textarea readonly>${esc(refs.prompt || '')}</textarea>
    <label>Prompt suffix</label>
    <textarea readonly>${esc(refs.prompt_suffix || '')}</textarea>
  `;
}

function upscaleSettingsHtml(upscale) {
  return `
    <h3>Upscaling Backend</h3>
    <div id="upscaleAdvanced" class="settings-fields">
      ${settingsPathField('realbasicvsr_repo', 'RealBasicVSR repo', 'folder', upscale.realbasicvsr_repo || 'tools/realbasicvsr')}
      ${settingsPathField('python_executable', 'Upscaler Python', 'file', upscale.python_executable || '')}
      ${settingsPathField('config', 'Config', 'file', upscale.config || '')}
      ${settingsPathField('checkpoint', 'Checkpoint', 'file', upscale.checkpoint || '')}
      <label>Max sequence length</label>
      <input data-upscale-field="max_seq_len" type="number" step="1" value="${esc(upscale.max_seq_len || '0')}">
      <div class="actions">
        <button type="button" onclick="saveUpscaleAdvanced()">Save Upscaling Settings</button>
      </div>
    </div>
  `;
}

function settingsPathField(key, label, kind, value) {
  return `
    <label>${label}</label>
    <div class="field-row">
      <input data-field="${key}" data-upscale-field="${key}" data-kind="${kind}" value="${esc(value)}">
      <button type="button" onclick="browseField('upscale','${key}','${kind}')">Browse</button>
    </div>
  `;
}

function pipelineDefaultsHtml(outpaint, colour, recomp) {
  return `
    <h3>Pipeline Defaults</h3>
    <div class="source-info">
      <div><span>Outpaint aspect</span><strong>${esc(outpaint.target_aspect || '16:9')}</strong></div>
      <div><span>Outpaint height</span><strong>${esc(outpaint.target_height || '720')}</strong></div>
      <div><span>Color CRF</span><strong>${esc(colour.crf || '18')}</strong></div>
      <div><span>Feather pixels</span><strong>${esc(recomp.feather_pixels || '80')}</strong></div>
    </div>
  `;
}

function logFileSettingsHtml() {
  return `
    <h3>Log file</h3>
    <div class="row">
      <input id="comfyLog" placeholder="path/to/comfy.log">
      <button onclick="loadLogFile()">Load</button>
    </div>
    <pre class="log" id="comfyLogText"></pre>
  `;
}

async function loadComfy() {
  const url = document.getElementById('comfyUrl').value;
  const result = await api('/api/comfy?url=' + encodeURIComponent(url));
  document.getElementById('queue').textContent = result.ok
    ? JSON.stringify(result.queue, null, 2)
    : result.error;
}

async function loadLogFile() {
  const path = document.getElementById('comfyLog').value;
  const result = await api('/api/logfile?path=' + encodeURIComponent(path));
  document.getElementById('comfyLogText').textContent = result.text;
}

async function saveUpscaleAdvanced() {
  const values = {};
  document.querySelectorAll('#upscaleAdvanced [data-upscale-field]').forEach(el => {
    values[el.dataset.upscaleField] = el.value;
  });
  await postJson('/api/settings', { stage: 'upscale', values });
  state = await api('/api/state');
  drawSettings();
}
