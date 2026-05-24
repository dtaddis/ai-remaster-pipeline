function drawSettings() {
  const refs = settings('references');
  const outpaint = settings('outpaint');
  const colour = settings('colour');
  const recomp = settings('recomp');

  document.getElementById('app').innerHTML = `
    <section class="card">
      <h2>Settings</h2>
      ${comfySettingsHtml()}
      ${qwenSettingsHtml(refs)}
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
