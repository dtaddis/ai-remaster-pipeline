function drawSettings() {
  const refs = settings('references');
  const cloud = settings('cloud');
  const outpaint = settings('outpaint');
  const colour = settings('colour');
  const recomp = settings('recomp');

  document.getElementById('app').innerHTML = `
    <section class="card">
      <h2>Settings</h2>
      ${comfySettingsHtml()}
      ${cloudSettingsHtml(cloud)}
      ${qwenSettingsHtml(refs)}
      ${openAISettingsHtml(refs)}
      ${pipelineDefaultsHtml(outpaint, colour, recomp)}
      ${logFileSettingsHtml()}
    </section>
  `;
}

function cloudSettingsHtml(cloud) {
  const format = cloud.intermediate_format || 'hevc_high';
  return `
    <h3>Cloud Compute</h3>
    <p class="field-help">RunPod stages use a secure reusable CUDA 13 worker. Paste an API key; ARP creates an SSH key, provisions the Pod, installs its worker, transfers only job assets, and retrieves the results automatically. A saved Pod ID reuses an existing worker.</p>
    <label>RunPod API key</label>
    <input id="runpodApiKey" type="password" autocomplete="off" value="${esc(cloud.runpod_api_key || '')}" placeholder="RunPod API key">
    <label>Existing Pod ID (optional)</label>
    <input id="runpodPodId" value="${esc(cloud.runpod_pod_id || '')}" placeholder="Leave blank to provision automatically">
    <label>Compatible GPU pool (RunPod chooses by live availability)</label>
    <input id="runpodGpuTypes" value="${esc(cloud.runpod_gpu_types || '')}">
    <label>CUDA 13 worker image</label>
    <input id="runpodImage" value="${esc(cloud.runpod_image || '')}">
    <label>Persistent storage</label>
    <select id="runpodStorageMode">
      <option value="network" ${(cloud.runpod_storage_mode || 'network') === 'network' ? 'selected' : ''}>Portable network volume — recommended</option>
      <option value="pod" ${cloud.runpod_storage_mode === 'pod' ? 'selected' : ''}>Pod volume — cheaper but tied to one physical host</option>
    </select>
    <small class="field-help">A network volume survives Pod replacement and is mounted at /workspace. If its ID is blank, ARP creates it once and remembers it. Standard storage is currently $0.07/GB/month.</small>
    <label>Storage size (GB)</label>
    <input id="runpodVolumeGb" type="number" min="50" step="10" value="${esc(cloud.runpod_volume_gb || '100')}">
    <label>Network volume data centre</label>
    <input id="runpodDataCenterId" value="${esc(cloud.runpod_data_center_id || 'EU-RO-1')}" placeholder="EU-RO-1">
    <label>Existing network volume ID (optional)</label>
    <input id="runpodNetworkVolumeId" value="${esc(cloud.runpod_network_volume_id || '')}" placeholder="Blank = create and remember automatically">
    <label>Keep Pod running after a stage (0 = stop automatically, 1 = keep running)</label>
    <input id="runpodIdleMinutes" type="number" min="0" max="1" step="1" value="${esc(cloud.runpod_idle_minutes || '0')}">
    <label>Hugging Face token (only needed for gated weights)</label>
    <input id="huggingfaceToken" type="password" autocomplete="off" value="${esc(cloud.huggingface_token || '')}">
    <h3>Intermediate Master Format</h3>
    <select id="intermediateFormat">
      ${[
        ['h264_standard', 'H.264 Standard — compact, broadly compatible'],
        ['h264_high', 'H.264 High — CRF 10, broadly compatible'],
        ['hevc_high', 'HEVC 10-bit High — recommended'],
        ['hevc_lossless', 'HEVC 10-bit Lossless — mathematically lossless'],
      ].map(([value, label]) => `<option value="${value}" ${format === value ? 'selected' : ''}>${label}</option>`).join('')}
    </select>
    <small class="field-help">HEVC High avoids the old low-bitrate bottleneck. HEVC Lossless preserves decoded pixels exactly in one modern compressed video file—not an uncompressed image sequence. Browser previews remain compact H.264.</small>
    <div class="actions">
      <button type="button" class="primary" onclick="saveCloudSettings()">Save Cloud & Format Settings</button>
    </div>
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
    <label>Masked edit workflow</label>
    <input id="qwenMaskedWorkflow" value="${esc(refs.masked_workflow || '')}" placeholder="Bundled Qwen inpaint workflow JSON">
    <label>Model backend</label>
    <input value="${esc(refs.model_backend || 'gguf')}" readonly>
    <label>GGUF model</label>
    <input value="${esc(refs.gguf_model || 'qwen-image-edit-2511-Q4_K_M.gguf')}" readonly>
    <label>Prompt</label>
    <textarea readonly>${esc(refs.prompt || '')}</textarea>
    <label>Prompt suffix</label>
    <textarea readonly>${esc(refs.prompt_suffix || '')}</textarea>
    <div class="actions">
      <button type="button" class="primary" onclick="saveQwenEditSettings()">Save Qwen Edit Settings</button>
    </div>
  `;
}

function openAISettingsHtml(refs) {
  return `
    <h3>OpenAI Reference Generation</h3>
    <label>API key</label>
    <input id="openaiApiKey" type="password" autocomplete="off" value="${esc(refs.openai_api_key || '')}" placeholder="sk-...">
    <label>Image model</label>
    <div class="row">
      <input id="openaiImageModel" value="${esc(refs.openai_image_model || 'gpt-image-2.5-sunburst')}">
      <button type="button" onclick="refreshOpenAIModels()">Query Models</button>
    </div>
    <label>Discovered image models</label>
    <select id="openaiModelList" onchange="selectOpenAIModel(this.value)">
      <option value="">No query run</option>
    </select>
    <label>Size</label>
    <select id="openaiImageSize">
      ${[
        ['max', 'Maximum (source aspect, up to 4K)'],
        ['auto', 'Auto'],
        ['3840x2160', '3840x2160 (4K landscape)'],
        ['2160x3840', '2160x3840 (4K portrait)'],
        ['2048x2048', '2048x2048 (2K square)'],
        ['1536x1024', '1536x1024 (landscape)'],
        ['1024x1536', '1024x1536 (portrait)'],
        ['1024x1024', '1024x1024 (square)'],
      ].map(([value, label]) => `<option value="${value}" ${(refs.openai_image_size || 'max') === value ? 'selected' : ''}>${label}</option>`).join('')}
    </select>
    <small class="field-help">Maximum preserves the extracted frame's aspect ratio and requests the largest supported multiple-of-16 dimensions. The returned master is kept at that resolution.</small>
    <label>Quality</label>
    <select id="openaiImageQuality">
      ${['max', 'xhigh', 'high', 'medium', 'low', 'auto'].map(value => `<option value="${value}" ${(refs.openai_image_quality || 'max') === value ? 'selected' : ''}>${value}</option>`).join('')}
    </select>
    <div class="actions">
      <button type="button" class="primary" onclick="saveOpenAISettings()">Save OpenAI Settings</button>
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
      <input id="comfyLog" value="output/logs/comfyui-startup.log" placeholder="path/to/comfy.log">
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

function selectOpenAIModel(model) {
  if (!model) return;
  const input = document.getElementById('openaiImageModel');
  if (input) input.value = model;
}

async function saveOpenAISettings() {
  await postJson('/api/settings', {
    stage: 'references',
    values: {
      openai_api_key: document.getElementById('openaiApiKey')?.value || '',
      openai_image_model: document.getElementById('openaiImageModel')?.value || 'gpt-image-2.5-sunburst',
      openai_image_size: document.getElementById('openaiImageSize')?.value || 'max',
      openai_image_quality: document.getElementById('openaiImageQuality')?.value || 'max',
    },
  });
  state = await api(stateUrl());
}

async function saveQwenEditSettings() {
  await postJson('/api/settings', {
    stage: 'references',
    values: {
      masked_workflow: document.getElementById('qwenMaskedWorkflow')?.value || '',
    },
  });
  state = await api(stateUrl());
}

async function saveCloudSettings() {
  await postJson('/api/settings', {
    stage: 'cloud',
    values: {
      runpod_api_key: document.getElementById('runpodApiKey')?.value || '',
      runpod_pod_id: document.getElementById('runpodPodId')?.value || '',
      runpod_gpu_types: document.getElementById('runpodGpuTypes')?.value || '',
      runpod_image: document.getElementById('runpodImage')?.value || '',
      runpod_storage_mode: document.getElementById('runpodStorageMode')?.value || 'network',
      runpod_volume_gb: document.getElementById('runpodVolumeGb')?.value || '100',
      runpod_data_center_id: document.getElementById('runpodDataCenterId')?.value || 'EU-RO-1',
      runpod_network_volume_id: document.getElementById('runpodNetworkVolumeId')?.value || '',
      runpod_idle_minutes: document.getElementById('runpodIdleMinutes')?.value || '0',
      huggingface_token: document.getElementById('huggingfaceToken')?.value || '',
      intermediate_format: document.getElementById('intermediateFormat')?.value || 'hevc_high',
    },
  });
  state = await api(stateUrl());
}

async function refreshOpenAIModels() {
  await saveOpenAISettings();
  const select = document.getElementById('openaiModelList');
  if (select) select.innerHTML = '<option value="">Querying...</option>';
  const result = await api('/api/openai-models');
  if (!result.ok) {
    if (select) select.innerHTML = '<option value="">Query failed</option>';
    return alert(result.error || 'Could not query OpenAI models');
  }
  const models = result.models || [];
  if (select) {
    select.innerHTML = models.length
      ? models.map(model => `<option value="${esc(model)}">${esc(model)}</option>`).join('')
      : '<option value="">No image models returned</option>';
  }
}
