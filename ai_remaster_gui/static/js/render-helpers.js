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
  'cleanup.ai_descratch':
    'Detect only scratch-shaped damage, reconstruct those masked pixels across neighbouring frames with ProPainter, and composite them back over the untouched source. Runs after DeVignette and before Dearchive. ProPainter is licensed for non-commercial use only.',
  'cleanup.scratch_sensitivity':
    'How readily vertical, temporally persistent damage enters the AI repair mask. Higher values catch fainter scratches but can mistake narrow scene detail for damage. The mask preview makes this trade-off visible.',
  'cleanup.scratch_mask_dilate':
    'Expand each detected scratch by this many working-resolution pixels so ProPainter also replaces its damaged edges. Larger values remove wider streaks but reconstruct more source detail.',
  'cleanup.ai_descratch_height':
    'Working height used by ProPainter; the result is composited back at the exact source resolution. 720 is the safe default for a 24 GB GPU. Source and 1080 use substantially more VRAM.',
  'cleanup.ai_chunk_frames':
    'Frames processed together by ProPainter. Longer windows provide more temporal evidence but use more VRAM. 41 is a conservative default for 720p on a 24 GB GPU.',
  'cleanup.save_scratch_mask':
    'Save a companion black-and-white mask video beside the Clean Up output. White shows the detected base mask; Scratch mask expansion adds the chosen safety margin around those marks during repair.',
  'cleanup.devignette':
    'Estimate stationary dark falloff or a pale additive edge veil from samples across the clip, then apply a bounded edge correction. Black presentation bars are excluded and preserved. Runs locally before Dearchive and keeps colour intact.',
  'cleanup.repair_device':
    'Auto uses CUDA through ARP\'s PyTorch installation when an NVIDIA GPU is available, otherwise it falls back to CPU. Dearchive continues to use ComfyUI separately.',
  'cleanup.dearchive':
    'Run the LTX 2.3 Dearchive LoRA after the selected DeVignette and AI DeScratch repairs.',
  'cleanup.chunk_seconds':
    'Requested duration, from 2 to 20 seconds. At the source frame rate it is rounded to the nearest frame count where frames % 8 = 1 (8n + 1). For example, 4.04 seconds becomes 97 frames at 24 fps. Longer chunks reduce seams but use more VRAM.',
  'cleanup.overlap_frames':
    'Frames shared by neighbouring LTX chunks. They are trimmed during stitching so the final frame count stays identical to the source.',
  'cleanup.source_fidelity':
    'How strongly the complete input video controls LTX. 1.0 is the safe default and preserves the source most exactly, but can also preserve scratches. Lower values give Dearchive more freedom to repaint damage; very low values can change faces, hands, fine motion, or period detail.',
  'cleanup.lora_strength':
    'How strongly Dearchive rewrites the archive footage. 1.0 is the model author\'s workflow default.',
  'stabilize.smoothing':
    'Radius of the low-pass camera path in frames. 12 removes short gate-weave impulses while retaining deliberate pans; higher values produce a calmer camera path but can over-smooth intentional movement.',
  'stabilize.max_shift':
    'Caps horizontal and vertical correction at this many source pixels. Use 0 for no cap. The default prevents a bad match from making a very large jump.',
  'stabilize.max_angle':
    'Caps rotational correction in degrees. Use 0 for no cap. Three degrees is enough for the stronger weave measured in the Berlin test without allowing extreme rotations.',
  'stabilize.zoom':
    'Fixed safety crop used to hide transformed edges. Larger values reduce edge exposure but discard more of the frame. ARP deliberately avoids automatic zoom so the framing cannot pulse or unexpectedly crop far beyond this value.',
  'stabilize.shot_threshold':
    'Uses the same visual shot detector as Reference Generation. Lower values detect subtler cuts and dissolves; values that are too low can split energetic camera movement into false shots.',
  'stabilize.min_shot_seconds':
    'Prevents closely spaced detections from creating tiny stabilization spans. Tracking and camera smoothing restart at every accepted boundary.',
  'stabilize.encoder':
    'FFV1 is mathematically lossless and recommended for pipeline intermediates. ProRes HQ is larger and visually lossless, but more convenient in editing applications.',
  'outpaint.offset_x':
    'Shift the source horizontally for the whole video before outpainting. Positive values move it right; negative values move it left. Chunks inherit this unless overridden.',
  'outpaint.offset_y':
    'Shift the source vertically for the whole video before outpainting. Positive values move it down; negative values move it up. Chunks inherit this unless overridden.',
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
  'recomp.reference_luminance_match':
    'Derive one stable tonal curve for each shot by comparing its original black-and-white reference with its approved colour reference. This brings the moving result closer to the reference lighting without frame-by-frame exposure flicker.',
  'recomp.reference_luminance_strength':
    'Blend between the original film luminance and the colour-reference luminance. 70% retains most source shadow detail while giving the references a clear influence; 100% applies the complete bounded curve.',
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
  'upscale.blend_strength':
    'How much of the AI reconstruction is used by default. The remainder is a conventional resize of the source, which restores source-derived motion blur and reduces the stop-motion look. Each shot can override this.',
  'upscale.chunk_seconds':
    'The clip is upscaled in chunks of roughly this many seconds; each chunk restarts the model’s temporal stream. 0 sends the whole clip at once.',
  'upscale.overlap_frames':
    'Warm-up frames repeated before each chunk and trimmed afterwards. Raise to 16-24 if chunk starts look unstable or faces flip identity mid-scene.',
};

const OUTPAINT_FIELD_TOOLTIPS = {
  target_aspect:
    'Shape of the expanded canvas. The cropped source is fitted inside it and LTX generates the new pillarbox or letterbox regions.',
  target_height:
    'Requested delivery height. Source keeps the source height; numbered choices set a new height. LTX may render at a nearby model-safe multiple of 32 before Recomposition scales to the requested size.',
  offset_x:
    'Shift the fitted source horizontally before outpainting. Positive values move it right and create more room on the left; negative values move it left. Chunks inherit this unless overridden.',
  offset_y:
    'Shift the fitted source vertically before outpainting. Positive values move it down and create more room above; negative values move it up. Chunks inherit this unless overridden.',
  chunk_seconds:
    'Approximate duration sent to LTX in each job. Longer chunks improve continuity and create fewer joins but use more VRAM; 0 sends the whole clip at once.',
  overlap_frames:
    'Frames repeated between neighbouring LTX chunks to give each join temporal context. They are trimmed during stitching, so they do not lengthen the finished video.',
  generation_mask_overlap:
    'Extend the generation mask this many pixels beneath the protected source edge. A small overlap helps the mask survive LTX spatial compression; too much can make edge objects get regenerated, changed, or omitted. Default: 8.',
  mask_blend_dilation:
    'How far the Laplacian seam blend reaches into the protected source when the generated plate is assembled. Higher values soften a hard join but can create halos or ghosting. Default: 2.',
  seed_qwen_guides:
    'Generate a Qwen-outpainted guide frame at every detected shot change before LTX renders. Use this when LTX returns the original black bars. It is slower, but helps stubborn shots begin from an already-filled frame.',
  outpaint_all_black_regions:
    'Treat every near-black region as an outpaint target instead of protecting black pixels inside the source. Useful for mixed-size footage or changing bars, but it can also replace genuine shadows, silhouettes, or black objects.',
  black_mask_threshold:
    'Maximum pixel brightness treated as black when Outpaint all black regions is enabled. Raise it only when encoded bars are dark grey rather than true black; higher values risk selecting real picture detail.',
  prompt:
    'Instruction used for every LTX outpaint chunk. Keep the word "outpaint" in it so the IC-LoRA activates; add scene, period, lighting, or style guidance when the generated sides need direction.',
  negative_prompt:
    'Details and failure modes LTX should avoid in every chunk. Chunk-specific negative text is appended to this global list.',
  crop_left:
    'Permanently discard this many pixels from the source left edge before fitting it to the new canvas. Use it for baked-in bars or damaged borders; discarded picture content is not an outpaint target.',
  crop_right:
    'Permanently discard this many pixels from the source right edge before fitting it to the new canvas. Use it for baked-in bars or damaged borders; discarded picture content is not an outpaint target.',
  crop_top:
    'Permanently discard this many pixels from the source top edge before fitting it to the new canvas. Use it for baked-in bars or damaged borders; discarded picture content is not an outpaint target.',
  crop_bottom:
    'Permanently discard this many pixels from the source bottom edge before fitting it to the new canvas. Use it for baked-in bars or damaged borders; discarded picture content is not an outpaint target.',
};

function fieldDescription(stageKey, key) {
  return FIELD_DESCRIPTIONS[`${stageKey}.${key}`] || FIELD_DESCRIPTIONS[key] || '';
}

function fieldTooltip(stageKey, key) {
  return stageKey === 'outpaint' ? (OUTPAINT_FIELD_TOOLTIPS[key] || '') : '';
}

function tooltipTitle(tooltip) {
  return tooltip ? ` title="${esc(tooltip)}"` : '';
}

function fieldHelpHtml(help) {
  return help ? `<small class="field-help">${esc(help)}</small>` : '';
}

function fieldHtml(st, field) {
  const [key, label, kind, defaultValue] = field;
  const value = settings(st.key)[key] ?? defaultValue ?? '';
  const help = fieldDescription(st.key, key);
  const tooltip = fieldTooltip(st.key, key);
  const title = tooltipTitle(tooltip);

  if (kind.startsWith('select:')) return selectFieldHtml(key, label, kind, value, tooltip) + fieldHelpHtml(help);
  if (kind.startsWith('range:')) return rangeFieldHtml(key, label, kind, value, tooltip) + fieldHelpHtml(help);
  if (kind === 'checkbox') return checkboxFieldHtml(key, label, value, help, tooltip);

  const input = `
    <input data-field="${key}" data-kind="${kind}" type="${kind === 'number' ? 'number' : 'text'}" step="any" value="${esc(value)}"${title}>
  `;

  if (['file', 'folder', 'save'].includes(kind)) {
    return `
      <label${title}>${label}</label>
      <div class="field-row">
        ${input}
        <button type="button" onclick="browseField('${st.key}','${key}','${kind}')">Browse</button>
      </div>
      ${fieldHelpHtml(help)}
    `;
  }

  return `<label${title}>${label}</label>${input}${fieldHelpHtml(help)}`;
}

function selectFieldHtml(key, label, kind, value, tooltip = '') {
  const options = kind.slice(7).split('|')
    .map(option => `<option value="${esc(option)}" ${value === option ? 'selected' : ''}>${esc(selectOptionLabel(key, option))}</option>`)
    .join('');
  const title = tooltipTitle(tooltip);
  return `<label${title}>${label}</label><select data-field="${key}"${title}>${options}</select>`;
}

function selectOptionLabel(key, option) {
  if (key === 'repair_device' && option === 'auto') return 'Auto (prefer GPU)';
  if (key === 'repair_device' && option === 'cuda') return 'NVIDIA GPU (CUDA)';
  if (key === 'repair_device' && option === 'cpu') return 'CPU';
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
  chunk_seconds: ' s',
  feather_pixels: ' px',
  saturation: '%',
  temperature: ' K',
  color_opacity: '%',
};

const RANGE_NUDGE_FIELDS = new Set(['chunk_seconds', 'source_fidelity', 'crop_left', 'crop_right', 'crop_top', 'crop_bottom', 'feather_pixels', 'saturation', 'temperature', 'color_opacity']);

function rangeDisplayValue(key, value) {
  const number = Number(value);
  const shown = Number.isFinite(number) && Number.isInteger(number) ? String(number) : String(value);
  return shown + (RANGE_FIELD_UNITS[key] || '');
}

function rangeFieldHtml(key, label, kind, value, tooltip = '') {
  const [min, max, step] = kind.slice(6).split('|');
  const hasNudge = RANGE_NUDGE_FIELDS.has(key);
  const nudgeAmount = step || '1';
  const title = tooltipTitle(tooltip);
  const controls = hasNudge ? `
    <div class="pixel-nudge-row">
      <button type="button" onclick="nudgeRangeField('${key}',-1)">-${esc(nudgeAmount)}</button>
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
      <button type="button" onclick="nudgeRangeField('${key}',1)">+${esc(nudgeAmount)}</button>
    </div>
  ` : '';
  return `
    <label${title}>${label}: <span id="${key}Value">${esc(rangeDisplayValue(key, value))}</span></label>
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
      ${title}
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

function checkboxFieldHtml(key, label, value, help = '', tooltip = '') {
  const description = tooltip || CHECKBOX_DESCRIPTIONS[key];
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

  if (st.key === 'cleanup') return drawCleanup(st, s, expected, sp);
  if (st.key === 'stabilize') return drawStabilization(st, s, expected, sp);
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
  `;

  bindStageFields(st.key);
  showCommand(st.key);
}

function drawCleanup(st, s, expected, sp) {
  const comparison = state.cleanup_comparison || {};
  document.getElementById('app').innerHTML = `
    <div class="editor-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        ${progressHtml(sp.percent, sp.label)}
        ${cleanupLicenseWarning(s)}
        ${st.fields.map(f => fieldHtml(st, f)).join('')}
        ${shotOutputList(expected, null)}
        ${stageCheckboxes(s)}
        <div class="actions">
          <button class="primary" onclick="runStage('cleanup')" ${state.running ? 'disabled' : ''}>Run Clean Up</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card editor-viewer">
        <h2>${esc(comparison.title || 'Clean Up Comparison')}</h2>
        ${cleanupComparisonHtml(comparison)}
      </section>
    </div>
  `;

  bindStageFields('cleanup');
  bindCleanupComparison();
  showCommand('cleanup');
}

function drawStabilization(st, s, expected, sp) {
  const comparison = state.stabilization_comparison || {};
  document.getElementById('app').innerHTML = `
    <div class="editor-page">
      <section class="card">
        <h2>${st.title}</h2>
        <p>${st.description}</p>
        ${progressHtml(sp.percent, sp.label)}
        ${st.fields.map(f => fieldHtml(st, f)).join('')}
        ${shotOutputList(expected, null)}
        ${stageCheckboxes(s)}
        <div class="actions">
          <button class="primary" onclick="runStage('stabilize')" ${state.running ? 'disabled' : ''}>Run Stabilization</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card editor-viewer">
        <h2>${esc(comparison.title || 'Stabilization Comparison')}</h2>
        ${stabilizationComparisonHtml(comparison)}
      </section>
    </div>
  `;

  bindStageFields('stabilize');
  bindVideoComparison('stabilizationCompareSlider');
  showCommand('stabilize');
}

function stabilizationComparisonHtml(comparison) {
  const before = comparison.source || '';
  const after = comparison.exists === 'true' ? comparison.output : '';
  if (!before || comparison.source_exists !== 'true') {
    return '<p class="shot-empty">Choose source material on the Overview page, then run any enabled Clean Up phase.</p>';
  }
  if (!after) {
    return `
      <video src="${media(before)}" controls preload="metadata"></video>
      <p class="shot-empty">The synchronized before/after comparison will appear here when Stabilization finishes.</p>
    `;
  }
  return `
    <div class="comparison-player">
      <video class="compare-before" src="${media(before)}" controls preload="metadata"></video>
      <video class="compare-after" src="${media(after)}" muted preload="metadata"></video>
      <div class="compare-after-mask" style="width:50%"></div>
      <div class="compare-handle" style="left:50%"></div>
    </div>
    <input id="stabilizationCompareSlider" class="compare-slider" type="range" min="0" max="100" value="50" aria-label="Stabilization before after split">
    <div class="source-info">
      <div><span>Before master</span><strong>${esc(comparison.master_source || before)}</strong></div>
      <div><span>After master</span><strong>${esc(comparison.master_output || after)}</strong></div>
    </div>
  `;
}

function cleanupLicenseWarning(s) {
  if (s.ai_descratch !== 'true') return '';
  return `
    <div class="inline-warning">
      <strong>AI DeScratch is non-commercial only.</strong>
      ProPainter's code and models use the NTU S-Lab License 1.0. Do not use this option for paid,
      monetised, or other commercial work unless you have separate written permission from the authors.
      <a href="https://github.com/sczhou/ProPainter/blob/main/LICENSE" target="_blank" rel="noopener">View licence</a>.
    </div>
  `;
}

function cleanupComparisonHtml(comparison) {
  const before = comparison.source || '';
  const after = comparison.exists === 'true' ? comparison.output : '';
  if (!before || comparison.source_exists !== 'true') {
    return '<p class="shot-empty">Choose source material on the Overview page, then run Clean Up.</p>';
  }
  if (!after) {
    return `
      <video src="${media(before)}" controls preload="metadata"></video>
      <p class="shot-empty">The before/after comparison will appear here when Clean Up finishes.</p>
    `;
  }
  return `
    <div class="comparison-player">
      <video class="compare-before" src="${media(before)}" controls preload="metadata"></video>
      <video class="compare-after" src="${media(after)}" muted preload="metadata"></video>
      <div class="compare-after-mask" style="width:50%"></div>
      <div class="compare-handle" style="left:50%"></div>
    </div>
    <input id="cleanupCompareSlider" class="compare-slider" type="range" min="0" max="100" value="50" aria-label="Clean Up before after split">
    <div class="source-info">
      <div><span>Before</span><strong>${esc(before)}</strong></div>
      <div><span>After</span><strong>${esc(after)}</strong></div>
    </div>
  `;
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
