function drawCache() {
  const cache = state.cache || { categories: [], count: 0, total_label: '0 B' };

  document.getElementById('app').innerHTML = `
    <section class="card cache-page">
      <div class="cache-header">
        <div>
          <h2>Cached Files</h2>
          <p class="shot-empty">Generated previews, chunks, intermediate renders, references, and manifests stored by ARP.</p>
        </div>
        <div class="cache-summary">
          <strong>${esc(cache.total_label)}</strong>
          <span>${cache.count || 0} files</span>
          <button class="warn" type="button" onclick="clearAllCache()" ${(cache.count || 0) ? '' : 'disabled'}>Clear Cache</button>
        </div>
      </div>
      <div class="cache-categories">
        ${(cache.categories || []).map(cacheCategoryHtml).join('')}
      </div>
    </section>
  `;
}

function cacheCategoryHtml(category) {
  const files = category.files || [];
  return `
    <section class="cache-category">
      <div class="cache-category-head">
        <div>
          <h3>${esc(category.title)}</h3>
          <p>${esc(category.description || '')}</p>
        </div>
        <div class="cache-category-total">
          <strong>${esc(category.total_label)}</strong>
          <span>${category.count || 0} files</span>
          <button class="icon-button inline" type="button" title="Clear ${esc(category.title)}" onclick='clearCacheCategory(${jsArg(category.key)},${jsArg(category.title)})' ${files.length ? '' : 'disabled'}>&#128465;</button>
        </div>
      </div>
      ${files.length ? cacheFileTable(category) : '<p class="shot-empty">No cached files in this category.</p>'}
    </section>
  `;
}

function cacheFileTable(category) {
  const rows = (category.files || []).map(file => `
    <tr>
      <td class="cache-path">${esc(file.path)}</td>
      <td>${esc(file.size_label)}</td>
      <td>
        <button class="icon-button inline" type="button" title="Delete file" onclick='deleteCacheFile(${jsArg(file.path)})'>&#128465;</button>
      </td>
    </tr>
  `).join('');

  return `
    <table class="cache-table">
      <tr><th>File</th><th>Size</th><th></th></tr>
      ${rows}
    </table>
  `;
}
