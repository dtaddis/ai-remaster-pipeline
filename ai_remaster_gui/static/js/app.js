function draw(followLogs = false) {
  if (active === 'global') return drawGlobal(followLogs);
  if (active === 'settings') return drawSettings();
  if (active === 'cache') return drawCache();
  if (active === 'output') return drawOutput();
  if (active === 'shots') return drawShots(followLogs);
  if (active === 'references') return drawReferences(followLogs);
  if (active === 'colour') return drawColour(followLogs);
  if (active === 'recomp') return drawRecomp(followLogs);
  if (active === 'upscale') return drawUpscale(followLogs);

  return drawStage(stage(active), followLogs);
}

setInterval(refresh, 4000);
refresh();
