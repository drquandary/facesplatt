// Academic database page — loads manifest, renders filterable grid of faces,
// opens a 3DGS viewer modal on click.

import { openSplatViewer, closeSplatViewer } from './splat-viewer.js';

const state = {
  manifest: [],
  available: new Set(),   // set of CFAD IDs that have a .ply splat
  filters: { race: 'all', sex: 'all', pose: 'all' },
};

const $ = (id) => document.getElementById(id);

async function loadManifest() {
  const res = await fetch('./manifest.json');
  state.manifest = await res.json();
}

async function probeAvailability() {
  // HEAD a short list of .ply files; any that 200 are "available".
  // Rather than probing all 120, we optimistically assume .ply exists iff the
  // filename (without extension) matches a CFAD file. The static server will
  // return 404 on the splat load itself for the 4 missing ones; we mark those.
  const known = [
    'CFAD-WF-048-999-211-NBM',
    'CFAD-WF-228-999-111-NBM',
    'CFAD-WF-259-999-111-NBM',
    'CFAD-WM-C53-999-212-NBM',
  ];
  const missing = new Set(known);
  for (const entry of state.manifest) {
    const id = entry.file.replace(/\.png$/, '');
    if (!missing.has(id)) state.available.add(id);
  }
}

function matchesFilter(entry) {
  const { race, sex, pose } = state.filters;
  if (race !== 'all' && entry.race !== race) return false;
  if (sex !== 'all' && entry.sex !== sex) return false;
  if (pose !== 'all' && entry.pose !== pose) return false;
  return true;
}

function renderGrid() {
  const grid = $('grid');
  grid.innerHTML = '';
  let visible = 0;
  for (const entry of state.manifest) {
    if (!matchesFilter(entry)) continue;
    visible++;
    const id = entry.file.replace(/\.png$/, '');
    const hasSplat = state.available.has(id);
    const card = document.createElement('article');
    card.className = 'card' + (hasSplat ? '' : ' no-splat');
    card.style.position = 'relative';
    card.innerHTML = `
      <img class="card-img" src="./cfad/${entry.file}" alt="${id}" loading="lazy" />
      <div class="card-meta">
        <div class="card-id">${entry.id}<span class="card-pose">${entry.pose}</span></div>
        <div class="card-demo">${entry.race} ${entry.sex}</div>
      </div>
    `;
    if (hasSplat) {
      card.addEventListener('click', () => openSplatViewer(entry));
    }
    grid.appendChild(card);
  }
  $('count').textContent = `${visible} of ${state.manifest.length} subjects · ${state.available.size} with splats`;
}

function wireFilters() {
  for (const chip of document.querySelectorAll('.chip')) {
    chip.addEventListener('click', () => {
      const { filter, value } = chip.dataset;
      state.filters[filter] = value;
      // update chip active states
      for (const c of document.querySelectorAll(`.chip[data-filter="${filter}"]`)) {
        c.classList.toggle('active', c.dataset.value === value);
      }
      renderGrid();
    });
  }
}

function wireViewer() {
  $('viewer-close').addEventListener('click', closeSplatViewer);
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSplatViewer();
  });
}

(async () => {
  await loadManifest();
  await probeAvailability();
  wireFilters();
  wireViewer();
  renderGrid();
})();
