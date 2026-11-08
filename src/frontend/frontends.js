const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const CONFIG_PATH = path.join(ROOT, 'config.json');
const FRONTENDS_DIR = path.join(ROOT, 'src', 'frontend');
const DEFAULT_FRONTEND = 'default';
const VIEWERS_DIR = path.join(FRONTENDS_DIR, DEFAULT_FRONTEND, 'viewers');

const MEDIA = {
  '3d': {
    label: '3D models (.glb files)',
    python: ['pip install -r requirements-3d.txt', 'python -m playwright install chromium'],
  },
  pdf: {
    label: 'PDF documents',
    python: ['pip install -r requirements-pdf.txt'],
  },
  zoom: {
    label: 'maps and other large images served by IIIF image servers',
    python: [],
  },
};

const LEGACY_TYPES = { photographs: [], maps: ['zoom'], documents: ['pdf'] };

const readConfig = () => (fs.existsSync(CONFIG_PATH) ? JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) : {});

const writeConfig = (config) => fs.writeFileSync(CONFIG_PATH, `${JSON.stringify(config, null, 2)}\n`);

const choiceOf = (config) => ({
  frontend: config.frontend || DEFAULT_FRONTEND,
  media: config.media || LEGACY_TYPES[config.collection_type] || [],
});

const isPath = (frontend) => path.isAbsolute(frontend) || frontend.startsWith('.') || /[\\/]/.test(frontend);

const frontendDir = (frontend) => (
  isPath(frontend) ? path.resolve(ROOT, frontend) : path.join(FRONTENDS_DIR, frontend)
);

const viewerDir = (medium) => path.join(VIEWERS_DIR, medium);

const dependenciesOf = (dir) => {
  const manifest = path.join(dir, 'package.json');
  if (!fs.existsSync(manifest)) {
    return [];
  }
  return Object.keys(JSON.parse(fs.readFileSync(manifest, 'utf8')).dependencies || {});
};

const missingPackages = (dir) => dependenciesOf(dir).filter(
  (name) => !fs.existsSync(path.join(dir, 'node_modules', name, 'package.json')),
);

module.exports = {
  CONFIG_PATH,
  DEFAULT_FRONTEND,
  LEGACY_TYPES,
  MEDIA,
  ROOT,
  choiceOf,
  dependenciesOf,
  frontendDir,
  missingPackages,
  readConfig,
  viewerDir,
  writeConfig,
};
