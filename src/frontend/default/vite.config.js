import { defineConfig, mergeConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import { createRequire } from 'module';
import path from 'path';
import { pathToFileURL } from 'url';

const require = createRequire(import.meta.url);
const { choiceOf, missingPackages, readConfig, viewerDir } = require('../frontends.js');

const VIEWERS_MODULE = 'virtual:dce-viewers';
const BACKEND_PATHS = ['/api', '/images', '/static', '/models', '/documents'];

const config = readConfig();
const api = config.api_config || {};
const apiHost = !api.host || api.host === '0.0.0.0' ? 'localhost' : api.host;
const apiOrigin = `http://${apiHost}:${api.port || 8000}`;

const media = [...new Set(
  (process.env.DCE_MEDIA !== undefined ? process.env.DCE_MEDIA.split(',') : choiceOf(config).media)
    .map((name) => name.trim())
    .filter(Boolean),
)];

const viewers = media.filter((name) => fs.existsSync(path.join(viewerDir(name), 'index.js')));

const viewerModules = () => ({
  name: 'dce-viewers',
  resolveId(id) {
    return id === VIEWERS_MODULE ? `\0${VIEWERS_MODULE}` : null;
  },
  load(id) {
    if (id !== `\0${VIEWERS_MODULE}`) {
      return null;
    }
    const imports = viewers.map((name, index) => {
      const file = path.join(viewerDir(name), 'index.js').split(path.sep).join('/');
      return `import viewer${index} from ${JSON.stringify(file)};`;
    });
    const entries = viewers.map((name, index) => `${JSON.stringify(name)}: viewer${index}`);
    return `${imports.join('\n')}\nexport default { ${entries.join(', ')} };\n`;
  },
  generateBundle() {
    this.emitFile({
      type: 'asset',
      fileName: 'frontend.json',
      source: `${JSON.stringify({ frontend: 'default', media }, null, 2)}\n`,
    });
  },
});

const viewerConfig = async (name) => {
  const file = path.join(viewerDir(name), 'vite.js');
  return fs.existsSync(file) ? (await import(pathToFileURL(file).href)).default : {};
};

export default defineConfig(async ({ command }) => {
  const unready = viewers.filter((name) => missingPackages(viewerDir(name)).length);
  if (unready.length) {
    throw new Error(
      `The packages of the ${unready.join(', ')} viewer are not installed. Run: npm run setup -- --with=${media.join(',')}`,
    );
  }

  const base = {
    base: './',
    plugins: [react(), viewerModules()],
    server: {
      proxy: Object.fromEntries(BACKEND_PATHS.map((prefix) => [prefix, { target: apiOrigin, changeOrigin: true }])),
    },
    define: {
      'import.meta.env.API_BASE_URL': JSON.stringify(command === 'build' ? '' : apiOrigin),
    },
  };
  const extras = await Promise.all(viewers.map(viewerConfig));
  return extras.reduce((merged, extra) => mergeConfig(merged, extra), base);
});
