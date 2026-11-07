import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let apiHost;
let apiPort;

try {
  const configPath = path.resolve(__dirname, '../../..', 'config.json');
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  
  if (config.api_config) {
    apiHost = config.api_config.host === '0.0.0.0' ? 'localhost' : config.api_config.host;
    apiPort = config.api_config.port;
  }
} catch (error) {
  console.warn('Could not read config.json, using default API settings:', error);
}

const THREE_LIBS = path.resolve(__dirname, 'node_modules/three/examples/jsm/libs');
const DECODERS = {
  'decoders/draco/draco_decoder.js': 'draco/draco_decoder.js',
  'decoders/draco/draco_decoder.wasm': 'draco/draco_decoder.wasm',
  'decoders/draco/draco_wasm_wrapper.js': 'draco/draco_wasm_wrapper.js',
  'decoders/basis/basis_transcoder.js': 'basis/basis_transcoder.js',
  'decoders/basis/basis_transcoder.wasm': 'basis/basis_transcoder.wasm',
};

const modelDecoders = () => ({
  name: 'model-decoders',
  configureServer(server) {
    server.middlewares.use((request, response, next) => {
      const name = decodeURIComponent((request.url || '').split('?')[0]).replace(/^\//, '');
      if (!DECODERS[name]) {
        next();
        return;
      }
      response.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
      response.end(fs.readFileSync(path.join(THREE_LIBS, DECODERS[name])));
    });
  },
  generateBundle() {
    Object.entries(DECODERS).forEach(([fileName, source]) => {
      this.emitFile({ type: 'asset', fileName, source: fs.readFileSync(path.join(THREE_LIBS, source)) });
    });
  },
});

// https://vite.dev/config/
export default defineConfig(({ command }) => {
  const isProduction = command === 'build';
  
  return {
    base: './',
    plugins: [react(), modelDecoders()],
    build: {
      // The 3D viewer is about 1 MB and loads only on the page of an object with a model.
      chunkSizeWarningLimit: 1100,
    },
    server: {
      proxy: {
        '/api': {
          target: `http://${apiHost}:${apiPort}`,
          changeOrigin: true,
        },
        '/images': {
          target: `http://${apiHost}:${apiPort}`,
          changeOrigin: true,
        },
        '/static': {
          target: `http://${apiHost}:${apiPort}`,
          changeOrigin: true,
        },
        '/models': {
          target: `http://${apiHost}:${apiPort}`,
          changeOrigin: true,
        }
      }
    },
    define: {
      'import.meta.env.API_BASE_URL': isProduction 
        ? JSON.stringify('') 
        : JSON.stringify(`http://${apiHost}:${apiPort}`)
    }
  }
});
