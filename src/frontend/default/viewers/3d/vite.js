import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const THREE_LIBS = path.join(path.dirname(fileURLToPath(import.meta.url)), 'node_modules/three/examples/jsm/libs');
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

export default {
  plugins: [modelDecoders()],
  build: {
    // The 3D viewer is about 1 MB and loads only on the page of an object with a model.
    chunkSizeWarningLimit: 1100,
  },
};
