import React, { useEffect, useState } from 'react';
import './ModelStage.css';

const decoders = (name) => new URL(`decoders/${name}/`, document.baseURI).href;

let viewerLoading = null;

const loadViewer = () => {
  if (!viewerLoading) {
    self.ModelViewerElement = {
      ...(self.ModelViewerElement || {}),
      dracoDecoderLocation: decoders('draco'),
      ktx2TranscoderLocation: decoders('basis'),
    };
    viewerLoading = import('@google/model-viewer');
  }
  return viewerLoading;
};

function ModelStage({ src, orbit, poster, alt }) {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    let isCurrent = true;
    loadViewer()
      .then(() => isCurrent && setIsReady(true))
      .catch(() => {});
    return () => {
      isCurrent = false;
    };
  }, []);

  if (!isReady) {
    return poster ? <img className="model-stage-poster" src={poster} alt={alt} /> : null;
  }

  return (
    <model-viewer
      class="model-stage"
      src={src}
      poster={poster}
      alt={alt}
      camera-orbit={orbit}
      camera-controls=""
      touch-action="pan-y"
      interaction-prompt="none"
      shadow-intensity="0"
    />
  );
}

export default ModelStage;
