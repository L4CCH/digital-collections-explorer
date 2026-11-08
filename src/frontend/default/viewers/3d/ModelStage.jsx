import React, { useEffect, useRef, useState } from 'react';
import { loadViewer } from './viewer.js';
import './ModelStage.css';

function ModelStage({ src, orbit, poster, alt }) {
  const element = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [shownSrc, setShownSrc] = useState(null);
  const [failedSrc, setFailedSrc] = useState(null);

  useEffect(() => {
    let isCurrent = true;
    loadViewer()
      .then(() => isCurrent && setIsReady(true))
      .catch(() => isCurrent && setFailedSrc(src));
    return () => {
      isCurrent = false;
    };
  }, [src]);

  useEffect(() => {
    const viewer = element.current;
    if (!isReady || !viewer) {
      return undefined;
    }
    const shown = () => setShownSrc(src);
    const failed = () => setFailedSrc(src);
    viewer.addEventListener('load', shown);
    viewer.addEventListener('error', failed);
    if (viewer.loaded) {
      shown();
    }
    return () => {
      viewer.removeEventListener('load', shown);
      viewer.removeEventListener('error', failed);
    };
  }, [isReady, src]);

  const isShown = shownSrc === src;
  const hasFailed = !isShown && failedSrc === src;

  return (
    <div className="model-stage">
      {isReady ? (
        <model-viewer
          ref={element}
          class="model-stage-viewer"
          src={src}
          poster={poster}
          alt={alt}
          camera-orbit={orbit}
          camera-controls=""
          touch-action="pan-y"
          interaction-prompt="none"
          shadow-intensity="0"
        />
      ) : (
        poster && <img className="model-stage-poster" src={poster} alt={alt} />
      )}
      <span
        className={`model-stage-status${isShown ? ' is-done' : ''}${hasFailed ? ' has-failed' : ''}`}
        role="status"
        aria-label={hasFailed ? 'The 3D model could not be loaded' : 'Loading the 3D model'}
        aria-hidden={isShown || undefined}
        title={hasFailed ? 'The 3D model could not be loaded' : undefined}
      >
        <span className="model-stage-badge">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" />
            <path d="M4 7.5l8 4.5 8-4.5M12 12v9" />
            {hasFailed && <path d="M3 21L21 3" />}
          </svg>
        </span>
      </span>
    </div>
  );
}

export default ModelStage;
