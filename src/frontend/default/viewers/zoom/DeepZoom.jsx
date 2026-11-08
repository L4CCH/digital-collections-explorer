import React, { useEffect, useRef } from 'react';
import './DeepZoom.css';

let libraryLoading = null;

const loadLibrary = () => {
  libraryLoading = libraryLoading || import('openseadragon').then((module) => module.default);
  return libraryLoading;
};

function DeepZoom({ info, alt, onClose, onFail }) {
  const stage = useRef(null);
  const closeButton = useRef(null);

  useEffect(() => {
    let isCurrent = true;
    let viewer = null;
    loadLibrary()
      .then((OpenSeadragon) => {
        if (!isCurrent) {
          return;
        }
        viewer = OpenSeadragon({
          element: stage.current,
          tileSources: info,
          showNavigationControl: false,
          showNavigator: true,
          navigatorPosition: 'BOTTOM_RIGHT',
          visibilityRatio: 0.5,
          crossOriginPolicy: 'Anonymous',
        });
        viewer.addHandler('open-failed', () => isCurrent && onFail());
      })
      .catch(() => isCurrent && onFail());
    return () => {
      isCurrent = false;
      viewer?.destroy();
    };
  }, [info, onFail]);

  useEffect(() => {
    closeButton.current?.focus();
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  return (
    <div className="deep-zoom" role="dialog" aria-modal="true" aria-label={alt}>
      <div className="deep-zoom-stage" ref={stage} />
      <button type="button" className="deep-zoom-close" onClick={onClose} aria-label="Close" ref={closeButton}>
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </button>
    </div>
  );
}

export default DeepZoom;
