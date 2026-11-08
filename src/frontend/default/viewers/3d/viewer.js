const decoders = (name) => new URL(`decoders/${name}/`, document.baseURI).href;

let viewerLoading = null;

export const loadViewer = () => {
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
