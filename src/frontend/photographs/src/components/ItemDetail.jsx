import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Lightbox from 'yet-another-react-lightbox';
import Zoom from 'yet-another-react-lightbox/plugins/zoom';
import 'yet-another-react-lightbox/styles.css';
import CatalogText from './CatalogText';
import CubeIcon from './CubeIcon';
import ImageGrid from './ImageGrid';
import ModelStage from './ModelStage';
import RelatedSources from './RelatedSources';
import { itemHref } from '../hooks/useHashRoute';
import {
  getItem, getSimilar, getSimilarShape, imageUrl, modelUrl,
} from '../services/api';
import { plainText } from '../services/catalogText';
import { titleOf } from '../services/items';
import './ItemDetail.css';

const SIMILAR_COUNT = 24;
// Fields longer than this many characters open folded, so the rest of the record stays in view.
const FOLDED_LENGTH = 600;

const fullImage = (id) => imageUrl(id, 'full');
const thumbnail = (id) => imageUrl(id, 'thumbnail');

const hostnameOf = (url) => {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return 'the source';
  }
};

const Chevron = ({ direction }) => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d={direction === 'left' ? 'M15 5l-7 7 7 7' : 'M9 5l7 7-7 7'} />
  </svg>
);

const ExternalLinkIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M14 5h5v5M19 5l-8 8M11 7H6a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-5" />
  </svg>
);

function ItemDetail({ id, collection, canGoBack }) {
  const [object, setObject] = useState(null);
  const [currentId, setCurrentId] = useState(id);
  const [similar, setSimilar] = useState([]);
  const [similarShapes, setSimilarShapes] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [zoomIsOpen, setZoomIsOpen] = useState(false);
  const [unfolded, setUnfolded] = useState([]);
  const [showModel, setShowModel] = useState(false);

  const photos = useMemo(() => object?.photos || [], [object]);
  const model = object?.model;
  const viewOrbits = useMemo(
    () => new Map((object?.model?.views || []).map((view) => [view.id, view.orbit])),
    [object],
  );
  const stills = useMemo(
    () => photos.filter((photo) => !viewOrbits.has(photo.id)),
    [photos, viewOrbits],
  );
  const entries = useMemo(
    () => (model ? [{ id: null, isModel: true }, ...stills] : photos),
    [model, stills, photos],
  );
  const recordFields = useMemo(() => {
    const byLabel = new Map();
    (object?.record?.fields || []).forEach(({ label, value }) => {
      byLabel.set(label, [...(byLabel.get(label) || []), value]);
    });
    return [...byLabel.entries()];
  }, [object]);
  const groupLabels = useMemo(
    () => Object.fromEntries((collection?.groups || []).map((group) => [group.id, group.label])),
    [collection],
  );
  const current = showModel
    ? photos.find((photo) => photo.id === currentId) || photos[0]
    : stills.find((photo) => photo.id === currentId) || stills[0] || photos[0];
  const currentIndex = showModel
    ? 0
    : Math.max(0, entries.findIndex((entry) => entry.id === current?.id));
  const stillIndex = Math.max(0, stills.findIndex((photo) => photo.id === current?.id));
  const viewId = viewOrbits.has(currentId) ? currentId : model?.views?.[0]?.id;

  useEffect(() => {
    let isCurrent = true;
    setIsLoading(true);
    setError(null);

    getItem(id)
      .then((loaded) => {
        if (isCurrent) {
          const views = new Set((loaded.model?.views || []).map((view) => view.id));
          const hasStills = (loaded.photos || []).some((photo) => !views.has(photo.id));
          setObject(loaded);
          setCurrentId(id);
          setShowModel(Boolean(loaded.model) && (views.has(id) || !hasStills));
          setUnfolded([]);
        }
      })
      .catch(() => isCurrent && setError('This item could not be found.'))
      .finally(() => isCurrent && setIsLoading(false));

    return () => {
      isCurrent = false;
    };
  }, [id]);

  useEffect(() => {
    let isCurrent = true;
    getSimilar(currentId, SIMILAR_COUNT)
      .then((loaded) => isCurrent && setSimilar(loaded))
      .catch(() => {});
    return () => {
      isCurrent = false;
    };
  }, [currentId]);

  useEffect(() => {
    let isCurrent = true;
    setSimilarShapes([]);
    if (model) {
      getSimilarShape(id, SIMILAR_COUNT)
        .then((loaded) => isCurrent && setSimilarShapes(loaded))
        .catch(() => {});
    }
    return () => {
      isCurrent = false;
    };
  }, [id, model]);

  useEffect(() => {
    [entries[currentIndex - 1], entries[currentIndex + 1]]
      .filter((entry) => entry && !entry.isModel)
      .forEach((photo) => {
        new Image().src = fullImage(photo.id);
      });
  }, [entries, currentIndex]);

  const showPhoto = useCallback((photoId) => {
    setShowModel(false);
    setCurrentId(photoId);
    window.history.replaceState(window.history.state, '', itemHref(photoId));
  }, []);

  const showTheModel = useCallback(() => {
    setShowModel(true);
    if (viewId) {
      setCurrentId(viewId);
      window.history.replaceState(window.history.state, '', itemHref(viewId));
    }
  }, [viewId]);

  const step = useCallback((offset) => {
    const next = entries[currentIndex + offset];
    if (next?.isModel) {
      showTheModel();
    } else if (next) {
      showPhoto(next.id);
    }
  }, [entries, currentIndex, showPhoto, showTheModel]);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (zoomIsOpen || e.target.matches('input, textarea') || e.target.closest?.('model-viewer')) {
        return;
      }
      if (e.key === 'ArrowLeft') {
        step(-1);
      } else if (e.key === 'ArrowRight') {
        step(1);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [step, zoomIsOpen]);

  if (error) {
    return (
      <div className="error-message">
        <p>{error}</p>
        <p><a href="#/">Back to the collection</a></p>
      </div>
    );
  }

  if (!current) {
    return (
      <div className="loading-indicator">
        <div className="spinner"></div>
      </div>
    );
  }

  const { metadata } = current;
  const title = titleOf(current);
  const name = plainText(title);
  const sourceUrl = metadata.source_url;
  const sourceName = collection?.source_name || (sourceUrl && hostnameOf(sourceUrl));
  const license = collection?.license;
  const groupLabel = groupLabels[metadata.group];
  const hasSeveralEntries = entries.length > 1;
  const viewMetadata = photos.find((photo) => photo.id === viewId)?.metadata || {};
  const shown = showModel ? viewMetadata : metadata;
  const aspectRatio = shown.width && shown.height ? shown.width / shown.height : undefined;

  return (
    <article className={`item-detail ${isLoading ? 'item-detail-loading' : ''}`}>
      <nav className="item-detail-nav">
        {canGoBack ? (
          <>
            <button type="button" className="link-button" onClick={() => window.history.back()}>
              ← Back
            </button>
            <a href="#/">Start over</a>
          </>
        ) : (
          <a href="#/">← Browse the collection</a>
        )}
      </nav>

      <div className="item-detail-main">
        <div className="item-detail-viewer">
          <div className="item-detail-stage" style={{ '--ratio': aspectRatio }}>
            {showModel ? (
              <ModelStage
                src={model.url || modelUrl(model.id)}
                orbit={viewOrbits.get(viewId)}
                poster={viewId ? fullImage(viewId) : undefined}
                alt={name ? `3D model of ${name}` : '3D model'}
              />
            ) : (
              <button
                type="button"
                className="item-detail-image"
                onClick={() => setZoomIsOpen(true)}
                aria-label="View larger"
              >
                <img src={fullImage(current.id)} alt={name || 'Uncataloged image'} />
              </button>
            )}
            {hasSeveralEntries && (
              <>
                <button
                  type="button"
                  className="item-detail-step item-detail-step-previous"
                  onClick={() => step(-1)}
                  disabled={currentIndex === 0}
                  aria-label="Previous photo"
                >
                  <Chevron direction="left" />
                </button>
                <button
                  type="button"
                  className="item-detail-step item-detail-step-next"
                  onClick={() => step(1)}
                  disabled={currentIndex === entries.length - 1}
                  aria-label="Next photo"
                >
                  <Chevron direction="right" />
                </button>
                <span className="item-detail-count" aria-hidden="true">
                  {currentIndex + 1} / {entries.length}
                </span>
              </>
            )}
          </div>

          {hasSeveralEntries && (
            <div className="item-detail-photos">
              <ul>
                {entries.map((entry, index) => {
                  const isShown = entry.isModel ? showModel : !showModel && entry.id === current.id;
                  const href = itemHref(entry.isModel ? viewId || model.id : entry.id);
                  const photoNumber = index + (model ? 0 : 1);
                  const classes = [isShown && 'is-current', entry.isModel && 'item-detail-model-tile'];
                  return (
                    <li key={entry.id || 'model'}>
                      <a
                        href={href}
                        className={classes.filter(Boolean).join(' ') || undefined}
                        aria-current={isShown ? 'true' : undefined}
                        aria-label={entry.isModel ? '3D model' : `Photo ${photoNumber} of ${stills.length}`}
                        onClick={(e) => {
                          e.preventDefault();
                          if (entry.isModel) {
                            showTheModel();
                          } else {
                            showPhoto(entry.id);
                          }
                        }}
                      >
                        {entry.isModel ? (
                          <>
                            {viewId && <img src={thumbnail(viewId)} alt="" loading="lazy" />}
                            <span className="item-detail-model-mark"><CubeIcon />3D</span>
                          </>
                        ) : (
                          <img src={thumbnail(entry.id)} alt="" loading="lazy" />
                        )}
                      </a>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>

        <div className="item-detail-facts">
          <h2>{title ? <CatalogText text={title} /> : 'Uncataloged image'}</h2>
          {groupLabel && <p className="item-detail-group">{groupLabel}</p>}

          {!title && (
            <p className="item-detail-note">
              This image has no catalog record. It can be found here because the search
              looks at the picture itself, not at its description.
            </p>
          )}

          {sourceUrl && (
            <a className="item-detail-source" href={sourceUrl} target="_blank" rel="noopener noreferrer">
              <span>View the full record at {sourceName}</span>
              <ExternalLinkIcon />
            </a>
          )}

          {recordFields.length > 0 && (
            <dl className="item-detail-record">
              {recordFields.map(([label, values]) => {
                const isLong = values.join(' ').length > FOLDED_LENGTH;
                const isFolded = isLong && !unfolded.includes(label);
                return (
                  <div key={label}>
                    <dt>{label}</dt>
                    {(isFolded ? values.slice(0, 1) : values).map((value, index) => (
                      <dd key={`${label}-${index}`} className={isFolded ? 'is-folded' : undefined}>
                        <CatalogText text={value} links />
                      </dd>
                    ))}
                    {isLong && (
                      <dd>
                        <button
                          type="button"
                          className="link-button"
                          onClick={() => setUnfolded(isFolded
                            ? [...unfolded, label]
                            : unfolded.filter((other) => other !== label))}
                        >
                          {isFolded ? 'Show more' : 'Show less'}
                        </button>
                      </dd>
                    )}
                  </div>
                );
              })}
            </dl>
          )}

          <dl>
            {current.object_id !== current.id && (
              <>
                <dt>Object</dt>
                <dd>{current.object_id}</dd>
              </>
            )}
            {!showModel && metadata.width && metadata.height && (
              <>
                <dt>Original size</dt>
                <dd>{metadata.width.toLocaleString()} × {metadata.height.toLocaleString()} px</dd>
              </>
            )}
            <dt>File</dt>
            <dd>{showModel ? model.file_name : metadata.file_name}</dd>
          </dl>

          {license && (
            <p className="item-detail-license">
              {license.url ? (
                <a href={license.url} target="_blank" rel="noopener noreferrer">{license.name}</a>
              ) : license.name}
              {license.note ? ` — ${license.note}` : ''}
            </p>
          )}

          <RelatedSources related={object?.record?.related} sources={collection?.sources} />
        </div>
      </div>

      {similar.length > 0 && (
        <section>
          <h3>Looks similar</h3>
          <ImageGrid items={similar} groupLabels={groupLabels} />
        </section>
      )}

      {similarShapes.length > 0 && (
        <section>
          <h3>Similar in shape</h3>
          <ImageGrid items={similarShapes} groupLabels={groupLabels} />
        </section>
      )}

      <Lightbox
        open={zoomIsOpen}
        close={() => setZoomIsOpen(false)}
        index={stillIndex}
        slides={stills.map((photo) => ({ src: fullImage(photo.id), alt: name || '' }))}
        on={{ view: ({ index }) => stills[index] && showPhoto(stills[index].id) }}
        plugins={[Zoom]}
        carousel={{ finite: true }}
      />
    </article>
  );
}

export default ItemDetail;
