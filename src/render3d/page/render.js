import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const FIELD_OF_VIEW = 30;
// Leaves a little room around the model's bounding sphere.
const FRAMING_MARGIN = 1.04;
const JPEG_QUALITY = 0.92;
// Space kept around the object when a view is cropped to it, as a share of its size.
const CROP_MARGIN = 0.05;
// Blur of the room the color views are lit by, as RoomEnvironment's examples use.
const ROOM_SIGMA = 0.04;
const CLAY_COLOR = 0xd2d0cc;
const CLAY_ROUGHNESS = 0.85;
const RAKING_INTENSITY = 2.6;
const FILL_INTENSITY = 0.45;
// Up and to the left of the camera, low over the surface, as a raking light falls.
const RAKING_DIRECTION = [-1, 0.75, 0.3];

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setClearColor(0x000000, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const room = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), ROOM_SIGMA).texture;
const camera = new THREE.PerspectiveCamera(FIELD_OF_VIEW, 1, 0.01, 1000);
const raking = new THREE.DirectionalLight(0xffffff, RAKING_INTENSITY);
raking.position.set(...RAKING_DIRECTION);
camera.add(raking);
camera.add(raking.target);
raking.target.position.set(0, 0, -1);
const fill = new THREE.HemisphereLight(0xffffff, 0x3a3a3a, FILL_INTENSITY);
camera.add(fill);
scene.add(camera);

const loader = new GLTFLoader()
  .setDRACOLoader(new DRACOLoader().setDecoderPath('/three/examples/jsm/libs/draco/'))
  .setKTX2Loader(new KTX2Loader().setTranscoderPath('/three/examples/jsm/libs/basis/').detectSupport(renderer))
  .setMeshoptDecoder(MeshoptDecoder);

let current = null;
let loads = 0;

const asList = (material) => (Array.isArray(material) ? material : [material]);

const clay = (material) => new THREE.MeshStandardMaterial({
  color: CLAY_COLOR,
  roughness: CLAY_ROUGHNESS,
  metalness: 0,
  normalMap: material.normalMap || null,
  normalScale: material.normalScale ? material.normalScale.clone() : new THREE.Vector2(1, 1),
  side: material.side,
  alphaMap: material.alphaMap || null,
  alphaTest: material.alphaTest,
});

const dispose = (root) => {
  root.traverse((node) => {
    node.geometry?.dispose();
    asList(node.material || []).forEach((material) => {
      Object.values(material).forEach((value) => value?.isTexture && value.dispose());
      material.dispose();
    });
  });
};

const forget = () => {
  if (!current) {
    return;
  }
  scene.remove(current.root);
  current.relief.forEach((material) => asList(material).forEach((m) => m.dispose()));
  dispose(current.root);
  current = null;
};

window.dceLoad = async (url) => {
  loads += 1;
  const load = loads;
  forget();
  const gltf = await loader.loadAsync(url);
  const root = gltf.scene || gltf.scenes[0];
  if (load !== loads) {
    dispose(root);
    throw new Error('another model was asked for meanwhile');
  }
  scene.add(root);
  root.updateMatrixWorld(true);

  const original = new Map();
  const relief = new Map();
  let triangles = 0;
  let textured = false;
  root.traverse((node) => {
    if (!node.isMesh) {
      return;
    }
    const { geometry } = node;
    if (!geometry.attributes.normal) {
      geometry.computeVertexNormals();
    }
    triangles += (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3;
    textured = textured || asList(node.material).some((material) => material.map);
    original.set(node, node.material);
    relief.set(node, Array.isArray(node.material) ? node.material.map(clay) : clay(node.material));
  });

  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) {
    throw new Error('the model has nothing to show');
  }
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  current = { root, original, relief, center: sphere.center, radius: sphere.radius };
  return {
    meshes: original.size,
    triangles: Math.round(triangles),
    textured,
    size: box.getSize(new THREE.Vector3()).toArray(),
  };
};

const silhouette = (size) => {
  const gl = renderer.getContext();
  const pixels = new Uint8Array(size * size * 4);
  gl.readPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  let covered = 0;
  let left = size;
  let right = -1;
  let low = size;
  let high = -1;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (pixels[(y * size + x) * 4 + 3] > 0) {
        covered += 1;
        left = Math.min(left, x);
        right = Math.max(right, x);
        low = Math.min(low, y);
        high = Math.max(high, y);
      }
    }
  }
  return { covered, left, right, top: size - 1 - high, bottom: size - 1 - low };
};

const cropped = (box, size, background) => {
  const margin = Math.round(Math.max(box.right - box.left, box.bottom - box.top) * CROP_MARGIN);
  const x = Math.max(0, box.left - margin);
  const y = Math.max(0, box.top - margin);
  const width = Math.min(size, box.right + margin + 1) - x;
  const height = Math.min(size, box.bottom + margin + 1) - y;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);
  context.drawImage(renderer.domElement, x, y, width, height, 0, 0, width, height);
  return canvas.toDataURL('image/jpeg', JPEG_QUALITY);
};

window.dceRender = ({ theta, phi, mode, size, background }) => {
  if (!current) {
    throw new Error('no model is loaded');
  }
  renderer.setSize(size, size, false);
  scene.background = null;

  const isRelief = mode === 'relief';
  const materials = isRelief ? current.relief : current.original;
  materials.forEach((material, node) => {
    node.material = material;
  });
  scene.environment = isRelief ? null : room;
  raking.visible = isRelief;
  fill.visible = isRelief;

  const distance = (current.radius / Math.sin(THREE.MathUtils.degToRad(FIELD_OF_VIEW / 2))) * FRAMING_MARGIN;
  camera.position.setFromSphericalCoords(
    distance,
    THREE.MathUtils.degToRad(phi),
    THREE.MathUtils.degToRad(theta),
  ).add(current.center);
  camera.near = (distance - current.radius) / 2;
  camera.far = (distance + current.radius) * 2;
  camera.aspect = 1;
  camera.updateProjectionMatrix();
  camera.lookAt(current.center);

  renderer.render(scene, camera);
  const box = silhouette(size);
  if (!box.covered) {
    return { coverage: 0 };
  }
  return { coverage: box.covered / (size * size), image: cropped(box, size, background) };
};

window.dceReady = true;
