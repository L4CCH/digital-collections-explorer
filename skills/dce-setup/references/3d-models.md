# 3D models

When `media` in the profile includes `3d`, draw the models before building the index:

```bash
$PY -m playwright install chromium
$PY -m src.render3d render SOURCE --data-dir data/collections/NAME
```

This draws views of each model from several sides, eight by default, keeps them
under `data/collections/NAME/models3d`, and writes a manifest of those views and
the collection's photos. Rendering can be stopped and run again.

Then build the index as in step 5, with
`data/collections/NAME/models3d/views.parquet` as SOURCE. The profile's time
estimate counts each model once, so indexing its views takes longer: say so.

- Each model gets a viewer on its object's page, and its views are found by
  description and by example image like any photo.
- Views that show too little of a model, such as the bare back of a relief, are
  left out. Tell the person how many views each model kept if some kept few.
- It also builds the shape index behind "Similar in shape". The first run
  downloads an image model of about 350 MB; `--no-shape` skips it.
- Only `.glb` files, and `.gltf` files at a web address, are shown. Models in
  other formats (`.obj`, `.ply`, `.fbx`, or `.gltf` kept on disk) are listed as
  left out. Say so, and that exporting them as glTF binary from Blender makes
  them usable.
- A model hosted elsewhere is shown from where it lives, so its server must let
  other sites load it. Smithsonian 3D does.
