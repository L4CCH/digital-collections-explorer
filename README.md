# Digital Collections Explorer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15744570.svg)](https://doi.org/10.5281/zenodo.15744570) [![Paper](https://img.shields.io/badge/paper-Computational%20Humanities%20Research-blue.svg)](https://doi.org/10.1017/chr.2025.10017) [![arXiv](https://img.shields.io/badge/arXiv-2507.00961-b31b1b.svg)](https://arxiv.org/abs/2507.00961)

**Digital Collections Explorer is an open-source search site for image collections that have little or no metadata.** A library, archive or museum points it at a folder, an S3 bucket or a manifest of images. Visitors can then search the collection in plain language ("a steam locomotive", "a handwritten letter with a red seal"), search with a picture, wander through it, and follow every image back to the institution's own record. No catalog, tags or OCR are needed, and where a catalog exists, search reads its records too.

The [project site](https://l4cch.github.io/dce-website/) has the documentation and the collections searched with it, among them the [maps of the Library of Congress](https://digital-collections-explorer.com/) from our paper. The project started at the [Lab for Computing Cultural Heritage](https://l4cch.github.io/lab-website/) at the University of Washington Information School, is developed in the open with contributors from other institutions, and is described in [our paper in *Computational Humanities Research*](https://doi.org/10.1017/chr.2025.10017).

![A diagram showing an overview of our Digital Collections Explorer, including its various components.](overview.png)

## What It Does

- **Searches by meaning.** Natural-language queries and reverse image search over photographs, maps, PDFs and 3D models, together in one site.
- **Installs only what a collection needs.** Photos and other images are always shown. 3D models, PDFs and deep zoom into IIIF images are options, so a site without 3D models never installs the 3D viewer.
- **Shows 3D models and PDFs.** An object with a `.glb` model can be turned and zoomed on its page, views drawn from the model join the search, and objects of similar shape can be found. A PDF is searched page by page and opens at the page found.
- **Needs no metadata, and uses it when it exists.** A catalog groups images into objects, links each one to its record, and makes the records searchable beside the pictures.
- **Scales on ordinary hardware.** Indexing reads from disk, S3 or a manifest without copying the originals, and serving needs no GPU.
- **Is safe to put online.** Rate, upload and concurrency limits protect a small machine, and every refusal is explained in plain words.
- **Can be searched by AI assistants.** An optional MCP server lets Claude, ChatGPT and other assistants search the collection and link back to it.

## Quick Start

You need Python 3.10 or later and Node.js 20.19 or later.

```bash
git clone https://github.com/L4CCH/digital-collections-explorer.git
cd digital-collections-explorer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
npm install && npm run setup   # asks what the collection holds besides photos

python -m src.ingest /path/to/images --data-dir data/collections/my-collection
DCE_DATA_DIR=data/collections/my-collection python -m src.backend.main
```

Then open http://localhost:8000. The images can also come from an S3 prefix (add `--anonymous` for a public bucket) or a parquet manifest of image addresses, and `python -m src.profiling <source>` says beforehand what the collection holds and what indexing will cost. Indexing can be stopped and run again; it continues where it left off. Search uses SigLIP by default; SigLIP 2 and CLIP can be chosen in `config.json`.

Setup can also be told what to show, as in `npm run setup -- --with=pdf,3d`, and installs only those:

- `pdf`: PDFs in the folder, bucket or manifest are indexed page by page by `python -m src.ingest`. Needs `pip install -r requirements-pdf.txt`.
- `3d`: models are drawn before indexing. `python -m src.render3d render /path/to/models --data-dir data/collections/my-collection` takes a folder of `.glb` files or a manifest with a `model_url` column, and writes a manifest of views to index with `python -m src.ingest`. Needs `pip install -r requirements-3d.txt` and `python -m playwright install chromium`.
- `zoom`: an image whose catalog record names an IIIF image service, in a column such as `iiif_url`, opens in a deep zoom viewer, for maps and other large images.

The site's interface lives in `src/frontend/default`. Another frontend can take its place; [CONTRIBUTING.md](CONTRIBUTING.md#making-a-frontend) explains what it needs.

## Other Ways to Start

With an AI coding agent such as Claude Code, Codex, Gemini CLI or Cursor, ask it to make your collection searchable. The `dce-setup` skill walks it through profiling, indexing and serving, and agents started in this repository find it on their own.

With Docker, one command builds the index and another serves it. The image runs on x86-64 and ARM machines, Apple Silicon and AWS Graviton among them. The published image shows every kind of material; an image of your own built with `docker build --build-arg MEDIA=pdf .` holds only what you list.

```bash
docker run --rm -v "$PWD/my-collection:/data" -v /path/to/images:/source:ro ghcr.io/l4cch/digital-collections-explorer ingest /source
docker run -p 8000:8000 -v "$PWD/my-collection:/data" ghcr.io/l4cch/digital-collections-explorer
```

On a Linux server, `deploy/bootstrap.sh` installs everything, indexes and keeps the site running across reboots. On AWS, `deploy/aws/deploy.sh` does the same on a machine it creates, after showing what it will cost.

```bash
sudo deploy/bootstrap.sh install --name my-collection --source /path/to/images
deploy/aws/deploy.sh my-collection --source s3://bucket/prefix --anonymous --https
```

## Documentation

The [documentation](https://l4cch.github.io/dce-website/docs/) covers:

- [Indexing large or remote collections](https://l4cch.github.io/dce-website/docs/indexing/)
- [Kinds of material: 3D models, PDFs and deep zoom](https://l4cch.github.io/dce-website/docs/kinds-of-material/)
- [Describing a collection to visitors](https://l4cch.github.io/dce-website/docs/describing-a-collection/)
- [Collections split into groups](https://l4cch.github.io/dce-website/docs/groups/)
- [Searching by appearance and by catalog](https://l4cch.github.io/dce-website/docs/search/)
- [Models: SigLIP, SigLIP 2 and CLIP](https://l4cch.github.io/dce-website/docs/models/)
- [Limits that protect a public site](https://l4cch.github.io/dce-website/docs/limits/)
- Deploying [to a Linux server](https://l4cch.github.io/dce-website/docs/deploy/server/), [to AWS](https://l4cch.github.io/dce-website/docs/deploy/aws/), or [under a path of an existing website](https://l4cch.github.io/dce-website/docs/deploy/under-a-path/)
- [Making a frontend](https://l4cch.github.io/dce-website/docs/making-a-frontend/)
- [The MCP server for AI assistants](https://l4cch.github.io/dce-website/docs/mcp-server/)

## Citation

If you use Digital Collections Explorer in research, please cite the paper and the software release you used. Every release has its own DOI on [Zenodo](https://doi.org/10.5281/zenodo.15744570). The paper is open access in [*Computational Humanities Research*](https://doi.org/10.1017/chr.2025.10017), and the preprint is on [arXiv](https://arxiv.org/abs/2507.00961).

```bibtex
@article{huang2025digitalcollectionsexplorer,
  title   = {Digital Collections Explorer: An Open-Source, Multimodal Viewer for Searching Digital Collections},
  author  = {Huang, Ying-Hsiang and Lee, Benjamin Charles Germain},
  journal = {Computational Humanities Research},
  volume  = {1},
  pages   = {e14},
  year    = {2025},
  doi     = {10.1017/chr.2025.10017}
}
```

## License

The source code of Digital Collections Explorer is licensed under the [MIT License](LICENSE). Documentation and original figures, such as `overview.png`, are licensed under [CC BY 4.0](LICENSE-CC-BY-4.0) unless otherwise noted. Third-party code, models, data and media keep their respective licenses.

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers setting up for development, running the tests and sending a change, and the project follows a [Code of Conduct](CODE_OF_CONDUCT.md).
