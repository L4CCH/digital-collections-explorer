# Digital Collections Explorer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15744570.svg)](https://doi.org/10.5281/zenodo.15744570) [![Paper](https://img.shields.io/badge/paper-Computational%20Humanities%20Research-blue.svg)](https://doi.org/10.1017/chr.2025.10017) [![arXiv](https://img.shields.io/badge/arXiv-2507.00961-b31b1b.svg)](https://arxiv.org/abs/2507.00961)

**Digital Collections Explorer is an open-source search site for digital collections, with or without a catalog.** A gallery, library, archive, or museum points it at a folder, an S3 bucket, or a manifest of images and their catalog. Visitors can then search the collection in plain language ("a steam locomotive", "a handwritten letter with a red seal"), search with a picture, wander through it, and follow every image back to the institution's own record. No catalog or tags are needed, and where a catalog exists, search reads its records too.

The [project site](https://l4cch.github.io/dce-website/) shows what it does and the collections searched with it, among them the [maps of the Library of Congress](https://digital-collections-explorer.com/) from our paper, and holds the [documentation](https://l4cch.github.io/dce-website/docs/). The project started at the [Lab for Computing Cultural Heritage](https://l4cch.github.io/lab-website/) at the University of Washington Information School, is developed in the open with contributors from other institutions, and is described in [our paper in *Computational Humanities Research*](https://doi.org/10.1017/chr.2025.10017).

![A diagram showing an overview of our Digital Collections Explorer, including its various components.](overview.png)

## Quick Start

There are three recommended ways to start, each explained in the [documentation](https://l4cch.github.io/dce-website/docs/):

- **[With an AI coding agent](https://l4cch.github.io/dce-website/docs/indexing/#with-an-ai-coding-agent).** Ask Claude Code, Codex, Gemini CLI, or Cursor to make your collection searchable. The `dce-setup` skill in this repository walks it through profiling, indexing, and serving.
- **[With Docker](https://l4cch.github.io/dce-website/docs/indexing/#with-docker).** One command builds the index, another serves it.
- **[On a server or AWS](https://l4cch.github.io/dce-website/docs/deploy/server/).** One command keeps a site running. On AWS it shows the cost first.

## Citation

If you use Digital Collections Explorer in research, please cite [the paper](https://doi.org/10.1017/chr.2025.10017) and the release you used, each release having its own DOI on [Zenodo](https://doi.org/10.5281/zenodo.15744570).

## License

The source code of Digital Collections Explorer is licensed under the [MIT License](LICENSE). Documentation and original figures, such as `overview.png`, are licensed under [CC BY 4.0](LICENSES/CC-BY-4.0.txt) unless otherwise noted. Third-party code, models, data, and media keep their respective licenses.

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers setting up for development, running the tests, and sending a change, and the project follows a [Code of Conduct](CODE_OF_CONDUCT.md).
