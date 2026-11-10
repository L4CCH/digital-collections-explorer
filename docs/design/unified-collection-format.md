# One collection format, one pipeline

Status: proposal for review, 2026-11-10.

## Why

A collection reaches the site in one of two formats today, and each lacks what the other has.

| | Indexed by DCE (`embeddings.pt`, `item_ids.pt`, `metadata.json`) | Prepared by a pipeline (`index_info.json` format 2, `vectors/`, `metadata.sqlite`) |
| --- | --- | --- |
| Catalog fields on the object's page | no, only the title and the link back | yes, `record_fields` |
| Filters by date, place, kind of object | no | yes, `record_facets` |
| Search over the catalog text | no | yes, `records_fts` |
| Groups, related photos, several models | no | yes |
| PDF pages, IIIF services, keeping thumbnails only or linking | yes | no columns for them |

The writer of the second format lives outside DCE, in `smithsonian-search/dce_bundle.py`, so the next institution with a catalog would have to write it again. And adding catalog search to the first format took a second database, `catalog.sqlite`, beside the first.

The split was half right. Reading a source, such as Smithsonian EDAN records, its units, rights and Flickr links, belongs outside DCE. Writing the collection format belongs in it.

## Goals

- Every collection is stored in one format, whether DCE indexed it or a pipeline prepared it.
- DCE has the only code that writes that format, used by `src.ingest` and by any pipeline.
- Whatever a source holds reaches the site through one documented manifest, and the profile says what the data supports before indexing.
- Existing collections keep working: si-oa bundles as they are, older collections after one conversion that keeps their vectors.
- One site can search the collections of several institutions together, each keeping its own name and license.

Not goals: reading particular catalog systems inside DCE, editing records on a running site.

## 1. The collection format

```
my-collection/
  collection.json             title, introduction, suggested searches, links, labels
  embeddings/
    index_info.json           format 3
    vectors/<model>/<group>.npy        float32, one L2-normalized row per image
    vectors/<model>/<group>.ids.npy    the item id of each row
    metadata.sqlite           items, records and what the site shows and searches
    shape/                    optional, the shape index of 3D models
  thumbnails/  processed/  models3d/  related/   files the database points to
```

`index_info.json` keeps what format 2 keeps: `models` (each with `key`, `type`, `name`, `revision`, `dimensions`), `default_model`, `groups` (each with `id`, `label`, `images`), `built_at` and `source`. A collection that is not split has one group. The site hides the group filter when there is only one.

`metadata.sqlite`:

| Table | Columns | Required | What it powers |
| --- | --- | --- | --- |
| `items` | `id`, `group_id`, `record_id`, `file_name`, `width`, `height`, `thumbnail`, `processed`, `original` | yes | every image: where its sizes are kept and the object it shows |
| | `iiif`, `page`, `view_orbit` | no | deep zoom and sizes from an IIIF service, the page of a PDF, the camera of a 3D view |
| `records` | `record_id`, `title`, `source_url` | yes | one row per object: its title and the link back to its record |
| | `model_url`, `document`, `pages` | no | its 3D model, the PDF it is |
| `record_fields` | `record_id`, `position`, `label`, `value` | no | the catalog fields on the object's page, in order |
| `record_facets` | `record_id`, `facet`, `value` | no | filters: `date`, `object_type`, `place`, `culture`, `topic`, `name` |
| `records_fts`, `entity_vocab` | as today | no | hybrid search, and the names, places, topics and kinds of object Auto search recognizes |
| `related_sources`, `related_tags` | as today | no | where else a record appears, such as Flickr Commons |

Rules that hold throughout:

- An item without a record is its own object and has no catalog record.
- `thumbnail`, `processed` and `original` are paths relative to the collection folder or web addresses, and may be empty when `iiif` names a service that serves the sizes.
- Any column or table the site does not know is left alone, so a pipeline may keep its own, as si-oa keeps `unit_code`, `etag` and `guid`.

Format 3 is format 2 with these optional columns. The site reads both, so si-oa bundles need no change. An older site does not load format 3 at all, rather than showing a collection without its PDFs and IIIF services.

## 2. Several institutions in one site

A site with several institutions is one collection split into groups, as si-oa is split into museums: one index, one ranking, one filter. The groups come from a `group` column of the files table, or from a pipeline that hands over its vectors group by group. They are indexed together, so they share the model.

Each group may name its own institution and license in `collection.json`, beside the collection's:

```json
{
  "title": "...",
  "group_label": "Institution",
  "institution": "...",
  "license": {"name": "..."},
  "groups": {
    "loc": {"label": "Library of Congress", "institution": "the Library of Congress", "license": {"name": "No known restrictions"}},
    "nmah": {"label": "National Museum of American History", "institution": "the Smithsonian", "license": {"name": "CC0"}}
  }
}
```

An object's page links to its full record at its group's institution and shows its group's license, falling back to the collection's when its group names none. The MCP server and `llms.txt` name each group's institution. `institution` is the new name of `source_name`, which keeps working.

What institutions share is the format, not their catalogs:

- The format's core is the same for all: where each file is, the object it shows, the title and the link back.
- Each maps its own catalog columns onto the same searched fields, so one's `creator` and another's `artist` are both names (section 3).
- The filters use the same six categories for all, with values as each catalog spells them, except dates, which are read as years.
- The fields on an object's page are each record's own, so no institution has to change its catalog.

## 3. The collection manifest

What a person or a pipeline gives DCE. The smallest is a folder of files. A catalog comes as one table, or two.

**Files**, one row per file, which may be an image, a PDF or a 3D model: a folder, an S3 prefix, or a parquet table with

| Column | Holds | Needed for |
| --- | --- | --- |
| `image_url` (or `url`, `path`, `file`, `uri`) | the file's address, relative to the manifest or a web address | the file itself, unless `iiif` names an image |
| `iiif` (or `iiif_url`, `iiif_service`, `iiif_info`) | its IIIF image service | deep zoom, and sizes asked of the service |
| `thumbnail_url` | a thumbnail the source already makes | showing it without a copy |
| `object_id` (or `record_id`, `item_id`) | the object it belongs to | showing an object's files together, and joining the records table |
| `group` | the group it belongs to, such as a museum | the group filter |
| `model_url`, `view_orbit` | a 3D model, the camera of a rendered view | 3D models |
| any catalog column | as in the records table | when the catalog comes in the same table |

**Records**, optional, one row per object: `object_id`, then the catalog. Without it, the catalog columns of the files table are read, from the first file of each object. A record whose object has no file is left out, since nothing about it can be seen, and the profile says how many there are.

What DCE makes of each kind of file:

- **An image** is one item, searched by what it shows.
- **A PDF** is one object, its record's `document` pointing to the file and `pages` counting them. DCE draws each page as an image, keeps the copies, and indexes it as an item with its `page`; the object's page turns through them and opens the whole PDF. Keeping thumbnails only or linking does not apply, since no source serves the pages drawn.
- **A 3D model** is drawn from several sides first, by `src.render3d`; each view is an item with its `view_orbit`, and the record's `model_url` opens the model.

Catalog columns are read by name, case aside and plurals too:

| Read as | Columns | Shown on the page | Searched as | Filter |
| --- | --- | --- | --- | --- |
| Title | `title`, `label` | as the title | title | |
| Link back | `source_url`, `guid`, `record_url`, `landing_page`, `permalink` | as the link | | |
| Names | `creator`, `author`, `artist`, `maker`, `photographer`, `contributor`, `publisher`, `people`, `sitter` | yes | names | `name` |
| Dates | `date`, `year`, `date_created`, `created`, `period` | yes | dates, with each year's decade | `date` |
| Places | `place`, `location`, `geography`, `spatial`, `coverage`, `country`, `region`, `state`, `county`, `city` | yes | places | `place` |
| Cultures | `culture`, `nationality`, `people_group` | yes | topics | `culture` |
| Subjects | `subject`, `topic`, `keywords`, `tags`, `theme` | yes | topics | `topic` |
| Kinds of object | `object_type`, `genre`, `classification`, `kind`, `work_type` | yes | object types | `object_type` |
| Materials | `medium`, `material`, `technique`, `dimensions`, `extent` | yes | physical | |
| Catalog numbers | `identifier`, `accession_number`, `object_number`, `call_number`, `catalog_number`, `inventory_number` | yes | identifiers | |
| Notes | `description`, `notes`, `summary`, `abstract`, `caption`, `inscription`, `transcription`, any other column of text | yes | notes | |

Several values in one cell are separated by `;` or `|`. Web addresses, file names, media types and columns of numbers are not catalog text. A page shows each field under its column name, unless `collection.json` gives the column a label.

A pipeline that knows more than a table can say may add, beside the records:

- `fields`: `object_id`, `position`, `label`, `value`, to set the page's fields and their order itself
- `facets`: `object_id`, `facet`, `value`, to set the filters itself
- `related`: the rows of `related_sources` and `related_tags`

**Vectors**, optional: a pipeline that embeds the files itself, as si-oa does with two models, hands over `vectors/<model>/<group>.npy` and `.ids.npy` with the models' details, and DCE writes the rest without embedding again.

**What each part of the site needs.** The profile reports this for every collection before indexing, with the same rules the writer uses, so what an agent tells a person is what the site will do:

| The site can | when the data has |
| --- | --- |
| search by description and by image | images, PDFs or 3D models |
| show titles and link back | `title`, a link column |
| search the catalog as well (hybrid search) | any catalog text |
| filter by date, place, kind, culture, subject, name | the columns read as those |
| show an object's images together | an object column |
| filter by group | a group column |
| show 3D models, PDFs, deep zoom | `.glb` models, PDF files, an IIIF column |

## 4. The pipeline

One writer, `src/collection/`, takes items, records, fields, facets, related sources, groups and vectors and writes the format. Every route goes through it:

- `python -m src.ingest SOURCE --data-dir DIR`: reads the manifest, embeds the images, writes the collection. It continues where it stopped, as now, and writes the collection again at the end of each run.
- `python -m src.collection build MANIFEST --vectors VECTORS --data-dir DIR`: for a pipeline with its own vectors.
- `python -m src.collection convert DIR`: turns a collection in the old format into format 3, keeping its vectors as they are.

## 5. Existing collections

- si-oa bundles stay as they are. The next one is built by `src.collection build` from what smithsonian-search writes.
- The Library of Congress maps are converted once. They then show each map's title and date as fields, filter by date, and search the titles.
- The site keeps reading the old format for one release, warning that `src.collection convert` turns it into the new one, and then stops.
- `catalog.sqlite`, from the unpublished branch `feature/catalog-search`, is never released. Its column rules, dates with decades and vocabulary move into the writer.

## 6. Changes to smithsonian-search

It keeps everything about the Smithsonian: reading EDAN, units, rights, linking records to Flickr through Harvard LIL, embedding on EC2. It stops writing `metadata.sqlite` and instead writes the manifest: images, records, fields, facets, related sources, and its vectors, then calls `src.collection build`. `dce_bundle.py` shrinks to that.

## 7. Documentation

- A "Collection format" page replaces "Collections split into groups": the folder, `index_info.json` and every table, with what each powers.
- The "Collection format" page also covers several institutions in one site: the `group` column and each group's institution and license.
- A "Mapping your data" page: the manifest, the column table above, what each part of the site needs, and worked examples: a spreadsheet catalog, a Dublin Core export, IIIF manifests, an export from a museum's collection system, and Smithsonian EDAN as the full case.
- Indexing, search and describing a collection, and the setup skill, point to these two pages.

## 8. Phases

Each phase is one branch that works on its own.

1. The writer and format 3, with the site reading formats 2 and 3, and tests that a si-oa bundle reads as before.
2. `src.ingest` writes format 3: records, fields, facets, search over the catalog and groups for every collection, each group's institution and license on the object's page, and the profile uses the writer's rules.
3. PDF pages, IIIF services and the three image modes in format 3, and the warning on the old format.
4. `src.collection convert`, then the Library of Congress maps on format 3.
5. `src.collection build`, then smithsonian-search writing the manifest for the next si-oa bundle.
6. The documentation pages, the skill and the website.

## 9. To decide

1. **Format number.** Writing 3 and reading 2 and 3 (recommended) keeps an older site from loading a collection it would show incompletely. Keeping 2 with optional columns lets an older site show such a collection without its PDFs and IIIF services.
2. **Naming groups.** `group_id` now names an object, as `object_id` does, which collides with groups such as museums. Recommended: `group` names a group, and `group_id` keeps naming an object for old manifests, with a warning.
3. **Fields on the page.** Every catalog column that is not technical (recommended), or only the columns the table above knows.
4. **When the old format stops being read**: one release after the converter (recommended), or never.
