"""How a collection's folder is laid out, as the format in its index_info.json says."""

INDEX_INFO_FILE = "index_info.json"
INGEST_STATE_FILE = "ingest_state.sqlite"
# The format of a collection split into groups, one vector shard per group.
GROUPED_FORMAT = 2
# Format 2 with optional columns for IIIF services, PDF pages and their documents.
COLLECTION_FORMAT = 3
READABLE_FORMATS = (GROUPED_FORMAT, COLLECTION_FORMAT)
