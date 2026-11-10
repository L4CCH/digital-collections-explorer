"""
Tests for the EmbeddingService class
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.backend.services.embedding_service import EmbeddingService
from src.backend.services.index_info import describe_old_layout
from src.collection.writer import Group, Item, Model, Record, write_collection


def write_small_collection(root, monkeypatch):
    from src.backend.services import embedding_service

    monkeypatch.setattr(embedding_service.settings, "model_name", "fake/model")
    write_collection(
        root,
        [Model("fake", "fake", "fake/model", 3)],
        [Group("collection", "collection")],
        [Item("a", "collection", "rec-a"), Item("b", "collection")],
        [Record("rec-a", catalog={"title": "Star Quilt"})],
        {"fake": (["a", "b"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))},
    )
    return root


@pytest.fixture
def embedding_service():
    """Create a fresh EmbeddingService instance for each test"""
    return EmbeddingService()


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings tensor"""
    return torch.randn(10, 512)  # 10 items, 512-dimensional embeddings


@pytest.fixture
def mock_item_ids():
    """Create mock item IDs"""
    return [f"item_{i}" for i in range(10)]


@pytest.fixture
def mock_metadata():
    """Create mock metadata"""
    return {
        f"item_{i}": {
            "filename": f"image_{i}.jpg",
            "path": f"/path/to/image_{i}.jpg",
            "collection": "test_collection",
        }
        for i in range(10)
    }


class TestEmbeddingServiceInitialization:
    """Test EmbeddingService initialization"""

    def test_init_creates_service(self, embedding_service):
        """Test that service initializes correctly"""
        assert embedding_service.embeddings is None
        assert embedding_service.item_ids is None
        assert embedding_service.metadata is None
        assert embedding_service.is_loaded is False

    def test_init_sets_embeddings_dir(self, embedding_service):
        """Test that embeddings directory is set"""
        assert embedding_service.embeddings_dir is not None
        assert isinstance(embedding_service.embeddings_dir, Path)


class TestLoadEmbeddings:
    """Test loading a collection from disk"""

    def test_a_collection_is_loaded_with_what_the_site_shows(
        self, tmp_path, monkeypatch
    ):
        service = EmbeddingService()
        service.embeddings_dir = write_small_collection(tmp_path, monkeypatch)
        service.load_embeddings()
        assert service.is_loaded is True
        assert service.embeddings.shape == (2, 3)
        assert service.item_ids == ["a", "b"]
        assert service.metadata["a"]["title"] == "Star Quilt"

    def test_a_folder_without_a_collection_loads_nothing(self, tmp_path):
        service = EmbeddingService()
        service.embeddings_dir = tmp_path
        service.load_embeddings()
        assert service.is_loaded is False
        assert service.embeddings is None

    @patch("src.backend.services.embedding_service.load_grouped")
    def test_load_embeddings_already_loaded(self, mock_load, embedding_service):
        """Test that the collection is not read again once loaded"""
        embedding_service.is_loaded = True

        embedding_service.load_embeddings()

        mock_load.assert_not_called()

    def test_an_index_in_the_old_layout_is_not_read_and_says_what_to_do(
        self, tmp_path, caplog
    ):
        (tmp_path / "embeddings.pt").write_bytes(b"old")
        (tmp_path / "item_ids.pt").write_bytes(b"old")
        (tmp_path / "index_info.json").write_text(
            json.dumps(
                {"model_name": "openai/clip-vit-base-patch32", "dimensions": 512}
            )
        )
        service = EmbeddingService()
        service.embeddings_dir = tmp_path
        with caplog.at_level("ERROR"):
            service.load_embeddings()
        assert service.is_loaded is False
        message = describe_old_layout(tmp_path)
        assert "v2.0.0-beta.2" in message and "into a new folder" in message
        assert message in caplog.text
        (tmp_path / "ingest_state.sqlite").write_bytes(b"")
        assert "only writes the collection anew" in describe_old_layout(tmp_path)


class TestGetEmbeddingCount:
    """Test getting embedding count"""

    def test_get_embedding_count_loaded(
        self, embedding_service, mock_embeddings, mock_item_ids
    ):
        """Test getting count when embeddings are loaded"""
        embedding_service.is_loaded = True
        embedding_service.item_ids = mock_item_ids

        count = embedding_service.get_embedding_count()

        assert count == 10

    @patch("pathlib.Path.exists")
    def test_get_embedding_count_not_loaded(self, mock_exists, embedding_service):
        """Test that count is 0 when no embedding files exist on disk"""
        mock_exists.return_value = False

        count = embedding_service.get_embedding_count()

        assert embedding_service.is_loaded is False
        assert count == 0


class TestGetDocumentById:
    """Test getting document by ID"""

    def test_get_document_by_id_success(
        self, embedding_service, mock_item_ids, mock_metadata
    ):
        """Test successfully retrieving a document by ID"""
        embedding_service.is_loaded = True
        embedding_service.item_ids = mock_item_ids
        embedding_service.metadata = mock_metadata

        doc = embedding_service.get_document_by_id("item_5")

        assert doc is not None
        assert doc["id"] == "item_5"
        assert "metadata" in doc
        assert doc["metadata"]["filename"] == "image_5.jpg"

    def test_get_document_by_id_not_found(
        self, embedding_service, mock_item_ids, mock_metadata
    ):
        """Test retrieving a non-existent document"""
        embedding_service.is_loaded = True
        embedding_service.item_ids = mock_item_ids
        embedding_service.metadata = mock_metadata

        doc = embedding_service.get_document_by_id("nonexistent_item")

        assert doc is None

    def test_get_document_by_id_no_metadata(self, embedding_service, mock_item_ids):
        """Test retrieving document when no metadata is loaded"""
        embedding_service.is_loaded = True
        embedding_service.item_ids = mock_item_ids
        embedding_service.metadata = None

        doc = embedding_service.get_document_by_id("item_5")

        assert doc is None


class TestSearch:
    """Test search functionality"""

    def test_search_basic(
        self, embedding_service, mock_embeddings, mock_item_ids, mock_metadata
    ):
        """Test basic search functionality"""
        embedding_service.is_loaded = True
        embedding_service.embeddings = mock_embeddings
        embedding_service.item_ids = mock_item_ids
        embedding_service.metadata = mock_metadata

        # Create a query embedding
        query_embedding = torch.randn(512, 1)

        results = embedding_service.search(query_embedding, limit=5)

        assert len(results) <= 5
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)
        assert all("metadata" in r for r in results)

    def test_search_with_score_transform(
        self, embedding_service, mock_embeddings, mock_item_ids, mock_metadata
    ):
        """Test search with a score_transform callable applied to raw similarities"""
        embedding_service.is_loaded = True
        embedding_service.embeddings = mock_embeddings
        embedding_service.item_ids = mock_item_ids
        embedding_service.metadata = mock_metadata

        query_embedding = torch.randn(512, 1)

        baseline = embedding_service.search(query_embedding, limit=5)
        scaled = embedding_service.search(
            query_embedding, score_transform=lambda s: s * 2.5, limit=5
        )

        assert len(scaled) <= 5
        assert [r["id"] for r in scaled] == [r["id"] for r in baseline]
        for b, s in zip(baseline, scaled):
            assert s["score"] == pytest.approx(b["score"] * 2.5, rel=1e-5)

    def test_search_with_pagination(
        self, embedding_service, mock_embeddings, mock_item_ids, mock_metadata
    ):
        """Test search with offset pagination"""
        embedding_service.is_loaded = True
        embedding_service.embeddings = mock_embeddings
        embedding_service.item_ids = mock_item_ids
        embedding_service.metadata = mock_metadata

        query_embedding = torch.randn(512, 1)

        # Get first page
        results_page1 = embedding_service.search(query_embedding, limit=3, offset=0)
        # Get second page
        results_page2 = embedding_service.search(query_embedding, limit=3, offset=3)

        assert len(results_page1) <= 3
        assert len(results_page2) <= 3
        # Ensure different results (not overlapping)
        if results_page1 and results_page2:
            assert results_page1[0]["id"] != results_page2[0]["id"]

    def test_search_empty_results(self, embedding_service):
        """Test search with no embeddings loaded"""
        embedding_service.is_loaded = False
        embedding_service.embeddings = None

        query_embedding = torch.randn(512, 1)

        results = embedding_service.search(query_embedding)

        assert results == []

    def test_search_limit_exceeds_available(
        self, embedding_service, mock_embeddings, mock_item_ids, mock_metadata
    ):
        """Test search when limit exceeds available items"""
        embedding_service.is_loaded = True
        embedding_service.embeddings = mock_embeddings
        embedding_service.item_ids = mock_item_ids
        embedding_service.metadata = mock_metadata

        query_embedding = torch.randn(512, 1)

        results = embedding_service.search(query_embedding, limit=100)

        # Should return at most 10 items (the size of mock data)
        assert len(results) <= 10
