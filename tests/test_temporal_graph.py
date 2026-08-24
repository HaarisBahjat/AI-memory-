"""
============================================================
tests/test_temporal_graph.py -- Phase 7.5 Temporal GraphRAG Tests
============================================================
Covers:
  1. graph_service.extract_triples         -- LLM triple extraction + validation
  2. graph_service.resolve_node            -- Entity resolution (merge vs create)
  3. graph_service.upsert_edge             -- Edge creation vs reinforcement
  4. temporal_graph_engine.find_seed_nodes -- pgvector seed search
  5. temporal_graph_engine.traverse_*      -- Recursive CTE traversal (happy + empty paths)
  6. temporal_graph_engine.format_graph_context -- Prompt formatting
  7. temporal_graph_engine.retrieve_graph_context -- Full pipeline (empty graph)
  8. retrieval_engine.assemble_system_prompt      -- Graph context injected correctly
  9. Performance guard: graph retrieval does not raise on empty graph
============================================================
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.temporal_graph_engine import format_graph_context


# -------------------------------------------------------
# 1. extract_triples -- LLM extraction & strict enum validation
# -------------------------------------------------------

class TestExtractTriples:

    @pytest.mark.asyncio
    async def test_valid_triples_returned(self):
        """LLM response with valid triples is parsed and returned."""
        from app.services.graph_service import extract_triples

        mock_response = MagicMock()
        mock_response.choices[0].message.content = '''
        [
          {
            "source": {"name": "Exam Stress", "entity_type": "TRIGGER"},
            "relation": "TRIGGERS",
            "target": {"name": "Insomnia", "entity_type": "SYMPTOM"},
            "evidence": "user reported not sleeping before exam week",
            "weight": 0.85
          }
        ]
        '''
        with patch("app.services.graph_service.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            triples = await extract_triples("User felt anxious before their exam.")

        assert len(triples) == 1
        assert triples[0]["source"]["name"] == "Exam Stress"
        assert triples[0]["relation"] == "TRIGGERS"
        assert triples[0]["target"]["name"] == "Insomnia"
        assert 0 < triples[0]["weight"] <= 1.0

    @pytest.mark.asyncio
    async def test_invalid_relation_type_dropped(self):
        """Triples with relation types not in the enum are silently dropped."""
        from app.services.graph_service import extract_triples

        mock_response = MagicMock()
        # "CAUSES" is not in ALLOWED_RELATION_TYPES
        mock_response.choices[0].message.content = '''
        [
          {
            "source": {"name": "Coffee", "entity_type": "ACTIVITY"},
            "relation": "CAUSES",
            "target": {"name": "Anxiety", "entity_type": "SYMPTOM"},
            "evidence": "user mentioned coffee worsens anxiety",
            "weight": 0.7
          },
          {
            "source": {"name": "Coffee", "entity_type": "ACTIVITY"},
            "relation": "WORSENS",
            "target": {"name": "Anxiety", "entity_type": "SYMPTOM"},
            "evidence": "user mentioned coffee worsens anxiety",
            "weight": 0.7
          }
        ]
        '''
        with patch("app.services.graph_service.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client

            triples = await extract_triples("Short session.")

        # Only the WORSENS triple (valid) should survive
        assert len(triples) == 1
        assert triples[0]["relation"] == "WORSENS"

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_failure(self):
        """If the LLM call raises, extract_triples returns [] (never raises)."""
        from app.services.graph_service import extract_triples

        with patch("app.services.graph_service.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("OpenAI timeout")
            )
            mock_cls.return_value = mock_client

            triples = await extract_triples("Some summary.")

        assert triples == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_blank_summary(self):
        """Empty or whitespace-only summary returns [] without LLM call."""
        from app.services.graph_service import extract_triples

        result = await extract_triples("   ")
        assert result == []


# -------------------------------------------------------
# 2. resolve_node -- Entity resolution (merge vs create)
# -------------------------------------------------------

class TestResolveNode:

    @pytest.mark.asyncio
    async def test_creates_new_node_when_no_existing(self):
        """When no close node exists, a new node is inserted."""
        from app.services.graph_service import resolve_node

        db = AsyncMock()
        empty = MagicMock()
        empty.mappings.return_value.first.return_value = None
        db.execute = AsyncMock(return_value=empty)

        node_id = await resolve_node(
            db=db,
            user_id="user-001",
            name="Exam Stress",
            entity_type="TRIGGER",
            embedding_vector=[0.1] * 1536,
        )

        assert node_id is not None
        # 2 calls: SELECT + INSERT
        assert db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_merges_into_existing_node_when_similar(self):
        """When a near-duplicate node exists (cos_dist <= 0.10), it is merged."""
        from app.services.graph_service import resolve_node

        db = AsyncMock()
        existing_id = str(uuid.uuid4())
        found = MagicMock()
        found.mappings.return_value.first.return_value = {
            "id": existing_id,
            "name": "Exam Stress",
            "cos_dist": 0.05,   # well within merge threshold (0.10)
        }
        db.execute = AsyncMock(return_value=found)

        node_id = await resolve_node(
            db=db,
            user_id="user-001",
            name="Test Anxiety",
            entity_type="TRIGGER",
            embedding_vector=[0.1] * 1536,
        )

        assert node_id == existing_id
        # 2 calls: SELECT + UPDATE
        assert db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_creates_new_node_when_distance_too_large(self):
        """When closest node is too dissimilar (cos_dist > 0.10), insert new."""
        from app.services.graph_service import resolve_node

        db = AsyncMock()
        found = MagicMock()
        found.mappings.return_value.first.return_value = {
            "id": str(uuid.uuid4()),
            "name": "Work Deadline",
            "cos_dist": 0.35,   # too far from GRAPH_ENTITY_MERGE_DISTANCE (0.10)
        }
        db.execute = AsyncMock(return_value=found)

        node_id = await resolve_node(
            db=db,
            user_id="user-001",
            name="Family Conflict",
            entity_type="TRIGGER",
            embedding_vector=[0.9] * 1536,
        )

        assert node_id is not None
        # 2 calls: SELECT + INSERT
        assert db.execute.call_count == 2


# -------------------------------------------------------
# 3. upsert_edge -- Create vs Reinforce
# -------------------------------------------------------

class TestUpsertEdge:

    @pytest.mark.asyncio
    async def test_creates_new_edge_when_none_exists(self):
        """New edge is inserted when no active edge with same src/rel/tgt exists."""
        from app.services.graph_service import upsert_edge

        db = AsyncMock()
        empty = MagicMock()
        empty.mappings.return_value.first.return_value = None
        db.execute = AsyncMock(return_value=empty)

        edge_id = await upsert_edge(
            db=db,
            user_id="user-001",
            source_node_id=str(uuid.uuid4()),
            target_node_id=str(uuid.uuid4()),
            relation_type="TRIGGERS",
            weight=0.8,
            evidence="user reported this pattern",
        )

        assert edge_id is not None
        assert db.execute.call_count == 2  # SELECT + INSERT

    @pytest.mark.asyncio
    async def test_reinforces_existing_active_edge(self):
        """Existing active edge is reinforced (weight blended, count incremented)."""
        from app.services.graph_service import upsert_edge

        db = AsyncMock()
        existing_id = str(uuid.uuid4())
        found = MagicMock()
        found.mappings.return_value.first.return_value = {
            "id": existing_id,
            "weight": 0.70,
            "observation_count": 1,
        }
        db.execute = AsyncMock(return_value=found)

        edge_id = await upsert_edge(
            db=db,
            user_id="user-001",
            source_node_id=str(uuid.uuid4()),
            target_node_id=str(uuid.uuid4()),
            relation_type="TRIGGERS",
            weight=0.90,
            evidence="observed again",
        )

        assert edge_id == existing_id
        assert db.execute.call_count == 2  # SELECT + UPDATE


# -------------------------------------------------------
# 4. find_seed_nodes
# -------------------------------------------------------

class TestFindSeedNodes:

    @pytest.mark.asyncio
    async def test_returns_seed_nodes(self):
        """find_seed_nodes returns nodes ordered by cosine distance."""
        from app.services.temporal_graph_engine import find_seed_nodes

        db = AsyncMock()
        rows_mock = MagicMock()
        rows_mock.mappings.return_value.all.return_value = [
            {"id": str(uuid.uuid4()), "name": "Exam Stress", "entity_type": "TRIGGER",
             "mention_count": 3, "cos_dist": 0.04},
            {"id": str(uuid.uuid4()), "name": "Insomnia", "entity_type": "SYMPTOM",
             "mention_count": 2, "cos_dist": 0.08},
        ]
        db.execute = AsyncMock(return_value=rows_mock)

        seeds = await find_seed_nodes(db, "user-001", [0.1] * 1536)

        assert len(seeds) == 2
        assert seeds[0]["name"] == "Exam Stress"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_nodes(self):
        """Returns empty list when user has no knowledge nodes yet."""
        from app.services.temporal_graph_engine import find_seed_nodes

        db = AsyncMock()
        empty = MagicMock()
        empty.mappings.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=empty)

        seeds = await find_seed_nodes(db, "user-001", [0.1] * 1536)
        assert seeds == []


# -------------------------------------------------------
# 5. traverse_temporal_subgraph
# -------------------------------------------------------

class TestTraverseTemporalSubgraph:

    @pytest.mark.asyncio
    async def test_returns_paths(self):
        """Traversal returns structured path dicts."""
        from app.services.temporal_graph_engine import traverse_temporal_subgraph

        db = AsyncMock()
        rows_mock = MagicMock()
        rows_mock.mappings.return_value.all.return_value = [
            {
                "src_name": "Exam Stress", "src_type": "TRIGGER",
                "relation_type": "TRIGGERS",
                "tgt_name": "Insomnia", "tgt_type": "SYMPTOM",
                "weight": 0.82, "depth": 1,
                "evidence": "no sleep before exams",
                "valid_from": "2026-01-01", "observation_count": 3,
            }
        ]
        db.execute = AsyncMock(return_value=rows_mock)

        paths = await traverse_temporal_subgraph(db, "user-001", [str(uuid.uuid4())])

        assert len(paths) == 1
        assert paths[0]["relation_type"] == "TRIGGERS"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_seed_ids(self):
        """Short-circuits and returns [] when no seed IDs are provided."""
        from app.services.temporal_graph_engine import traverse_temporal_subgraph

        db = AsyncMock()
        paths = await traverse_temporal_subgraph(db, "user-001", [])
        assert paths == []
        db.execute.assert_not_called()


# -------------------------------------------------------
# 6. format_graph_context
# -------------------------------------------------------

class TestFormatGraphContext:

    def test_formats_single_path(self):
        """Single path is formatted with entity names, relation, and evidence."""
        paths = [
            {
                "src_name": "Exam Stress", "src_type": "TRIGGER",
                "relation_type": "TRIGGERS",
                "tgt_name": "Insomnia", "tgt_type": "SYMPTOM",
                "weight": 0.85, "depth": 1,
                "evidence": "not sleeping before exams",
                "observation_count": 4, "valid_from": None,
            }
        ]
        context = format_graph_context(paths)

        assert "Exam Stress" in context
        assert "Insomnia" in context
        assert "has historically triggered" in context
        assert "not sleeping before exams" in context

    def test_returns_empty_string_for_no_paths(self):
        """Empty path list returns empty string (no section injected)."""
        assert format_graph_context([]) == ""

    def test_orders_by_depth_then_weight(self):
        """Depth-1 paths appear before depth-2 paths."""
        paths = [
            {
                "src_name": "Coffee", "src_type": "ACTIVITY",
                "relation_type": "WORSENS",
                "tgt_name": "Anxiety", "tgt_type": "SYMPTOM",
                "weight": 0.70, "depth": 2, "evidence": "late night coffee",
                "observation_count": 1, "valid_from": None,
            },
            {
                "src_name": "Exam Stress", "src_type": "TRIGGER",
                "relation_type": "TRIGGERS",
                "tgt_name": "Insomnia", "tgt_type": "SYMPTOM",
                "weight": 0.85, "depth": 1, "evidence": "no sleep before exams",
                "observation_count": 3, "valid_from": None,
            },
        ]
        context = format_graph_context(paths)
        # Exam Stress (depth 1) should appear before Coffee (depth 2)
        assert context.index("Exam Stress") < context.index("Coffee")


# -------------------------------------------------------
# 7. retrieve_graph_context -- Full pipeline integration
# -------------------------------------------------------

class TestRetrieveGraphContext:

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_nodes(self):
        """When user has no knowledge nodes, returns empty string (no crash)."""
        from app.services.temporal_graph_engine import retrieve_graph_context

        db = AsyncMock()
        empty = MagicMock()
        empty.mappings.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=empty)

        result = await retrieve_graph_context(db, "user-001", [0.1] * 1536)
        assert result == ""


# -------------------------------------------------------
# 8. assemble_system_prompt -- Graph section injection
# -------------------------------------------------------

class TestAssembleSystemPromptWithGraph:

    def test_graph_section_injected_when_present(self):
        """When graph_context is non-empty, it appears in the system prompt."""
        from app.services.retrieval_engine import assemble_system_prompt

        result = assemble_system_prompt(
            session_messages=[],
            episodes=[],
            semantic_memories=[],
            graph_context="[Causal History]\n- Exam Stress triggers Insomnia",
        )

        assert "CAUSAL KNOWLEDGE GRAPH" in result
        assert "Exam Stress" in result
        assert "four layers" in result

    def test_graph_section_absent_when_empty(self):
        """When graph_context is empty string, the section is not added."""
        from app.services.retrieval_engine import assemble_system_prompt

        result = assemble_system_prompt(
            session_messages=[],
            episodes=[],
            semantic_memories=[],
            graph_context="",
        )

        assert "CAUSAL KNOWLEDGE GRAPH" not in result
