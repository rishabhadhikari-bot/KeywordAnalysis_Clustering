import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

import app2
from src import topic_retrieval as topics


class BERTopicQualityTests(unittest.TestCase):
    def test_all_outliers_and_exclusion_before_prototype_budget(self):
        rows = [{"story_id": str(i), "page_title": f"Oil title {i}", "total_views": 1}
                for i in range(162)]
        ids = [row["story_id"] for row in rows]
        embeddings = np.tile([1., 0.], (162, 1)).astype(np.float32)
        embeddings[-1] = [.9, .4358899]
        artifacts = {
            "manifest": {"embedding_model": "test"}, "story_ids": ids,
            "topic_ids": np.full(162, -1), "topic_probabilities": np.zeros(162),
            "unique_topic_ids": np.array([], dtype=int),
            "topic_embeddings": np.empty((0, 2)), "topic_labels": {},
        }
        with patch.object(topics, "load_or_build_bertopic_index", return_value=artifacts), \
             patch.object(topics, "encode_relationship_texts", return_value=np.array([[1., 0.]])), \
             patch.object(topics, "load_or_build_title_embedding_index", return_value=(embeddings, ids, {})), \
             patch.object(topics, "_build_query_prototypes", return_value=([embeddings[0]], [list(range(160))])):
            results, diagnostics = topics.retrieve_bertopic_candidates(
                "oil", rows, direct_story_ids=tuple(ids[:160]), excluded_story_ids=(ids[160],)
            )
        self.assertEqual([row["story_id"] for row in results], [ids[161]])
        self.assertEqual(diagnostics["outlier_rescue_count"], 1)

    def test_training_configuration_tracks_independent_density_settings(self):
        with patch.dict(os.environ, {"BERTOPIC_MIN_CLUSTER_SIZE": "50", "BERTOPIC_MIN_SAMPLES": "5"}):
            config = topics.get_bertopic_training_config()
        self.assertEqual(config["min_cluster_size"], 50)
        self.assertEqual(config["min_samples"], 5)
        self.assertEqual(config["min_df"], 1)

    def test_prototype_only_candidate_requires_query_or_topic_evidence(self):
        rows = [
            {"story_id": "direct", "page_title": "Vaibhav Suryavanshi cricket statement"},
            {"story_id": "ritual", "page_title": "Aaj ka panchang vrat puja"},
            *[
                {"story_id": str(index), "page_title": f"Unrelated title {index}"}
                for index in range(98)
            ],
        ]
        story_ids = [row["story_id"] for row in rows]
        artifacts = {
            "manifest": {"embedding_model": "test"},
            "story_ids": story_ids,
            "topic_ids": np.array([1, 39, *([-1] * 98)]),
            "topic_probabilities": np.array([1.0, 1.0, *([0.0] * 98)]),
            "unique_topic_ids": np.array([1, 39]),
            "topic_embeddings": np.array([[1.0, 0.0], [0.9, 0.4358899]]),
            "topic_labels": {"1": "cricket vaibhav", "39": "statue vrat puja"},
        }
        title_embeddings = np.vstack([
            [[1.0, 0.0], [0.8, 0.6]],
            np.tile([[0.0, 1.0]], (98, 1)),
        ]).astype(np.float32)
        with patch.object(topics, "load_or_build_bertopic_index", return_value=artifacts), \
             patch.object(topics, "encode_relationship_texts", return_value=np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])), \
             patch.object(topics, "load_or_build_title_embedding_index", return_value=(title_embeddings, story_ids, {})), \
             patch.object(topics, "_build_query_prototypes", return_value=([title_embeddings[0]], [[0]])):
            results, diagnostics = topics.retrieve_bertopic_candidates(
                "Vaibhav Suryavanshi",
                rows,
                direct_story_ids=("direct",),
                excluded_story_ids=("direct",),
            )
        self.assertEqual(results, [])
        self.assertGreaterEqual(diagnostics["prototype_gate_rejected_count"], 1)

    def test_selected_cluster_still_requires_title_level_relevance(self):
        rows = [
            {"story_id": "direct", "page_title": "Hanuman Chalisa lyrics and benefits"},
            {"story_id": "unrelated", "page_title": "Mamata Banerjee election clarification"},
            *[
                {"story_id": str(index), "page_title": f"Unrelated title {index}"}
                for index in range(98)
            ],
        ]
        story_ids = [row["story_id"] for row in rows]
        artifacts = {
            "manifest": {"embedding_model": "test"},
            "story_ids": story_ids,
            "topic_ids": np.array([1, 1, *([-1] * 98)]),
            "topic_probabilities": np.ones(100),
            "unique_topic_ids": np.array([1]),
            "topic_embeddings": np.array([[1.0, 0.0]]),
            "topic_labels": {"1": "hanuman chalisa devotional"},
        }
        title_embeddings = np.vstack([
            [[1.0, 0.0], [0.55, 0.835]],
            np.tile([[0.0, 1.0]], (98, 1)),
        ]).astype(np.float32)
        with patch.object(topics, "load_or_build_bertopic_index", return_value=artifacts), \
             patch.object(topics, "encode_relationship_texts", return_value=np.array([[1.0, 0.0], [1.0, 0.0]])), \
             patch.object(topics, "load_or_build_title_embedding_index", return_value=(title_embeddings, story_ids, {})), \
             patch.object(topics, "_build_query_prototypes", return_value=([], [])):
            results, diagnostics = topics.retrieve_bertopic_candidates(
                "hanuman chalisa",
                rows,
                direct_story_ids=("direct",),
                excluded_story_ids=("direct",),
            )
        self.assertEqual(results, [])
        self.assertGreaterEqual(diagnostics["title_gate_rejected_count"], 1)

    def test_immutable_artifacts_and_config_cache_reuse(self):
        rows = [{"story_id": str(i), "page_title": f"Title {i}"} for i in range(100)]
        config = topics.get_bertopic_training_config()
        manifest = {"title_count": 100, "index_version": topics.BERTOPIC_INDEX_VERSION,
                    "embedding_model": topics.get_relationship_embedding_model_name(),
                    "training_config": config,
                    "corpus_fingerprint": topics.build_title_corpus_fingerprint(topics._valid_sorted_rows(rows))}
        # Keep temporary test artifacts inside the permitted workspace.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".tmp") as directory:
            root = Path(directory)
            topics._save_artifacts(manifest=manifest, assignments_path=root / "assignments.npz",
                topics_path=root / "topics.json", manifest_path=root / "manifest.json",
                story_ids=[str(i) for i in range(100)], topic_ids=np.zeros(100, dtype=int),
                topic_probabilities=np.ones(100), unique_topic_ids=np.array([0]),
                topic_embeddings=np.ones((1, 2)), topic_labels={"0": "energy"})
            with patch.object(topics, "load_or_build_title_embedding_index", side_effect=AssertionError("Unexpected rebuild")):
                loaded = topics.load_or_build_bertopic_index(rows, index_dir=root)
            self.assertEqual(loaded["topic_labels"], {"0": "energy"})
            self.assertTrue((root / manifest["assignments_file"]).exists())
            self.assertEqual(json.loads((root / "manifest.json").read_text())["training_config"], config)

    def test_catalog_keeps_unassigned_but_excludes_refined_title_samples(self):
        artifacts = {"story_ids": ["1", "2"], "topic_ids": np.array([0, -1]),
                     "topic_probabilities": np.array([1., 0.]), "unique_topic_ids": np.array([0]),
                     "topic_labels": {"0": "energy"}}
        with patch.object(topics, "load_or_build_bertopic_index", return_value=artifacts):
            catalog = topics.get_bertopic_topic_catalog([
                {"story_id": "1", "page_title": "Forbidden refined title"},
                {"story_id": "2", "page_title": "Unassigned title"},
            ], excluded_story_ids=("1",))
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0]["topic_label"], "Unassigned")
        self.assertNotIn("Forbidden", str(catalog))

    def test_app2_suppresses_duplicate_titles_and_anchor_previews(self):
        frame = pd.DataFrame([
            {"story_id": "direct", "page_title": "An oil story"},
            {"story_id": "duplicate", "page_title": "AN  OIL STORY"},
            {"story_id": "related", "page_title": "A related story"},
        ])
        candidates = [{"story_id": "related", "retrieval_evidence": [
            {"prototype_anchor_titles": ["An oil story"]}]}]
        diagnostics = {"query_prototypes": [{"representative_titles": "An oil story"}]}
        with patch.object(app2, "get_bertopic_view_data", return_value=(candidates, diagnostics)) as retrieve, \
             patch.object(app2, "get_bertopic_catalog_data", return_value=[]) as catalog:
            result = app2.run_bertopic_tab_flow("oil", frame, ("direct",), ("direct",))
        self.assertEqual(set(retrieve.call_args.kwargs["excluded_story_ids"]), {"direct", "duplicate"})
        self.assertEqual(set(catalog.call_args.kwargs["excluded_story_ids"]), {"direct", "duplicate"})
        self.assertNotIn("An oil story", str(result))

    def test_refined_exclusion_includes_fuzzy_and_nonvisible_pages(self):
        frame = pd.DataFrame([{"story_id": "1", "page_title": "oil", "total_views": 1}])
        state = {"ready": True, "fingerprint": app2.title_corpus_fingerprint(frame), "physical_index": "v1"}
        client = Mock(); client.index_state.return_value = state
        with patch.object(app2, "load_opensearch_settings", return_value=Mock(configured=True)), \
             patch.object(app2, "OpenSearchRefinedClient", return_value=client), \
             patch.object(app2, "collect_all_refined_story_id_sets", return_value=({"1", "fuzzy", "page2"}, {"1", "page2"})):
            all_ids, primary = app2.get_bertopic_refined_exclusions("oil", "All keywords", frame)
        self.assertEqual(set(all_ids), {"1", "fuzzy", "page2"})
        self.assertNotIn("fuzzy", primary)

    def test_refined_unavailable_fails_closed(self):
        with patch.object(app2, "load_opensearch_settings", return_value=Mock(configured=False)):
            with self.assertRaises(app2.OpenSearchRefinedError):
                app2.get_bertopic_refined_exclusions("oil", "All keywords", pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
