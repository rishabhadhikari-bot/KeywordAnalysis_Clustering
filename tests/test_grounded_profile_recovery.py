import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import app2 as app


class SavedGroundedProfileRecoveryTests(unittest.TestCase):
    def test_stale_grounded_profile_is_not_reused(self) -> None:
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(hours=app.GROUNDED_PROFILE_MAX_AGE_HOURS + 1)

        self.assertFalse(
            app.grounded_profile_timestamp_is_fresh(stale_time.isoformat(), now=now)
        )

    def test_running_loader_is_animated_accessible_and_escaped(self) -> None:
        loader_html = app.build_running_loader_html("Working <now>")

        self.assertIn("@keyframes related-loader-spin", loader_html)
        self.assertIn("role='status'", loader_html)
        self.assertIn("Working &lt;now&gt;", loader_html)
        self.assertNotIn("Working <now>", loader_html)

    def test_malformed_profile_is_refreshed_even_if_routes_can_be_recovered(self) -> None:
        profile = {
            "quality_warnings": [
                "The grounded research was not valid structured JSON."
            ]
        }

        with patch.object(app, "extract_relationship_routes", return_value=[object()]):
            needs_recovery = app.saved_grounded_profile_needs_recovery(
                keyword_query="Shreyas Iyer",
                grounded_research=profile,
            )

        self.assertTrue(needs_recovery)

    def test_malformed_profile_without_relationships_is_refreshed(self) -> None:
        profile = {
            "quality_warnings": [
                "The grounded research was not valid structured JSON."
            ]
        }

        with patch.object(app, "extract_relationship_routes", return_value=[]):
            needs_recovery = app.saved_grounded_profile_needs_recovery(
                keyword_query="Shreyas Iyer",
                grounded_research=profile,
            )

        self.assertTrue(needs_recovery)

    def test_failed_recovery_without_relationships_is_refreshed(self) -> None:
        profile = {
            "relationship_recovery_status": "failed",
            "quality_warnings": [],
        }

        with patch.object(app, "extract_relationship_routes", return_value=[]):
            needs_recovery = app.saved_grounded_profile_needs_recovery(
                keyword_query="Shreyas Iyer",
                grounded_research=profile,
            )

        self.assertTrue(needs_recovery)


class RelatedStoriesProcessTimingTests(unittest.TestCase):
    def test_primary_traffic_title_is_not_sent_to_vertex_research(self) -> None:
        title_summary = pd.DataFrame(
            [
                {"story_id": "primary", "page_title": "Israel update", "total_views": 10},
                {"story_id": "eligible", "page_title": "Regional update", "total_views": 5},
            ]
        )

        with (
            patch.object(app, "load_saved_grounded_profile", return_value=None),
            patch.object(
                app,
                "research_query_relationships_with_vertex",
                return_value={"research_text": "saved", "sources": []},
            ) as research,
            patch.object(app, "save_grounded_profile"),
            patch.object(app, "load_cached_generative_retrieval", return_value=None),
            patch.object(app, "extract_relationship_routes", return_value=[]),
            patch.object(
                app,
                "retrieve_relationship_candidates",
                return_value=(
                    [],
                    {
                        "mode": "bounded_retrieval_no_candidates",
                        "relationship_count": 0,
                        "route_count": 0,
                        "candidate_count": 0,
                        "fallback_reason": "",
                    },
                ),
            ),
            patch.object(app, "load_cached_candidate_evaluations", return_value={}),
        ):
            app.get_ai_related_candidate_pool.__wrapped__(
                keyword_query="Israel",
                match_type="All keywords",
                story_months=pd.DataFrame(),
                title_summary=title_summary,
                matched_titles=title_summary.iloc[[0]],
                direct_story_ids=("primary",),
            )

        self.assertEqual(research.call_args.kwargs["matched_titles"], [])

    def test_primary_traffic_title_is_removed_before_gemini_validation(self) -> None:
        title_summary = pd.DataFrame(
            [
                {"story_id": "legacy", "page_title": "Israel update", "total_views": 10},
                {
                    "story_id": "root-primary",
                    "page_title": "Hezbollah drones challenge Israeli defenses",
                    "total_views": 5,
                },
                {
                    "story_id": "eligible-related",
                    "page_title": "Regional defense development",
                    "total_views": 3,
                },
            ]
        )
        root_candidate = {
            **title_summary.iloc[1].to_dict(),
            "retrieval_evidence": [{"relationship_id": "R1"}],
        }
        eligible_candidate = {
            **title_summary.iloc[2].to_dict(),
            "retrieval_evidence": [{"relationship_id": "R2"}],
        }

        with (
            patch.object(
                app,
                "load_saved_grounded_profile",
                return_value=({"research_text": "saved"}, []),
            ),
            patch.object(app, "saved_grounded_profile_needs_recovery", return_value=False),
            patch.object(app, "load_cached_generative_retrieval", return_value=None),
            patch.object(app, "extract_relationship_routes", return_value=[]),
            patch.object(
                app,
                "retrieve_relationship_candidates",
                return_value=(
                    [root_candidate, eligible_candidate],
                    {
                        "mode": "relationship_embeddings",
                        "relationship_count": 1,
                        "route_count": 1,
                        "candidate_count": 1,
                        "fallback_reason": "",
                    },
                ),
            ) as retrieve,
            patch.object(app, "load_cached_candidate_evaluations", return_value={}),
            patch.object(
                app,
                "select_titles_generatively_with_vertex",
                return_value=[],
            ) as validate,
        ):
            result = app.get_ai_related_candidate_pool.__wrapped__(
                keyword_query="Israel",
                match_type="All keywords",
                story_months=pd.DataFrame(),
                title_summary=title_summary,
                matched_titles=title_summary.iloc[[0]],
                direct_story_ids=("legacy", "root-primary"),
            )

        self.assertTrue(result.empty)
        self.assertEqual(
            retrieve.call_args.kwargs["excluded_story_ids"],
            {"legacy", "root-primary"},
        )
        validated_story_ids = {
            str(row["story_id"])
            for row in validate.call_args.kwargs["candidate_titles"]
        }
        self.assertEqual(validated_story_ids, {"eligible-related"})

    def test_transient_research_failure_is_raised_instead_of_cached_as_empty(self) -> None:
        title_summary = pd.DataFrame(
            [
                {
                    "story_id": "story-1",
                    "page_title": "Shreyas Iyer direct story",
                    "total_views": 10,
                },
                {
                    "story_id": "story-2",
                    "page_title": "Potential related story",
                    "total_views": 5,
                },
            ]
        )

        with (
            patch.object(app, "load_saved_grounded_profile", return_value=None),
            patch.object(
                app,
                "research_query_relationships_with_vertex",
                side_effect=ValueError("temporary branch failure"),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Live grounded relationship research failed.*temporary branch failure",
            ):
                app.get_ai_related_candidate_pool.__wrapped__(
                    keyword_query="Shreyas Iyer",
                    match_type="All keywords",
                    story_months=pd.DataFrame(),
                    title_summary=title_summary,
                    matched_titles=title_summary.iloc[[0]],
                    direct_story_ids=("story-1",),
                )

    def test_duration_formatting_uses_readable_units(self) -> None:
        self.assertEqual(app.format_process_duration(0.125), "125 ms")
        self.assertEqual(app.format_process_duration(4.25), "4.2 s")
        self.assertEqual(app.format_process_duration(65.5), "1m 5.5s")

    def test_timing_component_highlights_and_escapes_slowest_step(self) -> None:
        timing_html = app.build_process_timings_html(
            [
                {"step": "Prepare <context>", "duration_seconds": 0.5},
                {"step": "Validate candidates", "duration_seconds": 7.25},
            ],
            total_elapsed_seconds=7.75,
        )

        self.assertIn("Most time: Validate candidates", timing_html)
        self.assertIn("SLOWEST", timing_html)
        self.assertIn("7.2 s", timing_html)
        self.assertIn("Prepare &lt;context&gt;", timing_html)
        self.assertNotIn("Prepare <context>", timing_html)

    def test_live_snapshot_includes_elapsed_active_phase(self) -> None:
        with patch.object(app.time, "perf_counter", return_value=15.0):
            timings = app.get_process_timing_snapshot(
                {
                    "phase": "Validate candidates with Gemini",
                    "phase_started_at": 10.0,
                    "process_timings": [
                        {"step": "Prepare title corpus", "duration_seconds": 2.0}
                    ],
                }
            )

        self.assertEqual(len(timings), 2)
        self.assertEqual(timings[-1]["duration_seconds"], 5.0)
        self.assertTrue(timings[-1]["is_active"])

    def test_candidate_flow_retains_completed_step_timings(self) -> None:
        title_summary = pd.DataFrame(
            [{"story_id": "story-1", "page_title": "A direct match"}]
        )
        progress_key = "process-timing-test"
        try:
            with (
                patch.object(
                    app,
                    "load_saved_grounded_profile",
                    return_value=({"research_text": "saved"}, []),
                ),
                patch.object(
                    app,
                    "saved_grounded_profile_needs_recovery",
                    return_value=False,
                ),
            ):
                result = app.get_ai_related_candidate_pool.__wrapped__(
                    keyword_query="direct",
                    match_type="Any keyword",
                    story_months=pd.DataFrame(),
                    title_summary=title_summary,
                    matched_titles=title_summary,
                    direct_story_ids=("story-1",),
                    _progress_key=progress_key,
                )
        finally:
            with app.RELATED_FLOW_PROGRESS_LOCK:
                app.RELATED_FLOW_PROGRESS.pop(progress_key, None)

        steps = [item["step"] for item in result.attrs["process_timings"]]
        self.assertEqual(
            steps,
            ["Prepare title corpus"],
        )
        self.assertGreaterEqual(result.attrs["total_elapsed_seconds"], 0.0)

    def test_zero_result_retains_local_rejection_audit_and_is_not_sweep_cached(self) -> None:
        title_summary = pd.DataFrame(
            [
                {"story_id": "direct", "page_title": "Iran direct", "total_views": 10},
                {
                    "story_id": "candidate",
                    "page_title": "Iranian general responds",
                    "total_views": 5,
                },
            ]
        )
        retrieval_candidate = {
            **title_summary.iloc[1].to_dict(),
            "retrieval_evidence": [
                {
                    "relationship_id": "R1",
                    "factual_bridge": "Iran has a verified relationship.",
                    "acceptance_condition": "The title expresses the relationship.",
                    "rejection_rule": "Reject unrelated titles.",
                    "can_retrieve_standalone": False,
                    "required_title_cues": ["Iran-US"],
                    "match_method": "embedding",
                    "similarity": 0.5,
                }
            ],
        }

        def reject_with_local_audit(**kwargs: object) -> list[dict[str, object]]:
            batch = kwargs["candidate_titles"]
            assert isinstance(batch, list)
            batch[0]["_selection_audit_reason"] = (
                "Local validation rejected Gemini's match: required relationship cue is absent."
            )
            callback = kwargs["batch_callback"]
            assert callable(callback)
            callback(batch, [], 1, 1)
            return []

        with (
            patch.object(
                app,
                "load_saved_grounded_profile",
                return_value=({"research_text": "{}"}, []),
            ),
            patch.object(app, "saved_grounded_profile_needs_recovery", return_value=False),
            patch.object(app, "load_cached_generative_retrieval", return_value=None),
            patch.object(app, "extract_relationship_routes", return_value=[]),
            patch.object(
                app,
                "retrieve_relationship_candidates",
                return_value=(
                    [retrieval_candidate],
                    {
                        "mode": "relationship_embeddings",
                        "relationship_count": 1,
                        "route_count": 1,
                        "candidate_count": 1,
                        "fallback_reason": "",
                    },
                ),
            ),
            patch.object(app, "load_cached_candidate_evaluations", return_value={}),
            patch.object(app, "select_titles_generatively_with_vertex", side_effect=reject_with_local_audit),
            patch.object(app, "save_candidate_evaluations"),
            patch.object(app, "save_generative_retrieval") as save_sweep,
        ):
            result = app.get_ai_related_candidate_pool.__wrapped__(
                keyword_query="Iran",
                match_type="All keywords",
                story_months=pd.DataFrame(),
                title_summary=title_summary,
                matched_titles=title_summary.iloc[[0]],
                direct_story_ids=("direct",),
            )

        self.assertTrue(result.empty)
        self.assertEqual(result.attrs["local_validation_rejected_count"], 1)
        self.assertEqual(result.attrs["gemini_excluded_count"], 0)
        save_sweep.assert_not_called()


class RefinedDashboardSchemaTests(unittest.TestCase):
    def test_refined_result_includes_dashboard_threshold_columns(self) -> None:
        title_summary = pd.DataFrame(
            [
                {
                    "story_id": "story-1",
                    "page_title": "Policy update",
                    "total_views": 150,
                }
            ]
        )
        story_months = pd.DataFrame(
            [
                {
                    "story_id": "story-1",
                    "month": pd.Timestamp("2026-01-01"),
                    "views": 100,
                    "above_monthly_threshold": True,
                },
                {
                    "story_id": "story-1",
                    "month": pd.Timestamp("2026-02-01"),
                    "views": 50,
                    "above_monthly_threshold": False,
                },
            ]
        )

        refined_result = app.build_refined_search_result(
            hits=[
                {
                    "story_id": "story-1",
                    "refined_match_tier": "Root variant",
                    "opensearch_score": 3.0,
                    "matched_queries": ["root_variant"],
                }
            ],
            total_hits=1,
            story_months=story_months,
            title_summary=title_summary,
        )
        analysis_result = app.build_refined_primary_analysis_result(refined_result)
        matched_titles = analysis_result["matched_title_summary"]

        self.assertEqual(matched_titles.iloc[0]["threshold_matched_months"], 1)
        self.assertEqual(matched_titles.iloc[0]["views_above_threshold"], 100)

        displayed = app.add_high_traffic_month_display(
            matched_titles,
            analysis_result["matched_story_months"],
        )
        self.assertEqual(displayed.iloc[0]["threshold_matched_months"], "1 (Jan 2026)")

if __name__ == "__main__":
    unittest.main()
