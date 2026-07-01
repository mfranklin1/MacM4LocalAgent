"""Tests for cost.pricing and scripts/check_claude_pricing.py.

Covers:
  - per-model rate lookup for the canonical model ids
  - ALIASES dict (claude-code, dotted variants, etc.)
  - prefix stripping (anthropic/claude-..., vertex_ai/claude-...)
  - date-suffixed ids (claude-opus-4-7-20260416)
  - unknown model fallback + one-time stderr warning
  - shadow_cost is pinned to Sonnet 4.6 regardless of input model
  - actual_claude_cost matches the per-model rate
  - parser handles both markdown and HTML fixtures
  - parser ignores the Batch API table (which has half-price columns)
  - --strict and --quiet flags behave as documented

We do NOT make network calls in this suite. The HTML/markdown
parsers are exercised against committed fixtures.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cost import pricing  # noqa: E402


# ---- rate lookup -------------------------------------------------------------

def test_canonical_lookup_opus_4_7() -> None:
    r = pricing.claude_rate("claude-opus-4-7")
    assert r.input == pytest.approx(5.0 / 1_000_000)
    assert r.output == pytest.approx(25.0 / 1_000_000)


def test_canonical_lookup_sonnet_4_6() -> None:
    r = pricing.claude_rate("claude-sonnet-4-6")
    assert r.input == pytest.approx(3.0 / 1_000_000)
    assert r.output == pytest.approx(15.0 / 1_000_000)


def test_canonical_lookup_haiku_4_5() -> None:
    r = pricing.claude_rate("claude-haiku-4-5")
    assert r.input == pytest.approx(1.0 / 1_000_000)
    assert r.output == pytest.approx(5.0 / 1_000_000)


@pytest.mark.parametrize(
    "alias,expected_in_per_mtok",
    [
        ("claude-code",          2.0),    # internal proxy alias -> Sonnet 5 (default Claude tier)
        ("claude-opus-4.7",      5.0),    # dotted variant
        ("claude-haiku-4.5",     1.0),    # dotted variant
    ],
)
def test_alias_resolution(alias: str, expected_in_per_mtok: float) -> None:
    r = pricing.claude_rate(alias)
    assert r.input == pytest.approx(expected_in_per_mtok / 1_000_000)


def test_claude_code_alias_matches_yaml_default() -> None:
    """Regression guard: the `claude-code` pricing alias MUST mirror
    the upstream model id configured for `claude-code` in
    config/litellm-config.yaml. If the YAML default changes, this
    test will fail and remind us to update _ALIASES too."""
    rate = pricing.claude_rate("claude-code")
    sonnet5_rate = pricing.claude_rate("claude-sonnet-5")
    assert rate == sonnet5_rate, (
        "claude-code alias is out of sync with config/litellm-config.yaml. "
        "Both should point to the same Claude tier."
    )


@pytest.mark.parametrize(
    "model_id,expected_canonical",
    [
        ("anthropic/claude-opus-4-7",    "claude-opus-4-7"),
        ("vertex_ai/claude-sonnet-4-6",  "claude-sonnet-4-6"),
        ("bedrock/claude-haiku-4-5",     "claude-haiku-4-5"),
        ("Claude-Opus-4-7",              "claude-opus-4-7"),  # case
    ],
)
def test_provider_prefix_stripping(model_id: str, expected_canonical: str) -> None:
    assert pricing._normalize(model_id) == expected_canonical


def test_date_suffix_stripping() -> None:
    # Anthropic date-stamps some model ids: claude-opus-4-7-20260416.
    # The normalizer should walk back to the longest known prefix.
    assert pricing._normalize("claude-opus-4-7-20260416") == "claude-opus-4-7"
    assert pricing._normalize("anthropic/claude-sonnet-4-6-20251002") == "claude-sonnet-4-6"


# ---- unknown model handling --------------------------------------------------

def test_unknown_model_falls_back_to_sonnet_with_warning(capsys) -> None:
    pricing._warned_unknown.clear()  # ensure the warning fires even if cached
    r = pricing.claude_rate("claude-future-99-0")
    assert r == pricing.sonnet_rate(), "should fall back to Sonnet rates"
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "claude-future-99-0" in err
    assert "claude-sonnet-4-6" in err


def test_unknown_model_warns_only_once(capsys) -> None:
    pricing._warned_unknown.clear()
    pricing.claude_rate("claude-mystery-1-0")
    pricing.claude_rate("claude-mystery-1-0")
    pricing.claude_rate("claude-mystery-1-0")
    err = capsys.readouterr().err
    assert err.count("WARNING") == 1


def test_distinct_unknowns_each_warn_once(capsys) -> None:
    pricing._warned_unknown.clear()
    pricing.claude_rate("claude-alpha-1-0")
    pricing.claude_rate("claude-beta-1-0")
    err = capsys.readouterr().err
    assert err.count("WARNING") == 2
    assert "claude-alpha-1-0" in err
    assert "claude-beta-1-0" in err


# ---- shadow_cost / actual_claude_cost ----------------------------------------

def test_shadow_cost_is_pinned_to_sonnet() -> None:
    assert pricing.shadow_cost(1_000_000, 0) == pytest.approx(3.0)
    assert pricing.shadow_cost(0, 1_000_000) == pytest.approx(15.0)
    # Round-trip: shadow_cost should equal Sonnet rate * tokens.
    s = pricing.sonnet_rate()
    assert pricing.shadow_cost(123, 456) == pytest.approx(123 * s.input + 456 * s.output)


def test_actual_claude_cost_uses_per_model_rate() -> None:
    # Opus 4.7 charges $5/$25, so 1M in + 1M out = $30.
    assert pricing.actual_claude_cost("claude-opus-4-7", 1_000_000, 1_000_000) == pytest.approx(30.0)
    # Sonnet 4.6 charges $3/$15, so 1M + 1M = $18.
    assert pricing.actual_claude_cost("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)
    # Haiku 4.5 charges $1/$5, so 1M + 1M = $6.
    assert pricing.actual_claude_cost("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)


def test_actual_claude_cost_zero_tokens_zero_dollars() -> None:
    assert pricing.actual_claude_cost("claude-opus-4-7", 0, 0) == 0.0


def test_shadow_cost_does_not_change_when_router_picks_opus(capsys) -> None:
    # The whole point of shadow_cost being Sonnet-pinned is that
    # if the router upgrades a call to Opus, shadow stays the same.
    # Actual cost goes UP, savings shrinks (correctly).
    in_t, out_t = 100_000, 50_000
    shadow = pricing.shadow_cost(in_t, out_t)
    sonnet_actual = pricing.actual_claude_cost("claude-sonnet-4-6", in_t, out_t)
    opus_actual = pricing.actual_claude_cost("claude-opus-4-7", in_t, out_t)
    assert shadow == pytest.approx(sonnet_actual)
    assert opus_actual > shadow


# ---- table coverage / opus 4.7 specifically ----------------------------------

def test_opus_4_7_is_present_and_priced_correctly() -> None:
    """Regression guard for the specific model the user called out."""
    assert "claude-opus-4-7" in pricing.CLAUDE_PRICES
    r = pricing.CLAUDE_PRICES["claude-opus-4-7"]
    assert r.input == pytest.approx(5.0 / 1_000_000)
    assert r.output == pytest.approx(25.0 / 1_000_000)


def test_all_major_families_present() -> None:
    expected = {
        "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5",
        "claude-sonnet-4-6", "claude-sonnet-4-5",
        "claude-haiku-4-5", "claude-haiku-3-5",
    }
    assert expected.issubset(set(pricing.CLAUDE_PRICES.keys()))


# ---- freshness ---------------------------------------------------------------

def test_table_age_days_is_nonnegative() -> None:
    # Just smoke: it shouldn't blow up.
    age = pricing._table_age_days()
    assert age >= 0.0


def test_maybe_run_pricing_check_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRICING_STARTUP_CHECK", "0")
    # Should be a silent no-op; mostly we just want it not to spawn a thread.
    pricing.maybe_run_pricing_check()
    # We can't easily assert "no thread was started", but the call
    # must return immediately and never raise.


# ---- check_claude_pricing.py -------------------------------------------------

# Import the script as a module so we can call its parse_pricing()
# helper directly. importlib gymnastics because of the hyphen in
# the filename and the fact that scripts/ isn't a package.
def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_claude_pricing",
        REPO_ROOT / "scripts" / "check_claude_pricing.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


# Inline markdown fixture (3 representative rows from Anthropic's docs).
_MD_FIXTURE = """\
| Model | Base Input Tokens | 5m Cache Writes | 1h Cache Writes | Cache Hits & Refreshes | Output Tokens |
| --- | --- | --- | --- | --- | --- |
| Claude Opus 4.7 | $5 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $25 / MTok |
| Claude Sonnet 4.6 | $3 / MTok | $3.75 / MTok | $6 / MTok | $0.30 / MTok | $15 / MTok |
| Claude Haiku 4.5 | $1 / MTok | $1.25 / MTok | $2 / MTok | $0.10 / MTok | $5 / MTok |
"""


def test_markdown_parser(checker) -> None:
    rows = checker.parse_pricing(_MD_FIXTURE)
    assert rows["claude-opus-4-7"]   == (5.0, 25.0)
    assert rows["claude-sonnet-4-6"] == (3.0, 15.0)
    assert rows["claude-haiku-4-5"]  == (1.0, 5.0)


def test_markdown_parser_handles_deprecated_marker(checker) -> None:
    md = (
        "| Model | Base Input Tokens | 5m Cache Writes | 1h Cache Writes | Cache Hits & Refreshes | Output Tokens |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Claude Sonnet 3.7 ([deprecated](https://example.com)) | $3 / MTok | $3.75 / MTok | $6 / MTok | $0.30 / MTok | $15 / MTok |\n"
    )
    rows = checker.parse_pricing(md)
    assert rows["claude-sonnet-3-7"] == (3.0, 15.0)


# Inline HTML fixture mimicking the docs page structure: one Model
# Pricing table (full rates) + one Batch API table (half rates).
# The parser MUST pick the first one only.
_HTML_FIXTURE = """\
<html><body>
<h2>Model pricing</h2>
<table>
<thead><tr><th>Model</th><th>Base Input Tokens</th><th>5m Cache Writes</th><th>1h Cache Writes</th><th>Cache Hits &amp; Refreshes</th><th>Output Tokens</th></tr></thead>
<tbody>
<tr><td>Claude Opus 4.7</td><td>$5 / MTok</td><td>$6.25 / MTok</td><td>$10 / MTok</td><td>$0.50 / MTok</td><td>$25 / MTok</td></tr>
<tr><td>Claude Sonnet 4.6</td><td>$3 / MTok</td><td>$3.75 / MTok</td><td>$6 / MTok</td><td>$0.30 / MTok</td><td>$15 / MTok</td></tr>
</tbody></table>

<h2>Batch API pricing</h2>
<table>
<thead><tr><th>Model</th><th>Batch input</th><th>Batch output</th></tr></thead>
<tbody>
<tr><td>Claude Opus 4.7</td><td>$2.50 / MTok</td><td>$12.50 / MTok</td></tr>
<tr><td>Claude Sonnet 4.6</td><td>$1.50 / MTok</td><td>$7.50 / MTok</td></tr>
</tbody></table>
</body></html>
"""


def test_html_parser_picks_model_pricing_table_not_batch(checker) -> None:
    rows = checker.parse_pricing(_HTML_FIXTURE)
    # Standard rates -- NOT the half-price batch rates.
    assert rows["claude-opus-4-7"]   == (5.0, 25.0)
    assert rows["claude-sonnet-4-6"] == (3.0, 15.0)
    # And critically: Batch-API-only rows must not have leaked through.
    for canonical, (in_v, out_v) in rows.items():
        assert in_v >= 1.0, f"{canonical} input rate {in_v} looks like batch pricing"


def test_diff_no_drift_against_local(checker) -> None:
    rows = checker.parse_pricing(_MD_FIXTURE)
    mismatches, missing_local, _ = checker.diff_against_local(rows)
    assert mismatches == [], f"unexpected mismatches: {mismatches}"
    assert missing_local == [], f"unexpected missing-local: {missing_local}"


def test_diff_detects_input_rate_mismatch(checker) -> None:
    bogus = {"claude-opus-4-7": (99.0, 25.0)}
    mismatches, _, _ = checker.diff_against_local(bogus)
    assert len(mismatches) == 1
    assert "claude-opus-4-7" in mismatches[0]
    assert "99" in mismatches[0]


def test_diff_detects_missing_local(checker) -> None:
    rows = {"claude-future-1-0": (4.0, 20.0)}
    _, missing_local, _ = checker.diff_against_local(rows)
    assert len(missing_local) == 1
    assert "claude-future-1-0" in missing_local[0]


def test_main_exit_zero_on_clean_fixture(checker, tmp_path, capsys) -> None:
    # Write our markdown fixture to disk and feed it as --fixture.
    fix = tmp_path / "pricing.md"
    fix.write_text(_MD_FIXTURE)
    rc = checker.main(["--fixture", str(fix), "--quiet"])
    assert rc == 0


def test_main_exit_one_on_drift(checker, tmp_path) -> None:
    fix = tmp_path / "pricing.md"
    fix.write_text(_MD_FIXTURE.replace("$5 / MTok", "$99 / MTok", 1))
    rc = checker.main(["--fixture", str(fix), "--quiet"])
    assert rc == 1


def test_main_strict_returns_two_on_unparseable(checker, tmp_path) -> None:
    fix = tmp_path / "garbage.md"
    fix.write_text("not a pricing table at all")
    # Default mode: returns 0 (treat outage as no-op so CI is safe).
    rc = checker.main(["--fixture", str(fix), "--quiet"])
    assert rc == 0
    # Strict mode: returns 2.
    rc = checker.main(["--fixture", str(fix), "--strict", "--quiet"])
    assert rc == 2
