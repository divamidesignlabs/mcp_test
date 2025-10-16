# Market Research MCP (Deployed)

This repository now exposes only the Market & Web Intelligence capability via a deployed Model Context Protocol (MCP) server. All file conversion / S3 archival features have been removed from the active deployment and are no longer documented here.

## Purpose
Provide fast, structured market and competitive intelligence by combining targeted open‑web retrieval (DuckDuckGo via DDGS) with LLM‑based synthesis validated through Pydantic schemas.

## Available MCP Tools
| Tool | Purpose | Returns |
|------|---------|---------|
| `quick_search` | Lightweight DuckDuckGo query returning raw snippets | `{ query, results:[{title, href, body}], count }` |
| `market_research` | Full workflow: search → context assembly → LLM analysis (summary, trends, competitors, opportunities, confidence, citations) | JSON matching `MarketResearchResult` schema |

## Output Schema (`market_research`)
| Field | Type | Notes |
|-------|------|-------|
| `query` | string | Original user query |
| `summary` | string | 2–4 sentence synthesized overview |
| `top_trends` | list[str] | 3–5 specific, evidence‑grounded trends |
| `competitors` | list[str] | 3–5 principal players (companies/products) |
| `opportunities` | list[str] | 3 actionable opportunity statements |
| `confidence` | float (0–1) | Calibrated subjective confidence |
| `citations` | list[SearchHit] | Each: `title, href, body` snippet |

## Model Selection Logic
1. If `GEMINI_API_KEY` present → use `GEMINI_MODEL` (e.g. `gemini-1.5-pro`).
2. Else if `OPENAI_API_KEY` present → use `OPENAI_MODEL` (e.g. `gpt-4.1-mini`).
3. Else → raise configuration error (returned in response envelope for tool calls made without startup validation).

## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | One of Gemini/OpenAI | Preferred model API key |
| `GEMINI_MODEL` | If Gemini used | Model name (defaults internally if omitted) |
| `OPENAI_API_KEY` | One of Gemini/OpenAI | Fallback model API key |
| `OPENAI_MODEL` | If OpenAI used | Model name (defaults internally if omitted) |

## Error Handling
| Situation | Behavior |
|-----------|----------|
| No results from search | Returns `{ error, query, suggestions }` |
| Missing all API keys | Raises/returns configuration error message |
| LLM schema deviation | Falls back to raw text in `summary`, empty structured lists, includes original citations, adds `note` |
| Async loop conflict | Returns error envelope with `type=RuntimeError` |

## Design Notes
- Retrieval and synthesis cleanly separated for inspectability.
- Defensive parsing ensures partial value even on model drift.
- Citations preserve source transparency for downstream verification.
- Minimal external dependencies beyond DDGS + chosen model provider.

## Roadmap (Focused)
- Add rate limiting and retry with jitter around search
- Confidence calibration heuristics (e.g., penalize low citation diversity)
- Optional per‑citation relevance scoring
- Increase robustness for multilingual queries
- Streaming incremental analysis (chunked JSON patches)

---
Internal deployment/run details intentionally omitted (service already live).