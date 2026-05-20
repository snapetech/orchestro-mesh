"""Server-rendered HTML dashboard for the mesh gateway.

Mounted at ``/dashboard`` by ``gateway.create_gateway_app``. Auth uses the
same requester dependency as the rest of the gateway, with HTTP Basic
support so browsers can prompt cleanly (paste the token as the password).

No JS framework, no external assets, no client-side templating — one big
HTML string. Auto-refreshes via meta tag.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Iterable

from orchestro_mesh.config import MeshConfig
from orchestro_mesh.models import FeedbackRating, JobRecord, NodeInventory
from orchestro_mesh.store import MeshStore

_CSS = """
:root {
    --bg: #0f1117; --fg: #e6e6e6; --muted: #8a8f99;
    --panel: #181b22; --border: #262a33; --accent: #6ea8fe;
    --good: #79e07b; --warn: #ffce6c; --bad: #ff7878;
    --font: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
body { background: var(--bg); color: var(--fg); font-family: var(--font);
       margin: 0; padding: 24px; font-size: 13px; line-height: 1.5; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 13px; margin: 24px 0 8px; color: var(--muted);
     text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }
.meta { color: var(--muted); margin-bottom: 12px; }
.panel { background: var(--panel); border: 1px solid var(--border);
         border-radius: 6px; padding: 0; margin-bottom: 16px; overflow: hidden; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 500; font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.05em; background: #14171d; }
tr:last-child td { border-bottom: 0; }
.status-online { color: var(--good); }
.status-degraded, .status-draining { color: var(--warn); }
.status-offline { color: var(--bad); }
.job-completed { color: var(--good); }
.job-error { color: var(--bad); }
.job-started { color: var(--warn); }
.rating-bar { display: inline-block; width: 48px; height: 8px; background: var(--border);
              border-radius: 2px; overflow: hidden; vertical-align: middle; margin-right: 6px; }
.rating-bar-fill { height: 100%; background: var(--good); }
.empty { padding: 16px; color: var(--muted); text-align: center; }
.id { color: var(--muted); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
"""


def _humanize_age(ts: datetime | None, now: datetime) -> str:
    if ts is None:
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta_s = (now - ts).total_seconds()
    if delta_s < 0:
        delta_s = 0
    if delta_s < 60:
        return f"{int(delta_s)}s ago"
    if delta_s < 3600:
        return f"{int(delta_s / 60)}m ago"
    if delta_s < 86400:
        return f"{int(delta_s / 3600)}h ago"
    return f"{int(delta_s / 86400)}d ago"


def _short_id(value: str | None, length: int = 8) -> str:
    if not value:
        return "—"
    return value[:length]


def _status_class(status: str) -> str:
    return f"status-{status.lower()}"


def _job_class(status: str) -> str:
    return f"job-{status.lower()}"


def _render_nodes(nodes: Iterable[NodeInventory], now: datetime, ttl: int | None) -> str:
    rows = []
    for node in nodes:
        age = _humanize_age(node.last_seen, now)
        ttl_warn = ""
        if ttl and node.last_seen and (now - node.last_seen.replace(tzinfo=node.last_seen.tzinfo or timezone.utc)).total_seconds() > ttl:
            ttl_warn = ' <span class="status-bad">stale</span>'
        rows.append(
            f"<tr>"
            f"<td>{html.escape(node.node_id)}</td>"
            f"<td>{html.escape(node.owner)}</td>"
            f"<td>{html.escape(node.trust_domain)}</td>"
            f'<td class="{_status_class(node.status.value)}">{node.status.value}</td>'
            f'<td class="num">{node.current_jobs}</td>'
            f'<td class="num">{node.queue_depth}</td>'
            f"<td>{age}{ttl_warn}</td>"
            f'<td class="num">{len(node.models)}</td>'
            f"</tr>"
        )
    if not rows:
        return '<div class="panel"><div class="empty">no nodes registered</div></div>'
    return (
        '<div class="panel"><table>'
        "<tr><th>node</th><th>owner</th><th>trust</th><th>status</th>"
        '<th class="num">jobs</th><th class="num">queue</th>'
        '<th>last seen</th><th class="num">models</th></tr>'
        + "".join(rows)
        + "</table></div>"
    )


def _render_models(nodes: Iterable[NodeInventory]) -> str:
    rows = []
    for node in nodes:
        for model in node.models:
            ctx = f"{model.context_window:,}" if model.context_window else "—"
            tps = model.benchmark.best_decode_tps()
            tps_str = f"{tps:.1f}" if tps else "—"
            conf = f"{model.benchmark.confidence:.2f}"
            rows.append(
                f"<tr>"
                f"<td>{html.escape(node.node_id)}</td>"
                f"<td>{html.escape(model.id)}</td>"
                f'<td class="{_status_class(model.state.value)}">{model.state.value}</td>'
                f'<td class="num">{ctx}</td>'
                f'<td class="num">{tps_str}</td>'
                f'<td class="num">{conf}</td>'
                f"<td>{html.escape(', '.join(t.value for t in model.task_classes))}</td>"
                f"</tr>"
            )
    if not rows:
        return '<div class="panel"><div class="empty">no models advertised</div></div>'
    return (
        '<div class="panel"><table>'
        "<tr><th>node</th><th>model</th><th>state</th>"
        '<th class="num">ctx</th><th class="num">dec tps</th>'
        '<th class="num">conf</th><th>task classes</th></tr>'
        + "".join(rows)
        + "</table></div>"
    )


def _render_jobs(jobs: list[JobRecord], now: datetime) -> str:
    if not jobs:
        return '<div class="panel"><div class="empty">no jobs yet</div></div>'
    rows = []
    for job in jobs:
        age = _humanize_age(job.created_at, now)
        ms = f"{job.total_ms:.0f}" if job.total_ms else "—"
        rows.append(
            f"<tr>"
            f'<td class="id">{html.escape(_short_id(job.id))}</td>'
            f"<td>{html.escape(job.requester)}</td>"
            f'<td class="{_job_class(job.status)}">{html.escape(job.status)}</td>'
            f"<td>{html.escape(job.node_id or '—')}</td>"
            f"<td>{html.escape(job.task_class.value)}</td>"
            f'<td class="num">{job.input_tokens}</td>'
            f'<td class="num">{job.output_tokens}</td>'
            f'<td class="num">{ms}</td>'
            f"<td>{age}</td>"
            f"</tr>"
        )
    return (
        '<div class="panel"><table>'
        "<tr><th>id</th><th>requester</th><th>status</th><th>node</th>"
        "<th>task</th>"
        '<th class="num">in</th><th class="num">out</th>'
        '<th class="num">ms</th><th>age</th></tr>'
        + "".join(rows)
        + "</table></div>"
    )


def _render_usage(totals: dict[str, float], quotas: dict[str, float], cpk: float) -> str:
    keys = sorted(set(totals) | set(quotas))
    if not keys:
        return '<div class="panel"><div class="empty">no usage yet</div></div>'
    rows = []
    for key in keys:
        used = totals.get(key, 0.0)
        quota = quotas.get(key)
        quota_str = f"{quota:.3f}" if quota is not None else "—"
        pct = f"{(used / quota * 100):.1f}%" if quota else ""
        rows.append(
            f"<tr>"
            f"<td>{html.escape(key)}</td>"
            f'<td class="num">{used:.4f}</td>'
            f'<td class="num">{quota_str}</td>'
            f'<td class="num">{pct}</td>'
            f"</tr>"
        )
    return (
        '<div class="panel"><table>'
        f"<tr><th>requester</th>"
        f'<th class="num">credits used</th>'
        f'<th class="num">quota</th>'
        f'<th class="num">% used</th></tr>'
        + "".join(rows)
        + "</table></div>"
        + f'<div class="meta">credits_per_1k_tokens = {cpk}</div>'
    )


def _render_feedback(items: list[FeedbackRating], now: datetime) -> str:
    if not items:
        return '<div class="panel"><div class="empty">no feedback yet</div></div>'
    rows = []
    for item in items:
        fill_pct = int(max(0.0, min(1.0, item.rating)) * 100)
        bar = (
            f'<span class="rating-bar"><span class="rating-bar-fill" '
            f'style="width:{fill_pct}%"></span></span>'
            f"{item.rating:.2f}"
        )
        age = _humanize_age(item.created_at, now)
        notes = html.escape((item.notes or "")[:80])
        rows.append(
            f"<tr>"
            f"<td>{bar}</td>"
            f"<td>{html.escape(item.verifier)}</td>"
            f"<td>{html.escape(item.node_id or '—')}</td>"
            f"<td>{html.escape(item.model_id or '—')}</td>"
            f"<td>{html.escape(item.requester or '—')}</td>"
            f"<td>{notes}</td>"
            f"<td>{age}</td>"
            f"</tr>"
        )
    return (
        '<div class="panel"><table>'
        "<tr><th>rating</th><th>verifier</th><th>node</th>"
        "<th>model</th><th>requester</th><th>notes</th><th>age</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def render_dashboard(
    *,
    store: MeshStore,
    config: MeshConfig,
    refresh_s: int = 5,
) -> str:
    now = datetime.now(tz=timezone.utc)
    nodes = store.list_nodes()
    jobs = store.recent_jobs(limit=20)
    feedback = store.recent_feedback(limit=10)
    totals = store.ledger_totals()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh_s}">
<title>orchestro-mesh dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Orchestro Mesh</h1>
<div class="meta">{html.escape(now.isoformat(timespec='seconds'))} · refresh {refresh_s}s · {len(nodes)} node(s)</div>

<h2>Nodes</h2>
{_render_nodes(nodes, now, config.node_ttl_seconds)}

<h2>Models</h2>
{_render_models(nodes)}

<h2>Recent jobs</h2>
{_render_jobs(jobs, now)}

<h2>Usage</h2>
{_render_usage(totals, config.quota_credits, config.credits_per_1k_tokens)}

<h2>Recent feedback</h2>
{_render_feedback(feedback, now)}
</body>
</html>"""
