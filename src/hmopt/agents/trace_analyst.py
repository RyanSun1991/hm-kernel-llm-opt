"""Trace Analyst agent."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hmopt.agents.safety import SafetyGuard
from hmopt.analysis.runtime.hotspot import HotspotCandidate
from hmopt.core.llm import ChatMessage, LLMClient

logger = logging.getLogger(__name__)


class TraceAnalystAgent:
    def __init__(self, llm: LLMClient, safety: SafetyGuard):
        self.llm = llm
        self.safety = safety
        self.default_system_prompt = "Trace analyst focusing on HM kernel perf."
        self.default_prompt_template = (
            "You are the Trace Analyst. Summarize performance symptoms and hotspot classes.\n"
            "Metrics: {metrics}\n"
            "Hotspots:\n"
            "{hotspots}\n"
            "{code_context_block}"
            "{insight_block}"
        )

    def analyze(
        self,
        metrics: Mapping[str, float],
        hotspots: Iterable[HotspotCandidate],
        trace_insights: Sequence[Mapping[str, object]] | None = None,
        *,
        include_code_context: bool = False,
        code_root: Path | None = None,
        max_code_lines: int = 120,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        hotspot_list = list(hotspots)
        logger.info(
            "TraceAnalyst analyzing metrics=%d hotspots=%d flamegraphs=%d",
            len(metrics),
            len(hotspot_list),
            len(trace_insights or []),
        )
        responses: list[str] = []
        for hotspot in hotspot_list or [None]:
            prompt = self._build_prompt(
                metrics,
                [hotspot] if hotspot else [],
                trace_insights,
                include_code_context=include_code_context,
                code_root=code_root,
                max_code_lines=max_code_lines,
                prompt_template=prompt_template,
            )
            messages = self._build_messages(prompt, system_prompt=system_prompt)
            responses.append(self.llm.chat(messages))
        return "\n\n".join([r for r in responses if r])

    def _build_prompt(
        self,
        metrics: Mapping[str, float],
        hotspots: Iterable[HotspotCandidate],
        trace_insights: Sequence[Mapping[str, object]] | None,
        *,
        include_code_context: bool,
        code_root: Path | None,
        max_code_lines: int,
        prompt_template: str | None,
    ) -> str:
        hotspot_list = list(hotspots)
        hotspot_lines = [self._format_hotspot_line(h) for h in hotspot_list]
        metric_lines = [f"{k}={v}" for k, v in metrics.items()]
        insight_lines = self._format_insight_lines(trace_insights)
        metrics_text = ", ".join(metric_lines)
        hotspots_text = "\n".join(hotspot_lines)
        code_context_text = ""
        if include_code_context:
            context_lines = self._format_hotspot_code_context(
                hotspot_list,
                code_root=code_root or Path.cwd(),
                max_code_lines=max_code_lines,
            )
            if context_lines:
                code_context_text = "Hotspot code context:\n" + "\n".join(context_lines) + "\n"
        insight_text = ""
        if insight_lines:
            insight_text = (
                "Flamegraph insights (compare patterns across files, highlight per-thread differences):\n"
                + "\n".join(insight_lines)
                + "\n"
            )
        call_stack_lines = self._format_hotspot_call_stacks(hotspot_list)
        if call_stack_lines:
            insight_text += "Hotspot call stacks (caller->callee and callee->caller):\n"
            insight_text += "\n".join(call_stack_lines) + "\n"
        template = prompt_template or self.default_prompt_template
        return template.format(
            metrics=metrics_text,
            hotspots=hotspots_text,
            code_context_block=code_context_text,
            insight_block=insight_text,
        ).rstrip()

    def _build_messages(self, prompt: str, *, system_prompt: str | None = None) -> list[ChatMessage]:
        system_content = system_prompt or self.default_system_prompt
        return [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=self.safety.redact(prompt)),
        ]

    @staticmethod
    def _format_hotspot_line(hotspot: HotspotCandidate) -> str:
        location = ""
        if hotspot.file_path:
            location = f" @ {hotspot.file_path}"
            if hotspot.line_start is not None:
                line_end = hotspot.line_end or hotspot.line_start
                location += f":{hotspot.line_start}-{line_end}"
        return f"- {hotspot.symbol}: score={hotspot.score:.2f}{location}"

    @staticmethod
    def _format_insight_lines(
        trace_insights: Sequence[Mapping[str, object]] | None,
    ) -> list[str]:
        insight_lines: list[str] = []
        if trace_insights:
            for idx, insight in enumerate(trace_insights, start=1):
                source = insight.get("source_path") or insight.get("source") or "flamegraph"
                total = insight.get("event_total", 0)
                top_symbols = insight.get("top_symbols", [])
                top_threads = insight.get("top_symbols_per_thread", [])
                insight_lines.append(f"Flamegraph {idx} ({source}) total_events={total}")
                for symbol in top_symbols[:5]:
                    name = symbol.get("name") or symbol.get("symbol")
                    weight = symbol.get("weight", 0.0)
                    ratio = symbol.get("ratio")
                    ratio_text = f" ({ratio:.2%})" if isinstance(ratio, float) else ""
                    insight_lines.append(f"  - top_symbol {name}: {weight:.2f}{ratio_text}")
                for thread in top_threads[:5]:
                    thread_name = thread.get("name") or thread.get("tid")
                    insight_lines.append(f"  - thread {thread_name}:")
                    for symbol in (thread.get("top_symbols") or [])[:5]:
                        name = symbol.get("name") or symbol.get("symbol")
                        weight = symbol.get("weight", 0.0)
                        ratio = symbol.get("ratio")
                        ratio_text = f" ({ratio:.2%})" if isinstance(ratio, float) else ""
                        insight_lines.append(f"    - {name}: {weight:.2f}{ratio_text}")
        return insight_lines

    @staticmethod
    def _format_hotspot_code_context(
        hotspots: Sequence[HotspotCandidate],
        *,
        code_root: Path,
        max_code_lines: int,
    ) -> list[str]:
        context_lines: list[str] = []
        for hotspot in hotspots:
            if not hotspot.file_path or hotspot.line_start is None:
                continue
            file_path = code_root / hotspot.file_path
            if not file_path.exists():
                context_lines.append(f"- {hotspot.symbol}: {hotspot.file_path} (file not found)")
                continue
            start_line = max(hotspot.line_start, 1)
            end_line = hotspot.line_end or hotspot.line_start
            if end_line < start_line:
                end_line = start_line
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                context_lines.append(
                    f"- {hotspot.symbol}: {hotspot.file_path} (read failed: {exc})"
                )
                continue
            excerpt = lines[start_line - 1 : end_line]
            if len(excerpt) > max_code_lines:
                excerpt = excerpt[:max_code_lines]
                context_lines.append(
                    f"- {hotspot.symbol}: {hotspot.file_path}:{start_line}-{start_line + max_code_lines - 1} (truncated)"
                )
            else:
                context_lines.append(
                    f"- {hotspot.symbol}: {hotspot.file_path}:{start_line}-{end_line}"
                )
            for offset, line in enumerate(excerpt, start=start_line):
                context_lines.append(f"    {offset:>5} | {line}")
        return context_lines

    @staticmethod
    def _format_hotspot_call_stacks(hotspots: Sequence[HotspotCandidate]) -> list[str]:
        lines: list[str] = []
        for hotspot in hotspots:
            call_stacks = hotspot.call_stacks or []
            if not call_stacks:
                continue
            lines.append(f"- {hotspot.symbol}:")
            for stack in call_stacks[:8]:
                if not isinstance(stack, dict):
                    continue
                path = " -> ".join(stack.get("stack", []))
                direction = stack.get("direction", "call")
                events = stack.get("self_events", 0)
                thread_id = stack.get("thread_id")
                thread_text = f" tid={thread_id}" if thread_id is not None else ""
                lines.append(f"  ({direction}){thread_text} {path} (events: {events:.0f})")
        return lines

    @staticmethod
    def hotspots_to_json(hotspots: Iterable[HotspotCandidate]) -> list[dict[str, object]]:
        return [
            {
                "symbol": hotspot.symbol,
                "file_path": hotspot.file_path,
                "line_start": hotspot.line_start,
                "line_end": hotspot.line_end,
                "score": hotspot.score,
                "evidence_artifacts": hotspot.evidence_artifacts,
            }
            for hotspot in hotspots
        ]


def _load_json_value(raw: str) -> object:
    raw = raw.strip()
    candidate_path = Path(raw)
    if candidate_path.exists():
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    return json.loads(raw)


def _load_hotspots(raw: str) -> list[HotspotCandidate]:
    payload = _load_json_value(raw)
    if not isinstance(payload, list):
        raise ValueError("hotspots payload must be a JSON list")
    hotspots: list[HotspotCandidate] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("hotspot entry must be an object")
        hotspots.append(
            HotspotCandidate(
                symbol=str(entry.get("symbol")),
                file_path=entry.get("file_path"),
                line_start=entry.get("line_start"),
                line_end=entry.get("line_end"),
                score=float(entry.get("score", 0.0)),
                evidence_artifacts=entry.get("evidence_artifacts"),
            )
        )
    return hotspots


def _print_section(title: str, body: str) -> None:
    divider = "=" * len(title)
    print(f"\n{title}\n{divider}\n{body}")


def _demo_config_payloads() -> dict[str, object]:
    return {
        "metrics": {"latency_ms": 12.3, "throughput_qps": 456.7},
        "hotspots": [
            {
                "symbol": "hm_kernel::matmul",
                "file_path": "src/hm_kernel/matmul.cc",
                "line_start": 120,
                "line_end": 185,
                "score": 0.92,
                "evidence_artifacts": ["flamegraph_001.svg"],
            }
        ],
        "trace_insights": [
            {
                "source_path": "data/flamegraph.svg",
                "event_total": 123456,
                "top_symbols": [
                    {"name": "hm_kernel::matmul", "weight": 4321.0, "ratio": 0.35}
                ],
                "top_symbols_per_thread": [
                    {
                        "name": "worker-0",
                        "top_symbols": [
                            {"name": "hm_kernel::matmul", "weight": 1111.0, "ratio": 0.28}
                        ],
                    }
                ],
            }
        ],
        "config": {
            "system_prompt": "Trace analyst focusing on HM kernel perf.",
            "prompt_template": (
                "You are the Trace Analyst. Summarize performance symptoms and hotspot classes.\n"
                "Metrics: {metrics}\n"
                "Hotspots:\n"
                "{hotspots}\n"
                "{code_context_block}"
                "{insight_block}"
            ),
            "include_code_context": True,
            "code_root": ".",
            "max_code_lines": 120,
        },
        "demo_command": (
            "python src/hmopt/agents/trace_analyst.py "
            "--metrics demo_metrics.json "
            "--hotspots demo_hotspots.json "
            "--trace-insights demo_trace_insights.json "
            "--config demo_config.json "
            "--print-only"
        ),
    }


def _load_config(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    payload = _load_json_value(raw)
    if not isinstance(payload, dict):
        raise ValueError("config payload must be a JSON object")
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug runner for TraceAnalystAgent prompt/LLM payloads.",
    )
    parser.add_argument("--metrics", required=True, help="JSON string or path to metrics JSON.")
    parser.add_argument("--hotspots", required=True, help="JSON string or path to hotspots JSON.")
    parser.add_argument(
        "--trace-insights",
        help="JSON string or path to flamegraph insight list.",
    )
    parser.add_argument("--api-base", default="http://10.90.56.33:20010/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen3-coder-30b")
    parser.add_argument("--print-only", action="store_true", help="Skip LLM call.")
    parser.add_argument(
        "--config",
        help="JSON string or path with prompt/system config overrides.",
    )
    parser.add_argument(
        "--print-demo",
        action="store_true",
        help="Print demo JSON payloads and exit.",
    )
    parser.add_argument(
        "--include-code-context",
        action="store_true",
        help="Include hotspot code excerpts in the prompt.",
    )
    parser.add_argument(
        "--code-root",
        default=".",
        help="Root directory for hotspot file paths.",
    )
    parser.add_argument(
        "--max-code-lines",
        type=int,
        default=120,
        help="Max lines per hotspot excerpt.",
    )
    args = parser.parse_args()

    if args.print_demo:
        print(json.dumps(_demo_config_payloads(), indent=2))
        return 0

    metrics = _load_json_value(args.metrics)
    if not isinstance(metrics, dict):
        raise ValueError("metrics payload must be a JSON object")
    hotspots = _load_hotspots(args.hotspots)
    trace_insights = None
    if args.trace_insights:
        trace_insights = _load_json_value(args.trace_insights)
        if not isinstance(trace_insights, list):
            raise ValueError("trace-insights payload must be a JSON list")

    config = _load_config(args.config)
    system_prompt = config.get("system_prompt")
    prompt_template = config.get("prompt_template")
    include_code_context = bool(
        config.get("include_code_context", args.include_code_context)
    )
    code_root = Path(config.get("code_root", args.code_root))
    max_code_lines = int(config.get("max_code_lines", args.max_code_lines))

    llm = LLMClient(api_base=args.api_base, api_key=args.api_key, default_model=args.model)
    safety = SafetyGuard()
    agent = TraceAnalystAgent(llm=llm, safety=safety)
    prompt = agent._build_prompt(
        metrics,
        hotspots,
        trace_insights,
        include_code_context=include_code_context,
        code_root=code_root,
        max_code_lines=max_code_lines,
        prompt_template=prompt_template if isinstance(prompt_template, str) else None,
    )
    messages = agent._build_messages(
        prompt,
        system_prompt=system_prompt if isinstance(system_prompt, str) else None,
    )

    _print_section("Raw prompt", prompt)
    _print_section("Redacted prompt", safety.redact(prompt))
    _print_section("LLM messages", json.dumps([m.__dict__ for m in messages], indent=2))
    _print_section(
        "LLM payload",
        json.dumps(
            {
                "model": args.model,
                "temperature": 0.2,
                "messages": [m.__dict__ for m in messages],
            },
            indent=2,
        ),
    )

    if args.print_only:
        return 0

    response = agent.analyze(
        metrics,
        hotspots,
        trace_insights,
        include_code_context=include_code_context,
        code_root=code_root,
        max_code_lines=max_code_lines,
        prompt_template=prompt_template if isinstance(prompt_template, str) else None,
        system_prompt=system_prompt if isinstance(system_prompt, str) else None,
    )
    _print_section("LLM response", response)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
