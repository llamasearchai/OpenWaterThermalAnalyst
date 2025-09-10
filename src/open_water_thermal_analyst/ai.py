from __future__ import annotations

from typing import Any, Dict, Optional
import os
import json
import shutil
import subprocess


def summarize_compliance_results(
    results: Dict[str, Any],
    model: Optional[str] = None,
    max_tokens: int = 400,
) -> str:
    """Generate a narrative summary of compliance results using OpenAI or 'llm' CLI.

    Prefers the OpenAI Python SDK if available and OPENAI_API_KEY is set.
    Falls back to the 'llm' CLI (https://github.com/simonw/llm) if installed.
    Raises RuntimeError if neither is available.
    """
    prompt = _compose_prompt(results)
    mdl = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Try OpenAI SDK first
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()  # reads OPENAI_API_KEY from env
        resp = client.chat.completions.create(
            model=mdl,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a hydrology and environmental compliance assistant. "
                        "Explain thermal compliance results clearly for regulators and operators."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        if not content:
            raise RuntimeError("Empty response from OpenAI.")
        return content.strip()
    except Exception:
        # Fall through to llm CLI if available
        pass

    # Fallback to llm CLI
    if shutil.which("llm"):
        cmd = [
            "llm",
            "-m",
            mdl,
            prompt,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"llm CLI failed: {r.stderr.strip()}")
        return r.stdout.strip()

    raise RuntimeError(
        "No AI backend available. Install 'openai' and set OPENAI_API_KEY, or install the 'llm' CLI."
    )


def _compose_prompt(results: Dict[str, Any]) -> str:
    stats = results.get("stats", {})
    thresholds = results.get("thresholds", {})
    warn = results.get("warning_exceedances", 0)
    crit = results.get("critical_exceedances", 0)

    payload = {
        "stats": stats,
        "thresholds": thresholds,
        "warning_exceedances": warn,
        "critical_exceedances": crit,
    }
    return (
        "Summarize the following thermal compliance results for a non-technical audience. "
        "Include whether thresholds are exceeded, potential operational implications, and recommended next steps.\n\n"
        + json.dumps(payload, indent=2)
    )

