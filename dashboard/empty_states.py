"""Enterprise empty states for dashboard pages — finished, not broken."""
from __future__ import annotations


def empty_state_html(
    *,
    icon: str,
    title: str,
    body: str,
    steps: list[str] | None = None,
) -> str:
    steps_html = ""
    if steps:
        items = "".join(
            f'<li style="margin:4px 0">{s}</li>' for s in steps
        )
        steps_html = (
            '<ol style="color:#94a3b8;font-size:0.9rem;line-height:1.65;'
            f'margin:12px 0 0 18px;padding:0">{items}</ol>'
        )
    return f"""
    <div style="border:1px solid #334155;border-radius:16px;
                padding:32px 28px;background:linear-gradient(135deg,#1e293b,#0f172a);
                margin:12px 0 24px;">
      <div style="font-size:2.2rem;margin-bottom:10px">{icon}</div>
      <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-bottom:8px">{title}</div>
      <div style="color:#94a3b8;font-size:0.95rem;line-height:1.65">{body}</div>
      {steps_html}
    </div>
    """
