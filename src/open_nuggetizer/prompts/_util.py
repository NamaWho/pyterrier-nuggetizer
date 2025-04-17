from jinja2 import Template
from typing import Dict, Any


def render_prompt(template: Template, context: Dict[str, Any]) -> str:
    """
    Render a Jinja2 template with the given context dictionary.

    Args:
        template_name: Filename (e.g. 'creator.txt') inside /prompts
        context: Dictionary of variables to pass into the template

    Returns:
        Rendered prompt string
    """
    return template.render(**context)


__all__ = ["render_prompt"]
