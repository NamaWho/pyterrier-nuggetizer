from jinja2 import Template
from typing import Dict, Any
import importlib.resources as pkg_resources
from src.open_nuggetizer.prompts import __name__ as PROMPTS_DIR

def load_template(name: str) -> Template:
    """
        Load a Jinja2 template from the /prompts directory.
    """
    try:
        content = pkg_resources.files(PROMPTS_DIR).joinpath(name).read_text(encoding='utf-8')
        return Template(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template '{name}' not found in /prompts")
    
def render_prompt(template_name: str, context: Dict[str, Any]) -> str:
    """
    Render a Jinja2 template with the given context dictionary.

    Args:
        template_name: Filename (e.g. 'creator.txt') inside /prompts
        context: Dictionary of variables to pass into the template

    Returns:
        Rendered prompt string
    """
    template = load_template(template_name)
    return template.render(**context)