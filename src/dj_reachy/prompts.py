import re
import logging
from pathlib import Path

from dj_reachy.config import config


logger = logging.getLogger(__name__)


PROFILES_DIRECTORY = Path(__file__).parent / "profiles"
PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r'^\[([a-zA-Z0-9/_-]+)\]$')

    lines = content.split('\n')
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return '\n'.join(expanded_lines)


def get_session_instructions() -> str:
    """Reload user-owned persona while keeping the app's confirmation policy."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    instructions_file = (
        PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME
        if profile else PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    )
    instructions = _expand_prompt_includes(instructions_file.read_text(encoding="utf-8").strip())
    user_instructions = config.APP_HOME / "instructions.md"
    if user_instructions.is_symlink():
        raise ValueError("user instructions cannot be a symbolic link")
    if user_instructions.is_file():
        persona = user_instructions.read_text(encoding="utf-8").strip()
        if len(persona) > 12000:
            raise ValueError("user instructions exceed 12000 characters")
        if persona:
            instructions += "\n\nUser personality preferences:\n" + persona
    language = config.CONVERSATION_LANGUAGE
    instructions += (
        "\n\nApp policy: Obtain explicit user confirmation before any paid music generation. "
        "Never generate another track automatically. Stop and pause requests take priority. "
        "Conversation language and lyric language are independent settings. "
        "Never change lyric language just because the conversation language changes."
    )
    instructions += (
        "\nFollow the user's conversation language naturally."
        if language == "auto" else f"\nSpeak naturally in {language} unless the user asks otherwise."
    )
    return instructions


def get_session_voice(default: str = "cedar") -> str:
    """Use the explicit conversation voice, never a singer or packaged override."""
    return str(getattr(config, "VOICE_NAME", default))
