import re


SECTION_PATTERN = re.compile(
    r"^(?P<section>\d+(?:\.\d+)*)[.)]\s+(?P<title>.+)$"
)


def detect_sections(text: str) -> list[dict]:
    sections = []
    current_section = None
    current_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        match = SECTION_PATTERN.match(line)

        if match:
            if current_section is not None:
                sections.append(
                    {
                        "section": current_section["section"],
                        "title": current_section["title"],
                        "text": "\n".join(current_lines).strip(),
                    }
                )

            current_section = {
                "section": match.group("section"),
                "title": match.group("title").strip(),
            }

            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    if current_section is not None:
        sections.append(
            {
                "section": current_section["section"],
                "title": current_section["title"],
                "text": "\n".join(current_lines).strip(),
            }
        )

    return sections
