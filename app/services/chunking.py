import re


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    sections = re.split(r"(?=Q\.)", text)

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            start = 0

            while start < len(section):
                end = start + chunk_size
                chunk = section[start:end]

                if chunk.strip():
                    chunks.append(chunk)

                start += chunk_size - overlap

    return chunks