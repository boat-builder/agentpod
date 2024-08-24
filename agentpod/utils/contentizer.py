from bs4 import BeautifulSoup

from agentpod.http.client import retryable_httpx


async def contentize(url: str) -> str:
    """
    Extract and structure content from a given URL.

    Args:
        url (str): The URL of the webpage to contentize.

    Returns:
        str: A string of extracted content formatted in markdown.
    """
    try:
        # Fetch and parse the webpage
        soup = await fetch_and_parse_url(url)
        # Define tags we want to extract content from
        relevant_tags = get_relevant_tags()

        # Extract content from the body of the document
        body_content = soup.body if soup.body else soup
        return extract_content_from_body(body_content, relevant_tags)
    except Exception:
        return None


async def fetch_and_parse_url(url: str) -> BeautifulSoup:
    """Fetch the webpage and parse it with BeautifulSoup."""
    async with retryable_httpx() as client:
        response = await client.get(url)
        response.raise_for_status()  # Ensure we got a successful response
        return BeautifulSoup(response.text, "html.parser")


def get_relevant_tags() -> list[str]:
    """Return a list of HTML tags we want to extract content from."""
    return [
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "article",
        "section",
        "blockquote",
        "li",
        "strong",
        "em",
        "span",
        "div",
    ]


def extract_content_from_body(body_content: BeautifulSoup, relevant_tags: list[str]) -> str:
    """Extract content from the body of the document and format it in markdown."""
    parsed_content = []
    for tag in body_content.find_all(relevant_tags, recursive=True):
        if tag.name.startswith("h"):
            level = int(tag.name[1])
            parsed_content.append(f"{'#' * level} {tag.get_text(strip=True)}")
        elif tag.name == "li":
            parsed_content.append(extract_list_item_content(tag))
        elif tag.name in relevant_tags:
            parsed_content.append(tag.get_text(separator=" ", strip=True))
    return "\n\n".join(parsed_content)


def extract_list_item_content(element: BeautifulSoup) -> str:
    """Extract content from a list item, preserving its structure in markdown."""
    return f"- {element.get_text(separator=' ', strip=True)}"


def extract_content(element: BeautifulSoup, relevant_tags: list[str]) -> str:
    """
    Recursively extract text content from relevant tags, maintaining structure in markdown.

    Args:
        element (BeautifulSoup): The BeautifulSoup element to extract content from.
        relevant_tags (list[str]): List of HTML tags to consider for content extraction.

    Returns:
        str: Extracted and structured content in markdown format.
    """
    content_list = []

    if element.name in relevant_tags:
        if element.name.startswith("h"):
            level = int(element.name[1])
            content_list.append(f"{'#' * level} {element.get_text(strip=True)}")
        elif element.name == "li":
            content_list.append(extract_list_item_content(element))
        else:
            for child in element.children:
                if isinstance(child, str):
                    content_list.append(child.strip())
                elif child.name in relevant_tags:
                    content_list.append(extract_content(child, relevant_tags))

    return "\n\n".join(filter(None, content_list))
