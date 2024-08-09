import pytest

from agentpod.utils import contentize


@pytest.mark.asyncio
async def test_contentize_boatbuilder():
    url = "https://boatbuilder.dev"
    content = await contentize(url)

    print(content)

    # Check if content is not empty
    assert content, "Content should not be empty"

    # Check if content is a string
    assert isinstance(content, str), "Content should be a string"

    # Check for expected content
    expected_phrases = ["boatbuilder", "RedisAI", "LightningAI", "SimpleWriter"]
    content_lower = content.lower()
    for phrase in expected_phrases:
        assert phrase.lower() in content_lower, f"Content should contain '{phrase}'"

    # Check that the content has multiple lines
    assert len(content.split("\n")) > 1, "Content should have multiple lines"

    # Check that there are no empty lines in the content
    assert all(line.strip() for line in content.split("\n")), "Content should not contain empty lines"
