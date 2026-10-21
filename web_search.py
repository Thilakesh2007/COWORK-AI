
from ddgs import DDGS


def search_web(query, max_results=5):
    """Search the web and return useful results."""
    try:
        results = DDGS().text(
            query,
            max_results=max_results
        )

        if not results:
            return "No web search results found."

        output = []

        for i, result in enumerate(results, start=1):
            output.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"URL: {result.get('href', 'No URL')}\n"
                f"Summary: {result.get('body', 'No summary')}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Web search failed: {e}"


if __name__ == "__main__":
    question = input("What would you like to search for? ")
    print("\nSearching the web...\n")
    print(search_web(question))
