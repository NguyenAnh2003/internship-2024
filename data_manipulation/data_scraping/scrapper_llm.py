from scrapegraphai.graphs import SmartScraperGraph
from urls import QUORA_URL
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API = os.environ.get("GEMINI_APIKEY")

graph_config = {
    "llm": {
        "model": "gemini-1.5-pro-latest",
        "api_key": GEMINI_API,
        "format": "json",
    },
    "verbose": True,
}

smart_scraper_graph = SmartScraperGraph(
    prompt="List all contents of all posts in link below",
    source=QUORA_URL[0],
    config=graph_config,
)

result = smart_scraper_graph.run()
print(result)
