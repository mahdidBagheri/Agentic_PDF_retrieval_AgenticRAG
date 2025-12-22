from typing import TypedDict, Optional, List


class GraphState(TypedDict):
    query: str
    route: Optional[str]
    contexts: Optional[List[dict]]
    answer: Optional[str]