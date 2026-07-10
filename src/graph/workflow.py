from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from src.graph.state import GraphState
from src.graph.nodes import GraphNodes


class GraphWorkflow:
    """Builds the LangGraph workflow."""
    def __init__(self):
        self.nodes = GraphNodes()
    def build(self):
        graph = StateGraph(GraphState)
        graph.add_node("retrieve",self.nodes.retrieve)

        graph.add_node("generate",self.nodes.generate)
        graph.add_edge(START,"retrieve")
        
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate",END)
        return graph.compile()