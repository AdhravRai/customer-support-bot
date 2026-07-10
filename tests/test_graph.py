from src.graph.workflow import GraphWorkflow


def main():
    workflow = GraphWorkflow()
    graph = workflow.build()

    initial_state = {
        "question": "What is your refund policy?",
        "context": [],
        "answer": "",
    }

    result = graph.invoke(initial_state)

    print("\nAnswer:\n")
    print(result["answer"])


if __name__ == "__main__":
    main()