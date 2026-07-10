from src.graph.workflow import GraphWorkflow


def main():
    workflow = GraphWorkflow()
    graph = workflow.build()

    first_state = {
        "question": "Do you ship to India?",
        "context": [],
        "answer": "",
        "sources": [],
        "chat_history": [],
    }
    config = {
        "configurable": {
            "thread_id": "demo-thread"
        }
    }

    result = graph.invoke(first_state,config=config,)

    print("\nAnswer:\n")
    print(result["answer"])

    second_state = {
        "question": "How much does shipping cost there?",
        "context": [],
        "answer": "",
        "sources": [],
        "chat_history": [],
    }
    
    result = graph.invoke(
        second_state,
        config=config,
    )
    
    print(result["answer"])



if __name__ == "__main__":
    main()