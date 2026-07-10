from src.vector_store.builder import VectorStoreBuilder


def main():
    builder = VectorStoreBuilder()

    builder.build_vector_store()

    print("Vector store built successfully!")


if __name__ == "__main__":
    main()