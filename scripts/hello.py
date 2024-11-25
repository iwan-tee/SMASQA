import os

def main():
    print("Hello, World!")
    print("\nEnvironment Variables:")
    # for key, value in os.environ.items():
    #     print(f"{key}: {value}")
    print(os.environ.get("OPENAI_API_KEY"))

if __name__ == "__main__":
    main()