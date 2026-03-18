from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

# Load environment variables from .env file
load_dotenv()

# Initialize LLM and tools
llm = ChatOllama(model="qwen3:0.6b")
tools = [TavilySearch(max_results=3)]

# Create agent with tools bound to LLM
agent = create_agent(llm, tools)


def main():
    print("Hello from langchain-course!")
    result=agent.invoke({"messages": [HumanMessage(content="What is the weather of bangalore?")]})
    print(result)


if __name__ == "__main__":
    main()
