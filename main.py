import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


load_dotenv()


def main():
    print("Hello from langchain-course!")
    info="""Anthropic PBC is an American artificial intelligence (AI) company headquartered in San Francisco. It has developed a family of large language models (LLMs) named Claude. Anthropic operates as a public benefit corporation, which researches and develops AI to "study their safety properties at the technological frontier" and use this research to deploy safe models for the public.[7][8]

Anthropic was founded in 2021 by former members of OpenAI, including siblings Daniela Amodei and Dario Amodei, who are president and CEO, respectively.[9] As of February 2026, Anthropic has an estimated value of $380 billion.[10]

History

Dario Amodei, co-founder and CEO
Anthropic was founded in 2021 by seven former employees of OpenAI, including siblings Daniela Amodei and Dario Amodei, the latter of whom was OpenAI's Vice President of Research.[11][12]

In the summer of 2022, Anthropic finished training the first version of Claude but did not release it, citing the need for further internal safety testing and a desire to avoid initiating a potentially hazardous race to develop increasingly powerful AI systems.[13]

In 2024, Anthropic attracted several notable employees from OpenAI, including Jan Leike, John Schulman, and Durk Kingma.[14]

In March, Databricks and Anthropic announced that Claude would be integrated into the Databricks Data Intelligence Platform.[15][16]

In May 2025, the company announced Claude 4, introducing both Claude Opus 4 and Claude Sonnet 4 with improved coding capabilities and other new features. It also introduced new API capabilities, including the Model Context Protocol (MCP) connector.[17] The company hosted its inaugural developer conference that month.[18] Also in May, Anthropic launched a web search API that enables Claude to access real-time information from the internet.[19] Claude Code, Anthropic's coding assistant, transitioned from research preview to general availability, featuring integrations with VS Code and JetBrains IDEs and support for GitHub Actions.[17]

In September 2025, Anthropic announced that it would stop selling its products to groups majority-owned by Chinese, Russian, Iranian, or North Korean entities due to national security concerns.[20]

In October 2025, Anthropic announced a cloud partnership with Google, giving it access to up to one million of Google's custom Tensor Processing Units (TPUs). According to Anthropic, the partnership will bring more than one gigawatt of AI compute capacity online by 2026.[21]

In November 2025, Nvidia, Microsoft and Anthropic announced a partnership deal. Nvidia and Microsoft were expected to invest up to $15 billion in Anthropic, and Anthropic said it would buy $30 billion of computing capacity from Microsoft Azure running on Nvidia AI systems.[22]

In November 2025, Anthropic said that hackers sponsored by the Chinese government used Claude to perform automated cyberattacks against around 30 global organisations. The hackers tricked Claude into carrying out automated subtasks by pretending it was for defensive testing.[23][24]

In December 2025, Anthropic acquired Bun to improve the speed and stability of Claude Code.[25] The same month, Anthropic signed a multi-year, $200 million partnership with Snowflake Inc. to make Claude models available through Snowflake's platform as the companies expanded enterprise deployments of AI tools and agents.[26]

In February 2026, Anthropic aired two commercials during Super Bowl LX[27][28] as part of a broader marketing campaign called "A Time and a Place", with four ads created by Mother. Each ad depicts AI assistants suddenly pivoting to promoting a fictional product in the middle of a conversation. Anthropic said that Claude will stay ad-free, in contrast to its competitor OpenAI, which introduced ads to the free version of ChatGPT.[28]
    """

    summary_template = """
    given the information {information} about a company, I want you to create:
    1. A short summary
    2. two interesting facts about it
    """

    summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
    )

    llm=ChatOllama(model="qwen3:0.6b", temperature=1.0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": info})
    print(response.content)


if __name__ == "__main__":
    main()
