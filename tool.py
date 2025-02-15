from langchain_core.tools import tool
from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions

#tool creation
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = OllamaFunctions(model="llama3.2:latest")
#llm=Ollama(model="llama3.2")

#tool binding
llm_with_tools = llm.bind(functions=[multiply])

#tool calling
result = llm_with_tools.invoke("Hello world!")

result = llm_with_tools.invoke("What is 2 multiplied by 3?")

print(result.tool_calls)