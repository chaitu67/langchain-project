from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSequence


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def get_value(a,b):
    return a,b

model = OllamaLLM(model="llama3.2")
llm_with_tools = model.bind(functions=[get_value,multiply])
runable1 =RunnableLambda(get_value)
print(runable1)
runable2 =RunnableLambda(multiply)

template = """Question: {question}

Answer: Let's think step by step."""

#prompt template
prompt = PromptTemplate.from_template(template)

chain=prompt|model|runable1|runable1
print(chain.invoke({"question": "What is LangChain?"}))


#chain = RunnableSequence([runable1, runable2])

#chain.invoke(1,4)

#template = """add 2 numbers {a} & {b}"""
#prompt template
#prompt = PromptTemplate.from_template(template)
#chain_prompt=prompt.invoke({"a":10,"b":20}).text
#print(chain_prompt)
#model
#model = OllamaLLM(model="llama3.2")
#llm_with_tools = model.bind(functions=[multiply])


#chain = chain_prompt | llm_with_tools

#call chain
#print(chain.invoke({"a":"10","b":"20"}))
