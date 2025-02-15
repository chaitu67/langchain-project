from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

#prompt template
prompt = PromptTemplate.from_template(template)

#model
model = OllamaLLM(model="llama3.2")

#chains
chain = prompt | model

#call chain
print(chain.invoke({"question": "What is LangChain?"}))