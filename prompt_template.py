from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm=Ollama(model="llama3.2")

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

print(llm(prompt_template.invoke({"topic": "cats"}).text))