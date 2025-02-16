import getpass
import os
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from typing import Optional
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import ChatOllama
import json
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from test import start_program
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage,SystemMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class Joke(BaseModel):
    input_path: str = Field(description="this  path for source csv file")
    output_path: str = Field(description="this  path for output parquet file")

class Program(BaseModel):
    config: dict = Field(description="this  path for configuration file ")



def read_json(data):
    with open(data) as f:
        d = json.load(f)
        return d

def test(a):
    return a

def main():
    data=read_json("/Users/chaitanyavarmamudundi/Desktop/workspace/langchain-project/config/pyspark.json")
    #data_path=read_json(data)
    #output_parser=PydanticOutputFunctionsParser(pydantic_object=Joke)
    model=ChatOllama(model="llama3.2:latest")
    structured_llm = model.with_structured_output(Joke)
    prompt = PromptTemplate(template=data['prompt'],input_variables=['input_path','output_path'])
    main1=RunnableLambda(start_program)
    chain=prompt|structured_llm|main1
    a=chain.invoke({"input_path": data['input_path'],"output_path": data['output_path']})
    print(a)

main()