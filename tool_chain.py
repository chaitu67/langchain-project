from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSequence
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
import json
from test import  init_spark_session,csv_to_df,df_to_parquet,start_program
from pydantic import BaseModel,Field

class Joke(BaseModel):
    input_path: str = Field(description="this  path for source csv file")
    output_path: str = Field(description="this  path for output parquet file")

def multiply(a) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    print(a)
    print(json.loads(a))
    #print(a["first"]+a["second"])
    return a["first"]


def get_value(a):
    print(a)
    return a

def chain_test():
    response_schemas = [
        ResponseSchema(name="a", description="this is the first number"),
        ResponseSchema(name="b", description="this is the second number")
    ]

    runable1 =RunnableLambda(get_value)
    runable2 =RunnableLambda(multiply)


    output_parser=StructuredOutputParser.from_response_schemas(response_schemas)
    #prompt template
    template = """you are intial prompt in langchain chain pass  these values to model {a} and {b} ask model to echo these values back"""
    input_variables=["a","b"]
    prompt = PromptTemplate(template=template,input_variables=input_variables,output_parser=output_parser)

    model = OllamaLLM(model="llama3.2",PromptTemplate=prompt,verbose=True)


    #chain=prompt|model|runable1|runable2

    chain=prompt|model|runable1

    a=10
    b=20
    print(chain.invoke({"a":a,"b":b}))

def elt_chain(data):
    
    response_schemas = [
        ResponseSchema(name="input_path", description="this  path for source csv file"),
        ResponseSchema(name="output_path", description="this  path for output parquet file")
    ]
    output_parser=StructuredOutputParser.from_response_schemas(response_schemas)
    #output_parser=JsonOutputParser(pydantic_object=Joke)
    format_instruction=output_parser.get_format_instructions()
    template=data['prompt']
    print(format_instruction)
    input_variables=['input_path','output_path']
    prompt = PromptTemplate(template=template
                            ,input_variables=input_variables
                            ,partial_vaiables={"etl_model": format_instruction})
    print(prompt.invoke({"input_path":data["input_path"],"output_path":data["output_path"]}))
    print(output_parser)
    #model = OllamaLLM(model="elt_model:latest",PromptTemplate=prompt)
    model=OllamaFunctions(model="llama3.2:latest")
    llm=model.with_structured_output(Joke)
    '''
    model = OllamaLLM(model="elt_model:latest",PromptTemplate=prompt)
    #spark_session=RunnableLambda(init_spark_session)
    pyspark=RunnableLambda(start_program)
    '''
    chain=prompt|llm
    print(chain.invoke({"input_path":data["input_path"],"output_path":data["output_path"]}))
    
def read_json(data):
    with open(data) as f:
        d = json.load(f)
        return d

def main():
    #chain_test()
    data=read_json("/Users/chaitanyavarmamudundi/Desktop/workspace/langchain-project/config/pyspark.json")
    #print(data)
    elt_chain(data)

main()

