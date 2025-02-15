#from langchain.llms import Ollama
from langchain_community.llms import Ollama
from langchain import chains

llm=Ollama(model="python_model")
with open('/Users/chaitanyavarmamudundi/Desktop/workspace/langchain_project/test.py', 'w') as f:
    response=llm("""<python-program>
    <functions>
        <function>
            <function-name> csv_to_df</function-name>
            <function-purpose> get csv file path as input and convert it into pyspark dataframe </function-purpose>
        </function>
        <function>
            <function-name> df_to_parquet</function-name>
            <function-purpose> get output from "function-name : csv_to_df" and convert in into a parquet file and write it to output-path </function-purpose>
        </function>
    </functions>
    </python-program>""")
    print(response,file=f)
