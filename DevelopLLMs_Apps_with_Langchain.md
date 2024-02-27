
# My Lecture from [Developing LLM Applications with LangChain from DataCamp](https://app.datacamp.com/learn/courses/developing-llm-applications-with-langchain)

<details> <summary>1. Introduction to LangChain & Chatbot Mechanics</summary>

**<li>code for using Langchain to call opensource LLMs</li>**
```python

from langchain_community.llms import HuggingFaceHub

# Set your Hugging Face API token for read model
huggingfacehub_api_token = '<HUGGING_FACE_TOKEN>' 

# Define the LLM
llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)

# Predict the words following the text in question
question = 'Question..?'
output = llm.invoke(question)

print(output)
```
**<li>code for using Langchain to call closesource LLMs such as OpenAI LLMs</li>**

```python
from lanchain_openai import OpenAI

# Set your API Key from OpenAI
openai_api_key = "<openai_api_key>"

# Define the LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)

# Predict the words following the text in question
question = 'Question..?'
output = llm.invoke(question)
print(output)
```

```python .invoke()``` is the method for predicting using an LLMs


</details>









 <details><summary>2. Loading and Preparing External Data for Chatbots</summary></details>
 <details><summary>3. LangChain Expression Language (LCEL), Chains, and Agents</summary></details>
 <details><summary>4. Tools, Troubleshooting, and Evaluation</summary></details>

