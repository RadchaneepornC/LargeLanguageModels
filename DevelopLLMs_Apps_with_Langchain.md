
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

**<li>Prompt templates and chaining</li>**
- Prompt template: used for creating prompts in a more modular way, so they can be reused and built on
- Chain: act as the glue in LangChain, bringing the other components together into workflows that pass inputs and outputs between the different components
  
```python
# Set your Hugging Face API token
huggingfacehub_api_token = '<HUGGING_FACE_TOKEN>'

# Create a prompt template from the template string
template = "You are an artificial intelligence assistant, answer the question. {question}"
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain to integrate the prompt template and LLM
llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "How does LangChain make LLM application development easier?"
print(llm_chain.run(question))

```
**<li>Chat prompt templates</li>**

```python

# Set your API Key from OpenAI
openai_api_key= '<OPENAI_API_TOKEN>'

# Define an OpenAI chat model
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)		

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Respond to question: {question}")
    ]
)

# Insert a question into the template and call the model
full_prompt = prompt_template.format_messages(question='How can I retain learning?')
llm(full_prompt)
```








</details>









 <details><summary>2. Loading and Preparing External Data for Chatbots</summary></details>
 <details><summary>3. LangChain Expression Language (LCEL), Chains, and Agents</summary></details>
 <details><summary>4. Tools, Troubleshooting, and Evaluation</summary></details>

