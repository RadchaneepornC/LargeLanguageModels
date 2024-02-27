
# My Lecture from [Developing LLM Applications with LangChain from DataCamp](https://app.datacamp.com/learn/courses/developing-llm-applications-with-langchain)

<details>
<summary>1. Introduction to LangChain & Chatbot Mechanics</summary>

```
python

from langchain_community.llms import HuggingFaceHub

# Set your Hugging Face API token 
huggingfacehub_api_token = '<HUGGING_FACE_TOKEN>'

# Define the LLM
llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)

print(output)
```
</details>

<details><summary>2. Loading and Preparing External Data for Chatbots</summary></details>

<details><summary>3. LangChain Expression Language (LCEL), Chains, and Agents</summary></details>

<details><summary>4. Tools, Troubleshooting, and Evaluation</summary></details>

