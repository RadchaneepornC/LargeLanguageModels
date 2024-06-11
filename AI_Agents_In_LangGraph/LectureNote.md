## Lecture Note from [AI Agents In LangGraph Course](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/?fbclid=IwZXh0bgNhZW0CMTAAAR1OAuQRQN9udwT1SR4wpBO0W90bnM45ofAeOwezLPyiRcSK3foTgrRs-2s_aem_AareDshNT8DFgmbXn6CAe4ARsEby1Ag0p1quU_AdTWoReIYX9u1pPeUHPE_VHOPPav1-4I8mJ-i0hrkmCYKjKWNv) from DeepLearning.AI


<details><summary>Introduction</summary> 
  <br>
Agentic workflow that LangChain support via LangGraph (a cyclical graph define agents and their behaviors)

![Alt text](https://github.com/RadchaneepornC/LargeLanguageModels/blob/main/AI_Agents_In_LangGraph/image/CyclicGraph.png)

- [ReAct](https://arxiv.org/pdf/2210.03629): Reasoning and action
- [SELF-REFINE](https://arxiv.org/pdf/2303.17651): iterative refinement that we were talking about
- [AlphaCodium](https://arxiv.org/pdf/2401.08500): alpha coing creating coding agent using flow engineering


</details>

<details><summary>
Build an Agent from Scratch
  
</summary>
<br>

![Alt text](https://github.com/RadchaneepornC/LargeLanguageModels/blob/main/AI_Agents_In_LangGraph/image/ReAct.png)

The agent built from scratch is based on **ReAct paper**, which ReAct stands for **Reasoning + Acting**
- LLM first think what to do
- Then, decide to action to take
- Action is ececuted in the environment, and observation is return
- With this observation, the LLM will repeat... by thinking  about what to do again, what action to take ...

**Steps of implementing code in this section:**

**1. To initiate LLM, OpenAI in this case**


```python
# based on https://til.simonwillison.net/llms/python-react-pattern
```


```python
import openai
import re
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv()
from openai import OpenAI
```


```python
client = OpenAI()
```

```python
chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}]
)
```


```python
chat_completion.choices[0].message.content
```

**2. Create an Agent class**

```python
class Agent:
    def __init__(self, system=""): #we want this agent will be parameterized by system message
        self.system = system  #save system message that user pass as an attribute
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                        model="gpt-4o", 
                        temperature=0, #very deterministic
                        messages=self.messages)
        return completion.choices[0].message.content
    
```

**3. Define prompt**

```python
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()
```

**4. Provide the tools for an agent**

```python
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
```

**5. **

```python
abot = Agent(prompt)
```


```python
result = abot("How much does a toy poodle weigh?")
print(result)
```

> Thought: I should look up the average weight of a toy poodle using the available action.
Action: average_dog_weight: Toy Poodle
PAUSE


```python
result = average_dog_weight("Toy Poodle")
```


```python
result
```
> 'a toy poodles average weight is 7 lbs'

```python
next_prompt = "Observation: {}".format(result)
```


```python
abot(next_prompt)
```
> 'Answer: A toy poodle weighs an average of 7 lbs.'

```python
abot.messages
```

```md
 [{'role': 'system',
  'content': 'You run in a loop of Thought, Action, PAUSE, Observation.\nAt the end of the loop you output an Answer\nUse Thought to describe your thoughts about the question you have been asked.\nUse Action to run one of the actions available to you - then return PAUSE.\nObservation will be the result of running those actions.\n\nYour available actions are:\n\ncalculate:\ne.g. calculate: 4 * 7 / 3\nRuns a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n\naverage_dog_weight:\ne.g. average_dog_weight: Collie\nreturns average weight of a dog when given the breed\n\nExample session:\n\nQuestion: How much does a Bulldog weigh?\nThought: I should look the dogs weight using average_dog_weight\nAction: average_dog_weight: Bulldog\nPAUSE\n\nYou will be called again with this:\n\nObservation: A Bulldog weights 51 lbs\n\nYou then output:\n\nAnswer: A bulldog weights 51 lbs'},
 {'role': 'user', 'content': 'How much does a toy poodle weigh?'},
 {'role': 'assistant',
  'content': 'Thought: I should look up the average weight of a toy poodle using the available action.\nAction: average_dog_weight: Toy Poodle\nPAUSE'},
 {'role': 'user',
  'content': 'Observation: a toy poodles average weight is 7 lbs'},
 {'role': 'assistant',
  'content': 'Answer: A toy poodle weighs an average of 7 lbs.'}]
```




```python
abot = Agent(prompt)
```


```python
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
abot(question)
```
> 'Thought: I need to find the average weight of both a Border Collie and a Scottish Terrier, then add them together to get the combined weight.\nAction: average_dog_weight: Border Collie\nPAUSE'


```python
next_prompt = "Observation: {}".format(average_dog_weight("Border Collie"))
print(next_prompt)
```
> Observation: a Border Collies average weight is 37 lbs



```python
abot(next_prompt)
```
> 'Thought: Now I need to find the average weight of a Scottish Terrier.\nAction: average_dog_weight: Scottish Terrier\nPAUSE'

```python
next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
print(next_prompt)
```

> Observation: Scottish Terriers average 20 lbs


```python
abot(next_prompt)
```
> 'Thought: I now have the average weights of both dogs. I will add them together to find their combined weight.\nAction: calculate: 37 + 20\nPAUSE'


```python
next_prompt = "Observation: {}".format(eval("37 + 20"))
print(next_prompt)
```
>Observation: 57


```python
abot(next_prompt)
```

>'Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs.'

```python
abot.messages
```

```md
[{'role': 'system',
  'content': 'You run in a loop of Thought, Action, PAUSE, Observation.\nAt the end of the loop you output an Answer\nUse Thought to describe your thoughts about the question you have been asked.\nUse Action to run one of the actions available to you - then return PAUSE.\nObservation will be the result of running those actions.\n\nYour available actions are:\n\ncalculate:\ne.g. calculate: 4 * 7 / 3\nRuns a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n\naverage_dog_weight:\ne.g. average_dog_weight: Collie\nreturns average weight of a dog when given the breed\n\nExample session:\n\nQuestion: How much does a Bulldog weigh?\nThought: I should look the dogs weight using average_dog_weight\nAction: average_dog_weight: Bulldog\nPAUSE\n\nYou will be called again with this:\n\nObservation: A Bulldog weights 51 lbs\n\nYou then output:\n\nAnswer: A bulldog weights 51 lbs'},
 {'role': 'user',
  'content': 'I have 2 dogs, a border collie and a scottish terrier. What is their combined weight'},
 {'role': 'assistant',
  'content': 'Thought: I need to find the average weight of both a Border Collie and a Scottish Terrier, then add them together to get the combined weight.\nAction: average_dog_weight: Border Collie\nPAUSE'},
 {'role': 'user',
  'content': 'Observation: a Border Collies average weight is 37 lbs'},
 {'role': 'assistant',
  'content': 'Thought: Now I need to find the average weight of a Scottish Terrier.\nAction: average_dog_weight: Scottish Terrier\nPAUSE'},
 {'role': 'user', 'content': 'Observation: Scottish Terriers average 20 lbs'},
 {'role': 'assistant',
  'content': 'Thought: I now have the average weights of both dogs. I will add them together to find their combined weight.\nAction: calculate: 37 + 20\nPAUSE'},
 {'role': 'user', 'content': 'Observation: 57'},
 {'role': 'assistant',
  'content': 'Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs.'}]
```

**6. Create loop for automating working of an agent**



</details>

<details><summary>
LangGraph Components
</summary>

</details>

<details><summary>
Agentic Search Tools
</summary>
<br>
This topic is about the capabilities of agentic search and how to use it.
</details>

<details><summary>
Persistence and Streaming
</summary>
<br>
This topics is about two additional capability that are helpful when building agents
  
- Human input: this allows you to guide an agent at critical points
- Persistence: the ability to store the current state of information so that we can return to it later, which great for debugging an agent and productionalizing them

</details>


<details><summary>
Human in the loop
</summary>

</details>

<details><summary>
Essay Writer
</summary>

</details>

<details><summary>
LangChain Resources
</summary></details>

<details><summary>
Conclusion
</summary></details>
