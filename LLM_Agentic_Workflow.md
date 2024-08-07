# My Lecture about Agentic Workflow with LLMs

## 📍 Multi-Agent Systems with AutoGen 



Below is my summarization about Agentic workflow with AutoGen from three resources:
- [1] Dibia, Victor, and Chi Wang. *Multi-Agent Systems with AutoGen*. Manning Publications, 2024. https://www.manning.com/books/multi-agent-systems-with-autogen.

- [2] [**Microsoft AutoGen tutorial**](https://microsoft.github.io/autogen/docs/tutorial/introduction)

- [3] DeepLearning.AI. (n.d.). AI Agentic Design Patterns with AutoGen. Retrieved August 6, 2024, from https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/



🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱🌱

<details><summary>
1. Understanding a Multi-Agent System (MAS)</summary>

## Multi-Agent system
- **agent:** entities that can reason, act, communicate and adapt to solve problems
- **A multi-agent system:** group of agents collaborating to solve tasks, each agent has their own abilities (reasoning, acting, communicating, adapting)
- **multi agent system has two core components** which can be driven by a combination of generative AI models, tools, and human input:
  - **agent capabilities:** method by which agents address tasks
    - mechanisms for **reasoning** (planning, deducing, etc. by applying some rules or logic), reasoning can be deductive, inductive, abductive
    - mechanisms for **acting** (utilizing tools)
    - mechanisms for **adapting** (learning, recalling information from memory), adjust their actions, and plans based on changing conditions, new information, or feedback from other agents and humans
  - **agent interactions:** how agents communicate and collaborate to solve tasks
    -  **agent workflows** (how agents are organized or grouped to address tasks, includes properties such as the entry point (how the task is initiated), details for each agent (tools, memory, models), conditions for task success or termination, and the orchestration pattern, workflow can be defined by developer or can be driven by some logic or LLMs )
    -  **agent orchestration** (the sequence in which agents take action as the task progresses, can be deterministic sequence, or sequence that involves agents reasoning over the task state)

![Alt text](image/multi-agent-system.png)

> Picture Reference: *Dibia, Victor, and Chi Wang. *Multi-Agent Systems with AutoGen*. Manning Publications, 2024.*

## Autogen core feature
- **autogen core feature** provides us
![Alt text](image/built-in-agent-classes.png)
> Picture Reference: *Dibia, Victor, and Chi Wang. *Multi-Agent Systems with AutoGen*. Manning Publications, 2024.*
  - **ConversableAgent:** class with methods  1) for sending and receiving messages between agents and  2) for defining how an agent acts on a received message, it includes a set of built-in agent classes

**NOTE** the `UserProxyAgent` and `AssistantAgent` are meant to be shortcuts to avoid writing the `system_message` instructions for the `ConversableAgent` class. They are not suitable for all use cases.


   - **UserProxyAgent**: has default capabilities to execute code and solicit human input, Existing AutoGen examples often create code executor agent using the `UserProxyAgent` class, which is a subclass of `ConversableAgent` with `human_input_mode`=`ALWAYS` and `llm_config`=`False` – it always requests human input for every message and does not use LLM. It also comes with default description field for each of the human_input_mode setting. This class is a convenient short-cut for creating an agent that is intended to be used as a code executor.
      
  - **AssistantAgent**: configured to address tasks using an LLM,  This class is a convenient short-cut for creating an agent that is intended to be used as a code writer and does not execute code, which is a subclass of ConversableAgent with `human_input_mode`=`NEVER` and `code_execution_config`=`False` – it never requests human input and does not use code executor.

see default message of `AssistantAgent`

```py
print("below is DEFAULT_SYSTEM_MESSAGE of AssistantAgent\n")
pprint.pprint(AssistantAgent.DEFAULT_SYSTEM_MESSAGE)
```

```
('You are a helpful AI assistant.\n'
 'Solve tasks using your coding and language skills.\n'
 'In the following cases, suggest python code (in a python coding block) or '
 'shell script (in a sh coding block) for the user to execute.\n'
 '    1. When you need to collect info, use the code to output the info you '
 'need, for example, browse or search the web, download/read a file, print the '
 'content of a webpage or a file, get the current date/time, check the '
 'operating system. After sufficient info is printed and the task is ready to '
 'be solved based on your language skill, you can solve the task by yourself.\n'
 '    2. When you need to perform some task with code, use the code to perform '
 'the task and output the result. Finish the task smartly.\n'
 'Solve the task step by step if you need to. If a plan is not provided, '
 'explain your plan first. Be clear which step uses code, and which step uses '
 'your language skill.\n'
 'When using code, you must indicate the script type in the code block. The '
 'user cannot provide any other feedback or perform any other action beyond '
 "executing the code you suggest. The user can't modify your code. So do not "
 "suggest incomplete code which requires users to modify. Don't use a code "
 "block if it's not intended to be executed by the user.\n"
 'If you want the user to save the code in a file before executing it, put # '
 "filename: <filename> inside the code block as the first line. Don't include "
 'multiple code blocks in one response. Do not ask users to copy and paste the '
 "result. Instead, use 'print' function for the output when relevant. Check "
 'the execution result returned by the user.\n'
 'If the result indicates there is an error, fix the error and output the code '
 'again. Suggest the full code instead of partial code or code changes. If the '
 "error can't be fixed or if the task is not solved even after the code is "
 'executed successfully, analyze the problem, revisit your assumption, collect '
 'additional info you need, and think of a different approach to try.\n'
 'When you find an answer, verify the answer carefully. Include verifiable '
 'evidence in your response if possible.\n'
 'Reply "TERMINATE" in the end when everything is done.\n'
 '    ')
```


- **GroupChatManager** : serves as a container abstraction for enabling interactions between groups of agents



## Basic Example

a setup where two agents converse to solve the task Generate a plot of the stock price of NVIDIA for the last 30 days:

```py
# define LLMs config
llm_config = {"config_list": [{ "model": "gpt-4-turbo-preview"}]}

# (I)
assistant = AssistantAgent(
    name= "assistant", llm_config=llm_config)

# (II)
user_proxy = UserProxyAgent(
    name = "user_proxy",
    code_execution_config={"work_dir": "scratch",
                           "use_docker": True},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    )

# (III)
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD. Save the result to a file named
 chart.png.")

```
- (I) defines an assistant agent which inherits the default system message for an AutoGen assistant agent and has access to an LLM via the specified llm_config parameter
- (II) defines a user proxy agent, which is configured with code writing capabilities (by providing a code_execution_config). Any code generated is written to the specified #directory work_dir and executed in a Docker container. We also define a task termination condition where the task ends when the user_proxy receives a message that #ends with the string TERMINATE
- (III) The user_proxy sends a message to the assistant agent to start the task. The assistant agent then responds with a plan to address the task, code snippets to execute, and any other relevant information. Given that the user_proxy has code execution capabilities, it can execute the code snippets provided by the assistant agent to address the task and share the results back with the assistant agent
</details>

<details><summary>
2. Building your first multi-agent application
</summary>

## Defining agents in code

- `ConversableAgent` class, list of important parameters for the ConversableAgent class are listed below:

    - `system_message`: System message useful for core agent behaviors
    - `is_termination_msg`: Function to determine if a message terminates the conversation
    - `max_consecutive_auto_reply`: Maximum consecutive auto replies
    - `human_input_mode`: Determines when to request human input (e.g., always, never, or just before a task terminates)
    - `function_map`: Mapping names to callable functions. This wraps the OpenAI tool calling functionality
    - `code_execution_config`: Configuration for code execution
    - `llm_config`: Configuration for LLM-based auto replies
 
- `UserProxyAgent`
  - `human_input_mode` set to `ALWAYS`
  - `llm_config` set to `False`
- `AssistantAgent`
  - `human_input_mode` is set by default to `NEVER`
- `GroupChat`
  - An abstraction to enable groups of ConversableAgents to collaborate on a task, with some plumbing to orchestrate their interactions (e.g., determining which agent speaks/acts next), maximum rounds in a conversation, etc.
  - GroupChat is wrapped by a GroupChatManager object which inherits from the ConversableAgent class
  - GroupChat abstraction receives a message, it broadcasts it to all agents, selects the next speaker based on the group chat orchestration policy, enables a turn for the selected speaker, checks for termination conditions, and continues this process until a termination condition is met


## Agent response strategies

- each agent can send or receive messages using the `send` and `receive` methods, to send a message to an agent, its 'receive' method is called
- Several agent response strategies to respond to messages they receive, as listed below:
  - leverage an LLM to generate text or code based on the conversation context, In AutoGen, we can set LLMs configurations in `AssistantAgent`
  - use external tools or APIs to perform specific actions, In AutoGen, the `code_execution_config` parameter can be used to grant an agent access to a code executor,or given access to functions that can be called to address a task
  - use custom logic response, to perform specific actions or tasks that are not covered by generative AI or tools, custom logic can be used to implement task-specific rules, decision-making processes, data processing or transformation, or other specialized tasks
  - use human feedback, enable agents to request input or clarification from human users in cases where the problem is too complex, ambiguous, or inappropriate for the agent to handle without explicit approval, In AutoGen, the 1human_input_mode` parameter can be set when initializing an agent. This setting determines when the agent should request human input, such as always, never, or only before a task terminates

## Controlling Agent Behaviors via a System Message

In AutoGen, the `AssistantAgent` has a default system message designed to guide the agent in generating code to solve a task. It is helpful to review this system message and extend it to fit the specific needs of your application.

```md
You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the
content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is
ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step
uses your language skill. When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action
beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't
use a code block if it's not intended to be executed by the user.

If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't
include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when
relevant. Check the execution result returned by the user.

If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code
changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit
your assumption, collect additional info you need, and think of a different approach to try.

When you find an answer, verify the answer carefully. Include verifiable
evidence in your response if possible.

Reply "TERMINATE" in the end when everything is done.

```
- The AssistantAgent is configured by default to implicitly generate plans (when none are provided) and create code to solve tasks.
- The system message is used to control task termination behaviors. Specifically, when the Assistant Agent receives a message and, based on the message and conversation history, determines that the task is complete, it will output a TERMINATE string as the last word in its response. This is a common pattern in AutoGen applications.
  

## Defining an Agent Workflow in AutoGen

**Ex:** a simple workflow with two agents: a UserProxyAgent and an AssistantAgent. The UserProxyAgent will act as a stand-in for a user, while the AssistantAgent will serve as a helper agent capable of generating responses using an LLM.

**I) define the UserProxyAgent**
- define a `UserProxyAgent` named user_proxy
- human_input_mode is set to `NEVER`, meaning the agent will not prompt for human input
- The `is_termination_msg` function checks if the last word in the message is "TERMINATE" and returns True if so.

```py
user_proxy = UserProxyAgent(
    name = "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    )

```
**II) define the AssistantAgent**

- define an AssistantAgent named assistant
- The agent is configured with an LLM configuration that specifies the model to use (in this case, GPT-4 Turbo Preview)

```py
llm_config = {"config_list": [{ "model": "gpt-4-turbo-preview"}]}
assistant = AssistantAgent(
    name= "assistant", llm_config=llm_config)

```

In this case, if want to use Thai LLM such as Typhoon1.5, scripts should be as below:

```py
llm_config = {
    "config_list": [{"model": "typhoon-v1.5x-70b-instruct",
                    "api_key": "put_you_api_here" ,
                    "base_url": 'https://api.opentyphoon.ai/v1'
                    }],
    "temperature": 0.9,
    "timeout": 300,
}
assistant = AssistantAgent(
    name= "assistant", llm_config=llm_config)
```

**III) start the conversation by sending a message from the UserProxyAgent to the AssistantAgent**

- The `UserProxyAgent` sends a message to the `AssistantAgent` --> `AssistantAgent` generates a response using the LLM --> The response is then sent back to the `UserProxyAgent` --> the task is complete


```py
user_proxy.initiate_chat(assistant, message="What is the height of the Eiffel Tower?")
```

##  Giving agents access to tools

while LLMs excel at generating text, they are severely limited in their ability to correctly address tasks that require extensive computation or information beyond their training data, halluciation often occur from this

Incidentally, we can improve the performance of our agents by giving them access to tools.

**Ex** we will provide the UserProxyAgent with a code executor tool that can execute code generated by the AssistantAgent.

```py
user_proxy = UserProxyAgent(
    name = "user_proxy",
    code_execution_config={"work_dir": "scratch", "use_docker": True},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    )

user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA
stock price change YTD. Save the result to a file named chart.png.")

```
> In AutoGen, the `code_execution_config` parameter can be used to specify the working directory and whether to use Docker for executing code. By setting the `use_docker` parameter to `True`, the agent can execute code in a secure and isolated environment, reducing the risk of malicious code execution.

## Representing task specific tools to agents

**Example: FUNCTION CALLING (TOOL CALLING) IN LLMS**
> the developer might have a function called get_weather that takes a location and date range as an argument and returns the weather in that location. When the user asks "What is the weather going to be like in San Francisco over the next 4 days?"

- the LLM can generate a function call to `get_weather` with the arguments `location=San Francisco` and `date_range=next 4 days`
- When function calling is enabled, the LLM typically needs a description of the function, the arguments it takes, the expected output and perhaps some examples of how the function is called.
- In return, it provides a structured representation of the function call typically in JSON that can be parsed to call the function.

**Function calling can be broken down into the following steps:**

**I) Function definition:** Define the function that the LLM will call. This includes the function name, description, arguments, and expected output.
**II) Register the function with the LLM (prompt)**  This is simply adding one or more function definitions to the LLM prompt. Most modern LLMs have api parameters (often in JSON) that can be used to present the function to the LLM as part of the prompt. 

In addition there may be settings that (i) force the LLM to always generate a a specific function call, (ii) or automatically select the relevant function based on the task

**III)Generate the function call:** When the LLM receives a message that requires the function, it generates a function call based on the function definition and arguments provided.

**Example: Plot a chart of NVDA and TESLA stock price change YTD. Save the result to a file named chart.png**
- I) define the function that generates images using an LLM

```py
from typing import List
import uuid
import requests
from pathlib import Path
from openai import OpenAI

def generate_and_save_images(query: str, image_size: str = "1024x1024") -> List[str]:
    client = OpenAI()
    response = client.images.generate(model="dall-e-3", prompt=query, n=1, size=image_size)
    saved_files = []
    if response.data:
        for image_data in response.data:
            file_name = str(uuid.uuid4()) + ".png"
            file_path = Path(file_name)
            img_url = image_data.url
            img_response = requests.get(img_url)
            if img_response.status_code == 200:
                with open(file_path, "wb") as img_file:
                    img_file.write(img_response.content)
                    saved_files.append(str(file_path))
    return saved_files

```

- II)  attach this tool to our AssistantAgent such that whenever it addresses a task relevant to generating images, it can call this tool to generate the images, also attach the tool to the UserProxyAgent such that it is aware of the tool and can execute it when needed

```py
from autogen import register_function

# Register the generate_and_save_images function
register_function(
    generate_and_save_images,
    caller=assistant,  # The assistant agent can suggest calls to the
  function
    executor=user_proxy,  # The user proxy agent can execute the
 function call.
    name="generate_and_save_images",  # By default, the function name is used as the tool name.
    description="Generate images using DALL-E",
)

```
## Task termination strategies

- Without a clear definition of task termination, agents may continue to exchange messages indefinitely, leading to inefficiency, resource wastage, or even incorrect task completion
- A solved task is complete (i.e., a set of steps required to solve it have been explored) but a complete task is not necessarily solved
- some strategies for task termination in multi-agent applications:
  -  be instructed to output a TERMINATE string as the last word in its response when it judges the task to be complete, `is_termination_msg` : This condition can trigger termination if the received message satisfies a particular condition, e.g., it contains the word “TERMINATE”. can customize this condition using the is_terminate_msg argument in the constructor of the ConversableAgent class.

```py

agent1 = ConversableAgent(
    "agent1",
    system_message="Your name is agent1 and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
    is_termination_msg=lambda msg: "good bye" in msg["content"].lower(),
)

result = agent1.initiate_chat(agent2, message="agent2, tell me a joke and then say the words GOOD BYE.")

```

```
agent1 (to agent2):

agent2, tell me a joke and then say the words GOOD BYE.

--------------------------------------------------------------------------------
agent2 (to agent1):

Why don't scientists trust atoms?

Because they make up everything!

GOOD BYE!

--------------------------------------------------------------------------------
```

 
  -  configuring rules that terminate the conversation after some set budget of resources has elapsed e.g., number of messages exchanged, time elapsed, LLM tokens consumed etc. This strategy is useful when the task is expected to be completed within a certain budget of resources and exceeding this budget is representative of a potential failure mode.
    -  used the `max_turns` parameter to limit the number of turns
   
```py

      result = agent1.initiate_chat(agent2, message="agent2, tell me a joke.", max_turns=2)

      # agent1 --> agent2, agent2 --> agent1, agent1 --> agent2, agent2 --> agent1
```


   - `max_consecutive_auto_reply` parameter that specifies the maximum number of times the sender can reply


```py

    agent1 = ConversableAgent(
    "agent1",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
    max_consecutive_auto_reply=1,  # Limit the number of consecutive auto-replies.
)

result = agent1.initiate_chat(agent2, message="agent1, tell me a joke.")

# agent1 --> agent2, agent2 --> agent1, agent1 --> agent2, agent2 --> agent1 (agent1 can reply once, not include initial chat)

```

 -  include custom logic that inspects task state, or may be a human-in-the-loop, for this The actual termination depends on the `human_input_mode` argument of the ConversableAgent class.
      - when mode is `NEVER` the termination conditions above will end the conversations
      - when mode is `ALWAYS` or `TERMINATE` , it will not terminate immediately
   
 
## Beyond Two Agents - Orchestrating Teams of Agents

Example: we create
- an `ArtPlanner` that extends the theme into a detailed plan
- an `ArtStyleSelector` that suggests relevant styles and artists
- an `ArtConceptualizer` that creates detailed conceptual descriptions
- an `ArtCataloguer` that generates the exhibition catalog
Each agent can be configured with the appropriate tools and LLMs to solve their specific subtask.

- **I) We define four agents: art_director, style_director, conceptualizer, and cataloguer.
  Each agent is configured with a specific system message that guides its behavior**
```py
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager,register_function

art_director = AssistantAgent(
    name= "art_director", llm_config=llm_config,
    system_message="You are a helpful assistant that is a highly skilled
art director. Given a general description of an art project or theme,
your goal is to provide a detailed and creative plan that develops the
provided description into a very high quality art exhibition. Your plan
should include details on the intended location, the layout of the
exhibition, the experience of the users as they engage etc.")

style_director = AssistantAgent(
    name= "style_director", llm_config=llm_config,
    system_message="You are a helpful assistant that is a highly skilled
art style director. Given a general description of an exhibition plan
provided by the art director, your goal is to design a well thought
out list of 5  art styles and artists that would be relevant to the
exhibition. Each style should be accompanied by a detailed rationale
and grounded in the context of the exhibition plan e.g. via location,
history etc.")

conceptualizer = AssistantAgent(
    name= "conceptualizer", llm_config=llm_config,
    system_message="You are a helpful assistant that is a highly skilled
art conceptualizer. Given a general description of an exhibition plan
 and a list of art styles and artists provided by the style director,
your goal is to create highly detailed conceptual descriptions for 5
art pieces that will be included in the exhibition. Each description
should be unique and should be grounded in the context of the
exhibition plan and the selected art styles. Each art piece should
have a title, a detailed description and some aspirational goals for
the viewer experience. Finally, you should generate an image of each
art piece.")

cataloguer = AssistantAgent(
    name= "cataloguer", llm_config=llm_config,
    system_message="You are a helpful assistant that is a highly skilled
art cataloguer. Given a general description of an exhibition plan, a
list of art styles and artists and detailed conceptual descriptions
for 5 art pieces provided by the conceptualizer, your goal is to create
a detailed exhibition catalog that describes the exhibition, the art
pieces, the artists and the styles. Your catalog should be detailed,
well written and should be suitable for publication. It should include
a cover page, a table of contents, a detailed description of the
exhibition, pictures of each of the art pieces, and the styles. When
all of this is completed, save the catalog as a well formatted and
designed PDF file.")
```
- **II) The user_proxy agent is configured to execute code and not prompt for human input.**
```py

user_proxy = UserProxyAgent(
    name = "user_proxy",
    code_execution_config={"work_dir":
                           "data/scratch",
                           "use_docker": False},
    human_input_mode="NEVER",
    is_termination_msg = lambda x: x.get("content", "").rstrip().endswith("TERMINATE") if x.get("content") else False,
    )
```

- **III) We also register the generate_and_save_images function as a tool for the conceptualizer agent, giving it the ability to generate images.**

```py

register_function(
    generate_and_save_images,
    caller=conceptualizer,  # The assistant agent can suggest calls to the function
    executor=user_proxy,  # The user proxy agent can execute the function call.
    name="generate_and_save_images",  # By default, the function name is used as the tool name.
    description="Generate images using DALL-E",
)
```
- **IV) The GroupChat abstraction orchestrates the conversation between the agents and selecting the next speaker based on the auto speaker selection method - i.e., at each step in the conversation, the group chat manager (driven by an LLM) determines the next agent to speak based on the state of the task, the description of each agent, and the task to be solved.**

```py
groupchat = GroupChat(agents=[art_director, style_director, conceptualizer, cataloguer, user_proxy], messages=[], speaker_selection_method="auto" )
groupchat_manager =  GroupChatManager(groupchat=groupchat, llm_config=llm_config)

```

- **V) start a conversation between the UserProxyAgent and the GroupChatManager to kick off the art exhibition planning process**
  
```py
user_proxy.initiate_chat(groupchat_manager, message="Plan an art exhibition with the theme 'Nature and Technology'.")
```

## Multi-Agent Orchestration Strategies

this is about `speaker_selection_method` parameter in the `GroupChat` abstraction
- **LLM Based Orchestration:** `speaker_selection_method` = `auto`,leverages large language models to determine which agent should take the next action based on the context and state of the ongoing task
  - **Pro:** useful for complex, unstructured tasks where predefined rules may not suffice
  - **Con:**
    -  demands significant computational resources and the performance quality may be contingent on the capability of the LLM
    -  risk of unpredictability as the model might produce unexpected outputs, making debugging and validation more challenging

- **Sequential Orchestration:** `speaker_selection_method` = `round_robin`, This strategy involves a predefined sequential order in which agents will act (may not include loops or any conditional branching capabilities)
  - **Pro:** simpler to debug and validate the process
  - **Con:** can be a drawback in scenarios where the task is dynamic or interdependent. Each agent may need information from others that could be more readily available if acting out of sequence, leading to inefficiency or missed opportunities for optimization


- **Random Orchestration:** `speaker_selection_method` = `random`, next agent to act is selected at random, This approach is typically used as a baseline for comparison with other strategies.
  - **Pro:** be useful for tasks where the order of actions is not critical or when the task is highly parallelizable
  - **Con:** the potential for inefficiency, as necessary steps might be delayed or skipped over in the randomness. This lack of structure can also make it difficult to track progress and debug issues, and it may lead to suboptimal use of resources as agents may end up working on irrelevant tasks or duplicating efforts
  
- **Graph Based Orchestration:** `speaker_selection_method` = `auto`, and a dictionary representing the orchestration graph `allowed_or_disallowed_speaker_transitions` is passed in as an argument, the developer defines a graph that encodes valid transitions between agents based on the task requirements. The graph can represent dependencies between agents, parallel actions, and termination conditions
  - **Pro:** useful for complex tasks with multiple interdependent steps
  - **Con:** setting up and maintaining the graph can be time-consuming, and changes to the task structure may require significant modifications to the graph

## Memory in agent
agents can maintain memory in two ways as shown in figure below: 

1. short-term memory via in-memory lists or metadata databases, typically implemented as a message history list that stores the last N messages (and actions) exchanged between agents. The history list is updated with each new message/action and is part of the prompt that is passed to the LLM model to generate agent responses.
In AutoGen, we can simulate additions to an agent’s history object by calling their send method without requesting a reply i.e. a silent send.

```py
history = [
    {"role": "user", "content": "What is the height of the Eiffel Tower?"},
    {"role": "assistant", "content": "The height of the Eiffel Tower is
324 meters."},
]
for message in history:
    if message["role"] == "user":
        user_proxy.send(message=message, request_reply=False)
    else:
        assistant.send(message=message, request_reply=False)

#AssistantAgent will use the history list to provide a contextually relevant response to the question
user_proxy.initiate_chat(assistant, message="When was it built?")
```

3.  long-term memory via vector databases, A common pattern adopted in implementing long term memory is retrieval augmented generation where data (across multiple modalities) is first converted to a fixed length vector representation and stored in a vector database.



![Alt text](image/Agent-Memory.png)
> Picture Reference: *Dibia, Victor, and Chi Wang. *Multi-Agent Systems with AutoGen*. Manning Publications, 2024.*


## Allowing Human Feedback in Agents

![Alt text](image/human_in_agent.png)

> Microsoft. (n.d.). Allowing Human Feedback in Agents. Retrieved August 6, 2024, from https://microsoft.github.io/autogen/docs/tutorial/human-in-the-loop

**The mode is specified through the `human_input_mode` argument of the ConversableAgent. The three modes are:**

- `NEVER` : human input is never requested, the termination conditions(message-based termination condition) are used to terminate. This mode is useful when you want your agents to act fully autonomously

**Example**

```py

import os

from autogen import ConversableAgent

agent_with_number = ConversableAgent(
    "agent_with_number",
    system_message="You are playing a game of guess-my-number. You have the "
    "number 53 in your mind, and I will try to guess it. "
    "If I guess too high, say 'too high', if I guess too low, say 'too low'. ",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    is_termination_msg=lambda msg: "53" in msg["content"],  # terminate if the number is guessed by the other agent
    human_input_mode="NEVER",  # never ask for human input
)

agent_guess_number = ConversableAgent(
    "agent_guess_number",
    system_message="I have a number in my mind, and you will try to guess it. "
    "If I say 'too high', you should guess a lower number. If I say 'too low', "
    "you should guess a higher number. ",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    human_input_mode="NEVER",
)

result = agent_with_number.initiate_chat(
    agent_guess_number,
    message="I have a number between 1 and 100. Guess it!",
)
```
```agent_with_number (to agent_guess_number):

I have a number between 1 and 100. Guess it!

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 50?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

Too low.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 75?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

Too high.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 63?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

Too high.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 57?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

Too high.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 54?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

Too high.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 52?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

Too low.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 53?

--------------------------------------------------------------------------------
```
  
- `TERMINATE` (default): human input is only requested when a termination condition is met. Note that in this mode if the human chooses to intercept and reply, the conversation continues and the counter used by max_consecutive_auto_reply is reset.

**Ex** playing the same game again, but this time the guessing agent will only have two chances to guess the number, and if it fails, the human will be asked to provide feedback, and the guessing agent gets two more chances. If the correct number is guessed eventually, the conversation will be terminated.

```py
agent_with_number = ConversableAgent(
    "agent_with_number",
    system_message="You are playing a game of guess-my-number. "
    "In the first game, you have the "
    "number 53 in your mind, and I will try to guess it. "
    "If I guess too high, say 'too high', if I guess too low, say 'too low'. ",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    max_consecutive_auto_reply=1,  # maximum number of consecutive auto-replies before asking for human input
    is_termination_msg=lambda msg: "53" in msg["content"],  # terminate if the number is guessed by the other agent
    human_input_mode="TERMINATE",  # ask for human input until the game is terminated
)

agent_guess_number = ConversableAgent(
    "agent_guess_number",
    system_message="I have a number in my mind, and you will try to guess it. "
    "If I say 'too high', you should guess a lower number. If I say 'too low', "
    "you should guess a higher number. ",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    human_input_mode="NEVER",
)

result = agent_with_number.initiate_chat(
    agent_guess_number,
    message="I have a number between 1 and 100. Guess it!",
)
```

```
agent_with_number (to agent_guess_number):

I have a number between 1 and 100. Guess it!

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 50?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
agent_with_number (to agent_guess_number):

Too low.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 75?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

It is too high my friend. 

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 60?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
agent_with_number (to agent_guess_number):

Too high.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 55?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

still too high, but you are very close.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 52?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
agent_with_number (to agent_guess_number):

Too low.

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 54?

--------------------------------------------------------------------------------
agent_with_number (to agent_guess_number):

Almost there! 

--------------------------------------------------------------------------------
agent_guess_number (to agent_with_number):

Is it 53?

--------------------------------------------------------------------------------
```
Each time after one auto-reply from the agent with the number, the human was asked to provide feedback. as you can see on these messages from human
- When the agent guessed “74”, the human said “It is too high my friend.”
- When the agent guessed “55”, the human said “still too high, but you are very close.”
- When the agent guessed “54”, the human said “Almost there!”

Once the human provided feedback, the counter was reset. The conversation was terminated after the agent correctly guessed “53”.




- `ALWAYS` : human input is always requested and the human can choose to skip and trigger an auto-reply, intercept and provide feedback, or terminate the conversation. Note that in this mode termination based on `max_consecutive_auto_reply` is ignored.

```py
human_proxy = ConversableAgent(
    "human_proxy",
    llm_config=False,  # no LLM used for human proxy
    human_input_mode="ALWAYS",  # always ask for human input
)

# Start a chat with the agent with number with an initial guess.
result = human_proxy.initiate_chat(
    agent_with_number,  # this is the same agent with the number as before
    message="10",
)
```

```
human_proxy (to agent_with_number):

10

--------------------------------------------------------------------------------
agent_with_number (to human_proxy):

Too low.

--------------------------------------------------------------------------------
human_proxy (to agent_with_number):

79

--------------------------------------------------------------------------------
agent_with_number (to human_proxy):

Too high.

--------------------------------------------------------------------------------
human_proxy (to agent_with_number):

76

--------------------------------------------------------------------------------
agent_with_number (to human_proxy):

Too high.

--------------------------------------------------------------------------------
human_proxy (to agent_with_number):

I give up

--------------------------------------------------------------------------------
agent_with_number (to human_proxy):

That's okay! The number I was thinking of was 53.

--------------------------------------------------------------------------------
```


</details>

<details><summary>
3. Agent with Code Executors</summary>




</details>



