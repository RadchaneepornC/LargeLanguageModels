## My Lecture about Agentic Workflow with LLMs

<details> <br> <summary> Multi-Agent Systems with AutoGen </summary>



Below is my summarization of **Dibia, Victor, and Chi Wang. *Multi-Agent Systems with AutoGen*. Manning Publications, 2024. https://www.manning.com/books/multi-agent-systems-with-autogen.**

<details><summary>
1. Understanding a Multi-Agent System (MAS)</summary>


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

*Multi-agent systems from Dibia, Victor, and Chi Wang. *Multi-Agent Systems with AutoGen*. Manning Publications, 2024.*


- **autogen core feature** provides us
  - **ConversableAgent:** class with methods  1) for sending and receiving messages between agents and  2) for defining how an agent acts on a received message, it includes a set of built-in agent classes
    - **UserProxyAgent**: has default capabilities to execute code and solicit human input
    - **AssistantAgent**: configured to address tasks using an LLM
    - **GroupChatManager** : serves as a container abstraction for enabling interactions between groups of agents





</details>

</details>

