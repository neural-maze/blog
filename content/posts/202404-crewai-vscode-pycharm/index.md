---
title: "CrewAI decides: VSCode or PyCharm?"
summary: "A funny weekend project where I created a crew of agents to give me a final verdict about the best Python IDE"
description: "A funny weekend project where I created a crew of agents to give me a final verdict about the best Python IDE"
categories: ["LLM", "Agents", "crewAI"]
date: 2024-04-14
draft: false
showauthor: false
---

The other day I was playing around with one of the hottest frameworks right now: [**CrewAI**](https://www.crewai.com/).

As defined in the documentation:

> Cutting-edge framework for orchestrating role-playing, autonomous AI agents.
By fostering collaborative intelligence, CrewAI empowers agents to work together
seamlessly, tackling complex tasks

So, as you can imagine, CrewAI is the right tool if you want to deal with [multi-agentic patterns](https://www.youtube.com/watch?v=sal78ACtGTc&ab_channel=SequoiaCapital) (
drawing inspiration from Andrew Ng's video on agentic patterns).

One **(huge)** advantage of CrewAI over other multi-agent frameworks (e.g. [Autogen](https://microsoft.github.io/autogen/), [ChatDev](https://github.com/OpenBMB/ChatDev)) is its simplicity.
To start creating your own **crews**, you just need to understand three key concepts, namely [Agent](https://docs.crewai.com/core-concepts/Agents/),
[Task](https://docs.crewai.com/core-concepts/Tasks/) and [Tool](https://docs.crewai.com/core-concepts/Tools/).

To make it easier for you, I have created the following simplified diagram 👇

![Alt text][image-2]

The **crew** above consists of three agents, each represented by the orange robots 🤖. As you can see, each agent
is assigned to a task. In order to accomplish their respective goals, these agents may require the use of 
specific tools (such as a web scraper, a search engine, a markdown formatter, etc.) and collaborate with one another. 
This collaboration could involve waiting for another agent's analysis or engaging in discussions from different perspectives. 
**Ultimately, their collective efforts aim to present the user with the final output.**

Easy, right? At least the high level description 😂

Now that you know the basics of CrewAI, let me detail the "problem" I wanted to solve in this
article (although it is pretty clear from the title).

**How about asking CrewAI to settle, once and for all, which IDE is better for programming in Python: PyCharm or VS Code?**

Let's approach this problem from a multi-agent perspective!

## Agents 🤖

As I said before, one of the central concepts in CrewAI is the concept of Agent. For this
application, we're going to define 3 agents.

### PyCharm Agent

Our first agent has a clear goal: **to conduct a thorough investigation into why PyCharm is a better
IDE than VS Code**

As you can see below, it's very easy to define an Agent in CrewAI. We just need to provide
a **role**, a **goal** , a **list of tools** that the agent can use (more on this later) and a **backstory**.

We can also set the `verbose` to `True` if you want to follow the application logs.

```python
def pycharm_agent():
    return Agent(
        role="PyCharm Python Programmer",
        goal="Conduct thorough research on why PyCharm is a BETTER IDE than VSCode for Python development",
        tools=TOOLS,
        backstory=dedent("""
        As a Python developer deeply enamored with the PyCharm IDE, you cherish its interface, debugging 
        capabilities, and the seamless integration of features like the terminal. It frustrates you when
        fellow programmers assert that VSCode surpasses PyCharm. Your mission is to persuade the Python
        IDE judge of PyCharm's superiority over VSCode.
        """),
        verbose=True,
        allow_delegation=False
    )
```

### VS Code Agent

Let's follow the same approach as before, but for the VS Code Agent.

```python
def vscode_agent():
    return Agent(
        role="VSCode Python Programmer",
        goal="Conduct thorough research on why VScode is a BETTER IDE than PyCharm for Python development",
        tools=TOOLS,
        backstory=dedent("""
        As a devoted VSCode programmer, you find its versatility, customizability, and extensive extension library
        irresistible. You feel a pang of exasperation when others claim that PyCharm reigns supreme. Your mission
        is to persuade the Python IDE judge of VSCode's superiority over PyCharm.
        """),
        verbose=True,
        allow_delegation=False,
    )
```

### IDE Judge Agent

Finally, let's create an agent to act as an impartial judge in this discussion. Given the
research carried out by both the PyCharm agent and the VS Code agent, this agent will be in charge of
providing a final verdict to the never-ending debate! 🤣

```python
def ide_judge_agent():
    return Agent(
        role="Python IDE judge",
        goal="Compile all gathered information about pros and cons of both PyCharm and VSCode into a concise,"
             "informative briefing document. You need to provide the final answer on which IDE is better for Python"
             ": VSCode or PyCharm?",
        tools=TOOLS,
        backstory="As the Python IDE Judge your role is to be impartial in your final decision about which"
                  " IDE is better for Python development: PyCharm or VSCode?",
        verbose=True
    )
```

Ok, so we have the agents ready to go! It's time to show you how to create a custom tool
in CrewAI 🛠️

# Tools 🛠️

For this application, we don't need anything particularly complex, just a way to search for content related
to PyCharm / VS Code on the web. In fact, I simply reused one of the [CrewAI Examples](https://github.com/joaomdmoura/crewAI-examples/blob/main/prep-for-a-meeting/tools/ExaSearchTool.py),
but updated the code to use CrewAI tools instead of Langchain tools. 

```python
import os
from exa_py import Exa
from crewai_tools import tool
from dotenv import load_dotenv

load_dotenv()

EXA = Exa(api_key=os.environ["EXA_API_KEY"])


@tool
def search(query: str):
	"""Search for a webpage based on the provided query"""
	return EXA.search(f"{query}", use_autoprompt=True, num_results=3)


@tool
def find_similar(url: str):
	"""Search for webpages similar to a given URL.
	The url passed in should be a URL returned from `search`.
	"""
	return EXA.find_similar(url, num_results=10)


@tool
def get_contents(ids: list):
	"""Get the contents of a webpage.
	The ids must be passed in as a list, a list of ids returned from `search`.
	"""
	contents = str(EXA.get_contents(ids))
	contents = contents.split("URL:")
	contents = [content[:1000] for content in contents]
	return "\n\n".join(contents)


TOOLS = [search, find_similar, get_contents]
```

As you can see in the code above, we are defining three tools (indicated by the `@tool` decorator) 
, all of them related to the [Exa search engine](https://exa.ai/search?c=all). By using these tools,
the agents will be able to look for promising information about PyCharm / VS Code using the Exa search engine. 

There's only one thing left ... the tasks!


## Tasks 🏗️

Defining tasks is also very simple in CrewAI. To define a task we only need to provide a 
**description**, the **expected output** and finally the **agent** (remember the diagram at
the beginning, an agent is responsible for a task).

We're going to define three tasks for our use case (as you can see, they are pretty straightforward)

```python
from textwrap import dedent
from crewai import Task


class VSCodeVSPyCharmTasks:
    @staticmethod
    def pycharm_research_task(agent):
        return Task(
            description=dedent("""
            Conduct a comprehensive research on the advantages of PyCharm over VSCode for Python development
            """),
            expected_output=dedent("""
            A detailed report on why PyCharm is a better Python IDE than VSCode and all the 
            disadvantages of using VSCode for Python development.
            """),
            agent=agent
        )

    @staticmethod
    def vscode_research_task(agent):
        return Task(
            description=dedent("""
            Conduct a comprehensive research on the advantages of VSCode over PyCharm for Python development
            """),
            expected_output=dedent("""
            A detailed report on why VSCode is a better Python IDE than PyCharm and all the 
            disadvantages of using PyCharm for Python development.
            """),
            agent=agent
        )

    @staticmethod
    def final_verdict_task(agent):
        return Task(
            description=dedent("""
            Gather all the information provided about the advantages and disadvantages of both PyCharm and VSCode,
            analyse it and summarise it into an informative briefing document. Give a final verdict on which
            IDE is better for Python development based on all the provided information.
            """),
            expected_output=dedent("""
            A summary of the advantages and disadvantages of both PyCharm and VSCode and a final verdict on
            which one is better for Python development.
            """),
            agent=agent
        )

```

We are almost there! During the article we have defined our agents, our tasks and our tools, so ...
what's left? 🤔

Well, we need to create the crew, i.e. make all the components work together!

Let's import each of the objects we've been creating and connect them together
using the `Crew` object. Then, simply `kickoff` the application!!

```python
from crewai import Crew

from tasks import VSCodeVSPyCharmTasks
from agents import VSCodeVSPyCharmAgents

tasks = VSCodeVSPyCharmTasks()
agents = VSCodeVSPyCharmAgents()

pycharm_agent = agents.pycharm_agent()
vscode_agent = agents.vscode_agent()
ide_judge_agent = agents.ide_judge_agent()

pycharm_research_task = tasks.pycharm_research_task(pycharm_agent)
vscode_research_task = tasks.vscode_research_task(vscode_agent)
final_verdict_task = tasks.final_verdict_task(ide_judge_agent)

final_verdict_task.context = [pycharm_research_task, vscode_research_task]

crew = Crew(
    agents=[
        pycharm_agent,
        vscode_agent,
        ide_judge_agent
    ],
    tasks=[
        pycharm_research_task,
        vscode_research_task,
        final_verdict_task
    ]
)

result = crew.kickoff()
```

## Final Verdict? ⚖️

So let's reveal the final verdict (SPOILER ALERT 🚨), which IDE is better according to CrewAI?
Well, sorry to disappoint you reader, but CrewAI's final response is as disappointing as it is 
extremely reasonable:

> In conclusion, it's hard to definitely state whether PyCharm or VSCode is better for Python development as both have substantial features that enhance the development experience. The choice ultimately depends on the individual developer's needs and preferences.

**Ladies and gentlemen, it seems the debate continues ... 🔥**


[image-1]: img/crewai_landing_page.png "CrewAI is one of the top Multi AI Agent frameworks right now"
[image-2]: img/crewai.drawio.correct.svg "A simplified architecture of CrewAI"
