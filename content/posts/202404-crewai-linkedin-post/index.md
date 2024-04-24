---
title: "Automating my LinkedIn posts with CrewAI"
summary: "Creating LinkedIn posts that replicate my writing style using Selenium, crewAI, GPT-3.5-turbo and Mistral Large"
description: "Creating LinkedIn posts that replicate my writing style using Selenium, CrewAI, GPT-3.5-turbo and Mistral Large"
categories: ["LLM", "Agents", "crewAI"]
date: 2024-04-22
draft: false
showauthor: false
---

We all know LinkedIn, the biggest social network for professionals. And we all know that, sometimes, it's very 
difficult to come up with engaging and quality content to share with the community. But ... what if we try to use
[crewAI](https://www.crewai.com/) to help us in these situations? Well, just to let you know, I've already 
tried it, and it works reasonably well. In this article, I'll show you the process I followed. 

Enjoy!

---

## Wait ... crewAI? 😰

Don't know what crewAI is? Don't worry! The official documentation gives a very nice and self-contained description
of crewAI's goal:

> Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence,
> CrewAI empowers agents to work together seamlessly, tackling complex tasks.

It may sound difficult but, believe me, one of the BIG advantages of **crewAI** is its extreme simplicity. Just pay
attention to these three concepts: **Agent**, **Task**, **Tool** and **Crew**. If you understand those
you'll be creating your own **crewAI** applications right after you finish this article  😀

### Agent

The agent is like a member of a team, with its specific skills and a particular goal to do (a task, as we'll see 
later). Agents have different roles (we'll create three agents in this article, with three different roles) and 
possess a differentiating backstory, which further improves the collaboration dynamics.

It's very easy to create an agent in crewAI. Let's take as example one agent I created a few weeks ago (see this
article if you are curious):

```python
from crewai import Agent

pycharm_agent = Agent(
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

As you can see, I've provided a `role` (my agent is a Python Programmer with preference for PyCharm over VSCode), 
a `goal` and a `backstory` (more about tools later). If you are curious about the `allow_delegation` param, it 
just enables / disables the action of delegating a task between different agents.

### Task

As I said before, an agent is assigned to a task. But, **what is a task?** You can think of it as specific assignments
completed by agents. These assignments provide all the necessary details such as which agent is responsible, the 
required tools (again, more on tools later, I swear ), etc. Let's see an example:

```python
from crewai import Task
from textwrap import dedent

pycharm_research_task =  Task(
            description=dedent("""
            Conduct a comprehensive research on the advantages of PyCharm over VSCode for Python development
            """),
            expected_output=dedent("""
            A detailed report on why PyCharm is a better Python IDE than VSCode and all the 
            disadvantages of using VSCode for Python development.
            """),
            agent=pycharm_agent
        )
```

Remember the Pycharm Agent? What you see above is the assigned task. **crewAI** makes this very simple, as we only
need to provide the task `description`, the `expected_output` and the assigned agent. Easy!!


### Tool

Some tasks are easy to solve, like telling me what's the capital of France 🥐. But, what happens if the agent needs
to find the share price of NVIDIA right now? Without any connection to the outside world, the LLM won't be able to
give as the answer! 

So ... tools to the rescue! 🦸‍♂️

A tool is nothing more than an interface that an agent can use to interact with the world. To use these tools, we 
have three options:

* [crewAI Tools](https://docs.crewai.com/core-concepts/Tools/#available-crewai-tools)
* [LangChain Tools](https://docs.crewai.com/core-concepts/Tools/#using-langchain-tools)
* [Custom Tools](https://docs.crewai.com/core-concepts/Tools/#creating-your-own-tools) (we'll create a custom tool later )

So, as you can see, there are plenty of options! 😁


### Crew

Last concept: **the Crew**. A crew is nothing more than putting all that we have seen until this point together. 
I took the liberty of creating the following diagram, which captures reasonably well (at least I think so 😅) the
overall behaviour of a crew.

![Alt text][crewai-architecture]

Do you want some code example? Let's take the one from **crewAI**'s official documentation:

```python
from crewai import Crew, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun

researcher = Agent(
    role='Senior Research Analyst',
    goal='Discover innovative AI technologies',
    tools=[DuckDuckGoSearchRun()]
)

writer = Agent(
    role='Content Writer',
    goal='Write engaging articles on AI discoveries',
    verbose=True
)

# Create tasks for the agents
research_task = Task(
    description='Identify breakthrough AI technologies',
    agent=researcher
)

write_article_task = Task(
    description='Draft an article on the latest AI technologies',
    agent=writer
)

# Assemble the crew with a sequential process
my_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_article_task],
    process=Process.sequential,
    full_output=True,
    verbose=True,
)

my_crew.kickoff()
```

Let's comment the code above.

In a few words, the crew is composed of **two agents*; the `researcher` and the `writer`. The former must search for 
**breakthrough AI technologies** (`research_task`) with the help of the DuckDuckGo Search Engine (`DuckDuckGoSearchRun()` tool)
The latter is in charge of writing an article from all the information gathered by the `researcher`.

To properly create crew, we need to use the `Crew` class, which is expecting a list of `agents`, `tasks` and the type 
of process (by default it's `Process.Sequentail` but in some cases we may need [other options](https://docs.crewai.com/core-concepts/Processes/)).

And ... I think we are done! By now, you already know the building blocks of crewAI so it's time to get to the funny
part. 

Let's automate my LinkedIn posts 😎


## Creating LinkedIn Posts


The approach I followed is really simple, as you can see in the diagram below.


The only thing a bit complex is the tool used by the first of the agents, because it is a custom tool that I have 
built around Selenium. In case you don't know it, Selenium is a browser automation tool that I will use to scrape
LinkedIn posts from my profile 

![Alt text][linkedin-crewai-architecture]



### Selenium Tool: Scraping my LinkedIn profile

### First Agent: LinkedIn Scraper Ninja 🥷

### Second Agent: Web Researcher 🧑‍🔬

### Third Agent: My own Doppelgänger 

![Alt text][spiderman-meme]

### Defining the Tasks

### Build the Crew

## Conclusion



[crewai-architecture]: img/crewai.drawio.correct.svg
[linkedin-crewai-architecture]: img/crewai.linkedin.influencer.diagram.svg
[spiderman-meme]: img/spiderman_meme.png 
