---
title: "Automating my LinkedIn posts with CrewAI"
summary: "Creating LinkedIn posts that replicate my writing style using Selenium, crewAI, GPT-3.5-turbo and Mistral Large"
description: "Creating LinkedIn posts that replicate my writing style using Selenium, CrewAI, GPT-3.5-turbo and Mistral Large"
categories: ["LLM", "Agents", "crewAI"]
date: 2024-04-22
draft: false
showauthor: false
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/oIb5JqZ5ylA" frameborder="0" allowfullscreen></iframe>


> If you prefer to go directly into the code, there's a [GitHub repo available](https://github.com/neural-maze/crewai_linkedin_post/tree/master)!!

We all know LinkedIn, the biggest social network for professionals. And we all know that, sometimes, it's very 
difficult to come up with engaging and quality content to share with the community. **But ... what if we try to use
[crewAI](https://www.crewai.com/) to help us in these situations?** 

Well, just to let you know, I've already 
tried it, and it works reasonably well. In this article, I'll show you the process I followed. 


---

## Wait ... crewAI? 😰

Don't know what crewAI is? Don't worry! The official documentation gives a very nice and self-contained description
of crewAI's goal:

> Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence,
> CrewAI empowers agents to work together seamlessly, tackling complex tasks.

It may sound difficult but, believe me, one of the **BIG** advantages of **crewAI** is its extreme simplicity. Just pay
attention to these four concepts: **Agent**, **Task**, **Tool** and **Crew**. If you understand those
you'll be creating your own **crewAI** applications right after you finish this article  😀

### Agent

**The agent is like a member of a team, with its specific skills and a particular goal achieve**. In addition, agents have roles and 
exhibit defining backstories, which further improves the collaboration dynamics.

It's very easy to create an agent in crewAI. Let's take as example one agent I created a few weeks ago ([see this
article](https://neural-maze.github.io/blog/posts/202404-crewai-vscode-pycharm/) if you are curious):

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
just enables/disables the delegation of tasks between different agents.

### Task

As I said before, an agent is assigned to a task. But, **what is a task?** 

**You can think of tasks as specific assignments
completed by agents**. These assignments provide all the necessary details such as which agent is responsible, the 
required tools (again, more on tools later, I swear 🙏), etc. Let's see an example:

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

Some tasks are easy to solve, like telling me the capital of France 🥐

But, what happens if the agent needs  to find the share price of NVIDIA right now? Without any connection to the
outside world, the LLM won't be able to give me the correct answer! 

So ... tools to the rescue! 🦸‍♂️

**A tool is an interface that an agent can use to interact with the world.** To use tools inside crewAI, we 
have three options:

* [Native crewAI Tools](https://docs.crewai.com/core-concepts/Tools/#available-crewai-tools)
* [LangChain Tools](https://docs.crewai.com/core-concepts/Tools/#using-langchain-tools)
* [Custom Tools](https://docs.crewai.com/core-concepts/Tools/#creating-your-own-tools) (we'll create a custom tool later )

So, as you can see, there are plenty of options! 😁


### Crew

Last concept: **the Crew**. 

**A crew is nothing more than putting together all the components we have seen**. I took the liberty of creating the
following diagram, which captures reasonably well (at least I think so 😅) the overall behaviour of a crew.

![Alt text][crewai-architecture]

Do you want some code example? Let's take one from **crewAI**'s official documentation:

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

Let's describe the code above.

In a few words, the crew is composed of **two agents**: the `researcher` and the `writer`. The former must search for 
**breakthrough AI technologies** (`research_task`) with the help of the DuckDuckGo Search Engine (`DuckDuckGoSearchRun` tool)
The latter is in charge of writing an article based on all the information gathered by the `researcher`.

To properly create crew, we need to use the `Crew` class, which is expecting a list of `agents`, `tasks` and the type 
of process (by default it's `Process.sequential` but in some cases we may need [other options](https://docs.crewai.com/core-concepts/Processes/)).

And ... I think we are done! At this point, you already know the building blocks of crewAI, so it's time to get to the funny
part. 

Let's automate my LinkedIn posts 😎


## Creating LinkedIn Posts


The crew I assembled consists of these components:

* **3 agents** 
* **1 custom tool (it uses Selenium for scraping LinkedIn profiles)**
* **2 crewAI tools (for doing web research basically)**
* **3 tasks**

In the following diagram, you can see how the application is going to work.

![Alt text][linkedin-crewai-architecture]

So, first of all, we need one agent (`LinkedIn Scraper Agent`) in charge of scraping some of my LinkedIn profiles. 
Why? **Because I want the generated LinkedIn post to follow my writing-style!!**. The agent will interact with
LinkedIn using a custom tool I've created, that basically wraps Selenium. 

We also need
another agent (`Web Researcher Agent`) for fetching relevant information about a given topic. In my case, I chose
the recent release of Llama 3 by Meta AI, but you could choose whatever you want. This agent is able to use the
`SerperDevTool` and the `ScrapeWebsiteTool`, both available as native crewAI tools.

After that, the last agent 
(`LinkedIn Influencer Agent`) will gather all the information provided by the previous agents and will generate
a LinkedIn post. **Ideally, it should be a LinkedIn post about Llama 3 following my writing style.**

Now that you know my general approach, let's investigate each component.


### Selenium Tool: Scraping my LinkedIn profile

This is probably the trickiest part of the application, since you need to know a little bit about what is [Selenium](https://www.selenium.dev/) and
how it works. What my custom tool does is to log into my LinkedIn account (using my email and password, stored in a `.env`
file) and parse relevant posts from my feed. You can see the tool implementation [here](https://github.com/neural-maze/crewai_linkedin_post/blob/master/tools/linkedin.py).

### First Agent: LinkedIn Scraper Ninja 🥷

As I said before, the first agent will be using the Selenium tool to scrape some of my LinkedIn profiles. The implementation
is pretty straightforward:

```python
linkedin_scraper_agent = Agent(
    role="LinkedIn Post Scraper",
    goal="Your goal is to scrape a LinkedIn profile to get a list of posts from the given profile",
    tools=[scrape_linkedin_posts_tool],
    backstory=dedent(
        """
        You are an experienced programmer who excels at web scraping. 
        """
    ),
    verbose=True,
    allow_delegation=False,
    llm=openai_llm
)
```

You may have noticed that I've added a `llm` param that I didn't show you before. Good catch!! 😀. This param is 
useful when you want to use different LLMs (by default crewAI uses gpt4). In my case, I've chosen **gpt-3.5-turbo**
for this agent. You can define the LLMs just using LangChain abstractions, like this:

```python
openai_llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-3.5-turbo-0125")
mistral_llm = ChatMistralAI(api_key=os.environ.get("MISTRAL_API_KEY"), model="mistral-large-latest")
```

### Second Agent: Web Researcher 🧑‍🔬

This is even easier to implement than the previous one, since we don't need any custom tool. We are going
to ask the agent to search for some key differences between Llama 2 and Llama 3.

```python
web_researcher_agent = Agent(
    role="Web Researcher",
    goal="Your goal is to search for relevant content about the comparison between Llama 2 and Llama 3",
    tools=[scrape_website_tool, search_tool],
    backstory=dedent(
        """
        You are proficient at searching for specific topics in the web, selecting those that provide
        more value and information.
        """
    ),
    verbose=True,
    allow_delegation=False,
    llm=openai_llm
)
```

### Third Agent: My own Doppelgänger 

![Alt text][spiderman-meme]

This agent has to deal with the information gathered by the two previous agents and write a high quality and engaging
LinkedIn post about the differences between Llama 2 and Llama 3. Additionally, the LinkedIn post **must follow
my writing style (I don't want the typical ChatGPT robotic style 😂)**. As you can see, I've chosen a [Mistral](https://mistral.ai/)
model for this agent.

```
doppelganger_agent = Agent(
    role="LinkedIn Post Creator",
    goal="You will create a LinkedIn post comparing Llama 2 and Llama 3 following the writing style "
         "observed in the LinkedIn posts scraped by the LinkedIn Post Scraper.",
    backstory=dedent(
        """
        You are an expert in writing LinkedIn posts replicating any influencer style
        """
    ),
    verbose=True,
    allow_delegation=False,
    llm=mistral_llm
)
```

### Defining the Tasks

I've defined the following tasks. Not much comment about them, since they are pretty self-explanatory ...

```python
scrape_linkedin_task = Task(
    description=dedent(
        "Scrape a LinkedIn profile to get some relevant posts"),
    expected_output=dedent("A list of LinkedIn posts obtained from a LinkedIn profile"),
    agent=linkedin_scraper_agent,
)

web_research_task = Task(
    description=dedent(
        "Get valuable and high quality web information about the comparison between Llama 2 and Llama 3"),
    expected_output=dedent("Your task is to gather high quality information about the comparison"
                           " between Llama 2 and Llama 3"),
    agent=web_researcher_agent,
)

create_linkedin_post_task = Task(
    description=dedent(
        "Create a LinkedIn post comparing Llama 2 and Llama 3 following the writing-style "
        "expressed in the scraped LinkedIn posts."
    ),
    expected_output=dedent("A high-quality and engaging LinkedIn post comparing Llama 2 and Llama 3."
                           " The LinkedIn post must follow"
                           " the same writing-style as the one expressed in the scraped LinkedIn posts"),
    agent=doppelganger_agent,
)
```

But there's something missing here ... The thing is that, by default, the execution of
tasks is sequential (remember `Process.sequential`?), but I need the last task to access the context from
the two previous tasks. How can we do this? Easy.

```python
create_linkedin_post_task.context = [scrape_linkedin_task, web_research_task]
```

### Build the Crew

Ok, we're almost done! Let's define the crew now.

```python
crew = Crew(
    agents=[
        linkedin_scraper_agent,
        web_researcher_agent,
        doppelganger_agent
    ],
    tasks=[
        scrape_linkedin_task,
        web_research_task,
        create_linkedin_post_task
    ]
)
```

And ... let's kick it off!!!

```
crew.kickoff()
```


## Conclusion

This is the final result I obtained after running the crew. It's not the best LinkedIn
post out there but, hey, it's pretty good, and it has its own personality (which is just a copy of mine 🙃)

```
🦙 🤯 𝐋𝐋𝐀𝐌𝐀 2 vs 𝐋𝐋𝐀𝐌𝐀 3: 𝐖𝐇𝐎 𝐑𝐄𝐈𝐆𝐍𝐒 𝐒𝐔𝐏𝐑𝐄𝐌𝐄? 🤯 🦙

This week, I've been diving deep into the world of AI models, specifically Llama 2 and Llama 3. At first glance, you might think they're just two versions of the same model, but let's dig deeper 👇

1️⃣ 𝐏𝐞𝐫𝐟𝐨𝐫𝐦𝐚𝐧𝐜𝐞 𝐌𝐞𝐭𝐫𝐢𝐜𝐬: Llama 3 outperforms Llama 2 in various metrics such as MMLU, ARC, and DROP. It's like comparing a cheetah to a leopard in a race - both are fast, but one is just a bit faster!

2️⃣ 𝐓𝐨𝐤𝐞𝐧𝐢𝐳𝐞𝐫: Llama 3 introduces a tokenizer with a vocabulary of 128K tokens, which encodes language more effectively than Llama 2. Think of it as having a larger toolbox to build better sentences.

3️⃣ 𝐒𝐩𝐞𝐜𝐢𝐚𝐥 𝐓𝐨𝐤𝐞𝐧𝐬: Llama 3 is noted for its flexibility in managing special tokens, essential for parsing distinct textual elements. It's like having a Swiss Army knife instead of a regular one - more tools for different tasks!

4️⃣ 𝐏𝐨𝐭𝐞𝐧𝐭𝐢𝐚𝐥 𝐈𝐦𝐩𝐚𝐜𝐭: Llama 3 has been highlighted as a top-performing AI model with the potential to shake up the AI landscape. It's not just an upgrade - it's a game changer!

So, 𝐰𝐡𝐚𝐭 𝐝𝐨𝐞𝐬 𝐢𝐭 𝐫𝐞𝐚𝐥𝐥𝐲 𝐦𝐞𝐚𝐧 𝐟𝐨𝐫 𝐭𝐡𝐞 𝐟𝐮𝐭𝐮𝐫𝐞 𝐨𝐟 𝐀𝐈? Does Llama 3's superior performance signal a new era in AI capabilities? Let me know what you think in the comments 🤔 👇
```

Hope you enjoyed this article. See you around! 👋


[crewai-architecture]: img/crewai.drawio.correct.svg
[linkedin-crewai-architecture]: img/crewai_linkedin_influencer.svg
[spiderman-meme]: img/spiderman_meme.png 
