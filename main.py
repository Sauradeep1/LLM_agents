from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
import operator
from typing import Annotated, TypedDict, Union
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import create_react_agent
from typing import List, Sequence
import os
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from dotenv import load_dotenv
load_dotenv()
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

print('executed this far')

#os.environ["OPENAI_API_KEY"] = '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'

llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)
