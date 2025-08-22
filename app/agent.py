from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from asteval import Interpreter

from app.rag import retriever, _format_docs
from app.memory import get_memory
from app.settings import settings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@tool
def search_corpus(query: str, k: int = 4) -> str:
    """RAG tool: search docs in vector DB"""
    r = retriever()
    docs = r.invoke(query)
    return _format_docs(docs[:k])

@tool
def python_calc(expression: str) -> str:
    """Safely eval a math expression"""
    aeval = Interpreter(minimal=True, no_print=True)
    try:
        return str(aeval(expression))
    except Exception as e:
        return f"Error: {e}"

TOOLS = [search_corpus, python_calc]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an agent. Use tools when needed. Cite sources like [1]."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])


TOOLS = [search_corpus, python_calc]

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful agent. You can call tools when needed.\n"
     "Prefer using 'search_corpus' before answering questions that need context.\n"
     "When you cite, use [1], [2] corresponding to results returned by the tool."
    ),
    MessagesPlaceholder("chat_history"),      # <-- memory messages go here
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # <-- REQUIRED for tool-calling agents
])

def run_agent(user_input: str, session_id: str = "default"):
    llm = ChatOllama(model=settings.chat_model, temperature=0.2)
    agent = create_tool_calling_agent(llm, TOOLS, PROMPT)
    memory = get_memory(session_id)
    executor = AgentExecutor(agent=agent, tools=TOOLS, memory=memory, verbose=False)
    result = executor.invoke({"input": user_input})
    return {"output": result.get("output", ""), "session_id": session_id}
