from vanna.integrations.google import GeminiLlmService
import os
from vanna.core.registry import ToolRegistry
from vanna import Agent
from vanna.core.agent import AgentConfig
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.core.user import UserResolver, User, RequestContext
from vanna.servers.fastapi.routes import register_chat_routes

from vanna.tools import RunSqlTool
from vanna.integrations.postgres import PostgresRunner
from vanna.tools.agent_memory import (
    SaveQuestionToolArgsTool,
    SearchSavedCorrectToolUsesTool,
    SaveTextMemoryTool
)
from vanna.integrations.pinecone.agent_memory import PineconeAgentMemory


# === LLM SETUP ===
llm = GeminiLlmService(
    model="gemini-2.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# === DATABASE SETUP ===
db_tool = RunSqlTool(
    sql_runner=PostgresRunner(
        connection_string=os.getenv("DATABASE_URL")
    )
)
sql_runner = PostgresRunner(
    connection_string=os.getenv("DATABASE_URL")
)

# === MEMORY SETUP (Pinecone) ===
agent_memory = PineconeAgentMemory(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment="us-east-1",
    index_name="vanna-memory"
)

# === USER RESOLVER ===
class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_id = request_context.get_cookie('user_id') or 'demo_user'
        return User(id=user_id, group_memberships=['read_sales'])


# === TOOL REGISTRATION ===
tools = ToolRegistry()
tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])


# === AGENT CONFIGURATION ===
config = AgentConfig(
    stream_responses=True,
    max_tool_iterations=10,
    auto_save_conversations=True
)


# === AGENT CREATION ===
agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    agent_memory=agent_memory,
    user_resolver=SimpleUserResolver(),
    config=config
)


# === FASTAPI SERVER SETUP ===
server = VannaFastAPIServer(agent, config={
    "cors": {
        "enabled": True,
        "allow_origins": ["*"],
        "allow_credentials": True
    },
    "dev_mode": True
})

# === APP ENTRY POINT ===
port = int(os.getenv("PORT", 8000))
app = server.create_app()
