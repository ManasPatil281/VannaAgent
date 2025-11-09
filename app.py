from vanna.integrations.google import GeminiLlmService
import os
from vanna.core.registry import ToolRegistry
from vanna import Agent
from vanna.core.agent import AgentConfig, UiFeatures
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.core.user import UserResolver, User, RequestContext
from vanna.servers.base import ChatHandler
from vanna.servers.fastapi.routes import register_chat_routes

llm = GeminiLlmService(
    model="gemini-2.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY")
)

from vanna.tools import RunSqlTool
from vanna.integrations.postgres import PostgresRunner

# Set up database connection
db_tool = RunSqlTool(
    sql_runner=PostgresRunner(
        connection_string=os.getenv("DATABASE_URL")
    )
)

sql_runner = PostgresRunner(
    connection_string=os.getenv("DATABASE_URL")
)

from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.integrations.pinecone.agent_memory import PineconeAgentMemory

# Set up Pinecone for cloud-based agent memory
agent_memory = PineconeAgentMemory(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment="us-east-1",
    index_name="vanna-memory"
)

class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_id = request_context.get_cookie('user_id') or 'demo_user'
        return User(id=user_id, group_memberships=['read_sales'])

# Create your agent with streaming configuration
tools = ToolRegistry()
tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin','user'])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])

# CRITICAL: Configure agent to stream tool results
config = AgentConfig(
    stream_responses=True,  # Enable streaming
    max_tool_iterations=10,
    auto_save_conversations=True,
    ui_features=UiFeatures(
        feature_group_access={
            'tool_names': ['admin', 'user'],
            'tool_arguments': ['admin', 'user'],
            'tool_results': ['admin', 'user']  # Show tool results
        }
    )
)

agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    agent_memory=agent_memory,
    user_resolver=SimpleUserResolver(),
    config=config  # Add the config here!
)

server = VannaFastAPIServer(agent, config={
    "cors": {
        "enabled": True,
        "allow_origins": ["*"],
        "allow_credentials": True
    },
    "dev_mode": True  # Enable for debugging
})

# Run the server
port = int(os.getenv("PORT", 8000))
app = server.create_app()
