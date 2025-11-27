import threading
import uuid
from typing import Optional, Dict

from app.agents.jira_langgraph_agent import JiraLangGraphAgent


class AgentFactory:
    _instance = None
    _lock = threading.Lock()
    _thread_local = threading.local()
    _agent_pool: Dict[str, JiraLangGraphAgent] = {}
    _pool_lock = threading.Lock()
    _max_pool_size = 100

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_agent() -> JiraLangGraphAgent:
        factory = AgentFactory()
        if hasattr(factory._thread_local, "agent"):
            return factory._thread_local.agent
        thread_id = threading.current_thread().ident
        agent_id = f"agent_{thread_id}_{uuid.uuid4().hex[:8]}"
        with factory._pool_lock:
            if len(factory._agent_pool) >= factory._max_pool_size:
                for old_id in list(factory._agent_pool.keys())[:10]:
                    del factory._agent_pool[old_id]
            agent = JiraLangGraphAgent()
            factory._agent_pool[agent_id] = agent
            factory._thread_local.agent = agent
            return agent

    @staticmethod
    def get_pool_stats() -> Dict:
        factory = AgentFactory()
        with factory._pool_lock:
            return {
                "pool_size": len(factory._agent_pool),
                "max_pool_size": factory._max_pool_size,
                "agents": [
                    {"id": agent_id} for agent_id in factory._agent_pool.keys()
                ],
            }

    @staticmethod
    def reset() -> None:
        factory = AgentFactory()
        with factory._pool_lock:
            factory._agent_pool.clear()
        if hasattr(factory._thread_local, "agent"):
            delattr(factory._thread_local, "agent")