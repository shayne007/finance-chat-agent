from fastapi import APIRouter, HTTPException
from app.models.schemas import AgentRequest, AgentResponse
from app.agents.jira_agent_factory import AgentFactory


router = APIRouter()


@router.post("/agent/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    try:
        agent = AgentFactory.get_agent()
        result = agent.invoke(request.user_input)
        return AgentResponse(
            output=result.get("final_output", ""),
            ticket_id=result.get("jira_ticket_id"),
            structured_data=result.get("structured_output"),
            error=result.get("error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/pool-stats")
async def pool_stats():
    return AgentFactory.get_pool_stats()