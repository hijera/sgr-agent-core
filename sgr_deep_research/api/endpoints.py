import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from sgr_deep_research.api.models import (
    AGENT_MODEL_MAPPING,
    AgentListItem,
    AgentListResponse,
    AgentModel,
    AgentStateResponse,
    ChatCompletionRequest,
    ClarificationRequest,
    HealthResponse,
)
from sgr_deep_research.core.agents import BaseAgent, SGRInfiniteAutoToolCallingAgent
from sgr_deep_research.core.models import AgentStatesEnum

logger = logging.getLogger(__name__)

router = APIRouter()

# ToDo: better to move to a separate service
agents_storage: dict[str, BaseAgent] = {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@router.get("/agents/{agent_id}/state", response_model=AgentStateResponse)
async def get_agent_state(agent_id: str):
    if agent_id not in agents_storage:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = agents_storage[agent_id]

    return AgentStateResponse(
        agent_id=agent.id,
        task=agent.task,
        sources_count=len(agent._context.sources),
        **agent._context.model_dump(),
    )


@router.get("/agents", response_model=AgentListResponse)
async def get_agents_list():
    agents_list = [
        AgentListItem(
            agent_id=agent.id,
            task=agent.task,
            state=agent._context.state,
            creation_time=agent.creation_time,
        )
        for agent in agents_storage.values()
    ]

    return AgentListResponse(agents=agents_list, total=len(agents_list))


@router.get("/v1/models")
async def get_available_models():
    """Get list of available agent models."""
    return {
        "data": [
            {"id": model.value, "object": "model", "created": 1234567890, "owned_by": "sgr-deep-research"}
            for model in AgentModel
        ],
        "object": "list",
    }


def extract_user_content_from_messages(messages):
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    raise ValueError("User message not found in messages")


@router.post("/agents/{agent_id}/provide_clarification")
async def provide_clarification(agent_id: str, request: ClarificationRequest):
    try:
        agent = agents_storage.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        logger.info(f"Providing clarification to agent {agent.id}: {request.clarifications[:100]}...")

        await agent.provide_clarification(request.clarifications)
        return StreamingResponse(
            agent.streaming_generator.stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Agent-ID": str(agent.id),
            },
        )

    except Exception as e:
        logger.error(f"Error completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _is_agent_id(model_str: str) -> bool:
    """Check if model string is an agent ID (contains underscore and UUID-like
    format)."""
    return "_" in model_str and len(model_str) > 20


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if not request.stream:
        raise HTTPException(status_code=501, detail="Only streaming responses are supported. Set 'stream=true'")

    # Check if this is a continuation of existing agent conversation
    if (
        request.model
        and isinstance(request.model, str)
        and _is_agent_id(request.model)
        and request.model in agents_storage
    ):
        agent = agents_storage[request.model]
        
        # Handle continuation for infinite agents (priority over clarification)
        if isinstance(agent, SGRInfiniteAutoToolCallingAgent):
            user_message = extract_user_content_from_messages(request.messages)
            logger.info(f"💬 Continuing infinite agent {agent.id} with message: {user_message[:100]}...")
            
            # Continue conversation
            await agent.continue_conversation(user_message)
            
            # If user didn't request stop, trigger clarification event to continue execution
            if not agent.user_requested_stop:
                agent._context.clarification_received.set()
            
            return StreamingResponse(
                agent.streaming_generator.stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Agent-ID": str(agent.id),
                    "X-Agent-Model": agent.name,
                },
            )
        
        # Handle clarification for non-infinite agents
        if agent._context.state == AgentStatesEnum.WAITING_FOR_CLARIFICATION:
            return await provide_clarification(
                agent_id=request.model,
                request=ClarificationRequest(clarifications=extract_user_content_from_messages(request.messages)),
            )
        
        # If agent exists but is not infinite and not waiting for clarification
        raise HTTPException(
            status_code=400,
            detail=f"Agent {request.model} exists but cannot continue conversation. State: {agent._context.state}",
        )

    try:
        task = extract_user_content_from_messages(request.messages)

        try:
            agent_model = AgentModel(request.model)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model '{request.model}'. Available models: {[m.value for m in AgentModel]}",
            )

        agent_class = AGENT_MODEL_MAPPING[agent_model]
        agent = agent_class(task=task)
        agents_storage[agent.id] = agent
        logger.info(f"Agent {agent.id} ({agent_model.value}) created and stored for task: {task[:100]}...")

        _ = asyncio.create_task(agent.execute())
        return StreamingResponse(
            agent.streaming_generator.stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Agent-ID": str(agent.id),
                "X-Agent-Model": agent_model.value,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
