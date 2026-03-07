# SkillMiddleware Deployment Guide

This guide covers the deployment and production considerations for the SkillMiddleware implementation.

## Overview

The SkillMiddleware implementation provides a production-ready progressive disclosure pattern for agent skills that reduces token consumption by 60-80% while maintaining full capabilities.

## Prerequisites

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

# Agent Configuration
AGENT_TEMPERATURE=0.7
MAX_ITERATIONS=20

# Redis Configuration (for state management)
REDIS_URL=redis://localhost:6379/0
```

### Dependencies

```txt
langchain>=0.2.0
langgraph>=0.2.0
langchain-openai>=0.2.0
fastapi>=0.104.0
uvicorn>=0.24.0
redis>=5.0.0
```

## Deployment Steps

### 1. Production Agent Setup

```python
from app.config.agents import AgentConfig

# Create production-ready agent with skills
production_agent = AgentConfig.create_finance_agent(
    enable_skills=True,
    model_name="gpt-4o",  # Use production-grade model
    temperature=0.3      # Lower temperature for consistency
)
```

### 2. Skill Management

#### Skill Registry

Skills are defined in `app/skills/registry.py`. Maintain skills in a version-controlled manner:

```python
# app/skills/registry.py
SKILLS_REGISTRY = [
    CODE_ANALYSIS_SKILL,
    DOCUMENTATION_GENERATION_SKILL,
    DIAGRAM_GENERATION_SKILL,
    # Add new skills here
]
```

#### Skill Versioning

Implement skill versioning for production:

```python
@dataclass
class Skill:
    name: str
    description: str
    content: str
    version: str = "1.0.0"  # Add version field
    tools: List[Callable] = None
    category: str = "general"
    token_budget: int = 2000
```

### 3. Performance Optimization

#### Caching

Implement caching for loaded skills:

```python
from functools import lru_cache

class CachedSkillManager:
    def __init__(self, skills):
        self.skills = skills
        self._skill_cache = {}

    @lru_cache(maxsize=32)
    def get_skill_content(self, skill_name: str) -> str:
        # Cache skill content to avoid repeated loading
        pass
```

#### Monitoring

Add monitoring for skill usage:

```python
# app/skills/monitor.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SkillMetrics:
    skill_name: str
    load_count: int = 0
    total_tokens_saved: int = 0
    avg_load_time_ms: float = 0
    last_loaded: Optional[datetime] = None
```

### 4. Production Configuration

#### FastAPI Configuration

```python
# app/main.py
app = FastAPI(
    title="Finance Chat Agent with Skills",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add monitoring middleware
app.add_middleware(
    MonitoringMiddleware,
    skill_monitor=skill_monitor
)
```

#### Environment-Specific Configurations

```python
# app/config/production.py
class ProductionConfig:
    MODEL = "gpt-4o"
    TEMPERATURE = 0.3
    MAX_ITERATIONS = 15
    ENABLE_SKILLS = True
    SKILL_CACHE_SIZE = 64
    MONITORING_ENABLED = True
```

## Testing Strategy

### Unit Tests

```python
# tests/test_skills_unit.py
def test_skill_loading():
    """Test skill loading functionality."""
    result = load_skill("code_analysis")
    assert "Loaded Skill:" in result

def test_skill_progressive_disclosure():
    """Test progressive disclosure mechanism."""
    # Verify only metadata is loaded initially
    # Verify full content is loaded on demand
```

### Integration Tests

```python
# tests/test_skills_integration.py
@pytest.mark.asyncio
async def test_skills_agent_integration():
    """Test full agent with skills."""
    agent = AgentConfig.create_skills_only_agent()
    result = await agent.invoke({"input": "test query"})
    assert "skills_used" in result["metadata"]
```

### Load Testing

```python
# tests/test_skills_performance.py
import asyncio
import time

async def test_concurrent_skill_loading():
    """Test concurrent skill loading performance."""
    agent = AgentConfig.create_skills_only_agent()

    # Concurrent queries
    tasks = [agent.invoke({"input": f"query {i}"}) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Verify all completed successfully
    assert all(result["success"] for result in results)
```

## Monitoring and Observability

### Metrics to Track

1. **Token Savings**
   - Tokens saved per request
   - Total tokens saved over time
   - Skill loading frequency

2. **Performance Metrics**
   - Skill load time
   - Response time with/without skills
   - Memory usage

3. **Usage Patterns**
   - Most frequently used skills
   - Skills per request distribution
   - User interaction patterns

### Implementation

```python
# app/skills/monitoring.py
class SkillMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()

    def record_skill_load(self, skill_name: str, tokens_saved: int):
        # Record skill loading event
        pass

    def get_dashboard_data(self):
        # Return formatted metrics for dashboard
        pass
```

## Scaling Considerations

### Horizontal Scaling

1. **Skill Cache Distribution**
   - Use Redis distributed cache for shared skill data
   - Implement cache invalidation strategy

2. **Load Balancing**
   - Distribute agent instances across multiple servers
   - Use sticky sessions for conversation state

```python
# app/skills/distributed_cache.py
class DistributedSkillCache:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url)

    async def get_skill(self, skill_name: str):
        # Get skill from Redis cache
        pass

    async def cache_skill(self, skill_name: str, content: str):
        # Cache skill in Redis with TTL
        pass
```

### Vertical Scaling

1. **Memory Management**
   - Monitor memory usage per agent instance
   - Implement skill garbage collection

2. **GPU Considerations**
   - For GPU-based models, monitor GPU memory
   - Batch skill processing when possible

## Security Considerations

### Skill Security

1. **Skill Validation**
   - Validate skill content before loading
   - Implement skill sandboxing for untrusted skills

```python
# app/skills/security.py
def validate_skill(skill: Skill) -> bool:
    """Validate skill content before loading."""
    # Check for malicious code
    # Validate input/output schemas
    # Verify skill dependencies
    return True
```

2. **Access Control**
   - Implement skill-level permissions
   - Audit skill usage

### API Security

1. **Rate Limiting**
   - Implement rate limiting for skill endpoints
   - Different limits for different skill types

```python
# app/api/security.py
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/analyze")
@limiter.limit("100/minute")
async def analyze_code(request: Request, data: Dict):
    # Rate-limited endpoint
    pass
```

## Rollout Strategy

### Phased Rollout

1. **Canary Deployment**
   - Deploy to 10% of users first
   - Monitor performance and errors
   - Gradually increase percentage

2. **Feature Flags**
   ```python
   # app/config/feature_flags.py
   class FeatureFlags:
       SKILLS_ENABLED = True  # Set to False to disable
       NEW_SKILLS_ROLLOUT = False
       MONITORING_ENABLED = True
   ```

### Rollback Plan

1. **Quick Rollback**
   - Set feature flag to disable skills
   - Monitor system health
   - Implement automatic rollback on errors

2. **Data Migration**
   - Plan for skill data migration
   - Maintain backward compatibility

## Maintenance

### Regular Tasks

1. **Skill Updates**
   - Monthly skill content review
   - Quarterly skill performance analysis

2. **Performance Tuning**
   - Monitor token savings metrics
   - Adjust cache sizes based on usage

### Troubleshooting

1. **Common Issues**
   - Skills not loading: Check skill registry and permissions
   - High token usage: Verify skill loading pattern
   - Slow responses: Check skill cache and network latency

2. **Debug Mode**
   ```python
   # Enable debug logging
   import logging
   logging.getLogger("app.skills").setLevel(logging.DEBUG)
   ```

## Best Practices

1. **Start Small**: Begin with 2-3 core skills
2. **Monitor Everything**: Track both performance and usage metrics
3. **Iterate**: Add new skills based on usage patterns
4. **Document**: Keep skill documentation updated
5. **Test**: Always test new skills in staging first

For more information, see:
- [Implementation Guide](./SkillMiddleware_Implementation.md)
- [Skills Examples](../../examples/skills_agent_example.py)
- [API Documentation](../../app/api/routes/skills.py)