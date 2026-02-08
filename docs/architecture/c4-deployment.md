# C4 Deployment Diagram - Production Architecture

## Overview
This diagram shows the deployment architecture for the finance chat agent system in a production environment.

```mermaid
C4Deployment
  title Deployment Diagram - Finance Chat Agent Production

  Deployment_Node(client, "Client Device", "Web Browser") {
    Container(spa, "Web Application", "React/Vue", "Chat interface")
  }

  Deployment_Node(aws, "AWS Cloud", "us-east-1") {
    Deployment_Node(alb, "Application Load Balancer", "AWS ALB") {
      Container(api, "FastAPI Application", "Python", "REST API server")
    }

    Deployment_Node(ecs, "ECS Cluster", "Fargate") {
      Container(worker1, "Celery Worker", "Python", "Background processing")
      Container(worker2, "Celery Worker", "Python", "Background processing")
    }

    Deployment_Node(rds, "RDS", "db.t3.medium") {
      ContainerDb(sqlite, "SQLite Database", "SQLite", "Application data")
    }

    Deployment_Node(elasticache, "ElastiCache", "Redis cluster") {
      Container(redis, "Redis Cache", "Redis", "Session and task queue")
    }

  Deployment_Node(s3, "S3", "Standard storage") {
      Container(bucket, "Documentation Storage", "S3", "Generated docs and assets")
    }
  }

  Deployment_Node(openai, "OpenAI Cloud", "API endpoint") {
    Container(openai_api, "OpenAI Service", "REST API", "LLM processing")
  }

  Deployment_Node(github, "GitHub Cloud", "api.github.com") {
    Container(github_api, "GitHub Service", "REST API", "Repository data")
  }

  Deployment_Node(jira, "Atlassian Cloud", "api.atlassian.com") {
    Container(jira_api, "JIRA Service", "REST API", "Project management")
  }

  Rel(spa, api, "HTTPS", "REST API calls")
  Rel(api, sqlite, "Reads/writes", "SQLite/JDBC")
  Rel(api, redis, "Reads/writes", "Redis protocol")
  Rel(api, openai_api, "REST calls", "HTTPS/JSON")
  Rel(api, github_api, "REST calls", "HTTPS/JSON")
  Rel(api, jira_api, "REST calls", "HTTPS/JSON")

  Rel(worker1, redis, "Consumes tasks", "Celery protocol")
  Rel(worker2, redis, "Consumes tasks", "Celery protocol")
  Rel(worker1, sqlite, "Reads/writes", "SQLite/JDBC")
  Rel(worker1, bucket, "Writes docs", "S3 API")
  Rel(worker2, bucket, "Writes docs", "S3 API")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Deployment Architecture

### Client Layer
- **Web Application**: Frontend chat interface built with React or Vue.js
- **Responsibilities**: User interaction, real-time messaging, display of responses

### Application Layer
- **Application Load Balancer**: Distributes traffic across API instances
- **FastAPI Application**: Main API server handling HTTP requests
- **Auto-scaling**: Horizontal scaling based on request volume

### Processing Layer
- **ECS Cluster**: Container orchestration for Celery workers
- **Celery Workers**: Background task processing
  - Repository analysis and documentation generation
  - Data fetching from external APIs
  - Indexing and caching operations
- **Auto-scaling**: Workers scale based on task queue length

### Data Layer
- **RDS with SQLite**: Persistent data storage
  - Conversation history
  - User data
  - Application state
- **ElastiCache (Redis)**: Caching and message broker
  - Session storage
  - Task queue
  - Temporary data caching
- **S3 Documentation Storage**: Generated documentation and assets
  - Static documentation files
  - Diagrams and images
  - Exported documents

### External Services
- **OpenAI API**: LLM processing for AI capabilities
- **GitHub API**: Repository data and operations
- **JIRA API**: Project management data

## Infrastructure Details

### High Availability
- **Load Balancer**: Distributes traffic across multiple API instances
- **Multi-AZ Deployment**: Resources span multiple availability zones
- **Auto Scaling**: Automatic scaling based on demand

### Security
- **VPC**: Isolated network environment
- **Security Groups**: Network access controls
- **IAM Roles**: Least-privilege access
- **SSL/TLS**: Encrypted communication

### Monitoring
- **CloudWatch**: Metrics and logging
- **Health Checks**: Automated monitoring of services
- **Alarms**: Notification for critical events

### Performance Optimization
- **Caching**: Redis for frequently accessed data
- **CDN**: Static asset delivery
- **Connection Pooling**: Efficient API connections
- **Asynchronous Processing**: Non-blocking operations

## Deployment Strategy

1. **Blue-Green Deployment**: Zero-downtime updates
2. **Canary Releases**: Gradual rollout of new features
3. **Rolling Updates**: Maintain service availability
4. **Auto-healing**: Automatic replacement of unhealthy instances

## Disaster Recovery

1. **Backup Strategy**:
   - Automated database backups
   - S3 versioning for documentation
   - Multi-region replication

2. **Recovery Procedures**:
   - Automated failover to standby instances
   - Database restore from backups
   - Service restart with graceful degradation
