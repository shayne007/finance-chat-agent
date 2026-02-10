from typing import List, Dict, Any
from app.skills.types import Skill

# Define the tools for the skill (placeholders for now, or simple implementations if possible)
# In a real implementation, these would use AST parsing, etc.
# For now, we will define them as dummy functions or simple parsers if needed by the agent.
# However, the skill definition mainly provides instructions to the LLM. 
# The actual tools need to be implemented and passed to the agent.
# The guide shows them as part of the skill definition text, but also mentions `tools: List[Callable]`.

# Let's define the skill content first.

CODE_ANALYSIS_CONTENT = """# Code Analysis Skill

## Purpose
Parse and analyze source code to extract classes, functions, dependencies, and architectural patterns.

## Capabilities
- **Python Analysis**: Use AST to extract classes, functions, decorators, imports, docstrings
- **Java Analysis**: Parse classes, interfaces, methods, annotations using javalang
- **SQL Analysis**: Extract database schemas from CREATE TABLE statements
- **Pattern Detection**: Identify design patterns (Singleton, Factory, Repository, etc.)
- **Dependency Mapping**: Build graphs showing relationships between code entities

## Tools Available

### parse_python_file(file_path: str, content: str) -> List[CodeEntity]
Parse Python code using Abstract Syntax Tree (AST).

**Returns:** CodeEntity objects with:
- name: Class or function name
- type: 'class', 'function', 'module'
- methods: List of method signatures (for classes)
- imports: List of imported modules
- decorators: Applied decorators
- docstring: Extracted documentation

**Example:**
```python
entities = parse_python_file(
    'src/services/payment.py',
    file_content
)
# Returns: [CodeEntity(name='PaymentService', type='class', ...)]
```

### parse_java_file(file_path: str, content: str) -> List[CodeEntity]
Parse Java source files to extract classes, interfaces, and methods.

### extract_sql_schema(file_path: str, content: str) -> List[DatabaseEntity]
Extract table definitions from SQL CREATE TABLE statements.

### identify_design_patterns(entities: List[CodeEntity]) -> Dict[str, List[str]]
Analyze code entities to detect common design patterns.

## Business Logic Rules

**File Filtering:**
- Skip test files: paths containing 'test', 'tests', '__pycache__', 'node_modules', 'vendor'
- Skip large files: >50KB typically binary or generated
- Handle encoding errors gracefully

**Relationship Extraction:**
- Import statements → module dependencies
- Class inheritance → parent-child relationships
- Method calls → inter-class dependencies
- Decorator usage → framework/library integration points

**API Endpoint Detection:**
- Python: Look for @app.route, @get, @post decorators (Flask/FastAPI)
- Java: Look for @RequestMapping, @GetMapping annotations (Spring)

## Output Format

### CodeEntity
```python
{
    'name': 'PaymentService',
    'type': 'class',
    'file_path': 'src/services/payment.py',
    'language': 'python',
    'methods': [
        {
            'name': 'process_payment',
            'params': ['amount', 'card_details'],
            'return_type': 'PaymentResult',
            'decorators': [],
            'docstring': 'Process a payment transaction'
        }
    ],
    'attributes': [
        {'name': 'gateway', 'type': 'PaymentGateway'}
    ],
    'dependencies': ['PaymentGateway', 'Database', 'Logger'],
    'docstring': 'Service for processing payments',
    'patterns': ['Repository', 'Dependency Injection']
}
```

## Example Usage

```python
# Analyze Python codebase
python_files = find_files(repo_path, '*.py')
all_entities = []

for file_path in python_files:
    with open(file_path, 'r') as f:
        content = f.read()
    
    entities = parse_python_file(file_path, content)
    all_entities.extend(entities)

# Identify patterns
patterns = identify_design_patterns(all_entities)
print(f"Found patterns: {patterns}")
# Output: {'Singleton': ['Config'], 'Factory': ['PaymentFactory'], ...}
```

## Best Practices

1. **Error Handling**: Catch parsing exceptions and continue with other files
2. **Incremental Processing**: Process files in batches to manage memory
3. **Documentation Extraction**: Always extract docstrings for later use
4. **Relationship Tracking**: Build dependency graphs for visualization
5. **Pattern Recognition**: Use heuristics (naming, structure) to identify patterns
6. **Performance**: Cache parsing results for large codebases
"""

CODE_ANALYSIS_SKILL = Skill(
    name="code_analysis",
    description="Analyze Python, Java, and SQL code to extract entities, dependencies, and patterns",
    content=CODE_ANALYSIS_CONTENT,
    category="analysis",
    token_budget=2500
)
