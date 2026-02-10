from typing import List, Dict, Any, Optional
from langchain.tools import tool
from app.services.repository_service import RepositoryCloner
from app.clients.github_client import GitHubClient
from app.core.config import github_settings
import ast
import re

# We need a client instance for RepositoryCloner
def get_github_client():
    token = github_settings.token if github_settings.token else "dummy_token"
    return GitHubClient(token=token)

@tool
def parse_python_file(file_path: str, content: str) -> List[Dict[str, Any]]:
    """
    Parse Python code using Abstract Syntax Tree (AST).
    
    Args:
        file_path: Path to the file
        content: Content of the file
        
    Returns:
        List of CodeEntity dictionaries
    """
    entities = []
    
    try:
        tree = ast.parse(content)
        
        # Helper to extract decorators
        def get_decorators(node):
            return [d.id for d in node.decorator_list if isinstance(d, ast.Name)]

        # Helper to extract method info
        def get_method_info(node):
            args = [a.arg for a in node.args.args]
            return {
                "name": node.name,
                "params": args,
                "decorators": get_decorators(node),
                "docstring": ast.get_docstring(node)
            }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [get_method_info(m) for m in node.body if isinstance(m, ast.FunctionDef)]
                entities.append({
                    "name": node.name,
                    "type": "class",
                    "file_path": file_path,
                    "language": "python",
                    "methods": methods,
                    "decorators": get_decorators(node),
                    "docstring": ast.get_docstring(node),
                    "dependencies": [], # TODO: Extract from imports
                    "attributes": [] # TODO: Extract from __init__
                })
            elif isinstance(node, ast.FunctionDef):
                # Check if it's a top-level function (simplified check)
                # We can't easily check parent in ast.walk, but we can iterate body separately
                pass
                
        # Top-level functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                entities.append({
                    "name": node.name,
                    "type": "function",
                    "file_path": file_path,
                    "language": "python",
                    "methods": [],
                    "decorators": get_decorators(node),
                    "docstring": ast.get_docstring(node),
                    "params": [a.arg for a in node.args.args]
                })
                
    except Exception as e:
        # Log error in a real app
        pass
        
    return entities

@tool
def parse_java_file(file_path: str, content: str) -> List[Dict[str, Any]]:
    """
    Parse Java source files to extract classes, interfaces, and methods.
    (Simplified Regex-based implementation)
    """
    entities = []
    # Simple regex to find class definitions
    class_pattern = re.compile(r'public\s+(class|interface)\s+(\w+)')
    matches = class_pattern.findall(content)
    
    for type_, name in matches:
        entities.append({
            "name": name,
            "type": type_,
            "file_path": file_path,
            "language": "java",
            "docstring": "Extracted via regex"
        })
    return entities

@tool
def extract_sql_schema(file_path: str, content: str) -> List[Dict[str, Any]]:
    """
    Extract table definitions from SQL CREATE TABLE statements.
    """
    entities = []
    table_pattern = re.compile(r'CREATE\s+TABLE\s+(\w+)', re.IGNORECASE)
    matches = table_pattern.findall(content)
    
    for table_name in matches:
        entities.append({
            "name": table_name,
            "type": "table",
            "file_path": file_path,
            "language": "sql"
        })
    return entities

@tool
def identify_design_patterns(entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Analyze code entities to detect common design patterns.
    
    Args:
        entities: List of CodeEntity objects
        
    Returns:
        Dictionary mapping pattern names to list of entity names
    """
    patterns = {
        "Singleton": [],
        "Factory": [],
        "Repository": []
    }
    
    for entity in entities:
        name = entity.get("name", "")
        if "Factory" in name:
            patterns["Factory"].append(name)
        if "Repository" in name:
            patterns["Repository"].append(name)
            
    return patterns

@tool
async def clone_repository(repo_url: str) -> str:
    """Clone a GitHub repository to a local temporary directory.
    
    Args:
        repo_url: The full URL of the repository (e.g., https://github.com/owner/repo)
        
    Returns:
        The local file path where the repository was cloned.
    """
    client = get_github_client()
    cloner = RepositoryCloner(client)
    path = await cloner.clone_repository(repo_url)
    return str(path)
