"""
Repository Service for transforming GitHub repositories into markdown documentation.

This service provides functionality to:
- Clone GitHub repositories
- Extract existing markdown files
- Analyze code structure (Python, Java, SQL)
- Generate comprehensive documentation
- Create diagrams and visualizations
- Extract file relationships
"""

import os
import re
import tempfile
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
import shutil

from app.core.config import github_settings
from app.clients.github_client import GitHubClient, GitHubAPIError
import git


logger = logging.getLogger(__name__)


@dataclass
class FileAnalysis:
    """Represents analysis of a code file."""
    file_path: str
    language: str
    file_type: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    imports: List[str]
    dependencies: List[str]
    business_logic: str
    complexity_score: int
    documentation: str


@dataclass
class DocumentationResult:
    """Result of documentation generation."""
    output_dir: str
    files_created: List[str]
    summary: Dict[str, Any]
    errors: List[str]


class RepositoryCloner:
    """Handles git operations and repository cloning."""

    def __init__(self, github_client: GitHubClient):
        self.github_client = github_client
        self.temp_dir = Path(tempfile.gettempdir()) / "repo_cloning"
        self.temp_dir.mkdir(exist_ok=True)

    async def clone_repository(self, url: str, branch: str = "main") -> Path:
        """Clone a GitHub repository to temporary directory."""
        repo_name = url.split('/')[-1].replace('.git', '')
        local_path = self.temp_dir / repo_name

        try:
            if local_path.exists():
                # Pull latest changes
                repo = git.Repo(local_path)
                repo.git.fetch('--all')
                repo.git.checkout(branch)
                repo.git.pull('origin', branch)
                logger.info(f"Updated existing repository: {local_path}")
            else:
                # Clone new repository
                git.Repo.clone_from(url, local_path, branch=branch)
                logger.info(f"Cloned new repository: {local_path}")

            return local_path

        except git.GitCommandError as e:
            error_msg = f"Git operation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def download_single_file(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Download a single file from GitHub repository."""
        try:
            # This would use the GitHub client implementation
            # For now, we'll implement a basic version
            raise NotImplementedError("GitHub API file download not yet implemented")
        except Exception as e:
            logger.error(f"Failed to download {path} from {owner}/{repo}: {str(e)}")
            return None

    def extract_existing_markdown(self, repo_path: Path) -> List[Dict[str, str]]:
        """Extract existing markdown files from repository."""
        markdown_files = []
        md_patterns = ['README.md', 'readme.md', 'CONTRIBUTING.md',
                      'ARCHITECTURE.md', 'DESIGN.md', 'API.md', 'docs/*.md']

        for pattern in md_patterns:
            if pattern.startswith('docs/'):
                # Handle docs directory
                docs_path = repo_path / 'docs'
                if docs_path.exists():
                    for md_file in docs_path.glob('*.md'):
                        markdown_files.append({
                            'path': str(md_file.relative_to(repo_path)),
                            'content': md_file.read_text(encoding='utf-8'),
                            'type': 'documentation'
                        })
            else:
                # Handle root level files
                md_file = repo_path / pattern
                if md_file.exists():
                    markdown_files.append({
                        'path': pattern,
                        'content': md_file.read_text(encoding='utf-8'),
                        'type': 'documentation'
                    })

        return markdown_files


class CodeAnalyzer:
    """Analyzes code structure and extracts meaningful information."""

    def __init__(self):
        self.supported_languages = {
            '.py': 'python',
            '.java': 'java',
            '.sql': 'sql',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c'
        }

    def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Analyze a single file based on its language."""
        try:
            if not file_path.exists():
                return None

            file_extension = file_path.suffix.lower()
            language = self.supported_languages.get(file_extension)

            if not language:
                logger.warning(f"Unsupported language: {file_extension}")
                return None

            content = file_path.read_text(encoding='utf-8')

            if language == 'python':
                return self._analyze_python(file_path, content)
            elif language == 'java':
                return self._analyze_java(file_path, content)
            elif language == 'sql':
                return self._analyze_sql(file_path, content)
            else:
                return self._analyze_generic(file_path, language, content)

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
            return None

    def _analyze_python(self, file_path: Path, content: str) -> FileAnalysis:
        """Analyze Python code structure."""
        import ast

        classes = []
        functions = []
        imports = []
        dependencies = []

        try:
            tree = ast.parse(content)

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                'name': item.name,
                                'args': [arg.arg for arg in item.args.args],
                                'docstring': ast.get_docstring(item)
                            })

                    classes.append({
                        'name': node.name,
                        'methods': methods,
                        'docstring': ast.get_docstring(node)
                    })

                elif isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node)
                    })

            # Analyze dependencies by looking for common imports
            dep_keywords = ['requests', 'django', 'flask', 'numpy', 'pandas', 'sqlalchemy', 'fastapi']
            for imp in imports:
                for keyword in dep_keywords:
                    if keyword in imp.lower():
                        dependencies.append(keyword)
                        break

            # Generate business logic summary
            business_logic = self._generate_business_logic_summary(content, file_path)

            # Calculate complexity
            complexity_score = self._calculate_complexity(tree)

            return FileAnalysis(
                file_path=str(file_path),
                language='python',
                file_type='code',
                classes=classes,
                functions=functions,
                imports=imports,
                dependencies=dependencies,
                business_logic=business_logic,
                complexity_score=complexity_score,
                documentation=self._generate_file_documentation(file_path, content, classes, functions)
            )

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {str(e)}")
            return self._analyze_generic(file_path, 'python', content)

    def _analyze_java(self, file_path: Path, content: str) -> FileAnalysis:
        """Analyze Java code structure."""
        classes = []
        functions = []
        imports = []
        dependencies = []

        # Extract imports
        import_pattern = r'import\s+(?:static\s+)?([\w\.]+)\s*;'
        imports = re.findall(import_pattern, content)

        # Extract classes
        class_pattern = r'(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w\s,]+)?\s*\{'
        class_matches = re.findall(class_pattern, content)

        for class_name in class_matches:
            # Extract methods for each class
            method_pattern = r'(?:public|private|protected)\s+(?:static\s+)?[\w<>]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{'
            methods = re.findall(method_pattern, content)

            classes.append({
                'name': class_name,
                'methods': [{'name': m} for m in methods],
                'docstring': ''
            })

        # Extract standalone functions
        functions = re.findall(r'static\s+[\w<>]+\s+(\w+)\s*\([^)]*\)\s*{', content)

        # Analyze dependencies
        dep_packages = ['spring', 'hibernate', 'junit', 'log4j', 'mysql', 'postgres', 'mongodb']
        for imp in imports:
            for pkg in dep_packages:
                if pkg in imp.lower():
                    dependencies.append(pkg)
                    break

        business_logic = self._generate_business_logic_summary(content, file_path)
        complexity_score = self._estimate_complexity(content)

        return FileAnalysis(
            file_path=str(file_path),
            language='java',
            file_type='code',
            classes=classes,
            functions=functions,
            imports=imports,
            dependencies=dependencies,
            business_logic=business_logic,
            complexity_score=complexity_score,
            documentation=self._generate_file_documentation(file_path, content, classes, functions)
        )

    def _analyze_sql(self, file_path: Path, content: str) -> FileAnalysis:
        """Analyze SQL code structure."""
        tables = []
        functions = []
        dependencies = []
        relationships = []

        # Extract tables
        table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\(([\s\S]*?)\)'
        table_matches = re.finditer(table_pattern, content, re.IGNORECASE)

        for match in table_matches:
            table_name = match.group(1)
            columns = match.group(2)

            # Extract columns and their types
            column_pattern = r'(\w+)\s+(\w+)(?:\s+\w+)*'
            columns_data = re.findall(column_pattern, columns)

            tables.append({
                'name': table_name,
                'columns': [{'name': c[0], 'type': c[1].upper()} for c in columns_data],
                'docstring': ''
            })

            # Find relationships
            for other_table in tables[:-1]:
                if table_name != other_table['name']:
                    # Look for foreign keys
                    fk_pattern = f'FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s*{other_table["name"]}\s*\(\s*(\w+)\s*\)'
                    if re.search(fk_pattern, columns, re.IGNORECASE):
                        relationships.append({
                            'from': table_name,
                            'to': other_table['name'],
                            'column': match.group(1)
                        })

        # Extract stored procedures/functions
        function_pattern = r'(?:CREATE\s+(?:PROCEDURE|FUNCTION)\s+(\w+)[\s\S]*?BEGIN[\s\S]*?END)'
        functions = re.findall(function_pattern, content, re.IGNORECASE)

        business_logic = self._generate_sql_business_logic(content)
        complexity_score = self._estimate_sql_complexity(content)

        return FileAnalysis(
            file_path=str(file_path),
            language='sql',
            file_type='code',
            classes=[],
            functions=[{'name': f} for f in functions],
            imports=[],
            dependencies=dependencies,
            business_logic=business_logic,
            complexity_score=complexity_score,
            documentation=self._generate_sql_documentation(file_path, content, tables, relationships)
        )

    def _analyze_generic(self, file_path: Path, language: str, content: str) -> FileAnalysis:
        """Analyze code for unsupported languages generically."""
        # Basic analysis using patterns
        functions = re.findall(r'(?:function|def|proc)\s+(\w+)', content, re.IGNORECASE)

        return FileAnalysis(
            file_path=str(file_path),
            language=language,
            file_type='code',
            classes=[],
            functions=[{'name': f} for f in functions],
            imports=[],
            dependencies=[],
            business_logic=self._generate_business_logic_summary(content, file_path),
            complexity_score=self._estimate_complexity(content),
            documentation=f"# {file_path.name}\n\n## File Overview\n\nThis file contains {language} code."
        )

    def _generate_business_logic_summary(self, content: str, file_path: Path) -> str:
        """Generate a summary of the business logic in the code."""
        # Extract comments and docstrings
        comments = re.findall(r'#.*|//.*|/\*[\s\S]*?\*/', content)

        # Look for key business terms
        business_terms = [
            'user', 'customer', 'order', 'payment', 'transaction', 'product',
            'inventory', 'auth', 'login', 'session', 'api', 'endpoint',
            'service', 'controller', 'model', 'view', 'database'
        ]

        found_terms = []
        for term in business_terms:
            if term.lower() in content.lower():
                found_terms.append(term)

        summary = f"File: {file_path.name}\n\n"
        summary += f"Business Areas: {', '.join(found_terms)}\n\n"
        summary += "Key Comments:\n"
        for comment in comments[:5]:  # Show first 5 comments
            summary += f"- {comment}\n"

        return summary

    def _generate_sql_business_logic(self, content: str) -> str:
        """Generate business logic summary for SQL code."""
        # Identify transaction patterns
        transaction_patterns = ['BEGIN', 'COMMIT', 'ROLLBACK', 'INSERT', 'UPDATE', 'DELETE']
        has_transactions = any(pattern in content.upper() for pattern in transaction_patterns)

        # Identify data operations
        operation_counts = {
            'SELECT': len(re.findall(r'SELECT', content, re.IGNORECASE)),
            'INSERT': len(re.findall(r'INSERT', content, re.IGNORECASE)),
            'UPDATE': len(re.findall(r'UPDATE', content, re.IGNORECASE)),
            'DELETE': len(re.findall(r'DELETE', content, re.IGNORECASE)),
        }

        summary = "SQL Business Logic Summary:\n\n"
        summary += f"Transaction Handling: {'Yes' if has_transactions else 'No'}\n"
        summary += "Data Operations:\n"
        for op, count in operation_counts.items():
            if count > 0:
                summary += f"- {op}: {count} occurrences\n"

        return summary

    def _calculate_complexity(self, ast_tree) -> int:
        """Calculate cyclomatic complexity using AST."""
        complexity = 1  # Base complexity

        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                complexity += 1

        return min(complexity, 10)  # Cap at 10

    def _estimate_complexity(self, content: str) -> int:
        """Estimate code complexity based on patterns."""
        complexity = 1

        # Count control structures
        complexity += len(re.findall(r'\b(if|while|for|switch|case)\b', content, re.IGNORECASE))
        complexity += len(re.findall(r'\b(try|except|finally)\b', content, re.IGNORECASE))

        # Count nested blocks
        complexity += content.count('{') // 4  # Estimate nesting level

        return min(complexity, 10)

    def _estimate_sql_complexity(self, content: str) -> int:
        """Estimate SQL complexity."""
        complexity = 1

        # Count JOINs
        complexity += len(re.findall(r'\b(INNER|LEFT|RIGHT|FULL)\s+JOIN\b', content, re.IGNORECASE))

        # Count subqueries
        complexity += len(re.findall(r'\bSELECT\b.*\bFROM\b.*\(.*SELECT', content, re.IGNORECASE, re.DOTALL))

        # Count CTEs
        complexity += len(re.findall(r'\bWITH\b', content, re.IGNORECASE))

        return min(complexity, 10)

    def _generate_file_documentation(self, file_path: Path, content: str, classes: List, functions: List) -> str:
        """Generate markdown documentation for a file."""
        doc = f"# {file_path.name}\n\n"
        doc += f"**File Path:** `{file_path}`\n\n"

        if classes:
            doc += "## Classes\n\n"
            for cls in classes:
                doc += f"### {cls['name']}\n"
                if cls.get('docstring'):
                    doc += f"{cls['docstring']}\n"
                if cls.get('methods'):
                    doc += "\n**Methods:**\n"
                    for method in cls['methods']:
                        doc += f"- `{method['name']}`"
                        if method.get('args'):
                            doc += f"({', '.join(method['args'])})"
                        doc += "\n"
                doc += "\n"

        if functions:
            doc += "## Functions\n\n"
            for func in functions:
                doc += f"### {func['name']}\n"
                if func.get('args'):
                    doc += f"**Parameters:** {', '.join(func['args'])}\n"
                if func.get('docstring'):
                    doc += f"{func['docstring']}\n"
                doc += "\n"

        return doc

    def _generate_sql_documentation(self, file_path: Path, content: str, tables: List, relationships: List) -> str:
        """Generate markdown documentation for SQL files."""
        doc = f"# {file_path.name}\n\n"
        doc += f"**File Path:** `{file_path}`\n\n"

        if tables:
            doc += "## Database Tables\n\n"
            for table in tables:
                doc += f"### {table['name']}\n\n"
                doc += "**Columns:**\n"
                for col in table['columns']:
                    doc += f"- `{col['name']}` ({col['type']})\n"
                doc += "\n"

        if relationships:
            doc += "## Relationships\n\n"
            for rel in relationships:
                doc += f"- **{rel['from']}** → **{rel['to']}**\n"
            doc += "\n"

        return doc


class DiagramGenerator:
    """Creates diagrams for documentation."""

    def __init__(self):
        self.mermaid_templates = {
            'er': """erDiagram
    {tables}
    """,
            'sequence': """sequenceDiagram
    {actors}
    {actions}
    """,
            'architecture': """graph TD
    {components}
    """
        }

    def generate_er_diagram(self, tables: List[Dict], relationships: List[Dict]) -> str:
        """Generate Entity-Relationship diagram using Mermaid syntax."""
        er_content = "erDiagram\n"

        # Add tables
        for table in tables:
            er_content += f"    {table['name']} {{\n"
            for column in table['columns']:
                er_content += f"        {column['type']} {column['name']}\n"
            er_content += "    }\n"

        # Add relationships
        for rel in relationships:
            er_content += f"    {rel['from']} ||--o{{ {rel['to']}}}\n"

        return er_content

    def generate_sequence_diagram(self, flow: Dict) -> str:
        """Generate sequence diagram for workflows."""
        sequence = "sequenceDiagram\n"

        # Add actors
        if 'actors' in flow:
            for actor in flow['actors']:
                sequence += f"    participant {actor}\n"

        # Add interactions
        if 'interactions' in flow:
            for interaction in flow['interactions']:
                sequence += f"    {interaction}\n"

        return sequence

    def generate_architecture_diagram(self, components: List[Dict]) -> str:
        """Generate architecture diagram."""
        arch = "graph TD\n"

        for comp in components:
            arch += f"    {comp['name']}[{comp['label']}]\n"

        # Add relationships
        if 'connections' in components:
            for conn in components['connections']:
                arch += f"    {conn['from']} --> {conn['to']}\n"

        return arch

    def generate_data_flow_diagram(self, flows: List[Dict]) -> str:
        """Generate data flow diagram."""
        dfd = "graph LR\n"

        # Add entities
        entities = set()
        for flow in flows:
            entities.add(flow['source'])
            entities.add(flow['target'])

        for entity in entities:
            dfd += f"    {entity}[{entity}]\n"

        # Add flows
        for flow in flows:
            dfd += f"    {flow['source']} -->|{flow['type']}| {flow['target']}\n"

        return dfd


class RelationshipExtractor:
    """Extracts relationships between files."""

    def __init__(self):
        self.import_map = {}
        self.call_map = {}
        self.dependency_map = {}

    def extract_import_relationships(self, repo_path: Path) -> Dict[str, List[str]]:
        """Extract import relationships between files."""
        import_relationships = {}

        for py_file in repo_path.rglob('*.py'):
            content = py_file.read_text(encoding='utf-8')
            relative_path = str(py_file.relative_to(repo_path))

            # Find imports
            imports = re.findall(r'from\s+([\w.]+)\s+import|import\s+([\w.]+)', content)

            imported_files = []
            for imp in imports:
                module = imp[0] or imp[1]
                # Find corresponding file
                imported_file = self._find_file_for_module(repo_path, module)
                if imported_file:
                    imported_files.append(imported_file)

            import_relationships[relative_path] = imported_files

        return import_relationships

    def _find_file_for_module(self, repo_path: Path, module: str) -> Optional[str]:
        """Find file path for a given module."""
        # Try common module patterns
        possible_paths = [
            repo_path / f"{module}.py",
            repo_path / module / "__init__.py",
            repo_path / "src" / f"{module}.py",
            repo_path / "src" / module / "__init__.py"
        ]

        for path in possible_paths:
            if path.exists():
                return str(path.relative_to(repo_path))

        return None

    def extract_function_calls(self, repo_path: Path) -> Dict[str, List[str]]:
        """Extract function call relationships."""
        function_calls = {}

        for py_file in repo_path.rglob('*.py'):
            content = py_file.read_text(encoding='utf-8')
            relative_path = str(py_file.relative_to(repo_path))

            # Find function calls
            calls = re.findall(r'(\w+)\s*\(', content)

            # Filter out built-in functions
            builtin_functions = {'print', 'len', 'range', 'list', 'dict', 'str', 'int', 'float', 'bool'}
            filtered_calls = [call for call in calls if call not in builtin_functions]

            function_calls[relative_path] = filtered_calls

        return function_calls

    def generate_dependency_graph(self, repo_path: Path) -> str:
        """Generate dependency graph in DOT format."""
        imports = self.extract_import_relationships(repo_path)

        dot = "digraph Dependencies {\n"
        dot += "    rankdir=LR;\n"
        dot += "    node [shape=box, style=rounded];\n\n"

        # Add nodes
        for file_path in imports.keys():
            dot += f'    "{file_path}";\n'

        # Add edges
        for file_path, dependencies in imports.items():
            for dep in dependencies:
                dot += f'    "{file_path}" -> "{dep}";\n'

        dot += "}"
        return dot


class DocumentationGenerator:
    """Creates comprehensive markdown documentation."""

    def __init__(self):
        self.diagram_generator = DiagramGenerator()

    def generate_architecture_docs(self, repo_path: Path, analysis_results: List[FileAnalysis]) -> str:
        """Generate architecture documentation."""
        # Extract project structure
        structure = self._extract_project_structure(repo_path)

        # Identify main components
        components = self._identify_components(analysis_results)

        # Generate architecture documentation
        arch_doc = "# Project Architecture\n\n"
        arch_doc += f"**Repository:** {repo_path.name}\n\n"
        arch_doc += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        arch_doc += "## Overview\n\n"
        arch_doc += f"This project follows a {'monolithic' if len(components['services']) == 0 else 'modular'} architecture with the following key components:\n\n"

        if components['controllers']:
            arch_doc += "### Controllers\n"
            for ctrl in components['controllers']:
                arch_doc += f"- `{ctrl}` - Handles incoming requests and coordinates responses\n"
            arch_doc += "\n"

        if components['services']:
            arch_doc += "### Services\n"
            for svc in components['services']:
                arch_doc += f"- `{svc}` - Implements business logic\n"
            arch_doc += "\n"

        if components['models']:
            arch_doc += "### Models\n"
            for model in components['models']:
                arch_doc += f"- `{model}` - Data structures and database entities\n"
            arch_doc += "\n"

        # Add architecture diagram
        arch_doc += "## Architecture Diagram\n\n"
        arch_doc += "```mermaid\n"
        arch_diagram = self.diagram_generator.generate_architecture_diagram(components)
        arch_doc += arch_diagram
        arch_doc += "\n```\n"

        # Add project structure
        arch_doc += "## Project Structure\n\n"
        arch_doc += "```\n"
        arch_doc += structure
        arch_doc += "\n```\n"

        return arch_doc

    def generate_data_flow_docs(self, repo_path: Path, analysis_results: List[FileAnalysis]) -> str:
        """Generate data flow documentation."""
        # Extract data flow patterns
        data_flows = self._extract_data_flows(analysis_results)

        flow_doc = "# Data Flow Documentation\n\n"
        flow_doc += f"**Repository:** {repo_path.name}\n\n"

        flow_doc += "## Overview\n\n"
        flow_doc += "This document describes the data flow and lineage through the application:\n\n"

        # Add data sources
        flow_doc += "## Data Sources\n\n"
        sources = self._identify_data_sources(analysis_results)
        for source in sources:
            flow_doc += f"- **{source['name']}**: {source['type']} - {source['description']}\n"
        flow_doc += "\n"

        # Add data transformations
        flow_doc += "## Data Transformations\n\n"
        transformations = self._identify_transformations(analysis_results)
        for trans in transformations:
            flow_doc += f"### {trans['name']}\n"
            flow_doc += f"**Input:** {trans['input']}\n"
            flow_doc += f"**Output:** {trans['output']}\n"
            flow_doc += f"**Description:** {trans['description']}\n\n"

        # Add data sinks
        flow_doc += "## Data Sinks\n\n"
        sinks = self._identify_data_sinks(analysis_results)
        for sink in sinks:
            flow_doc += f"- **{sink['name']}**: {sink['type']}\n"
        flow_doc += "\n"

        # Add data flow diagram
        flow_doc += "## Data Flow Diagram\n\n"
        flow_doc += "```mermaid\n"
        flow_diagram = self.diagram_generator.generate_data_flow_diagram(data_flows)
        flow_doc += flow_diagram
        flow_doc += "\n```\n"

        return flow_doc

    def generate_workflow_docs(self, repo_path: Path, analysis_results: List[FileAnalysis]) -> str:
        """Generate business workflow documentation."""
        workflows = self._extract_workflows(analysis_results)

        workflow_doc = "# Business Workflows\n\n"
        workflow_doc += f"**Repository:** {repo_path.name}\n\n"

        for i, workflow in enumerate(workflows, 1):
            workflow_doc += f"## {i}. {workflow['name']}\n\n"
            workflow_doc += f"**Description:** {workflow['description']}\n\n"

            workflow_doc += "### Steps\n\n"
            for step in workflow['steps']:
                workflow_doc += f"1. **{step['name']}** - {step['description']}\n"
            workflow_doc += "\n"

            workflow_doc += "### Sequence Diagram\n\n"
            workflow_doc += "```mermaid\n"
            seq_diagram = self.diagram_generator.generate_sequence_diagram(workflow)
            workflow_doc += seq_diagram
            workflow_doc += "\n```\n\n"

        return workflow_doc

    def generate_code_relationships(self, repo_path: Path, analysis_results: List[FileAnalysis]) -> str:
        """Generate code relationships documentation."""
        extractor = RelationshipExtractor()

        # Extract relationships
        import_relationships = extractor.extract_import_relationships(repo_path)

        rel_doc = "# Code Relationships\n\n"
        rel_doc += f"**Repository:** {repo_path.name}\n\n"

        # Add import dependencies
        rel_doc += "## Import Dependencies\n\n"
        for file_path, dependencies in import_relationships.items():
            rel_doc += f"### {file_path}\n"
            if dependencies:
                for dep in dependencies:
                    rel_doc += f"- `{dep}`\n"
            else:
                rel_doc += "*No imports*\n"
            rel_doc += "\n"

        # Add dependency graph
        rel_doc += "## Dependency Graph\n\n"
        rel_doc += "```mermaid\n"
        dependency_graph = extractor.generate_dependency_graph(repo_path)
        rel_doc += dependency_graph
        rel_doc += "\n```\n"

        return rel_doc

    def _extract_project_structure(self, repo_path: Path) -> str:
        """Extract project structure as a tree."""
        def build_tree(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
            if current_depth >= max_depth:
                return []

            items = []
            try:
                entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
                for i, entry in enumerate(entries):
                    # Skip hidden files and common directories
                    if entry.name.startswith('.') or entry.name in ['__pycache__', 'node_modules', '.git']:
                        continue

                    is_last = i == len(entries) - 1
                    current = "└── " if is_last else "├── "
                    items.append(f"{prefix}{current}{entry.name}")

                    if entry.is_dir():
                        extension = "    " if is_last else "│   "
                        items.extend(build_tree(entry, prefix + extension, max_depth, current_depth + 1))
            except PermissionError:
                pass

            return items

        tree_lines = [repo_path.name + "/"]
        tree_lines.extend(build_tree(repo_path))
        return "\n".join(tree_lines)

    def _identify_components(self, analysis_results: List[FileAnalysis]) -> Dict:
        """Identify main components in the codebase."""
        components = {
            'controllers': [],
            'services': [],
            'models': [],
            'utils': [],
            'configs': []
        }

        for analysis in analysis_results:
            if 'controller' in analysis.file_path.lower():
                components['controllers'].append(analysis.file_path)
            elif 'service' in analysis.file_path.lower():
                components['services'].append(analysis.file_path)
            elif 'model' in analysis.file_path.lower():
                components['models'].append(analysis.file_path)
            elif 'util' in analysis.file_path.lower() or 'helper' in analysis.file_path.lower():
                components['utils'].append(analysis.file_path)
            elif 'config' in analysis.file_path.lower():
                components['configs'].append(analysis.file_path)

        # Add connections for diagram
        components['connections'] = []
        for analysis in analysis_results:
            if 'controller' in analysis.file_path.lower() and analysis.dependencies:
                for dep in analysis.dependencies:
                    for other_path in analysis_results:
                        if other_path.file_path != analysis.file_path and dep in other_path.file_path:
                            components['connections'].append({
                                'from': analysis.file_path.split('/')[-1].replace('.py', ''),
                                'to': other_path.file_path.split('/')[-1].replace('.py', '')
                            })

        return components

    def _extract_data_flows(self, analysis_results: List[FileAnalysis]) -> List[Dict]:
        """Extract data flow patterns."""
        flows = []

        # Simple data flow detection
        for analysis in analysis_results:
            if analysis.language in ['python', 'java']:
                # Look for database operations
                db_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'save()', 'create()', 'find()', 'query()']
                for pattern in db_patterns:
                    if pattern.upper() in analysis.business_logic or pattern in analysis.file_path.lower():
                        flows.append({
                            'source': analysis.file_path,
                            'target': 'Database',
                            'type': 'Data Access'
                        })

        return flows

    def _identify_data_sources(self, analysis_results: List[FileAnalysis]) -> List[Dict]:
        """Identify data sources in the codebase."""
        sources = []

        # Look for common data source patterns
        for analysis in analysis_results:
            if 'database' in analysis.file_path.lower() or 'db' in analysis.file_path.lower():
                sources.append({
                    'name': analysis.file_path,
                    'type': 'Database',
                    'description': 'Database schema and operations'
                })
            elif 'api' in analysis.file_path.lower():
                sources.append({
                    'name': analysis.file_path,
                    'type': 'API',
                    'description': 'External API integration'
                })

        return sources if sources else [{
            'name': 'Unknown',
            'type': 'Internal',
            'description': 'Internal data processing'
        }]

    def _identify_transformations(self, analysis_results: List[FileAnalysis]) -> List[Dict]:
        """Identify data transformations."""
        transformations = []

        for analysis in analysis_results:
            if analysis.functions:
                for func in analysis.functions[:3]:  # Top 3 functions
                    transformations.append({
                        'name': func['name'],
                        'input': 'Various data sources',
                        'output': 'Processed data',
                        'description': f'Function {func["name"]} processes and transforms data'
                    })

        return transformations

    def _identify_data_sinks(self, analysis_results: List[FileAnalysis]) -> List[Dict]:
        """Identify data sinks."""
        sinks = []

        # Look for output-related files
        for analysis in analysis_results:
            if 'output' in analysis.file_path.lower() or 'response' in analysis.file_path.lower():
                sinks.append({
                    'name': analysis.file_path,
                    'type': 'Response Handler'
                })
            elif 'report' in analysis.file_path.lower():
                sinks.append({
                    'name': analysis.file_path,
                    'type': 'Report Generator'
                })

        return sinks if sinks else [{
            'name': 'Default Output',
            'type': 'Console/File'
        }]

    def _extract_workflows(self, analysis_results: List[FileAnalysis]) -> List[Dict]:
        """Extract business workflows."""
        workflows = []

        # Common workflow patterns
        common_workflows = [
            {
                'name': 'User Authentication',
                'description': 'Process of user login and authentication',
                'steps': [
                    {'name': 'Login Request', 'description': 'User submits credentials'},
                    {'name': 'Validation', 'description': 'System validates credentials'},
                    {'name': 'Session Creation', 'description': 'System creates user session'},
                    {'name': 'Response', 'description': 'System returns authentication result'}
                ]
            },
            {
                'name': 'Data Processing',
                'description': 'Generic data processing workflow',
                'steps': [
                    {'name': 'Input Collection', 'description': 'Collect data from sources'},
                    {'name': 'Validation', 'description': 'Validate data integrity'},
                    {'name': 'Transformation', 'description': 'Apply business rules'},
                    {'name': 'Storage', 'description': 'Store processed data'}
                ]
            }
        ]

        # Enhance workflows based on actual code
        for workflow in common_workflows:
            # Find related files
            related_files = []
            for analysis in analysis_results:
                keywords = [workflow['name'].lower().replace(' ', '_')]
                if any(keyword in analysis.file_path.lower() for keyword in keywords):
                    related_files.append(analysis.file_path)

            if related_files:
                workflow['related_files'] = related_files

        return workflows


class RepositoryService:
    """Main service for transforming GitHub repositories into markdown documentation."""

    def __init__(self):
        self.cloner = RepositoryCloner(None)  # Will be set with GitHub client
        self.analyzer = CodeAnalyzer()
        self.doc_generator = DocumentationGenerator()
        self.relationship_extractor = RelationshipExtractor()

    async def transform_repository_to_markdown(
        self,
        url: str,
        branch: str = "main",
        output_dir: Optional[str] = None,
        max_files: int = 100
    ) -> DocumentationResult:
        """Transform a GitHub repository into comprehensive markdown documentation."""

        # Set output directory
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), "codebase-to-knowledge-docs")

        os.makedirs(output_dir, exist_ok=True)

        # Initialize result
        result = DocumentationResult(
            output_dir=output_dir,
            files_created=[],
            summary={
                'repository': url,
                'branch': branch,
                'timestamp': datetime.now().isoformat(),
                'total_files_analyzed': 0,
                'markdown_files_copied': 0,
                'code_files_transformed': 0,
                'errors': []
            },
            errors=[]
        )

        try:
            # Step 1: Clone repository
            logger.info(f"Cloning repository: {url}")
            repo_path = await self.cloner.clone_repository(url, branch)

            # Step 2: Extract existing markdown files
            logger.info("Extracting existing markdown files...")
            markdown_files = self.cloner.extract_existing_markdown(repo_path)
            result.summary['markdown_files_copied'] = len(markdown_files)

            # Copy markdown files to output directory
            for md_file in markdown_files:
                output_path = os.path.join(output_dir, md_file['path'])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(md_file['content'])
                result.files_created.append(output_path)

            # Step 3: Analyze code files
            logger.info("Analyzing code files...")
            code_files = []

            # Find all supported code files
            for ext in self.analyzer.supported_languages.keys():
                code_files.extend(repo_path.rglob(f"*{ext}"))

            # Limit number of files
            code_files = list(set(code_files))[:max_files]
            result.summary['total_files_analyzed'] = len(code_files)

            analysis_results = []
            for code_file in code_files:
                try:
                    analysis = self.analyzer.analyze_file(code_file)
                    if analysis:
                        analysis_results.append(analysis)
                except Exception as e:
                    error_msg = f"Error analyzing {code_file}: {str(e)}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

            result.summary['code_files_transformed'] = len(analysis_results)

            # Step 4: Generate documentation
            logger.info("Generating documentation...")

            # Generate architecture documentation
            arch_doc = self.doc_generator.generate_architecture_docs(repo_path, analysis_results)
            arch_path = os.path.join(output_dir, "ARCHITECTURE.md")
            with open(arch_path, 'w', encoding='utf-8') as f:
                f.write(arch_doc)
            result.files_created.append(arch_path)

            # Generate data flow documentation
            data_flow_doc = self.doc_generator.generate_data_flow_docs(repo_path, analysis_results)
            data_flow_path = os.path.join(output_dir, "DATA_FLOW.md")
            with open(data_flow_path, 'w', encoding='utf-8') as f:
                f.write(data_flow_doc)
            result.files_created.append(data_flow_path)

            # Generate workflow documentation
            workflow_doc = self.doc_generator.generate_workflow_docs(repo_path, analysis_results)
            workflow_path = os.path.join(output_dir, "WORKFLOWS.md")
            with open(workflow_path, 'w', encoding='utf-8') as f:
                f.write(workflow_doc)
            result.files_created.append(workflow_path)

            # Generate code relationships documentation
            relationships_doc = self.doc_generator.generate_code_relationships(repo_path, analysis_results)
            relationships_path = os.path.join(output_dir, "CODE_RELATIONSHIPS.md")
            with open(relationships_path, 'w', encoding='utf-8') as f:
                f.write(relationships_doc)
            result.files_created.append(relationships_path)

            # Step 5: Generate transformed code documentation
            logger.info("Transforming code files to markdown...")
            code_docs_dir = os.path.join(output_dir, "code_documentation")
            os.makedirs(code_docs_dir, exist_ok=True)

            for analysis in analysis_results:
                doc_path = os.path.join(code_docs_dir, f"{analysis.file_path.split('/')[-1].replace('.', '_')}.md")
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(analysis.documentation)
                result.files_created.append(doc_path)

            # Clean up
            if repo_path.parent.parent == self.cloner.temp_dir:
                shutil.rmtree(repo_path)
                logger.info("Cleaned up temporary repository")

            logger.info(f"Documentation generated successfully in: {output_dir}")

        except Exception as e:
            error_msg = f"Failed to generate documentation: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return list(self.analyzer.supported_languages.values())

    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types."""
        return list(self.analyzer.supported_languages.keys())