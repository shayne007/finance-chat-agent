"""
Tests for repository service functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.repository_service import RepositoryService, CodeAnalyzer, DocumentationGenerator


class TestCodeAnalyzer(unittest.TestCase):
    """Test cases for CodeAnalyzer."""

    def setUp(self):
        self.analyzer = CodeAnalyzer()

    def test_analyze_python_file(self):
        """Test Python file analysis."""
        # Create a simple Python file
        python_code = '''
"""
A simple Python module.
"""

import os
import sys
from datetime import datetime

class User:
    """User class."""

    def __init__(self, name, email):
        self.name = name
        self.email = email

    def get_info(self):
        return f"{self.name} ({self.email})"

def greet(name):
    """Greet function."""
    return f"Hello, {name}!"

# Main execution
if __name__ == "__main__":
    print("Hello, World!")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            temp_file = Path(f.name)

        try:
            result = self.analyzer.analyze_file(temp_file)

            self.assertIsNotNone(result)
            self.assertEqual(result.language, 'python')
            self.assertEqual(result.file_type, 'code')
            self.assertTrue(len(result.classes) > 0)
            self.assertTrue(len(result.functions) > 0)
            self.assertTrue(len(result.imports) > 0)
            self.assertTrue('User' in [cls['name'] for cls in result.classes])
            self.assertTrue('greet' in [func['name'] for func in result.functions])

        finally:
            temp_file.unlink()

    def test_analyze_sql_file(self):
        """Test SQL file analysis."""
        sql_code = '''
-- Sample SQL database
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    product_name VARCHAR(100),
    amount DECIMAL(10, 2),
    order_date TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Stored procedure
CREATE PROCEDURE GetUserOrders(IN user_id INT)
BEGIN
    SELECT * FROM orders WHERE user_id = user_id;
END;
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(sql_code)
            temp_file = Path(f.name)

        try:
            result = self.analyzer.analyze_file(temp_file)

            self.assertIsNotNone(result)
            self.assertEqual(result.language, 'sql')
            self.assertTrue(len(result.classes) == 0)  # No classes in SQL
            self.assertTrue(len(result.functions) > 0)  # Has stored procedures
            self.assertTrue(len(result.functions) == 1)  # One stored procedure
            self.assertTrue('GetUserOrders' in [func['name'] for func in result.functions])

        finally:
            temp_file.unlink()

    def test_get_supported_languages(self):
        """Test supported languages."""
        languages = self.analyzer.get_supported_languages()
        self.assertIn('python', languages)
        self.assertIn('java', languages)
        self.assertIn('sql', languages)


class TestDocumentationGenerator(unittest.TestCase):
    """Test cases for DocumentationGenerator."""

    def setUp(self):
        self.generator = DocumentationGenerator()

    def test_generate_architecture_docs(self):
        """Test architecture documentation generation."""
        # Create mock analysis results
        from app.services.repository_service import FileAnalysis

        analysis_results = [
            FileAnalysis(
                file_path="src/controllers/user_controller.py",
                language="python",
                file_type="code",
                classes=[],
                functions=[],
                imports=[],
                dependencies=[],
                business_logic="Handles user authentication",
                complexity_score=3,
                documentation="# user_controller.py"
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs = self.generator.generate_architecture_docs(temp_path, analysis_results)

            self.assertIsInstance(docs, str)
            self.assertIn("# Project Architecture", docs)
            self.assertIn("User Authentication", docs)  # From business logic

    def test_generate_data_flow_docs(self):
        """Test data flow documentation generation."""
        from app.services.repository_service import FileAnalysis

        analysis_results = [
            FileAnalysis(
                file_path="src/services/data_service.py",
                language="python",
                file_type="code",
                classes=[],
                functions=[],
                imports=[],
                dependencies=[],
                business_logic="Processes user data and saves to database",
                complexity_score=2,
                documentation="# data_service.py"
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs = self.generator.generate_data_flow_docs(temp_path, analysis_results)

            self.assertIsInstance(docs, str)
            self.assertIn("# Data Flow Documentation", docs)
            self.assertIn("Database", docs)  # Should identify data source


class TestRepositoryService(unittest.TestCase):
    """Test cases for RepositoryService."""

    def setUp(self):
        self.service = RepositoryService()

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.service.get_supported_languages()
        self.assertIsInstance(languages, list)
        self.assertTrue(len(languages) > 0)

    def test_get_supported_file_types(self):
        """Test getting supported file types."""
        file_types = self.service.get_supported_file_types()
        self.assertIsInstance(file_types, list)
        self.assertIn('.py', file_types)
        self.assertIn('.java', file_types)
        self.assertIn('.sql', file_types)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    @patch('app.services.repository_service.RepositoryCloner.clone_repository')
    def test_repository_integration(self, mock_clone):
        """Test full repository processing flow."""
        # Mock the clone method to return a temp directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "README.md").write_text("# Test Repository\n\nThis is a test repository.")
            (temp_path / "main.py").write_text("""
"""
A simple Python script.
"""

print("Hello, World!")
""")
            (temp_path / "schema.sql").write_text("""
CREATE TABLE test_table (id INT PRIMARY KEY, name VARCHAR(100));
""")

            # Mock the clone method
            def mock_clone_impl(url, branch):
                return temp_path
            mock_clone.side_effect = mock_clone_impl

            # Test the service
            service = RepositoryService()

            with tempfile.TemporaryDirectory() as output_dir:
                result = service.transform_repository_to_markdown(
                    url="https://github.com/test/repo",
                    branch="main",
                    output_dir=output_dir,
                    max_files=10
                )

                # Check result
                self.assertEqual(result.summary['markdown_files_copied'], 1)
                self.assertEqual(result.summary['code_files_transformed'], 2)
                self.assertTrue(len(result.files_created) > 0)
                self.assertIn('README.md', [f for f in result.files_created if 'README' in f])


if __name__ == '__main__':
    unittest.main()
