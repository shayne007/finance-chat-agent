#!/usr/bin/env python3
"""
Command-line interface for generating documentation from GitHub repositories.

This script provides a simple way to test the repository documentation generation
functionality without needing to go through the web interface.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

from app.services.repository_service import RepositoryService
from app.core.config import github_settings


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_github_token():
    """Validate GitHub token configuration."""
    if not github_settings.token:
        print("Error: GitHub token is required. Please set GITHUB_TOKEN environment variable")
        return False

    if not github_settings.enabled:
        print("Warning: GitHub integration is disabled in configuration")

    return True


def generate_docs(repo_url: str, branch: str = "main", output_dir: str = None, max_files: int = 100):
    """Generate documentation from a GitHub repository."""

    # Validate inputs
    if not repo_url:
        print("Error: Repository URL is required")
        return False

    # Set up GitHub client
    if not validate_github_token():
        return False

    # Initialize repository service
    try:
        repository_service = RepositoryService()
        print(f"Initializing repository service...")
    except Exception as e:
        print(f"Error initializing service: {str(e)}")
        return False

    # Generate documentation
    try:
        print(f"Generating documentation from: {repo_url}")
        print(f"Branch: {branch}")
        print(f"Max files: {max_files}")
        if output_dir:
            print(f"Output directory: {output_dir}")
        print("-" * 50)

        result = repository_service.transform_repository_to_markdown(
            url=repo_url,
            branch=branch,
            output_dir=output_dir,
            max_files=max_files
        )

        # Display results
        print("\nDocumentation Generation Complete!")
        print("=" * 50)
        print(f"Output directory: {result.output_dir}")
        print(f"Files created: {len(result.files_created)}")
        print(f"Total files analyzed: {result.summary['total_files_analyzed']}")
        print(f"Markdown files copied: {result.summary['markdown_files_copied']}")
        print(f"Code files transformed: {result.summary['code_files_transformed']}")

        if result.errors:
            print("\nErrors encountered:")
            for error in result.errors:
                print(f"  - {error}")

        print("\nGenerated files:")
        for file_path in result.files_created:
            print(f"  - {file_path}")

        return True

    except Exception as e:
        print(f"Error generating documentation: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate markdown documentation from GitHub repositories"
    )
    parser.add_argument(
        "repo_url",
        help="GitHub repository URL (e.g., https://github.com/user/repo)"
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to clone (default: main)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for generated documentation"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum number of code files to analyze (default: 100)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Generate documentation
    success = generate_docs(
        repo_url=args.repo_url,
        branch=args.branch,
        output_dir=args.output_dir,
        max_files=args.max_files
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()