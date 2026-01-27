# Chat Agent Project Constitution
# Version: 1.0, Ratified: 2026-01-25

This document defines the unshakeable core development principles for this project. All AI Agents must unconditionally follow these rules during technical planning and code implementation.

---

## Article 1: Simplicity First

**Core:** Follow the "Pythonic" philosophy (Explicit is better than implicit). Never create unnecessary abstractions; never introduce non-essential dependencies.

- **1.1 (YAGNI):** You Ain't Gonna Need It. Only implement features explicitly required in `spec.md`.
- **1.2 (Lightweight Dependencies):** Prioritize the Python Standard Library (e.g., `pathlib`, `json`, `sqlite3`) and established ecosystem libraries defined in `requirements.txt` (e.g., `FastAPI`, `Pydantic`, `SQLAlchemy`). Avoid introducing libraries with overlapping functionality or heavy dependencies for minor features.
- **1.3 (Anti-Over-Engineering):** Avoid complex class inheritance hierarchies. Prioritize simple functions, data classes (Data Classes/Pydantic Models), and composition. If a feature can be implemented with a simple function, do not create a class.

---

## Article 2: Test-First Imperative - Non-Negotiable

**Core:** All new features or bug fixes must begin with writing one (or more) failing tests.

- **2.1 (TDD Cycle):** Strictly follow the "Red-Green-Refactor" cycle (Write failing test -> Make test pass -> Refactor).
- **2.2 (Parameterized Tests):** Unit tests must prioritize `pytest`'s parameterized testing style (`@pytest.mark.parametrize`) to cover multiple inputs and edge cases.
- **2.3 (Reject Excessive Mocking):** Prioritize integration tests. For external dependencies (e.g., OpenAI, Jira), use mock servers or Fake implementations rather than excessively mocking internal function calls in unit tests.

---

## Article 3: Clarity and Explicitness

**Core:** The primary purpose of code is for humans to understand, and secondarily for machines to execute.

- **3.1 (Type Hints & Error Handling):**
    - **Non-Negotiable**: All public functions and class methods must include Type Hints.
    - **Explicit Error Handling**: Never use bare `except:` to discard errors. Specific exceptions must be caught. When propagating exceptions, preserve the original stack trace or wrap them in custom exceptions.
- **3.2 (No Implicit State):** Avoid using module-level global variables to pass state. Dependencies should be explicitly injected via constructors or function arguments.
- **3.3 (Documentation & Comments):** Comments should explain "why", not "what". All public modules, classes, and functions must have clear Docstrings (Google style recommended).

---

## Article 4: Single Responsibility

**Core:** Every module, every file, and every function should do one thing and do it well.

- **4.1 (Module Cohesion):** Packages under the `app` directory (e.g., `agents`, `services`) should remain highly cohesive. For example, `jira_agent` is responsible only for Jira-related logic and should not contain generic database operations.
- **4.2 (Interface Segregation):** Define clear Protocols or Abstract Base Classes (ABCs), avoiding massive "God Objects".

---

## Governance

This Constitution holds the highest priority, overriding any `CLAUDE.md` or single-session instructions. Any plan (`plan.md`) generated must first undergo a "Constitutionality Review".
