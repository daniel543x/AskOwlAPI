# AskOwlAPI - Project Roadmap

## Stage 1: LLM Foundations (Core Ask Refactoring)
*Version: 0.1.0-alpha*

**Description:**  
Complete replacement of the communication engine with LangChain. Create an abstraction for models so that the `ask` module is independent of a specific provider (OpenAI, Anthropic, Ollama). Fix basic search routing and introduce persistent chat history tied to users.

### To do:

- **Part 1: Core Refactoring & Basic Route (`/ask/search`)**
    - [x] Rewrite LLM factory to use LangChain (implement `factory.py` with provider classes).
    - [x] Fix the `/ask/search` route – ensure it correctly invokes the LangChain chain.
    - [x] Verify and fix the number of links returned by the ranker (web search).
    - [ ] Add support for both streaming and non-streaming responses in `/ask/search`.
        - [x] Implement streaming version using Server-Sent Events (SSE) for token-by-token delivery.
        - [ ] **Ensure the streaming format is "backend-friendly"** – use a simple JSON structure per event, easy to parse by external services.
        - [ ] Ensure the non-streaming fallback returns a complete JSON response.

*Version: 0.2.0-alpha*

- **Part 2: History Module (PostgreSQL, User-bound)**
    - [ ] Design the database schema for conversation history in PostgreSQL.
        - Tables: `users`, `conversations` (`id`, `user_id`, `created_at`), `messages` (`id`, `conversation_id`, `role` (user/assistant), `content`, `created_at`).
    - [ ] Create the `history` module structure (`src/module/history/`).
        - [ ] Implement repository pattern for database operations (add message, get conversation history by `conversation_id` and `user_id`).
    - [ ] Integrate history with the `/ask/search` endpoint.
        - [ ] On each request, load previous messages from the current conversation (scoped to the authenticated user).
        - [ ] Save the user query and the assistant's response to the database.
    - [ ] Add an optional `conversation_id` parameter to the route. If not provided, create a new conversation.

*Version: 0.3.0-alpha*

- **Part 3: API Key Authentication (User-bound)**
    - [ ] Design the `api_keys` table: `id`, `key_hash` (store hash, not raw key), `name`, `user_id` (NOT NULL, foreign key to `users`), `last_used_at`, `expires_at` (optional), `is_active`.
    - [ ] Implement API key authentication middleware.
        - [ ] Extract key from `Authorization: Bearer <key>` header.
        - [ ] Validate key against the database (check existence, `is_active` true, not expired).
        - [ ] **Check if the associated user is active/not blocked.** If the user is blocked, reject the request even if the key is valid.
    - [ ] Secure the `/ask/search` route – require a valid API key.
    - [ ] Create user endpoints to manage API keys:
        - `POST /api-keys` – generate a new key (return raw key once, store hash).
        - `GET /api-keys` – list own keys (without hashes).
        - `DELETE /api-keys/{id}` – revoke a key.
    - [ ] Add rate limiting based on API key (e.g., using Redis, alternatives to Redis or in-memory store).

*Version: 0.4.0-alpha*

---

## Stage 2: Agentic Logic (Core Agentic Module)
*Version: 0.0.0*

**Description:**  

### To do:
- [ ] Create new branch per new feature. 
 
---

## Stage 3: Custom Agents (Core <Good Name> Module)
*Version: 0.0.0*

**Description:**  

### To do:

---

## Stage 4: Big Beautiful Tests 
*Version: N/A*

**Description:**  

### To do:

---

## Stage 5: Release Candidate (V1-RC)
*Version: 1.0.0-rc*

**Description:**
Comprehensive testing phase to ensure system stability, security compliance, and correct behavior of all implemented features before the final production launch.

### To do:

- **Part 1: Authentication & Authorization Testing**
    - [ ] Verify user registration and login flows.
        - [ ] Test `POST /auth/register` with valid and invalid data.
        - [ ] Test `POST /auth/login` and `POST /auth/refresh` token logic.
        - [ ] Verify password hashing implementation (bcrypt).
        - [ ] Validate that the `is_active` flag correctly blocks user access.
    - [ ] Verify Role-Based Access Control (RBAC) isolation.
        - [ ] Ensure `user` cannot access `admin` endpoints.
        - [ ] Verify `admin` privileges (manage models, logs, block users).
    - [ ] Test protection on all endpoints (verify JWT enforcement vs API key usage).

- **Part 2: Database & Migration Validation**
    - [ ] Validate Alembic migration scripts.
        - [ ] Test applying migrations on a clean database.
        - [ ] Test migration rollback (downgrade) paths.
    - [ ] Verify data integrity for all tables (`users`, `api_keys`, `conversations`, `messages`, `agents`).
    - [ ] Validate soft-delete mechanism (ensure data is hidden but preserved in DB).

- **Part 3: Containerization & Deployment Testing**
    - [ ] Verify the build process of the `Containerfile` (image size, layers).
    - [ ] Test `compose.yml` orchestration (network connectivity, volume persistence).
    - [ ] Validate environment variable loading (check `.env` handling and defaults).
    - [ ] Execute deployment dry-run on target environment (bare metal or Swarm) following the written instructions.

- **Part 4: Documentation & CI/CD Audit**
    - [ ] Verify API documentation accuracy (request/response schemas match implementation).
    - [ ] Validate `README.md` setup steps (fresh clone test).
    - [ ] Review CI/CD pipeline execution.
        - [ ] Confirm tests block PRs on failure.
        - [ ] Verify linting tools (`ruff`, `black`) catch style issues.
        - [ ] Validate Docker image build and push workflow on tag.
    - [ ] Conduct security verification (run dependency scans, check for exposed secrets).

---

## Additional Tasks (Sometime)

### Future Tool Integrations
- [ ] **Integrate external search APIs:** Add support for Tavily, SerpAPI, or Bing Search as optional backends for `SearchTool` (configurable per request or per agent).
- [ ] **API Caller Tool:** Allow agent to call external REST APIs (with user-provided credentials).

### Code Quality & Refactoring
- [ ] Standardize comments across the codebase and add doc for it.
- [ ] Add more inline comments in complex modules.
- [ ] Breaking down single-line loops.
- [x] Make a pre-vision for docs.
- [x] Refactor `cryptography.py` to an object-oriented design (create a `CryptoService` class).
- [ ] Looking for / check out code review tools.
