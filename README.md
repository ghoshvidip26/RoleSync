# RoleSync — AI Career & Outreach Engine

> **Align Your Profile. Accelerate Your Career.**

RoleSync is an automated, AI-assisted career platform and outreach engine. It ingests your resume PDF, indexes your real background using semantic vector retrieval (RAG), matches and tailors your resume for specific job descriptions without hallucination, reconstructs ATS-safe PDFs (or preserves your original document's layout), and drafts personalized cold outreach emails with your tailored resume attached.

---

## Key Features

- **Semantic Resume Parsing & RAG Indexing**: Extracts contact details, experience, skills, and projects using PyMuPDF and builds an isolated, local FAISS vector store per candidate profile.
- **Strict Anti-Hallucination Tailoring**: The Resume Tailor agent rewrites summaries and bullet points strictly using verified facts from your indexed history.
- **Dual PDF Generation Engine**:
  - **Original Layout Preservation**: Analyzes bounding boxes, typography, margins, and column layouts to reproduce your original resume's visual identity.
  - **ATS-Optimized Templates**: Fallback gallery of clean, recruiter-friendly layouts validated against text overlaps and page spillover.
- **Cold Outreach Agent**: Crafts high-converting emails tailored to company culture, answering recruiter questions with specific proof points from your background.
- **Human-in-the-Loop Review**: Full manual edit and approval gates at every phase (profile confirmation, tailored draft review, email editing, and sending).
- **Dual-Model Reliability**: Runs locally on Ollama (`qwen:latest`) by default, with seamless automatic fallback to Groq (`openai/gpt-oss-120b`) if local resources fail or time out.

---

## The 6-Step Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. Resume  │ ──► │   2. Job    │ ──► │   3. Fit    │
│ Upload & RAG│     │ Target & JD │     │ Match & Gaps│
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       ▼                                       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  6. E-mail  │ ◄── │  5. Design  │ ◄── │  4. Tailor  │
│ Send & Attach     │ ATS PDF Pre │     │ AI Customize│
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Resume (`UploadStep`)**: Upload a PDF resume (up to 5 MB). Review extracted sections, confirm profile details, and build the FAISS search index.
2. **Job (`JobStep`)**: Provide the job title, target company, and paste the job description (minimum 40 characters).
3. **Fit (`FitStep`)**: Review the match score, covered qualifications, missing skill gaps, and AI advice on whether resume tailoring is recommended.
4. **Tailor (`DraftStep`)**: Review an AI-customized draft aligning your real experience to the job description. Edit any bullet or section, then approve.
5. **Design (`DesignStep`)**: Choose between preserving your original resume format or selecting an ATS-friendly design from the gallery. Inspect the live rendered PDF preview.
6. **E-mail (`EmailStep`)**: Answer custom recruiter questions and add personal notes. Generate, edit, and approve the email, which is delivered via SMTP with the tailored PDF attached.

---

## Tech Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.14+)
- **LLM & Embeddings**: [LangChain](https://www.langchain.com/) + [Ollama](https://ollama.com/) (`qwen:latest`, `qwen3-embedding:latest`)
- **LLM Fallback**: [Groq](https://groq.com/) API client (`llm_fallback.py`)
- **Vector Retrieval**: [FAISS](https://github.com/facebookresearch/faiss)
- **PDF Extraction & Layout Analysis**: PyMuPDF (`fitz`), PyPDF
- **Database**: [Supabase](https://supabase.com/) / PostgreSQL
- **Email Delivery**: Standard Python SMTP client

### Frontend
- **Framework**: [React 19](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Bundler & Dev Server**: [Vite 8](https://vitejs.dev/)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **State Management**: Context API backed by `sessionStorage` (workflow persists across reloads)

---

## Project Structure

```
Automation/
├── app/
│   ├── agents/                     # AI agents (review, role_recommender, resume_tailor, tailoring_decision)
│   ├── db/                         # Supabase database operations & models
│   ├── models/                     # Pydantic schemas (profile, company, recipient, email, draft)
│   ├── services/                   # Business logic (resume extraction, outreach, email SMTP, fallback)
│   │   └── resume_pdf_renderers/   # PDF rendering, layout analyzer, templates, and validator
│   └── main.py                     # FastAPI application endpoints
├── frontend/
│   ├── src/
│   │   ├── components/             # Reusable components (e.g., RoleSyncLogo)
│   │   ├── steps/                  # Multi-step views (Upload, Job, Fit, Draft, Design, Email)
│   │   ├── api.ts                  # Typed client for backend communication
│   │   ├── flowModel.ts / flow.tsx # Multi-step state machine & sessionStorage sync
│   │   ├── index.css               # Midnight Indigo + Electric Cyan Tailwind CSS theme
│   │   └── ui.tsx                  # Shared form fields, buttons, badges, and alerts
│   ├── package.json
│   └── vite.config.ts              # Vite proxy configuration to backend
├── supabase/
│   └── schema.sql                  # PostgreSQL schema definitions and RLS policies
├── data/
│   ├── indexes/                    # Profile-specific FAISS vector stores
│   └── resumes/                    # Cached uploaded PDF files and manifests
├── tests/
│   ├── outreach/                   # Tests for email generation & SMTP delivery
│   ├── pdf/                        # Tests for layout manifest and PDF renderers
│   └── resume/                     # Tests for resume parsing, RAG, and tailoring
├── pytest.ini
└── requirements.txt
```

---

## Getting Started

### 1. Prerequisites
- **Python 3.11+** (Python 3.12+ recommended)
- **Node.js 20+** and **npm**
- **Ollama** installed and running locally
- A **Supabase** project (free tier works)

### 2. Backend Setup

1. **Clone the repository and enter the directory**:
   ```bash
   cd Automation
   ```

2. **Set up a Python virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Pull required Ollama models**:
   ```bash
   ollama pull qwen:latest
   ollama pull qwen3-embedding:latest
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   # Supabase Configuration
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-supabase-service-role-key

   # SMTP Delivery (for sending emails)
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_EMAIL=your-email@gmail.com
   SMTP_PASSWORD=your-app-password

   # Optional: Groq Fallback (used if Ollama fails)
   GROQ_API_KEY=gsk_your_groq_api_key
   GROQ_CHAT_MODEL=openai/gpt-oss-120b

   # Optional Overrides
   # OLLAMA_CHAT_MODEL=qwen:latest
   # OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest
   # LLM_FALLBACK=on
   ```

5. **Set up the Database**:
   Run the SQL statements from [`supabase/schema.sql`](supabase/schema.sql) in your Supabase SQL Editor to create the required tables (`profiles`, `companies`, `recipients`, `emails`, `resume_drafts`).

6. **Start the FastAPI backend server**:
   ```bash
   .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

---

### 3. Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```
   Open [http://localhost:5173](http://localhost:5173) in your browser.

> **Note**: The frontend proxy in `vite.config.ts` automatically forwards all `/api` requests to `http://127.0.0.1:8000`, so no manual CORS configuration is necessary.

---

## Running Checks & Tests

### Backend Tests (pytest)
To execute the comprehensive Python test suite covering resume extraction, layout reconstruction, PDF rendering, and outreach generation:
```bash
.venv/bin/pytest tests/
```

### Frontend Tests & Typecheck (vitest & tsc)
From the `frontend/` directory:
```bash
cd frontend

# Run unit and screen tests
npm test

# Run strict TypeScript verification
npm run typecheck

# Run linter
npm run lint

# Build production bundle
npm run build
```

---

## Configuration Reference

| Environment Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `SUPABASE_URL` | **Yes** | — | Supabase project API URL |
| `SUPABASE_KEY` | **Yes** | — | Supabase service role key (bypasses RLS) |
| `SMTP_HOST` | **Yes** | — | SMTP mail server host |
| `SMTP_PORT` | **Yes** | `587` | SMTP port |
| `SMTP_EMAIL` | **Yes** | — | Sender email address for authentication |
| `SMTP_PASSWORD` | **Yes** | — | SMTP app password or credential |
| `GROQ_API_KEY` | No | — | Groq API key for zero-downtime LLM fallback |
| `GROQ_CHAT_MODEL` | No | `openai/gpt-oss-120b` | Model used when falling back to Groq |
| `OLLAMA_CHAT_MODEL` | No | `qwen:latest` | Local chat model for parsing and drafting |
| `OLLAMA_EMBEDDING_MODEL`| No | `qwen3-embedding:latest` | Embedding model for FAISS vector indexing |
| `INDEX_DIR` | No | `data/indexes` | Directory path for local FAISS index storage |
| `LLM_FALLBACK` | No | `on` (if key set) | Set to `off` to disable remote fallback entirely |

---

## License

Private project. All rights reserved.

