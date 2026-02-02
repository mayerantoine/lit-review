# LitReview

Automated literature review generation using AI. Upload a CSV of research papers and generate a cohesive "Related Work" section for your research idea.

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.12+
- OpenAI API key

### Local Development

1. **Install dependencies**
   ```bash
   # Frontend
   npm install

   # Backend
   pip install uv
   uv sync
   ```

2. **Set environment variable**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run development servers**
   ```bash
   # Terminal 1: Frontend (Next.js)
   npm run dev

   # Terminal 2: Backend (FastAPI)
   cd api
   uvicorn index:app --reload --port 8000
   ```

4. **Open application**
   - Navigate to http://localhost:3000
   - Upload a CSV file with columns: `id`, `title`, `abstract`
  - Enter your research idea
  - Generate literature review

### Docker

```bash
docker build -t lit-review .
docker run -p 8000:8000 -e OPENAI_API_KEY="your-api-key" lit-review
```

Navigate to http://localhost:8000
