import os
import uuid
import re
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
import time
import shutil
from pathlib import Path
from pipeline import LiteratureReviewPipeline, PipelineConfig, ValidationError, ProcessingError
from dataclasses import asdict

#TODO
### README, commit and push to github
### Develop branch

### Add link references - paper titles - Citations
### Download link
### Set OpenAI Key in UI

## Add list top papers with - score, pro, cons
## Change selection of top k papers
## Add agentic retrieval on top of hybrid retreival
## Review UI for generate to be on the right
## Decoupled Ranking from upload and Index
## Ui input for top-k,hybrid k, display pro and cons,

## Change embeddings to openai

#### Use terraform to help for both deployment

## AWS branch
## should work both local on AWS
## Move from Chroma to Beckrok KB
## Test docker deploy App Runner AWS
##-----> Frontend and backend stored in container  + Bedrock KB
## Move away from App Runnner to Lambda
## Uses terraform for provisioning
##-----> Frontend static S3+CloudFront and backend in Lambda + Bedrock KB

#### Azure branch
## work both with AzureOpenAI and OpenAI
## should work both local and AzureWebApp
## Keep using Chroma but add mount AzureShareFile
##-----> Frontend and backend in container + ChromaDB in AzureShare File

## Test docker deploy App Service Azure
## Develop 3 tests datasets with comparable generated work section
## Improve Both UI


class ResearchIdeaRequest(BaseModel):
    research_idea: str
    hybrid_k: int | None = 50
    selected_paper_ids: list[int] | None = None

app = FastAPI()
# Configure CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Attach session ID to all requests via cookie"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())[:8]  # 8-char session ID

    # Attach to request state
    request.state.session_id = session_id

    # Process request
    response = await call_next(request)

    # Set cookie on response (24 hour expiry)
    response.set_cookie(
        key="session_id",
        value=session_id,
        max_age=86400,  # 24 hours
        httponly=True,
        samesite="lax"
    )

    return response

def sanitize_filename(filename: str) -> str:
    """Convert filename to safe collection name"""
    base = Path(filename).stem  # Remove extension
    # Replace special chars with underscore, keep alphanumeric
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', base)
    # Lowercase and limit length
    return safe.lower()[:50]

# Directory setup
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Session store (in-memory - upgrade to Redis/DB for production)
SESSIONS: Dict[str, Dict[str, Any]] = {}
# Format: {session_id: {csv_path, collection_name, timestamp}}

# Global variables to track state (backward compatibility)
LAST_CSV_PATH = None  # Track last uploaded CSV file
# NOTE: TOP_RANKED_PAPERS and LAST_QUERY moved to per-session storage in SESSIONS dict

@app.post("/api/upload-and-index")
async def upload_and_index(file: UploadFile = File(...), request: Request = None):
    """
    Upload a CSV file and build the vector index for literature review.

    Returns:
        JSON with indexing statistics
    """
    global LAST_CSV_PATH, SESSIONS

    # Get session ID
    session_id = request.state.session_id

    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Generate collection name: {session_id}_{sanitized_filename}
    safe_filename = sanitize_filename(file.filename)
    collection_name = f"{session_id}_{safe_filename}"

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Store session metadata
        SESSIONS[session_id] = {
            "csv_path": str(file_path),
            "collection_name": collection_name,
            "timestamp": time.time()
        }
        print(session_id,SESSIONS[session_id])

        # Store the path globally for backward compatibility
        LAST_CSV_PATH = str(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Initialize pipeline with collection name
    config = PipelineConfig(
        persist_directory="./corpus-data/chroma_db",
        collection_name=collection_name,
        recreate_index=True,  # Always recreate on new upload
        hybrid_k=50,
        num_abstracts_to_score=None,
        top_k=3,
        relevance_model="gpt-4o-mini",
        generation_model="gpt-4o-mini",
        random_seed=42
    )

    pipeline = LiteratureReviewPipeline(config)

    # Build index
    try:
        result = pipeline.build_index(str(file_path))

        # Convert result to dict for JSON response
        response_data = {
            "success": True,
            "message": "Index created successfully",
            "collection_name": collection_name,
            "session_id": session_id,
            "csv_path": result.csv_path,
            "total_abstracts": result.total_abstracts,
            "chunks_created": result.chunks_created,
            "total_indexed": result.total_indexed,
            "persist_directory": result.persist_directory,
            "recreated": result.recreated
        }

        print(session_id,response_data)

        return JSONResponse(content=response_data)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/retrieve-and-rank")
def retrieve_and_rank(request: ResearchIdeaRequest, http_request: Request = None):
    """
    Retrieve and rank papers for a given research idea (Steps 4-6 of pipeline).

    This endpoint must be called after /api/upload-and-index and before /api/generate.
    It performs retrieval, relevance scoring, and top-k selection.

    Returns:
        JSON with top-k papers, retrieval stats, and scoring stats
    """
    global LAST_CSV_PATH, SESSIONS

    # Get session ID
    session_id = http_request.state.session_id

    # Check if session has uploaded data
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=400,
            detail="No active session. Please upload a CSV file first using 'Upload & Index File'."
        )

    # Get session data
    session_data = SESSIONS[session_id]
    collection_name = session_data["collection_name"]
    csv_path = session_data["csv_path"]

    # Validate CSV still exists
    if not Path(csv_path).exists():
        raise HTTPException(
            status_code=400,
            detail="CSV file not found. Please upload the file again."
        )

    # Get the research idea from request
    query = request.research_idea.strip()

    # Validate and clamp hybrid_k to valid range
    hybrid_k_value = min(max(request.hybrid_k or 50, 1), 200)

    # Initialize pipeline with session's collection
    config = PipelineConfig(
        persist_directory="./corpus-data/chroma_db",
        collection_name=collection_name,
        recreate_index=False,  # Use existing index
        hybrid_k=hybrid_k_value,
        num_abstracts_to_score=None,
        top_k=3,
        relevance_model="gpt-4o-mini",
        generation_model="gpt-4o-mini",
        random_seed=42
    )

    try:
        pipeline = LiteratureReviewPipeline(config)

        # Load abstracts from the session's CSV
        pipeline.load_abstracts_only(csv_path)

        # Retrieve and rank papers (Steps 4-6)
        top_k_abstracts, all_scored_papers, retrieval_stats, scoring_stats = pipeline.retrieve_and_rank_papers(query)

        # Store results in session for use in /api/generate
        # Store ALL scored papers so users can select from any of them
        SESSIONS[session_id]["ranked_papers"] = all_scored_papers
        SESSIONS[session_id]["last_query"] = query

        # Convert top-k DataFrame to list of dicts for JSON response
        top_papers_list = []
        for _, paper in top_k_abstracts.iterrows():
            top_papers_list.append({
                "id": int(paper['id']),
                "title": str(paper['title']),
                "abstract": str(paper['abstract']),
                "relevance_score": float(paper['relevance_score'])
            })

        # Convert ALL scored papers DataFrame to list of dicts
        all_scored_papers_list = []
        for _, paper in all_scored_papers.iterrows():
            all_scored_papers_list.append({
                "id": int(paper['id']),
                "title": str(paper['title']),
                "abstract": str(paper['abstract']),
                "relevance_score": float(paper['relevance_score'])
            })

        # Build response
        response_data = {
            "success": True,
            "query": query,
            "top_papers": top_papers_list,
            "all_scored_papers": all_scored_papers_list,
            "retrieval_stats": {
                "total_papers_in_corpus": retrieval_stats.total_papers_in_corpus,
                "papers_retrieved": retrieval_stats.papers_retrieved,
                "retrieval_rate": retrieval_stats.retrieval_rate,
                "retrieval_k": retrieval_stats.retrieval_k
            },
            "scoring_stats": {
                "papers_scored": scoring_stats.papers_scored,
                "mean_score": scoring_stats.mean_score,
                "std_score": scoring_stats.std_score,
                "min_score": scoring_stats.min_score,
                "max_score": scoring_stats.max_score,
                "median_score": scoring_stats.median_score
            }
        }

        return JSONResponse(content=response_data)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/generate")
def lit_review(request: ResearchIdeaRequest, http_request: Request = None):
    """
    Generate related work section using pre-ranked papers (streaming).

    Requires that /api/retrieve-and-rank has been called first to rank papers.
    This endpoint only performs Step 7 (text generation) using the pre-ranked papers.
    """
    global SESSIONS

    # Get session ID
    session_id = http_request.state.session_id

    # Verify session exists
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=400,
            detail="No active session. Please upload and rank papers first."
        )

    # Get the research idea from request
    query = request.research_idea.strip()

    # Check if retrieve-and-rank has been called for THIS session
    if "ranked_papers" not in SESSIONS[session_id]:
        raise HTTPException(
            status_code=400,
            detail="No ranked papers found. Please call /api/retrieve-and-rank first to rank papers."
        )

    # Get session-specific ranked papers and query
    ranked_papers = SESSIONS[session_id]["ranked_papers"]
    last_query = SESSIONS[session_id]["last_query"]

    # Verify query matches the one used for ranking
    if last_query != query:
        raise HTTPException(
            status_code=400,
            detail=f"Query mismatch. Ranked papers were retrieved for a different query. Please call /api/retrieve-and-rank again with the current query."
        )

    # Filter papers based on selected_paper_ids if provided
    papers_to_use = ranked_papers
    if request.selected_paper_ids is not None and len(request.selected_paper_ids) > 0:
        # Filter to only include selected papers
        papers_to_use = ranked_papers[ranked_papers['id'].isin(request.selected_paper_ids)]
        if len(papers_to_use) == 0:
            raise HTTPException(
                status_code=400,
                detail="None of the selected paper IDs were found in the ranked papers."
            )

    # Get session's collection name (for consistency)
    collection_name = SESSIONS[session_id]["collection_name"]

    # Initialize pipeline config for generation
    config = PipelineConfig(
        persist_directory="./corpus-data/chroma_db",
        collection_name=collection_name,
        recreate_index=False,
        hybrid_k=50,
        num_abstracts_to_score=None,
        top_k=3,
        relevance_model="gpt-4o-mini",
        generation_model="gpt-4o-mini",
        random_seed=42
    )

    try:
        # Import here to avoid circular dependency
        from pipeline import generate_related_work_text

        # Step 7: Generate related work text (streaming mode) using selected papers
        streaming_response = generate_related_work_text(
            query,
            papers_to_use,
            config.generation_model,
            stream=True
        )

        # Return the streaming response
        return streaming_response

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/api/collections")
async def list_collections(request: Request):
    """List all collections for current session"""
    session_id = request.state.session_id

    try:
        from chromadb import PersistentClient
        client = PersistentClient(path="./corpus-data/chroma_db")
        all_collections = client.list_collections()

        print("collections:",all_collections)

        # Filter to this session's collections
        user_collections = []
        for col in all_collections:
            if col.name.startswith(f"{session_id}_"):
                user_collections.append({
                    "name": col.name,
                    "count": col.count(),
                    "metadata": col.metadata
                })

        return {
            "session_id": session_id,
            "collections": user_collections,
            "total": len(user_collections)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str, request: Request):
    """Delete a specific collection (user must own it)"""
    global SESSIONS

    session_id = request.state.session_id

    # Verify ownership: collection must start with user's session_id
    if not collection_name.startswith(f"{session_id}_"):
        raise HTTPException(
            status_code=403,
            detail="Forbidden: You can only delete your own collections"
        )

    try:
        from chromadb import PersistentClient
        client = PersistentClient(path="./corpus-data/chroma_db")

        # Delete collection from ChromaDB
        client.delete_collection(name=collection_name)

        # Clean up session store if this was the active collection
        if session_id in SESSIONS and SESSIONS[session_id]["collection_name"] == collection_name:
            del SESSIONS[session_id]

        return {
            "success": True,
            "message": f"Collection '{collection_name}' deleted successfully"
        }

    except ValueError as e:
        # Collection doesn't exist
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint for AWS App Runner"""
    return {"status": "healthy"}

# Serve static files (our Next.js export) - MUST BE LAST!
static_path = Path("static")
if static_path.exists():
    # Serve index.html for the root path
    @app.get("/")
    async def serve_root():
        return FileResponse(static_path / "index.html")
    
    # Mount static files for all other routes
    app.mount("/", StaticFiles(directory="static", html=True), name="static")