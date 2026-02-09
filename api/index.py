import os

# Environment configuration for deployment flexibility
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "local")

# Auto-detect ChromaDB persist directory based on environment
if DEPLOYMENT_ENV == "azure":
    DEFAULT_PERSIST_DIR = "/data/chromadb"
else:
    DEFAULT_PERSIST_DIR = "./corpus-data/chroma_db"

# Allow override via environment variable
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)

from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Directory setup
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global variables to track state
LAST_CSV_PATH = None  # Track last uploaded CSV file
TOP_RANKED_PAPERS = None  # Store ranked papers DataFrame after retrieve-and-rank
LAST_QUERY = None  # Track which query was used for ranking

@app.post("/api/upload-and-index")
async def upload_and_index(file: UploadFile = File(...)):
    """
    Upload a CSV file and build the vector index for literature review.

    Returns:
        JSON with indexing statistics
    """
    global LAST_CSV_PATH


    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Store the path globally for later use in generate endpoint
        LAST_CSV_PATH = str(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Initialize pipeline with config
    config = PipelineConfig(
        persist_directory=CHROMA_PERSIST_DIR,
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
            "csv_path": result.csv_path,
            "total_abstracts": result.total_abstracts,
            "chunks_created": result.chunks_created,
            "total_indexed": result.total_indexed,
            "persist_directory": result.persist_directory,
            "recreated": result.recreated
        }

        return JSONResponse(content=response_data)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/retrieve-and-rank")
def retrieve_and_rank(request: ResearchIdeaRequest):
    """
    Retrieve and rank papers for a given research idea (Steps 4-6 of pipeline).

    This endpoint must be called after /api/upload-and-index and before /api/generate.
    It performs retrieval, relevance scoring, and top-k selection.

    Returns:
        JSON with top-k papers, retrieval stats, and scoring stats
    """
    global LAST_CSV_PATH, TOP_RANKED_PAPERS, LAST_QUERY

    # Get the research idea from request
    query = request.research_idea.strip()

    # Check if vector index exists
    index_db_path = Path(CHROMA_PERSIST_DIR) / "chroma.sqlite3"
    if not index_db_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No index found. Please upload and index a CSV file first using the 'Upload & Index File' button."
        )

    # Check if we have CSV path
    if LAST_CSV_PATH is None or not Path(LAST_CSV_PATH).exists():
        raise HTTPException(
            status_code=400,
            detail="CSV file not found. Please upload the file again."
        )

    # Validate and clamp hybrid_k to valid range
    hybrid_k_value = min(max(request.hybrid_k or 50, 1), 200)

    # Initialize pipeline with same config as indexing
    config = PipelineConfig(
        persist_directory=CHROMA_PERSIST_DIR,
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

        # Load abstracts from the uploaded CSV
        pipeline.load_abstracts_only(LAST_CSV_PATH)

        # Retrieve and rank papers (Steps 4-6)
        top_k_abstracts, all_scored_papers, retrieval_stats, scoring_stats = pipeline.retrieve_and_rank_papers(query)

        # Store results globally for use in /api/generate
        # Store ALL scored papers so users can select from any of them
        TOP_RANKED_PAPERS = all_scored_papers
        LAST_QUERY = query

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
def lit_review(request: ResearchIdeaRequest):
    """
    Generate related work section using pre-ranked papers (streaming).

    Requires that /api/retrieve-and-rank has been called first to rank papers.
    This endpoint only performs Step 7 (text generation) using the pre-ranked papers.
    """
    global TOP_RANKED_PAPERS, LAST_QUERY

    # Get the research idea from request
    query = request.research_idea.strip()

    # Check if retrieve-and-rank has been called
    if TOP_RANKED_PAPERS is None:
        raise HTTPException(
            status_code=400,
            detail="No ranked papers found. Please call /api/retrieve-and-rank first to rank papers."
        )

    # Verify query matches the one used for ranking
    if LAST_QUERY != query:
        raise HTTPException(
            status_code=400,
            detail=f"Query mismatch. Ranked papers were retrieved for a different query. Please call /api/retrieve-and-rank again with the current query."
        )

    # Filter papers based on selected_paper_ids if provided
    papers_to_use = TOP_RANKED_PAPERS
    if request.selected_paper_ids is not None and len(request.selected_paper_ids) > 0:
        # Filter to only include selected papers
        papers_to_use = TOP_RANKED_PAPERS[TOP_RANKED_PAPERS['id'].isin(request.selected_paper_ids)]
        if len(papers_to_use) == 0:
            raise HTTPException(
                status_code=400,
                detail="None of the selected paper IDs were found in the ranked papers."
            )

    # Initialize pipeline config for generation
    config = PipelineConfig(
        persist_directory=CHROMA_PERSIST_DIR,
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