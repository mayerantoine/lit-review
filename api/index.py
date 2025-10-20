from fastapi import FastAPI
from fastapi.responses import PlainTextResponse,StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
import os
import time
from dotenv import load_dotenv



class ResearchIdeaRequest(BaseModel):
    research_idea: str

abstracts =[

    {'id':1,
     'title':"PaperQA: Retrieval-Augmented Generative Agent for Scientific Research",
     "abstract":"Large Language Models (LLMs) generalize well across language"
     " tasks, but suffer from hallucinations and uninterpretability, making it difficult to assess their accuracy without ground-truth. Retrieval-Augmented Generation (RAG) models have been proposed to reduce hallucinations and provide provenance for how an answer was generated. Applying such models to the scientific literature may enable large-scale, systematic processing of scientific knowledge. We present PaperQA, a RAG agent for answering questions over the scientific literature. PaperQA is an agent that performs information retrieval across full-text scientific articles, assesses the relevance of sources and passages, and uses RAG to provide answers. Viewing this agent as a question-answering model, we find it exceeds performance of existing LLMs and LLM agents on current science QA benchmarks. To push the field closer to how humans perform research on scientific literature, we also introduce LitQA, a more complex benchmark that requires retrieval and synthesis of information from full-text scientific papers across the literature. Finally, "
     "we demonstrate PaperQA’s matches expert human researchers on LitQA."
     },
    {'id':2,
     'title':"Improving Retrieval "
     "for RAG based Question Answering Models on Financial Documents",
     "abstract":"The effectiveness of Large Language Models (LLMs) "
     "in generating accurate responses relies heavily on the quality of input provided, particularly when employing Retrieval Augmented Generation (RAG) techniques. RAG enhances LLMs by sourcing the most relevant text chunk(s) to base queries upon. Despite the significant advancements in LLMs’ response quality in recent years, users may still encounter inaccuracies or irrelevant answers; these issues often stem from suboptimal text chunk retrieval by RAG rather than the inherent capabilities of LLMs. To augment the efficacy of LLMs, it is crucial to refine the RAG process. This paper explores the existing constraints of RAG pipelines and introduces methodologies for enhancing text retrieval. It delves into strategies such as sophisticated chunking techniques, query expansion, the incorporation of metadata annotations, the application of re-ranking algorithms, and the fine-tuning of embedding algorithms. Implementing these approaches can substantially improve the retrieval quality, thereby elevating the overall performance "
     "and reliability of LLMs in processing and responding to queries."
     },
    {'id':3,
     'title':"Retrieval augmented generation for large language models in healthcare: A systematic review",
     "abstract":"Large Language Models (LLMs) have demonstrated promising "
     "capabilities to solve complex tasks in critical sectors such as healthcare. However, LLMs are limited by their training data which is often outdated, the tendency to generate inaccurate (“hallucinated”) content and a lack of transparency in the content they generate. To address these limitations, retrieval augmented generation (RAG) grounds the responses of LLMs by exposing them to external knowledge sources. However, in the healthcare domain there is currently a lack of systematic understanding of which datasets, RAG methodologies and evaluation frameworks are available. This review aims to bridge this gap by assessing RAG-based approaches employed by LLMs in healthcare, focusing on the different steps of retrieval, augmentation and generation. Additionally, we identify the limitations, strengths and gaps in the existing literature. Our synthesis shows that 78.9% of studies used English datasets and 21.1% of the datasets are in Chinese. We find that a range of techniques are employed RAG-based LLMs in healthcare, including Naive RAG, Advanced RAG, and Modular RAG. Surprisingly, proprietary models such as GPT-3.5/4 are the most used for RAG applications in healthcare. We find that there is a lack of standardised evaluation frameworks for RAG-based applications. In addition, the majority of the studies do not assess or address ethical considerations related to RAG in healthcare. It is important to account for ethical challenges that are inherent when AI systems are implemented in the clinical setting. Lastly, we highlight the need for further research and development to ensure "
     "responsible and effective adoption of RAG in the medical domain."
     },
]

app = FastAPI()
# Configure CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api")
def lit_review(request: ResearchIdeaRequest, response_class=PlainTextResponse):
    # Load environment variables from api/.env
    load_dotenv(override=True)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get the research idea from request
    query = request.research_idea.strip()

    # Define instructions for related work generation
    INSTRUCTIONS_RELATED_WORK = """ 
    You are an expert research assistant who is helping with literature review for a research idea or abstract. 
    You will be provided with an abstract or research idea and a list of reference abstracts. 
    Your task is to write the related work section of the document using only the provided reference abstracts. 
    Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work 
    in the reference abstracts comparing the strengths and weaknesses while also motivating the proposed approach. 
    You should cite the reference abstracts as [id] whenever you are referring it in the related work. 
    Do not write it as Reference #. Do not cite abstract or research Idea. 
    Do not include any extra notes or newline characters at the end. 
    Do not copy the abstracts of reference papers directly but compare and contrast to the main work concisely. 
    Do not provide the output in bullet points or markdown. 
    Do not provide references at the end. 
    Please cite all the provided reference papers if needed.
    """

    input_related_work = f"Given the Research Idea or abstract: {query}"
    input_related_work += "\n\n## Given references abstracts list below:"

    for item in abstracts:
        input_related_work += f"\n\n[{item['id']}]: {item['abstract']}"

    input_related_work += "\n\nWrite the related work section summarizing in a cohesive story prior works relevant to the research idea."
    input_related_work += "\n\n## Related Work:"

    response = openai_client.responses.create(
        model="gpt-4o-mini",
        instructions=INSTRUCTIONS_RELATED_WORK,
        input=input_related_work)

            

    return  response.output_text