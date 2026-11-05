import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DATA_PATH = "./docs"
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "pdf_documents"
SECRET_KEY = "Supper_Protected_Secret_Key"
ALGORITHM = "HS256"

# Multilingual embedding model (Arabic + English, cross-lingual retrieval)
#   ollama pull bge-m3
EMBED_MODEL = "bge-m3"

# LLM with solid Arabic support (llama3.2 does not officially support Arabic)
#   ollama pull qwen2.5:7b      (alternative: aya-expanse:8b)
LLM_MODEL = "qwen2.5:7b"
LLM_NUM_CTX = 8192  # Arabic uses many tokens; default 4096 would silently truncate the context

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 375

RETRIEVER_K = 4

# --- Chat memory (no extra LLM call) -----------------------------------------------
HISTORY_TURNS = 3          # past Q&A turns remembered per connection
FOLLOWUP_MAX_WORDS = 8     # a question this short is treated as a possible follow-up:
                           # the previous question is added to the search query
HISTORY_ANSWER_CHARS = 600 # old answers are cut to this length inside the prompt
# Cross-lingual similarity scores are lower than same-language ones, so keep this modest
SCORE_THRESHOLD = 0.5

prompt = (
    "You are an assistant for question-answering tasks.\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "Cite the exact document name and page number provided in the context tags for everything.\n"
    "And give the full answer to the question after stating where it is.\n"
    "ALWAYS write your answer in the same language as the question "
    "(Arabic question -> Arabic answer, English question -> English answer), DO NOT USE ANY OTHER LANGUAGE OTHER THAN THESE TWO, "
    "If the user asks multiple questions, answer the ones that are supported by the context.\n"
    "For any sub-question or topic NOT supported by the context, state that you dont know the answer.\n"
    "Do not reject the whole request if partial information is available.\n"
    "translating the relevant facts from the context when needed.\n"
    "Earlier messages in the conversation are only there to help you understand references such as \"it\" or \"the second one\"; "
    "always answer from the retrieved context below.\n"
    "IMPORTANT : If you do not know the answer, only say \"I dont know the answer to that question\" nothing extra.\n\n"
    "{context}"
    )