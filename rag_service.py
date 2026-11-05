# rag_service.py
import os
import asyncio
import logfire
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import config
from vector_store import initialize_vector_store
from lang_utils import normalize_text, is_arabic, has_cjk, strip_cjk

# Vector Store Setup
client, vectorstore, chunks_by_id = initialize_vector_store(force_recreate=False)

retriever = None
retriever_final = None

def refresh_retrievers():
    global retriever, retriever_final
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
    retriever_final = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": config.SCORE_THRESHOLD, "k": config.RETRIEVER_K},
    )

def rebuild_vector_store_service():
    global client, vectorstore, chunks_by_id
    try:
        client.close()
    except Exception as e:
        print(f"Notice: Error while closing old client: {e}")

    client, vectorstore, chunks_by_id = initialize_vector_store(force_recreate=True)
    refresh_retrievers()

def close_vector_client():
    client.close()

# Initialize Retrievers
refresh_retrievers()

# LLM & System Prompt Setup
llm = ChatOllama(
    model=config.LLM_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0,
    num_ctx=config.LLM_NUM_CTX,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", config.prompt),
    ("human", "{question}\n\n(Answer only in {answer_lang}. Do not use Chinese.)"),
])

def build_messages(context: str, question: str, answer_lang: str, history: list | None = None):
    """System prompt (with context) + past turns + the new question."""
    msgs = prompt.format_messages(context=context, question=question, answer_lang=answer_lang)
    past = []
    for q, a in (history or []):
        past.append(HumanMessage(content=q))
        past.append(AIMessage(content=a[: config.HISTORY_ANSWER_CHARS]))
    return msgs[:-1] + past + msgs[-1:]   # [system] + history + [current question]

def build_search_query(query: str, history: list | None) -> str:
    """Short follow-ups ("and the second one?") are searched together with the previous question."""
    if history and len(query.split()) <= config.FOLLOWUP_MAX_WORDS:
        return f"{history[-1][0]} {query}"
    return query

async def answer_with_guard(context: str, question: str, answer_lang: str, history: list | None = None):
    """Call the LLM; if CJK characters leak in, retry once with a stricter nudge."""
    msgs = build_messages(context, question, answer_lang, history)
    resp = await llm.ainvoke(msgs)
    if has_cjk(resp.content):
        msgs.append(("ai", resp.content))
        msgs.append(("human", f"Your answer contained Chinese. Rewrite the same answer ONLY in {answer_lang}."))
        retry = await llm.ainvoke(msgs)
        # sum tokens so the usage counter stays honest
        for k in ("prompt_eval_count", "eval_count"):
            retry.response_metadata[k] = retry.response_metadata.get(k, 0) + resp.response_metadata.get(k, 0)
        resp = retry
    return resp

def format_sources(docs) -> str:
    """Unique list of file name + pages that were used as context."""
    pages_by_file = {}
    for d in docs:
        name = d.metadata.get("file_name") or os.path.basename(d.metadata.get("source", "Unknown"))
        page = d.metadata.get("page_number") or (d.metadata.get("page", 0) + 1)
        pages_by_file.setdefault(name, set()).add(page)
    lines = [f"- {n} (pages: {', '.join(str(p) for p in sorted(pg))})" for n, pg in pages_by_file.items()]
    return "\n".join(lines)

def format_docs(docs) -> str:
    formatted = []
    for doc in docs:
        source_path = doc.metadata.get("source")
        file_name = doc.metadata.get("file_name") or (os.path.basename(source_path) if source_path else "Unknown")
        page_num = doc.metadata.get("page_number") or (doc.metadata.get("page", 0) + 1)
        cid = doc.metadata.get("chunk_id")
        formatted.append(
            f"--- SOURCE: {file_name} (Page {page_num}) [Chunk ID {cid}] ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

def _usage(msg) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) from a message or streamed chunk, if present."""
    meta = getattr(msg, "response_metadata", None) or {}
    p, c = meta.get("prompt_eval_count"), meta.get("eval_count")
    if p is None and c is None:
        um = getattr(msg, "usage_metadata", None) or {}
        return um.get("input_tokens", 0) or 0, um.get("output_tokens", 0) or 0
    return p or 0, c or 0

async def _safe_send(websocket, text: str):
    try:
        await websocket.send_text(text)
    except Exception:
        pass

# ------------------------------------------------------------------ real streaming
async def stream_answer(messages, websocket, answer_lang: str):
    """
    Streams tokens to the client as the model generates them.
    If Chinese/Japanese/Korean characters leak in, the partial answer is discarded
    (client gets ANSWER_RESET) and generated again with a stricter reminder; on the
    second attempt any stray CJK characters are simply stripped.
    Returns (answer_text, prompt_tokens, completion_tokens).
    """
    prompt_tokens = completion_tokens = 0
    parts = []
    await websocket.send_text("ANSWER_START:")
    try:
        for attempt in (1, 2):
            parts = []
            leaked = False
            p = c = 0
            stream = llm.astream(messages)
            try:
                async for chunk in stream:
                    piece = chunk.content or ""
                    cp, cc = _usage(chunk)
                    if cp or cc:
                        p, c = cp, cc
                    if piece and has_cjk(piece):
                        if attempt == 1:
                            leaked = True
                            break
                        piece = strip_cjk(piece)
                    if piece:
                        parts.append(piece)
                        await websocket.send_text(piece)
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            prompt_tokens += p
            completion_tokens += c
            if not leaked:
                break
            await websocket.send_text("ANSWER_RESET:")
            messages = list(messages) + [(
                "human",
                f"Reminder: your previous answer contained Chinese. Answer again ONLY in {answer_lang}.",
            )]
    finally:
        await _safe_send(websocket, "ANSWER_END:")
    return "".join(parts), prompt_tokens, completion_tokens

async def translate_query(query: str):
    target = "English" if is_arabic(query) else "Arabic"
    messages = [
        ("system", f"Translate the user's text into {target}. Output only the translation, nothing else."),
        ("human", query),
    ]
    try:
        resp = await llm.ainvoke(messages)
        meta = resp.response_metadata
        tokens = meta.get("prompt_eval_count", 0) + meta.get("eval_count", 0)
        text = resp.content.strip()
        return (text or None), tokens
    except Exception as e:
        print(f"Translation failed: {e}")
        return None, 0

def retrieve_multilingual(query: str, alt_query: str | None):
    seen, merged = set(), []
    for q in (query, alt_query):
        if not q:
            continue
        for d in retriever.invoke(q):
            key = d.metadata.get("chunk_id")
            if key not in seen:
                seen.add(key)
                merged.append(d)
    return merged

async def run_two_pass_rag(user_message: str, websocket, history: list | None = None) -> str:
    """Returns the final answer text (the caller stores it in the chat history)."""
    history = history or []
    query = normalize_text(user_message).strip()
    answer_lang = "Arabic" if is_arabic(query) else "English"
    await websocket.send_text(f"\n--- Running Two-Pass RAG Query: '{query}' ---\n")

    with logfire.span("rag_agent", openinference_span_kind="AGENT") as span:
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Search text: the question, plus the previous question when this looks like a short follow-up
        search_query = build_search_query(query, history)
        if search_query != query:
            await websocket.send_text(f"\n[Search query with context] {search_query}\n")

        alt_query, translate_tokens = await translate_query(search_query)
        total_prompt_tokens += translate_tokens
        if alt_query:
            await websocket.send_text(f"\n[Translated query] {alt_query}\n")

        # PASS 1
        candidates = await asyncio.to_thread(retrieve_multilingual, search_query, alt_query)
        pass1_response = await answer_with_guard(format_docs(candidates), query, answer_lang, history)

        p1_meta = pass1_response.response_metadata
        p1_prompt = p1_meta.get("prompt_eval_count", 0)
        p1_completion = p1_meta.get("eval_count", 0)
        total_prompt_tokens += p1_prompt
        total_completion_tokens += p1_completion

        await websocket.send_text(f"\n[Pass 1 Token Counter] Prompt: {p1_prompt} | Completion: {p1_completion} | Subtotal: {p1_prompt + p1_completion}\n")

        # ANCHOR SELECTION & WINDOW EXPANSION
        correct_sections = await asyncio.to_thread(retriever_final.invoke, pass1_response.content)
        chunk_sections = [
            section.metadata.get("chunk_id", 0) 
            for section in correct_sections 
            if "chunk_id" in section.metadata
        ]
        if not chunk_sections and candidates:
            chunk_sections = [candidates[0].metadata.get("chunk_id", 0)]

        unique_ids = set()
        for cs in chunk_sections:
            anchor = chunks_by_id.get(cs)
            anchor_src = anchor.metadata.get("source") if anchor else None
            for nid in (cs - 1, cs, cs + 1, cs + 2):
                neighbour = chunks_by_id.get(nid)
                if neighbour and neighbour.metadata.get("source") == anchor_src:
                    unique_ids.add(nid)
        neighbor_ids = sorted(unique_ids)
        expanded_chunks = [chunks_by_id[nid] for nid in neighbor_ids]

        expanded_context_str = format_docs(expanded_chunks)

        # PASS 2 (streamed to the client token by token)
        messages = build_messages(expanded_context_str, query, answer_lang, history)
        await websocket.send_text("\nAnswer:\n")
        final_answer, p2_prompt, p2_completion = await stream_answer(messages, websocket, answer_lang)
        total_prompt_tokens += p2_prompt
        total_completion_tokens += p2_completion

        await websocket.send_text(f"\n[Pass 2 Token Counter] Prompt: {p2_prompt} | Completion: {p2_completion} | Subtotal: {p2_prompt + p2_completion}\n")
        await websocket.send_text(f"\n[Cumulative Total Used] Prompt: {total_prompt_tokens} | Completion: {total_completion_tokens} | Combined Total: {total_prompt_tokens + total_completion_tokens}\n")
        if expanded_chunks:
            await websocket.send_text(f"\n\nSources used:\n{format_sources(expanded_chunks)}\n")

        query_total_tokens = total_prompt_tokens + total_completion_tokens
        await websocket.send_text(f'METADATA:{{"tokens": {query_total_tokens}}}')

        span.set_attribute("llm.token_count.prompt", total_prompt_tokens)
        span.set_attribute("llm.token_count.completion", total_completion_tokens)
        span.set_attribute("llm.token_count.total", query_total_tokens)
        span.set_attribute("output.value", final_answer)
        return final_answer
