import json
import os
from datetime import datetime, timedelta
import chromadb
import ollama

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_MODEL = "llama3.2"
EVOLVING_MODEL = "my_evolving_ai"
OBSERVER_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

MEMORY_FILE = "memories.json"
SHORT_TERM_WINDOW = timedelta(days=2)  # Consolidate memories older than 2 days

# ==========================================
# 2. VECTOR DATABASE SETUP (Permanent Facts)
# ==========================================
chroma_client = chromadb.PersistentClient(path="./vector_memory")
collection = chroma_client.get_or_create_collection(name="technical_facts")

def get_embedding(text):
    """Converts text into a vector using the local embedding model."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"]

# ==========================================
# 3. TEMPORAL MEMORY SETUP (Context & Personality)
# ==========================================
def load_memory_store():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {
        "personality": "You are a highly capable and thoughtful AI assistant. Your tone reflects past interactions.",
        "long_term_core": "No long-term history established yet.",
        "short_term_memories": []
    }

def save_memory_store(store):
    with open(MEMORY_FILE, "w") as f:
        json.dump(store, f, indent=2)

memory_store = load_memory_store()

def consolidate_memories_by_age():
    """Merges aged short-term memories into the long-term core."""
    global memory_store
    now = datetime.now()
    
    matured_memories = []
    active_short_term = []
    
    for entry in memory_store["short_term_memories"]:
        entry_time = datetime.fromisoformat(entry["timestamp"])
        if now - entry_time >= SHORT_TERM_WINDOW:
            matured_memories.append(entry)
        else:
            active_short_term.append(entry)
            
    if matured_memories:
        print(f"\n⏳ Consolidating {len(matured_memories)} matured memory/memories into long-term storage...")
        matured_text = "\n".join([f"- [{m['timestamp'][:10]}] {m['content']}" for m in matured_memories])
        
        consolidation_prompt = f"""
        You are managing the memory consolidation of an AI.
        
        Existing Long-Term Summary:
        {memory_store['long_term_core']}
        
        Older memories to consolidate:
        {matured_text}
        
        Task: Synthesize both into a single, cohesive, dense summary paragraph. Preserve vital facts and user preferences, but generalize details that are no longer immediately relevant.
        """
        
        response = ollama.generate(model=OBSERVER_MODEL, prompt=consolidation_prompt)
        memory_store["long_term_core"] = response["response"].strip()
        memory_store["short_term_memories"] = active_short_term
        save_memory_store(memory_store)
        print("🧠 Temporal consolidation complete.")

def build_base_system_prompt():
    """Constructs the foundational system prompt from temporal memory."""
    prompt = f"{memory_store['personality']}\n\n=== LONG-TERM MEMORY ===\n{memory_store['long_term_core']}\n"
    if memory_store["short_term_memories"]:
        prompt += "\n=== RECENT SHORT-TERM MEMORIES ===\n"
        for m in memory_store["short_term_memories"]:
            prompt += f"- [{m['timestamp'][:16]}] {m['content']}\n"
    return prompt

# Initialize the evolving model on startup
consolidate_memories_by_age()
ollama.create(
    model=EVOLVING_MODEL,
    from_=BASE_MODEL,
    system=build_base_system_prompt()
)
print("\nSystem ready. Start chatting! (Type '/bye' to exit)\n" + "-"*50)

# ==========================================
# 4. MAIN ORCHESTRATION LOOP
# ==========================================
chat_history = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['/bye', 'exit', 'quit']:
        break
        
    # --- A. Retrieval Phase (RAG) ---
    query_vector = get_embedding(user_input)
    results = collection.query(
        query_embeddings=query_vector,
        n_results=2
    )
    
    retrieved_facts = results['documents'][0] if results['documents'] else []
    
    # Prepend retrieved facts as a temporary system instruction for this turn only
    turn_messages = []
    if retrieved_facts:
        fact_string = "\n".join(retrieved_facts)
        turn_messages.append({
            "role": "system", 
            "content": f"Use these permanent technical facts if relevant to the user's query:\n{fact_string}"
        })
        
    turn_messages.extend(chat_history)
    turn_messages.append({"role": "user", "content": user_input})
    
    # --- B. Actor Phase ---
    response = ollama.chat(model=EVOLVING_MODEL, messages=turn_messages)
    ai_text = response['message']['content']
    print(f"\nAI: {ai_text}\n")
    
    # Update active chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": ai_text})

    # --- C. Observer Phase ---
    observer_prompt = f"""
    Analyze this interaction:
    User: {user_input}
    AI: {ai_text}
    
    Provide a JSON object with EXACTLY these three keys:
    "temporal_memory": A factual, 1-sentence summary of what was just discussed to add to the short-term buffer.
    "personality_update": A revised system instruction shaping the AI's tone, improving it based on this interaction. Keep it concise.
    "permanent_fact": Isolate any concrete technical facts, code snippets, or configurations discussed that should be remembered forever. Leave blank if none exist.
    """
    
    obs_response = ollama.generate(
        model=OBSERVER_MODEL, 
        prompt=observer_prompt, 
        format="json"
    )
    
    # --- D. Orchestration & Updates ---
    try:
        feedback = json.loads(obs_response['response'])
        
        # 1. Save Permanent Fact to Vector DB
        new_fact = feedback.get("permanent_fact", "")
        if new_fact and len(new_fact) > 10:
            fact_id = f"fact_{len(collection.get()['ids'])}"
            collection.add(
                ids=[fact_id],
                embeddings=get_embedding(new_fact),
                documents=[new_fact]
            )
            print("💾 Permanent fact saved to vector database.")
        
        # 2. Update Temporal Memory & Personality
        new_memory = {
            "timestamp": datetime.now().isoformat(),
            "content": feedback.get("temporal_memory", "")
        }
        memory_store["short_term_memories"].append(new_memory)
        memory_store["personality"] = feedback.get("personality_update", memory_store["personality"])
        save_memory_store(memory_store)
        
        # 3. Consolidate if necessary and Rebuild Model
        consolidate_memories_by_age()
        ollama.create(
            model=EVOLVING_MODEL,
            from_=BASE_MODEL,
            system=build_base_system_prompt()
        )
        
    except json.JSONDecodeError:
        print("Observer parsing error; skipping memory updates for this turn.")