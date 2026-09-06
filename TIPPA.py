import ollama
import json
import os

# Configuration
BASE_MODEL = "llama3.2"           
EVOLVING_MODEL = "my_evolving_ai" 
OBSERVER_MODEL = "llama3.2"       

# 1. Initial Setup
current_system = "You are a neutral AI assistant. You have no personality yet."
memory_bank = []    # This will now ONLY store permanent, long-term memories
chat_history = []   # This stores short-term memory and resets when the script closes

# --- Load previous long-term state if it exists ---
if os.path.exists("ai_state.json"):
    with open("ai_state.json", "r") as f:
        saved_state = json.load(f)
        current_system = saved_state.get("system", current_system)
        memory_bank = saved_state.get("long_term_memories", [])
        print("Loaded previous long-term memories and personality from disk!")

# Inject loaded memories into the starting prompt
memory_string = "\n\nLong-Term Memories:\n" + "\n".join(memory_bank) if memory_bank else ""
final_system_prompt = current_system + memory_string

# Create the model 
ollama.create(model=EVOLVING_MODEL, from_=BASE_MODEL, system=final_system_prompt)
print("System initialized. Start chatting!")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['/bye', 'exit', 'quit']:
        print("Saving state and exiting. Goodbye!")
        break
        
    # Keep short-term memory manageable (store only the last 20 messages)
    if len(chat_history) > 20: 
        chat_history = chat_history[-20:]

    chat_history.append({"role": "user", "content": user_input})
    
    # 2. Primary AI Responds using short-term and long-term context
    response = ollama.chat(model=EVOLVING_MODEL, messages=chat_history)
    ai_text = response['message']['content']
    print(f"\nAI: {ai_text}\n")
    chat_history.append({"role": "assistant", "content": ai_text})
    
    # 3. Observer AI Analyzes specifically for Long-Term Value
    print("--- Observer is updating Modelfile... ---")
    observer_prompt = f"""
    Analyze this recent exchange:
    User: {user_input}
    AI: {ai_text}
    
    Provide a JSON response with exactly these three keys:
    "long_term_fact": Extract ONLY permanent, long-term facts about the user (e.g., name, job, preferences, goals). Do NOT include short-term temporal events (e.g., "User said hello", "User is testing the script"). If there is no new permanent fact to save, return an empty string "".
    "tone_analysis": A brief note on how the AI's personality should shift or evolve based on this chat.
    "new_system": A revised system prompt that adopts this evolving personality and tone. Do not include the memories in this prompt.
    """
    
    # Force the observer to output strict JSON
    obs_response = ollama.generate(
        model=OBSERVER_MODEL, 
        prompt=observer_prompt, 
        format="json" 
    )
    
    # 4. Parse Feedback, Save Permanently, and Rebuild
    try:
        feedback = json.loads(obs_response['response'])
        
        # Process the memory (Only append if the observer actually found a long-term fact)
        new_fact = feedback.get("long_term_fact", "").strip()
        if new_fact:
            memory_bank.append(new_fact)
            print(f"New long-term memory recorded: {new_fact}")
        else:
            print("No new long-term memory detected.")

        current_system = feedback.get("new_system", current_system)
        print(f"Tone shift: {feedback.get('tone_analysis')}")
        
        # --- Save the state permanently to your hard drive ---
        with open("ai_state.json", "w") as f:
            json.dump({
                "system": current_system,
                "long_term_memories": memory_bank
            }, f, indent=4)
        
        # Combine the new personality with the historical long-term memory bank
        memory_string = "\n\nLong-Term Memories:\n" + "\n".join(memory_bank)
        final_system_prompt = current_system + memory_string
        
        # Rebuild the model directly through the API
        ollama.create(
            model=EVOLVING_MODEL, 
            from_=BASE_MODEL, 
            system=final_system_prompt
        )
        
    except json.JSONDecodeError:
        print("Observer failed to output valid JSON. Skipping model update this turn.")