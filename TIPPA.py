import ollama
import json

# Configuration
BASE_MODEL = "llama3.2"           
EVOLVING_MODEL = "my_evolving_ai" 
OBSERVER_MODEL = "llama3.2"       

# 1. Initial Setup
current_system = "You are a neutral AI assistant. You have no personality yet."
memory_bank = []
chat_history = []

# Create the model using the modern syntax
ollama.create(model=EVOLVING_MODEL, from_=BASE_MODEL, system=current_system)
print("System initialized. Start chatting!")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['/bye', 'exit', 'quit']:
        break
        
    chat_history.append({"role": "user", "content": user_input})
    
    # 2. Primary AI Responds
    response = ollama.chat(model=EVOLVING_MODEL, messages=chat_history)
    ai_text = response['message']['content']
    print(f"\nAI: {ai_text}\n")
    chat_history.append({"role": "assistant", "content": ai_text})
    
    # 3. Observer AI Analyzes
    print("--- Observer is updating Modelfile... ---")
    observer_prompt = f"""
    Analyze this recent exchange:
    User: {user_input}
    AI: {ai_text}
    
    Provide a JSON response with exactly these three keys:
    "memory": A brief 1-sentence summary of what was discussed to act as a memory.
    "tone_analysis": A brief note on how the AI's personality should shift or evolve based on this chat.
    "new_system": A revised system prompt that adopts this evolving personality and tone. Do not include the memories in this prompt.
    """
    
    # Force the observer to output strict JSON
    obs_response = ollama.generate(
        model=OBSERVER_MODEL, 
        prompt=observer_prompt, 
        format="json" 
    )
    
    # 4. Parse Feedback and Rebuild Model
    try:
        feedback = json.loads(obs_response['response'])
        
        # Save the new memory and update the core personality
        memory_bank.append(feedback.get("memory", ""))
        current_system = feedback.get("new_system", current_system)
        
        print(f"Memory recorded: {feedback.get('memory')}")
        print(f"Tone shift: {feedback.get('tone_analysis')}")
        
        # Combine the new personality with the historical memory bank
        memory_string = "\n\nPast Memories:\n" + "\n".join(memory_bank)
        final_system_prompt = current_system + memory_string
        
        # Rebuild the model directly through the API using the modern syntax
        ollama.create(
            model=EVOLVING_MODEL, 
            from_=BASE_MODEL, 
            system=final_system_prompt
        )
        
    except json.JSONDecodeError:
        print("Observer failed to output valid JSON. Skipping model update this turn.")