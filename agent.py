import os
import re
from dotenv import load_dotenv
from tools import AGENT_TOOLS

# Load environment variables
load_dotenv()

template = """You are an intelligent AI agent designed to answer user queries by dynamically selecting the best source of information.

You have access to the following tools:

{tools}

---

### 🎯 Responsibilities

1. Understand the user query deeply.
2. Decide the BEST tool (or multiple tools) to use.
3. Do NOT guess when data is required from tools.
4. Minimize unnecessary tool usage.
5. Combine results intelligently if multiple tools are used.
6. Always produce a clear, final answer.

---

### 🧭 Decision Rules

* If the query involves videos / tutorials / songs / YouTube content -> use youtube_search
* If the query involves current events, facts, or internet information -> use web_search
* If the query involves internal documents / knowledge base -> use rag
* Otherwise -> answer directly using your own knowledge

---

### ⚠️ Constraints

* NEVER hallucinate facts; rely on web_search or rag if unsure
* NEVER assume access to data you did not retrieve
* Keep responses concise but informative
* Prefer tool usage over guessing

---

### 🧠 Reasoning Format (STRICT)

Use the following format:

Question: the input question you must answer
Thought: Analyze the query and decide what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Respond clearly to the user

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def get_llm():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN missing from .env")

    repo_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
        task="conversational",
    )
    return ChatHuggingFace(llm=llm_endpoint)


def run_react_agent(llm, user_input):
    tools_str = "\n".join([f"{t.name}: {t.description}" for t in AGENT_TOOLS])
    tool_names = ", ".join([t.name for t in AGENT_TOOLS])
    
    prompt = template.replace("{tools}", tools_str).replace("{tool_names}", tool_names)
    prompt = prompt.replace("{input}", user_input)
    
    scratchpad = " "
    logs = []
    videos = []
    
    for _ in range(5):
        current_prompt = prompt.replace("{agent_scratchpad}", scratchpad)
        
        response_msg = llm.invoke(current_prompt, stop=["\nObservation:", "Observation:"])
        response = response_msg.content
        
        scratchpad += response
        
        if "Final Answer:" in response:
            parts = response.split("Final Answer:")
            thought = parts[0].replace("Thought:", "").strip()
            if thought:
                logs.append(f"🧠 Thought: {thought}")
            
            final_answer = parts[-1].strip()
            return {"logs": logs, "final_answer": final_answer, "videos": videos}
            
        action_match = re.search(r"Action:\s*(.*)", response)
        action_input_match = re.search(r"Action Input:\s*(.*)", response)
        
        if action_match and action_input_match:
            thought = response.split("Action:")[0].replace("Thought:", "").strip()
            if thought:
                logs.append(f"🧠 Thought: {thought}")
            
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            logs.append(f"⚙️ Action: {action}( \"{action_input}\" )")
            
            tool_found = False
            for tool in AGENT_TOOLS:
                if tool.name == action.strip():
                    try:
                        observation = str(tool.invoke({"query": action_input}))
                        
                        # Extract Youtube video IDs if this tool returned youtube links
                        if "youtube.com/watch?v=" in observation:
                            found_vids = re.findall(r"v=([a-zA-Z0-9_-]+)", observation)
                            videos.extend(found_vids)
                            
                    except Exception as e:
                        observation = f"Error: {e}"
                    tool_found = True
                    break
            
            if not tool_found:
                observation = f"Tool '{action}' not found."
                
            logs.append(f"👁️ Observation: {observation[:200]}...") # truncate very long observations in UI
            scratchpad += f"\nObservation: {observation}\nThought: "
        else:
            if scratchpad.strip():
                return {"logs": logs, "final_answer": response.strip(), "videos": videos}
            return {"logs": logs, "final_answer": "Could not parse answer.", "videos": videos}

    return {"logs": logs, "final_answer": "Max iterations reached without a Final Answer.", "videos": videos}

def main():
    print("Initializing LLM...")
    try:
        llm = get_llm()
    except Exception as e:
        print(f"Failed to initialize HuggingFace Endpoint: {e}")
        return

    print("\n" + "="*50)
    print("Agent Initialized! Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue
            
            print("\nAgent thinking...\n")
            result = run_react_agent(llm, user_input)
            
            for log in result["logs"]:
                print(log)
            print(f"\nFinal Answer: {result['final_answer']}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError processing query: {str(e)}\n")

if __name__ == "__main__":
    main()
