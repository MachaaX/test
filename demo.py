import time
import gradio as gr
from remembr.memory.milvus_memory import MilvusMemory
from remembr.memory.memory import MemoryItem
from remembr.agents.remembr_agent import ReMEmbRAgent

# ── memory -----------------------------------------------------------
memory = MilvusMemory("test_collection", db_ip="127.0.0.1")
# memory.reset()

# ── agent ------------------------------------------------------------
agent = ReMEmbRAgent(llm_type="llama3")
agent.set_memory(memory)
agent.system_prompt = "Answer using stored facts only."

def chat(user, hist):
    # 1. insert the raw user text (positional!) into memory
    memory.insert(MemoryItem(
        user,                   # text
        time=time.time(),       # timestamp
        position=[0.0,0.0,0.0], # placeholders
        theta=0.0               # placeholder
    ))
    # 2. then get your answer
    answer = agent.query(user).text
    return hist + [(user, answer)], ""

with gr.Blocks() as demo:
    chatbox = gr.Chatbot()
    msg = gr.Textbox()
    msg.submit(chat, [msg, chatbox], [chatbox, msg])

# 3. share=True gives you a public URL
demo.launch(share=True)
