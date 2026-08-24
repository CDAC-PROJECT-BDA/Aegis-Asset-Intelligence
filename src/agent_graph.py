import asyncio
import os
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

async def check_guardrails(state: AgentState):
    last_msg = state["messages"][-1].content
    
    try:
        guardrail_llm = ChatGroq(model_name="meta-llama/llama-prompt-guard-2-86m", temperature=0)
        response = await guardrail_llm.ainvoke([HumanMessage(content=last_msg)])
        
        try:
            injection_score = float(response.content.strip())
        except ValueError:
            injection_score = 0.0
            
        if injection_score > 0.5:
            return {"messages": [AIMessage(content="[GUARDRAILS BLOCKED] Potential prompt injection detected. Request denied.")]}
    except Exception as e:
        print(f"Guardrail check failed: {e}")
        return {"messages": [AIMessage(content=f"[GUARDRAILS BLOCKED] Safety check failed: {e}")]}

    return {"messages": []}

async def call_mcp_and_agent(state: AgentState):
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
    print("Using Groq model: openai/gpt-oss-120b")
    
    mcp_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    
    server_params = StdioServerParameters(
        command="python",
        args=["-X", "utf8", mcp_server_path],
        env=None
    )
    
    from datetime import timedelta
    messages = state["messages"]
    
    if messages and isinstance(messages[-1], AIMessage) and "[GUARDRAILS BLOCKED]" in messages[-1].content:
        return {"messages": []}

    system_prompt = SystemMessage(content="You are Aegis, an expert industrial pump maintenance AI. "
                                          "Use your tools to diagnose issues and provide step-by-step fixes. "
                                          "STRICT GROUNDING RULE: You must NEVER guess or answer maintenance questions from your own memory. "
                                          "You MUST ALWAYS use the get_manual_section tool to search the official manual before providing repair advice. "
                                          "If the manual does not contain the answer, say 'I don't know based on the provided documentation.' "
                                          "IMPORTANT: Be extremely concise to avoid exceeding token rate limits. Keep answers under 150 words when possible.")
    
    if state.get("context"):
        system_prompt.content += f"\n\nCURRENT FAILURE CONTEXT:\n{state['context']}"
        
    from datetime import timedelta
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as session:
            await session.initialize()
            
            mcp_tools_raw = await session.list_tools()
            
            lc_tools = []
            for t in mcp_tools_raw.tools:
                props = {}
                req = []
                if t.inputSchema and "properties" in t.inputSchema:
                    props = t.inputSchema["properties"]
                    req = t.inputSchema.get("required", [])
                    
                lc_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": {
                            "type": "object",
                            "properties": props,
                            "required": req
                        }
                    }
                })
            
            llm_with_tools = llm.bind_tools(lc_tools)
            
            current_messages = messages.copy()
            max_iterations = 5
            
            for _ in range(max_iterations):
                invoke_msgs = [system_prompt] + current_messages
                try:
                    response = await llm_with_tools.ainvoke(invoke_msgs)
                except Exception as e:
                    return {"messages": current_messages[len(messages):] + [AIMessage(content=f"⚠️ LLM Error: {str(e)}")]}
                
                if not getattr(response, "content", None) and not getattr(response, "tool_calls", None):
                    return {"messages": [AIMessage(content="⚠️ Unable to generate a response. Please verify your GROQ_API_KEY or network connectivity.")]}
                
                current_messages.append(response)
                
                if not response.tool_calls:
                    return {"messages": current_messages[len(messages):]} # Return only new messages
                    
                for tc in response.tool_calls:
                    print(f"DEBUG: Calling tool {tc['name']} with args {tc['args']}")
                    try:
                        result = await session.call_tool(tc["name"], arguments=tc["args"])
                        print(f"DEBUG: Tool {tc['name']} returned successfully")
                        tool_content = result.content[0].text if getattr(result, "content", None) else "Executed."
                    except Exception as e:
                        print(f"DEBUG: Tool {tc['name']} failed with exception: {e}")
                        tool_content = f"Error executing tool: {e}"
                    
                    current_messages.append(ToolMessage(content=tool_content, tool_call_id=tc["id"], name=tc["name"]))
            
            current_messages.append(AIMessage(content="I've reached the maximum number of reasoning steps. Based on what I've gathered, please check the manuals directly or ask a more specific question."))
            return {"messages": current_messages[len(messages):]}

workflow = StateGraph(AgentState)
workflow.add_node("guardrails", check_guardrails)
workflow.add_node("agent", call_mcp_and_agent)

workflow.add_edge(START, "guardrails")
workflow.add_edge("guardrails", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()

def run_agent(messages: list, context: str = ""):
    return asyncio.run(app.ainvoke({"messages": messages, "context": context}))
