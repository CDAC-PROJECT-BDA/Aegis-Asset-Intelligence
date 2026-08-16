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

def check_guardrails(state: AgentState):
    last_msg = state["messages"][-1].content.lower()
    malicious = ["ignore previous", "forget all", "system prompt", "bypass", "jailbreak"]
    for word in malicious:
        if word in last_msg:
            return {"messages": [AIMessage(content="[GUARDRAILS BLOCKED] Prompt injection attempt detected. Request denied.")]}
    
    if "recipe" in last_msg or "joke" in last_msg or "poem" in last_msg:
         return {"messages": [AIMessage(content="[GUARDRAILS BLOCKED] Query is out of domain. Please ask maintenance-related questions.")]}
         
    return {"messages": []}

async def call_mcp_and_agent(state: AgentState):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    

    server_params = StdioServerParameters(
        command="python",
        args=["src/mcp_server.py"],
        env=None
    )
    
    messages = state["messages"]
    
    if messages and isinstance(messages[-1], AIMessage) and "[GUARDRAILS BLOCKED]" in messages[-1].content:
        return {"messages": []}

    system_prompt = SystemMessage(content="You are Aegis, an expert industrial pump maintenance AI. "
                                          "Use your tools to diagnose issues and provide step-by-step fixes.")
    
    if state.get("context"):
        system_prompt.content += f"\n\nCURRENT FAILURE CONTEXT:\n{state['context']}"
        
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
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
            
            invoke_msgs = [system_prompt] + messages
            response = await llm_with_tools.ainvoke(invoke_msgs)
            
            if response.tool_calls:
                new_msgs = [response]
                for tc in response.tool_calls:
                    result = await session.call_tool(tc["name"], arguments=tc["args"])
                    tool_content = result.content[0].text if result.content else "Executed."
                    new_msgs.append(ToolMessage(content=tool_content, tool_call_id=tc["id"], name=tc["name"]))
                
                final_response = await llm_with_tools.ainvoke([system_prompt] + messages + new_msgs)
                return {"messages": new_msgs + [final_response]}
            
            return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("guardrails", check_guardrails)
workflow.add_node("agent", call_mcp_and_agent)

workflow.add_edge(START, "guardrails")
workflow.add_edge("guardrails", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()

def run_agent(messages: list, context: str = ""):
    return asyncio.run(app.ainvoke({"messages": messages, "context": context}))
