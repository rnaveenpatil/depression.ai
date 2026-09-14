"""Native Anthropic Messages API adapter."""
from __future__ import annotations
import asyncio, json
from typing import Any, Dict, List, Optional
import httpx
from agent.llm.provider import LLMProvider, LLMResponse, Message, ToolCall
from agent.utils.errors import LLMError

class AnthropicProvider(LLMProvider):
    name = "anthropic"
    async def complete(self, messages: List[Any], model: Optional[str]=None, temperature: float=0.1, max_tokens: int=4096, tools: Optional[List[Dict[str, Any]]]=None, **kwargs) -> LLMResponse:
        if not self.api_key: raise LLMError("Anthropic API key not configured")
        if not model: raise LLMError("Anthropic model not configured")
        system=[]; converted=[]
        for raw in messages:
            m=raw.to_dict() if isinstance(raw,Message) else raw; role=m.get("role","user")
            if role=="system": system.append(m.get("content", "")); continue
            if role=="tool": converted.append({"role":"user","content":[{"type":"tool_result","tool_use_id":m.get("tool_call_id",""),"content":m.get("content","")} ]}); continue
            converted.append({"role":role,"content":m.get("content","")})
            if role=="assistant" and m.get("tool_calls"):
                blocks=[{"type":"text","text":m.get("content","")}] if m.get("content") else []
                for tc in m["tool_calls"]:
                    fn=tc.get("function",{}); args=fn.get("arguments",{})
                    if isinstance(args,str):
                        try: args=json.loads(args)
                        except Exception: args={}
                    blocks.append({"type":"tool_use","id":tc.get("id",""),"name":fn.get("name",""),"input":args})
                converted[-1]["content"]=blocks
        payload={"model":model,"messages":converted,"max_tokens":max_tokens,"temperature":temperature}
        if system: payload["system"]="\n\n".join(system)
        if tools: payload["tools"]=[{"name":t.get("function",{}).get("name",""),"description":t.get("function",{}).get("description", ""),"input_schema":t.get("function",{}).get("parameters",{})} for t in tools]
        headers={"x-api-key":self.api_key,"anthropic-version":"2023-06-01","content-type":"application/json"}
        async with httpx.AsyncClient(base_url=self.base_url.rstrip("/"),headers=headers,timeout=self.timeout) as client:
            last=None
            for attempt in range(self.max_retries):
                try:
                    r=await client.post("/v1/messages",json=payload); r.raise_for_status(); return self._parse(r.json(),model)
                except httpx.HTTPStatusError as e:
                    last=e
                    if e.response.status_code in (429,500,502,503,504): await asyncio.sleep(min(2**attempt,8)); continue
                    raise LLMError(f"Anthropic API error {e.response.status_code}: {e.response.text[:500]}")
        raise LLMError(f"Anthropic request failed: {last}")
    def _parse(self,data,model):
        text=[]; calls=[]
        for block in data.get("content",[]):
            if block.get("type")=="text": text.append(block.get("text", ""))
            elif block.get("type")=="tool_use": calls.append(ToolCall(id=block.get("id",""),name=block.get("name",""),arguments=block.get("input") or {}))
        usage=data.get("usage") or {}; inp=usage.get("input_tokens",0); out=usage.get("output_tokens",0)
        return LLMResponse(content="".join(text),model=model,provider="anthropic",usage={"prompt_tokens":inp,"completion_tokens":out,"total_tokens":inp+out},tool_calls=calls,finish_reason=data.get("stop_reason","stop"),raw=data)
    def list_models(self): return []
