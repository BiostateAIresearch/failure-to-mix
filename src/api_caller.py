"""
Async API caller for LLM experiments with rate limiting, retries, and error handling.
"""
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class APIResponse:
    """Container for API response data."""
    content: str
    reasoning: str
    raw_response: Dict[str, Any]
    status: str
    error: Optional[str] = None


class APICaller:
    """Async API caller with rate limiting and retry logic."""
    
    def __init__(
        self,
        api_keys: List[str],
        api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_retries: int = 3,
        timeout: int = 120,
        initial_wait: float = 2.0
    ):
        self.api_keys = api_keys
        self.api_url = api_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.initial_wait = initial_wait
        self._key_index = 0
    
    def _get_next_key(self) -> str:
        """Round-robin through API keys."""
        key = self.api_keys[self._key_index % len(self.api_keys)]
        self._key_index += 1
        return key
    
    @staticmethod
    def parse_response(response: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Parse API response to extract content and reasoning.
        
        Returns:
            Tuple of (content, reasoning, merged_text)
        """
        if not isinstance(response, dict):
            return "", "", ""
        
        choices = response.get("choices", [{}])
        if not choices:
            return "", "", ""
        
        message = choices[0].get("message", {})
        content = message.get("content", "") or ""
        reasoning = message.get("reasoning", "") or ""
        
        # Handle reasoning_details format
        if not reasoning:
            rd = message.get("reasoning_details")
            if isinstance(rd, list) and rd:
                reasoning = rd[0].get("summary") or rd[0].get("data") or ""
        
        merged = (content or reasoning or "").strip()
        return content.strip(), reasoning.strip(), merged
    
    async def call(
        self,
        session: aiohttp.ClientSession,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        attempt: int = 1
    ) -> APIResponse:
        """
        Make a single API call with retry logic.
        
        Args:
            session: aiohttp session
            model: Model identifier
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            api_key: Specific API key to use (or auto-select)
            attempt: Current attempt number
            
        Returns:
            APIResponse with content and status
        """
        if api_key is None:
            api_key = self._get_next_key()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        
        try:
            async with session.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                data = await response.json()
                
                # Handle rate limiting
                if response.status == 429:
                    if attempt < self.max_retries:
                        wait_time = 3 * attempt
                        print(f"⏳ Rate limited, waiting {wait_time}s (attempt {attempt}/{self.max_retries})")
                        await asyncio.sleep(wait_time)
                        return await self.call(
                            session, model, messages, max_tokens, temperature, api_key, attempt + 1
                        )
                    return APIResponse(
                        content="",
                        reasoning="",
                        raw_response=data,
                        status="rate_limited",
                        error="Rate limit exceeded after retries"
                    )
                
                # Handle API errors
                if isinstance(data, dict) and "error" in data:
                    error_msg = str(data["error"])
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 * attempt)
                        return await self.call(
                            session, model, messages, max_tokens, temperature, api_key, attempt + 1
                        )
                    return APIResponse(
                        content="",
                        reasoning="",
                        raw_response=data,
                        status="error",
                        error=error_msg
                    )
                
                # Parse successful response
                content, reasoning, merged = self.parse_response(data)
                return APIResponse(
                    content=content,
                    reasoning=reasoning,
                    raw_response=data,
                    status="success"
                )
                
        except asyncio.TimeoutError:
            if attempt < self.max_retries:
                await asyncio.sleep(2 * attempt)
                return await self.call(
                    session, model, messages, max_tokens, temperature, api_key, attempt + 1
                )
            return APIResponse(
                content="",
                reasoning="",
                raw_response={},
                status="timeout",
                error="Request timed out"
            )
            
        except Exception as e:
            if attempt < self.max_retries:
                await asyncio.sleep(2 * attempt)
                return await self.call(
                    session, model, messages, max_tokens, temperature, api_key, attempt + 1
                )
            return APIResponse(
                content="",
                reasoning="",
                raw_response={},
                status="error",
                error=str(e)
            )
    
    async def call_batch(
        self,
        session: aiohttp.ClientSession,
        requests: List[Dict[str, Any]],
        batch_size: int = 20
    ) -> List[APIResponse]:
        """
        Call API for a batch of requests with rate limiting.
        
        Args:
            session: aiohttp session
            requests: List of request dicts with 'model', 'messages', etc.
            batch_size: Number of concurrent requests
            
        Returns:
            List of APIResponse objects
        """
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            tasks = []
            
            for j, req in enumerate(batch):
                api_key = self.api_keys[(i + j) % len(self.api_keys)]
                tasks.append(
                    self.call(
                        session=session,
                        model=req["model"],
                        messages=req["messages"],
                        max_tokens=req.get("max_tokens"),
                        temperature=req.get("temperature"),
                        api_key=api_key
                    )
                )
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Rate limiting wait between batches
            if i + batch_size < len(requests):
                await asyncio.sleep(self.initial_wait)
        
        return results
    
    async def call_with_conversation(
        self,
        session: aiohttp.ClientSession,
        model: str,
        turns: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> List[APIResponse]:
        """
        Make multi-turn conversation calls.
        
        Args:
            session: aiohttp session
            model: Model identifier
            turns: List of user messages for each turn
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            system_prompt: Optional system message
            
        Returns:
            List of APIResponse objects, one per turn
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        responses = []
        
        for user_msg in turns:
            messages.append({"role": "user", "content": user_msg})
            
            response = await self.call(
                session=session,
                model=model,
                messages=messages.copy(),
                max_tokens=max_tokens,
                temperature=temperature
            )
            responses.append(response)
            
            # Add assistant response to conversation
            if response.status == "success":
                messages.append({"role": "assistant", "content": response.content})
        
        return responses
