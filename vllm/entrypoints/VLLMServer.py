import argparse
import json
import os
from typing import AsyncGenerator, Union, List, Optional
from http import HTTPStatus

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.transformers_utils.tokenizer import get_tokenizer


TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
API_KEY_NAME = "Authorization"
SECRET_KEY = os.getenv("SECRET_KEY")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

app = FastAPI()


class TokenUsageRequest(BaseModel):
    prompt: Union[str, List[str]]


class TokenUsageResponse(BaseModel):
    token_count: int


async def auth_check(api_key: str = Depends(api_key_header)) -> None:
    if api_key != SECRET_KEY:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail="Invalid API key")
    

@app.post("/token_usage")
async def get_token_usage(request: TokenUsageRequest,
                          _auth: None = Depends(auth_check)
                          ) -> TokenUsageResponse:
    """Tokenize a string and return the token count
    """
    if isinstance(request.prompt, str):
        prompts = [request.prompt]
    else:
        prompts = request.prompt
    
    total_token_count = 0
    for prompt in prompts:
        input_ids = tokenizer(prompt).input_ids
        total_token_count += len(input_ids)
    
    response = TokenUsageResponse(token_count=total_token_count)
    return response


@app.post("/generate")
async def generate(request: Request, _auth: None = Depends(auth_check)) -> Response:
    """Generate completion for the request.
    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer_arg = engine_args.tokenizer
    if not tokenizer_arg:
        tokenizer_arg = ""

    tokenizer = get_tokenizer(tokenizer_arg,
                            tokenizer_mode=engine_args.tokenizer_mode,
                            trust_remote_code=engine_args.trust_remote_code)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)