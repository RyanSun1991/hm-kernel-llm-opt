"""Authenticated enterprise gateway routes, mounted beside the unified MCP endpoint."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, ValidationError
from starlette.concurrency import run_in_threadpool

from hmopt.evolution.approval import (
    Decision,
    apply_decision,
    approval_context,
    authenticate,
)
from hmopt.evolution.production import Strict
from hmopt.evolution.store import ConflictError


class ContextRequest(Strict):
    approval_request_id: str = Field(min_length=1, max_length=200)
    principal: str = Field(min_length=1, max_length=200)


def approval_router(get_service, config):
    router = APIRouter(prefix="/evolution/approval")

    async def envelope(request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 65536:
                raise HTTPException(413, "Approval request exceeds 64 KiB")
        body = bytes(raw)
        try:
            proof = authenticate(
                config,
                request.headers.get("x-evolution-timestamp"),
                request.headers.get("x-evolution-signature"),
                request.url.path,
                body,
            )
            return json.loads(body), proof
        except PermissionError:
            raise HTTPException(401, "Invalid gateway authentication") from None
        except (ValueError, UnicodeError):
            raise HTTPException(400, "Invalid approval envelope or gateway configuration") from None

    def invoke(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except PermissionError:
            raise HTTPException(403, "Principal is not the assigned expert") from None
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from None
        except (ValueError, KeyError):
            raise HTTPException(
                400, "Decision does not satisfy the evidence or stage gate"
            ) from None

    @router.post("/context")
    async def context(request: Request):
        payload, _proof = await envelope(request)
        try:
            data = ContextRequest.model_validate(payload)
        except ValidationError:
            raise HTTPException(422, "Invalid approval context schema") from None
        return await run_in_threadpool(
            invoke,
            approval_context,
            get_service(),
            config,
            data.approval_request_id,
            data.principal,
        )

    @router.post("/decide")
    async def decide(request: Request):
        payload, proof = await envelope(request)
        try:
            data = Decision.model_validate(payload)
        except ValidationError:
            raise HTTPException(422, "Invalid structured decision schema") from None
        return await run_in_threadpool(
            invoke, apply_decision, get_service(), config, data, identity_proof=proof
        )

    return router
