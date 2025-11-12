from __future__ import annotations
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

ALLOWED_ORDER_BY = {"year", "title", "rating", "popularity"}

@wrap_tool_call
async def sanitize_sql_args(request, handler):
    """
    Async tool-call wrapper so it works with agent.ainvoke()/astream().
    """
    call = request.tool_call
    name = call.get("name")
    args = call.get("args", {})

    if name == "query_sql_db" and isinstance(args, dict):
        cleaned = {}
        for k, v in args.items():
            # drop empty
            if isinstance(v, str) and v.strip() == "":
                v = None
            # list -> comma-separated for genre/cast
            if k in {"genre", "cast"} and isinstance(v, list):
                v = ", ".join([str(x).strip() for x in v if str(x).strip()])
            # numeric strings -> ints
            if k in {"year", "year_min", "year_max", "limit", "offset"} and isinstance(v, str):
                if v.lstrip("-").isdigit():
                    v = int(v)
            # enums
            if k == "order_dir" and isinstance(v, str):
                v = v.upper()
                if v not in {"ASC", "DESC"}:
                    v = None
            if k == "order_by" and isinstance(v, str):
                if v not in ALLOWED_ORDER_BY:
                    v = None
            if k == "response_format" and isinstance(v, str):
                vv = v.lower()
                v = vv if vv in {"summary", "detailed"} else None

            if v is not None:
                cleaned[k] = v

        call["args"] = cleaned

    return await handler(request)


@wrap_tool_call
async def tool_errors_to_message(request, handler):
    """
    Convert tool exceptions into a ToolMessage in async contexts.
    """
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: {e}",
            tool_call_id=request.tool_call["id"],
        )


@wrap_tool_call
async def log_tools(request, handler):
    """
    Async logging wrapper (plays nice with ainvoke()).
    """
    name = request.tool_call.get("name")
    args = request.tool_call.get("args")
    print(f"[TOOL_START] name={name} args={args}")
    result = await handler(request)
    preview = str(result)
    # if len(preview) > 800:
    #     preview = preview[:800] + "…"
    print(f"[TOOL_END] name={name} result={preview}")
    return result
