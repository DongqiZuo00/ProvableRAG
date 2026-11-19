from contextlib import closing
from dataclasses import asdict, dataclass
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import time
from urllib.parse import urlsplit

from provablerag.models import ModelResponse, RunBlocked, UsageRecord, parse_response
from .util import canonical, digest


@dataclass(frozen=True)
class VLLMConfig:
    base_url: str
    exact_model_id: str
    format_mode: str = "json_object"
    deployment_revision: str = ""
    api_key_env: str = "VLLM_API_KEY"
    timeout_seconds: float = 120
    temperature: float = 0.0
    provider: str = "local_vllm"

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in ("http","https") or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Invalid local model endpoint")
        name = parsed.hostname
        if name != "localhost" and not (name and ipaddress.ip_address(name).is_loopback):
            raise ValueError("The local vLLM backend requires a loopback address")
        if not parsed.path.rstrip("/").endswith("/v1") or not self.exact_model_id.strip():
            raise ValueError("Invalid model endpoint path or model ID")
        if self.format_mode not in ("json_object","json_schema"):
            raise ValueError("Unknown structured output protocol")
        if not (math.isfinite(self.timeout_seconds) and self.timeout_seconds > 0):
            raise ValueError("Invalid transport timeout")


def _format(mode, messages, stage):
    if mode == "json_object":
        return {"type":"json_object"}
    from . import acquisition, formalization
    prompt = messages[0]["content"]
    string = {"type":"string"}
    array = {"type":"array", "items":{}}
    strings = {"type":"array", "items":string}
    obj = {"type":"object"}
    check_fields = {"status":{"enum":["accepted","rejected","unknown"]}, "reason":string,
                    **{key:{"type":"boolean"} for key in acquisition.CHECKS}}
    def record(properties):
        return {"type":"object", "properties":properties, "required":list(properties)}
    if prompt == formalization.NODES_PROMPT:
        name, fields = "nodes", {"nodes":array}
    elif prompt == formalization.RELATIONS_PROMPT:
        name, fields = "relations", {"edges":array, "prerequisites":array, "goal_options":array}
    elif prompt == formalization.BRIDGE_PROMPT:
        name, fields = "formalization", {"node_formulas":obj, "theory":obj, "answer":obj}
    elif prompt == acquisition.QUERY + acquisition.SHARED:
        name, fields = "search_query", {"query":string}
    elif prompt == acquisition.PROPOSE + acquisition.SHARED:
        name, fields = "candidate", {"value":string, "evidence_ids":strings, "premise_ids":strings}
    elif prompt == acquisition.VERIFY + acquisition.SHARED:
        name, fields = "verification", {**check_fields, "arguments":strings}
    elif prompt == acquisition.published_prompts.ANSWER + acquisition.SHARED:
        name, fields = "answer_proposal", {"answer":string, "evidence_ids":strings,
                                           "supporting_facts":array, "evidences":array}
    elif prompt.startswith("Independently verify the whole proposed answer"):
        name, fields = "answer_verification", check_fields
    else:
        name, fields = "role_output", {}
    schema = record(fields) if fields else obj
    if name == "formalization":
        schema["properties"].update(computations=obj, question_definition=obj)
    if name in ("nodes","relations","formalization"):
        schema = {"type":"object","anyOf":[schema,record({"status":{"const":"formalization_unresolved"},"reason":string})]}
    return {"type":"json_schema","json_schema":{"name":name,"schema":schema}}


class VLLMModel:
    def __init__(self, config, ledger_path):
        self.config = config
        self.endpoint = urlsplit(config.base_url.rstrip("/"))
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect_db()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS calls(request_key TEXT PRIMARY KEY, stage TEXT NOT NULL, wire TEXT NOT NULL, status TEXT NOT NULL, response TEXT, error TEXT, raw_response TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS counts(input_key TEXT PRIMARY KEY, tokens INTEGER NOT NULL)")
        self.token_counter = VLLMTokenCounter(self)

    def connect_db(self):
        return sqlite3.connect(self.path, timeout=30)

    def post(self, path, payload):
        cls = http.client.HTTPSConnection if self.endpoint.scheme == "https" else http.client.HTTPConnection
        host = self.endpoint.hostname
        if host == "localhost":
            host = str(ipaddress.IPv4Address(socket.INADDR_LOOPBACK))
        headers = {"Content-Type":"application/json"}
        key = os.environ.get(self.config.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        connection = cls(host, self.endpoint.port, timeout=self.config.timeout_seconds)
        try:
            connection.connect()
        except OSError as exc:
            connection.close()
            raise ConnectionNotSent("Cannot connect to the local vLLM endpoint") from exc
        with closing(connection):
            connection.request("POST", path, body=canonical(payload).encode("utf-8"), headers=headers)
            response = connection.getresponse()
            data = response.read()
            if response.status != 200:
                raise HTTPRejection(response.status, data.decode("utf-8", errors="replace")[:1200])
            return json.loads(data)

    def generate(self, messages, *, request_key, stage, max_output_tokens=512):
        if not request_key or type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("A request key and a positive output limit are required")
        payload = dict(model=self.config.exact_model_id, messages=messages, temperature=self.config.temperature,
                       max_tokens=max_output_tokens, response_format=_format(self.config.format_mode,messages,stage),
                       stream=False)
        wire = canonical({"config":asdict(self.config),"stage":stage,"payload":payload})
        with closing(self.connect_db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT wire,status,response FROM calls WHERE request_key=?",(request_key,)).fetchone()
            if row:
                if row[0] != wire:
                    raise RunBlocked("Request key maps to a different local vLLM request")
                if row[1] == "complete":
                    saved = json.loads(row[2])
                    return ModelResponse(**{**saved,"usage":UsageRecord(**saved["usage"])})
                if row[1] != "not_sent":
                    raise RunBlocked(f"Local vLLM request {request_key} is {row[1]}; inspect its ledger before retry")
                db.execute("UPDATE calls SET status='pending',error=NULL WHERE request_key=?",(request_key,))
            else:
                db.execute("INSERT INTO calls(request_key,stage,wire,status) VALUES(?,?,?,'pending')",(request_key,stage,wire))
        started = time.monotonic()
        raw = None
        try:
            raw = self.post(self.endpoint.path+"/chat/completions",payload)
            result = parse_response(raw,self.config.exact_model_id,raw.get("id"),time.monotonic()-started)
            if result.exact_model_id != self.config.exact_model_id:
                raise ValueError("served model ID differs from requested model")
            if any(type(v) is not int or v < 0 for v in (result.usage.input_tokens,result.usage.output_tokens)):
                raise ValueError("prompt_tokens and completion_tokens must be measured integers")
            if result.usage.output_tokens > max_output_tokens or not result.finish_reason:
                raise ValueError("invalid finish reason or completion token count")
        except Exception as exc:
            status = "not_sent" if isinstance(exc,ConnectionNotSent) else "unresolved"
            with closing(self.connect_db()) as db, db:
                db.execute("UPDATE calls SET status=?,error=?,raw_response=? WHERE request_key=?",
                           (status,str(exc),canonical(raw) if raw is not None else None,request_key))
            raise RunBlocked(f"Local vLLM call {status}: {exc}") from exc
        with closing(self.connect_db()) as db, db:
            db.execute("UPDATE calls SET status='complete',response=?,raw_response=? WHERE request_key=?",
                       (canonical(asdict(result)),canonical(raw),request_key))
        return result

    def usage(self):
        totals = {"requests":0, "input_tokens":0, "output_tokens":0, "unresolved":0, "not_sent":0}
        with closing(self.connect_db()) as db:
            for status, response in db.execute("SELECT status,response FROM calls"):
                if status != "complete":
                    totals["not_sent" if status=="not_sent" else "unresolved"] += 1
                    continue
                totals["requests"] += 1
                usage = json.loads(response)["usage"]
                for key in ("input_tokens","output_tokens"):
                    totals[key] += usage[key]
        return totals


class ConnectionNotSent(RuntimeError):
    pass


class HTTPRejection(RuntimeError):
    pass


class VLLMTokenCounter:
    def __init__(self, model):
        self.model = model
        self.profile = {"version":"vllm-chat-tokenize-v1","base_url":model.config.base_url,
                        "model":model.config.exact_model_id,"deployment_revision":model.config.deployment_revision,
                        "add_generation_prompt":True, "add_special_tokens":False,
                        "uncertainty_policy":"server count for these messages; stop on actual prompt usage overrun"}
        self.identity = digest(self.profile)

    def metadata(self):
        return {**self.profile,"identity":self.identity}

    def count(self, messages):
        key = digest([self.identity,messages])
        with closing(self.model.connect_db()) as db:
            row = db.execute("SELECT tokens FROM counts WHERE input_key=?",(key,)).fetchone()
        if row:
            return row[0]
        payload = {"model":self.model.config.exact_model_id,"messages":messages,"add_generation_prompt":True,
                   "add_special_tokens":False}
        try:
            response = self.model.post(self.model.endpoint.path[:-len("/v1")]+"/tokenize",payload)
            n = response["count"]
            if type(n) is not int or n < 1:
                raise ValueError("Invalid count returned by vLLM /tokenize")
        except Exception as exc:
            raise RunBlocked(f"Local vLLM chat tokenization failed: {exc}") from exc
        with closing(self.model.connect_db()) as db, db:
            db.execute("INSERT OR IGNORE INTO counts VALUES(?,?)",(key,n))
            if db.execute("SELECT tokens FROM counts WHERE input_key=?",(key,)).fetchone()[0] != n:
                raise RunBlocked("Local vLLM tokenizer changed for the same request")
        return n

    def input_bound(self, messages):
        n = self.count(messages)
        return n,n
