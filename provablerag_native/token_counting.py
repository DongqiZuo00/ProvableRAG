from collections import OrderedDict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import threading

VERSION = "deepseek_v41_recipe_020_calibrated_v1"
RECIPE_VERSION = "0.2.0"
TOKENIZER_SHA256 = "81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99"
RECIPE_REVISION = "935817c66f5d6e64838f419692545f06787e485f"
RECIPE_NATIVE_SHA256 = "753bd43926f1b938bccb07b6e033054f1d1f1aab74eff47598baccafa612bb0d"
RELEASE = Path(__file__).resolve().parents[2] / "provablerag_paper_reference"


class TokenCountingError(RuntimeError):
    pass


class TokenCounter:
    def __init__(self, model_config=None, *, tokenizer_path=None, absolute_margin=64,
                 relative_margin=0.02, cache_size=2048):
        profile = asdict(model_config) if hasattr(model_config,"__dataclass_fields__") else dict(model_config or {})
        model = profile.get("exact_model_id",profile.get("model","deepseek-flash"))
        if model not in ("deepseek-flash","DeepSeek-V4.1-Flash"):
            raise TokenCountingError("tokenizer_model_identity_unsupported:"+str(model))
        if profile.get("thinking","disabled") != "disabled":
            raise TokenCountingError("this_counter_requires_explicit_nonthinking_text_protocol")
        if type(absolute_margin) is not int or absolute_margin < 0 or not 0 <= relative_margin <= 1:
            raise ValueError("invalid_token_counter_margin")
        self.absolute_margin,self.relative_margin = absolute_margin,float(relative_margin)
        self.cache_size=cache_size
        path=Path(tokenizer_path or RELEASE/"tokenizer_assets/tokenizer.json").resolve()
        try:asset_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:raise TokenCountingError("missing_pinned_tokenizer_asset") from exc
        if asset_sha != TOKENIZER_SHA256:
            raise TokenCountingError("tokenizer_asset_sha256_mismatch")
        vendor=RELEASE/"vendor_python"
        if vendor.is_dir() and str(vendor) not in sys.path:sys.path.insert(0,str(vendor))
        try:
            from deepseek_recipe import ChatCompletionRequest,ConversionOptions,DeepseekV41Encoding,Tokenizer
            import deepseek_recipe._native as native
            package_version=importlib.metadata.version("deepseek-recipe")
            if package_version != RECIPE_VERSION:raise TokenCountingError("recipe_package_version_mismatch")
            native_sha=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
            if native_sha != RECIPE_NATIVE_SHA256:raise TokenCountingError("recipe_native_asset_sha256_mismatch")
            self._request,self._options = ChatCompletionRequest,ConversionOptions
            self._encoding=DeepseekV41Encoding().with_tokenizer(Tokenizer.from_file(str(path)))
        except TokenCountingError:raise
        except Exception as exc:raise TokenCountingError("official_recipe_runtime_unavailable:"+type(exc).__name__) from exc
        self._profile={"model":"DeepSeek-V4.1-Flash","thinking":{"type":"disabled"},
                       "response_format":profile.get("response_format",{"type":"json_object"})}
        self._cache=OrderedDict();self._lock=threading.RLock()
        self._metadata={"version":VERSION,"requested_model_id":model,"canonical_encoding_model":"DeepSeek-V4.1-Flash",
            "tokenizer_sha256":asset_sha,"recipe_package_version":package_version,
            "recipe_documentation_revision":RECIPE_REVISION,"recipe_native_sha256":native_sha,
            "prompt_profile":self._profile,"absolute_margin":absolute_margin,"relative_margin":self.relative_margin,
            "absolute_server_token_guarantee":False,
            "uncertainty_policy":"official local encoding plus fixed/relative safety margin; actual provider usage is authoritative; any input-bound or context overrun stops the run and requires recalibration"}
        self.identity=hashlib.sha256(json.dumps(self._metadata,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def metadata(self):
        return json.loads(json.dumps({**self._metadata,"identity":self.identity}))

    def count(self,messages):
        if not isinstance(messages,list) or not messages:
            raise TokenCountingError("nonempty_text_messages_required")
        for message in messages:
            if not isinstance(message,dict) or message.get("role") not in ("system","developer","user","assistant"):
                raise TokenCountingError("unsupported_message_contract")
            if not isinstance(message.get("content"),str) or set(message)-{"role","content"}:
                raise TokenCountingError("text_only_counter_rejects_extra_fields_or_multimodal_messages")
        payload={**self._profile,"messages":messages}
        key=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key);return self._cache[key]
            try:
                converted=self._request(payload).convert(self._options())
                count=len(self._encoding.encode(converted.conversation))
            except Exception as exc:raise TokenCountingError("official_prompt_encoding_failed:"+type(exc).__name__) from exc
            self._cache[key]=count
            while len(self._cache)>self.cache_size:self._cache.popitem(last=False)
            return count

    def input_bound(self,messages):
        estimate=self.count(messages)
        return estimate,math.ceil(estimate*(1+self.relative_margin))+self.absolute_margin


_COUNTERS={}
_DEFAULT_LOCK=threading.RLock()


def counter_for_config(model_config=None):
    profile=asdict(model_config) if hasattr(model_config,"__dataclass_fields__") else dict(model_config or {})
    relevant={key:profile[key] for key in ("exact_model_id","model","thinking","response_format") if key in profile}
    key=json.dumps(relevant,sort_keys=True,ensure_ascii=False)
    with _DEFAULT_LOCK:
        if key not in _COUNTERS:_COUNTERS[key]=TokenCounter(model_config)
        return _COUNTERS[key]
