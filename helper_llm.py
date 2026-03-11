

import json
import os
import socket
import time
import urllib.error
import urllib.request


def get_token(openai_token_env):
    openai_token = os.environ.get(openai_token_env, "").strip()
    if openai_token == "":
        raise RuntimeError(f"Missing OpenAI token in environment variable '{openai_token_env}'")
    return openai_token


def call_openai(prompt, model, token):
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": str(model),
        "messages": [{"role": "user", "content": str(prompt)}],
        "temperature": 0.0,
    }

    last_error = None
    for attempt in range(3):
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8")
            except Exception:
                details = ""
            last_error = RuntimeError(f"HTTP {exc.code}: {exc.reason}. Response: {details}".strip())
            if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except Exception as exc:
            last_error = exc
            break



        data = json.loads(raw)
        choices = data.get("choices", [])
        if len(choices) == 0:
            last_error = RuntimeError(f"returned no choices: {data}")
            if attempt < 2:
                time.sleep(1.0)
                continue
            break

        content = choices[0].get("message", {}).get("content", "")
        value = str(content).strip()
        if value == "":
            last_error = RuntimeError("returned empty content.")
            if attempt < 2:
                time.sleep(1.0)
                continue
            break

        return value

    raise RuntimeError(f"call failed for model '{model}': {last_error}")


def make_llm_call(openai_model, openai_token):
    model = openai_model
    def llm_call(prompt):
        return call_openai(prompt, model, openai_token)

    return llm_call, model
