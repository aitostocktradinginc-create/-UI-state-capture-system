
import os
import time
import json
import hashlib
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
    WebDriverException,
)

# =========================
# 0. Config
# =========================

# Your Gemini API Key (replace with your own)
GEMINI_API_KEY = "replace with your on gemini api"  

# Model and endpoint
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent"
)

# Browser profile dir , replace with your own dir
AUT_PROFILE_DIR = r"C:\Users\89659\Desktop\final round"

# Root folder for screenshots and tutorials
SCREENSHOT_ROOT = "dataset"

# Max steps per task to avoid infinite loops
MAX_STEPS = 15


# =========================
# 1. Helpers & Data Classes
# =========================

@dataclass
class ActionDecision:
    action: str                      # "click" | "type" | "wait_for_user" | "sleep" | "finish"
    target_kind: Optional[str]       # "clickable" | "input" | None
    target_id: Optional[int]
    text: str
    seconds: float
    reason: str


def compute_state_signature(state: Dict[str, Any]) -> str:
    """Generate a signature for the current page to detect duplicate states/screens."""
    click_texts = [c.get("text", "") for c in state.get("clickables", [])]
    input_phs = [i.get("placeholder", "") for i in state.get("inputs", [])]

    sig_str = "||".join([
        state.get("url", ""),
        state.get("title", ""),
        "\n".join(click_texts),
        "\n".join(input_phs),
    ])

    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()


def compute_auth_page_key(state: Dict[str, Any]) -> str:
    """
    A coarser key for auth pages: only URL + title.
    This makes auth-waits more stable even if dynamic content changes.
    """
    key_str = state.get("url", "") + "||" + state.get("title", "")
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def is_auth_form_like(state: Dict[str, Any]) -> bool:
    """
    Detect whether the current page looks like a login / signup / verification page.

    **Important**: we deliberately avoid treating generic "Sign in" in the navbar + a random input
    (like a search box) as an auth page. That was causing some homepages to be misclassified.
    """
    body = (state.get("body_text") or "").lower()
    url = (state.get("url") or "").lower()
    title = (state.get("title") or "").lower()
    inputs = state.get("inputs") or []

    # 1) Strong URL signal: typical auth-related paths
    auth_path_keywords = [
        "/login", "/signin", "/sign-in", "/sign_in",
        "/signup", "/sign-up", "/sign_up",
        "/auth", "/session", "/account/login"
    ]
    if any(k in url for k in auth_path_keywords):
        return True

    # 2) Placeholder-based signal: email / username / phone / password / 验证码, etc.
    placeholder_auth_words = [
        "email", "mail", "e-mail", "username", "user name",
        "phone", "mobile", "手机号", "手机号码",
        "password", "passcode",
        "code", "验证码", "验证代码", "安全码",
        "邮箱", "密码", "帐户", "账号",
    ]
    for inp in inputs:
        ph = (inp.get("placeholder") or "").lower()
        if any(w in ph for w in placeholder_auth_words):
            return True

    # 3) Text-based signal: verification / password, etc. (NO 'sign in' / 'sign up' here!)
    text_keywords = [
        "password", "passcode", "verification", "verify your identity",
        "2-step", "two-factor", "two step", "2 step", "security code",
        "验证码", "验证", "两步验证", "安全验证", "邮箱验证", "手机验证",
    ]
    if inputs and any(k in body or k in title for k in text_keywords):
        return True

    return False


def extract_and_clean_json(text: str) -> str:
    """
    Extract the {...} block from LLM output and replace raw newlines inside strings with \\n,
    to avoid 'Invalid control character' errors in json.loads.
    """
    text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start: end + 1]

    result = []
    in_string = False
    escaped = False

    for ch in text:
        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
                escaped = False
        else:
            if escaped:
                result.append(ch)
                escaped = False
            else:
                if ch == '\\':
                    result.append(ch)
                    escaped = True
                elif ch == '"':
                    result.append(ch)
                    in_string = False
                elif ch in ['\r', '\n']:
                    result.append('\\n')
                else:
                    if ord(ch) < 32 and ch not in ['\t']:
                        result.append(' ')
                    else:
                        result.append(ch)

    return "".join(result)


# =========================
# 2. Gemini Client
# =========================

class GeminiClient:
    def __init__(self, api_key: str = GEMINI_API_KEY):
        if (not api_key) or api_key.startswith("YOUR_") or "你的" in api_key:
            raise RuntimeError("Please set your Gemini API key in GEMINI_API_KEY.")
        self.api_key = api_key

    def _call(self, prompt_text: str) -> str:
        """Low-level call. Automatically retries on 503 until success."""
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                }
            ]
        }
        params = {"key": self.api_key}

        while True:
            resp = requests.post(
                GEMINI_ENDPOINT,
                params=params,
                json=body,
                timeout=120,
            )

            if resp.status_code == 503:
                print("\n[Gemini] Model overloaded (503). Retrying in 5 seconds...")
                time.sleep(5)
                continue

            break

        if resp.status_code == 429:
            print("\n[Gemini] 429: rate / quota issue. Full response:")
            print(resp.text)
            raise RuntimeError("Gemini API returned 429. Please check quota or try later.")

        if not resp.ok:
            print("\n[Gemini] Error response:")
            print(resp.text)
            resp.raise_for_status()

        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Cannot extract 'text' from Gemini response: {e}, resp={data}")

        return text.strip()

    # ---------- 2.1 Decide app / goal / start_url ----------

    def decide_task_entry(self, user_prompt: str) -> Dict[str, str]:
        system = """
You are a planning agent for a web UI automation system.

Given a user's natural language instruction about how to do something in a web app,
you must decide:

- which app is being used (e.g. "Linear", "Notion", "Gmail", "Facebook")
- the high-level goal as a short snake_case string (e.g. "create_project", "post_photo")
- a single starting URL that should be opened in the browser for this task.

Return a SINGLE JSON object, with this exact schema:

{
  "app": "<AppName>",
  "goal": "<snake_case_goal>",
  "start_url": "<https://...>"
}

STRICT formatting rules:
- The JSON MUST be valid, minified (all on one line if possible).
- Strings MUST NOT contain raw line-breaks. If you need a line-break, use "\\n".
- Do NOT add any commentary. Output JSON only.
        """.strip()

        prompt = f"{system}\n\nUser instruction:\n{user_prompt}\n\nJSON:"
        raw = self._call(prompt)
        cleaned = extract_and_clean_json(raw)

        info = json.loads(cleaned)

        return {
            "app": info["app"],
            "goal": info["goal"],
            "start_url": info["start_url"],
        }

    # ---------- 2.2 Decide next action ----------

    def decide_next_action(
        self,
        app: str,
        goal: str,
        user_prompt: str,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> ActionDecision:
        system = """
You are a step decision agent for a browser automation system.

At each turn, you are given:
- The overall app and goal.
- The original user instruction.
- The CURRENT browser state (URL, title, text, list of clickables & inputs).
- A short history of previous actions.

You must decide EXACTLY ONE next atomic action for the executor to perform.

Valid actions:
- "click": click on one clickable element by its id.
- "type": type some text into one input element by its id.
- "wait_for_user": pause and let the human do something manually (login, 2FA, upload...).
- "sleep": wait a few seconds.
- "finish": the task is done (no more actions needed).

The browser state format:
- "clickables": a list of objects, each has:
    { "id": <int>, "text": "<visible_text>", "tag": "<tag_name>" }
- "inputs": a list of objects, each has:
    { "id": <int>, "placeholder": "<placeholder>", "tag": "<tag_name>" }

When you choose "click":
- You MUST set:
    "target": { "kind": "clickable", "id": <id_from_clickables> }
- Do NOT invent ids that are not present in the clickables list.

When you choose "type":
- You MUST set:
    "target": { "kind": "input", "id": <id_from_inputs> }
- And set "text": the exact string to type.

When you choose "sleep":
- You MUST set "seconds": a small number like 1.5, 2, 3, etc.

When you choose "wait_for_user":
- "target" can be null.
- This means: the user will manually complete something (e.g. login form, 2FA, file upload), then resume.
- You SHOULD ONLY choose "wait_for_user" when there is an obvious auth / signup / verification
  form visible on the current page (e.g. email/password/code inputs).

When you choose "finish":
- The executor will stop the loop.
- Only choose "finish" when the goal is clearly achieved or no further automated action makes sense.

Respond with a SINGLE JSON object, with this exact schema:

{
  "action": "click" | "type" | "wait_for_user" | "sleep" | "finish",
  "target": {
    "kind": "clickable" | "input" | null,
    "id": <integer or null>
  },
  "text": "<text to type when action='type', else empty string>",
  "seconds": <number of seconds for 'sleep', or 0>,
  "reason": "<very short natural language explanation>"
}

STRICT formatting rules:
- The JSON MUST be valid and minified (preferably all on one line).
- Strings MUST NOT contain raw line-breaks. If you need a line-break, use "\\n".
- DO NOT output anything except the JSON object.
        """.strip()

        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        history_json = json.dumps(history[-10:], ensure_ascii=False, indent=2)

        prompt = f"""
{system}

APP: {app}
GOAL: {goal}
ORIGINAL_USER_INSTRUCTION:
{user_prompt}

CURRENT_STATE_JSON:
{state_json}

HISTORY_JSON (last 10 actions):
{history_json}

Now respond with the JSON object for the NEXT action only:
"""

        raw = self._call(prompt)
        cleaned = extract_and_clean_json(raw)

        data = json.loads(cleaned)

        target = data.get("target") or {}
        return ActionDecision(
            action=data.get("action", "finish"),
            target_kind=target.get("kind"),
            target_id=target.get("id"),
            text=data.get("text") or "",
            seconds=float(data.get("seconds") or 0),
            reason=data.get("reason") or "",
        )

    # ---------- 2.3 Summarize to human-friendly tutorial steps ----------

    def summarize_tutorial_steps(
        self,
        app: str,
        goal: str,
        user_prompt: str,
        filtered_steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Turn denoised filtered_steps into 2–6 high-level, English tutorial steps.
        Each step has: title / description / source_indices.
        Note: Step 1 ("Open the website and log in") will be added by the caller,
        so DO NOT create a step that only says "open the website".
        """
        steps_brief = []
        for h in filtered_steps:
            steps_brief.append({
                "step_index": h.get("step_index"),
                "action": h.get("action"),
                "reason": h.get("reason"),
                "url": h.get("url"),
                "title": h.get("title"),
                "target_text": h.get("target_text", ""),
                "input_placeholder": h.get("input_placeholder", ""),
            })

        system = """
You are a senior UX writer helping to turn raw interaction logs into a clean, beginner-friendly English tutorial.

You will receive:
- app name (e.g. "Notion")
- goal (snake_case, e.g. "filter_database")
- the original user question in natural language
- a list of interaction steps (click / type / wait / finish), each with step_index, url, title, etc.

Your job:
1) Infer the correct conceptual steps a normal user should follow to achieve the goal in this app.
2) Ignore detours, repeated pages, and noisy actions. Focus on the canonical workflow.
3) Output 2–6 steps. Each step is high-level and understandable for a complete beginner.

Important:
- The caller will ALWAYS create "Step 1: Open the website and log in".
- Therefore, do NOT include a step that only tells the user to open the website or log in initially.
- Start your conceptual steps from what happens AFTER the user is already on the correct app page and logged in.

Output format (JSON only, no comments):

{
  "steps": [
    {
      "title": "<short English title, do NOT include 'Step X:'>",
      "description": "<1–4 sentences in English explaining what to do in this step>",
      "source_indices": [<one_or_more_original_step_index_integers>]
    },
    ...
  ]
}

Requirements:
- Use English in title and description.
- Make the tutorial understandable to a complete beginner.
- For well-known apps (e.g., Notion / Linear / Gmail), prefer the official / typical flow.
- Keep descriptions concise but clear (no more than ~120 English words per step).
- Titles must NOT include "Step X:"; the caller will number the steps.
- source_indices should reference which raw steps this tutorial step roughly corresponds to.
- JSON MUST be valid and minified, no extra text, no trailing commas.
        """.strip()

        steps_brief_json = json.dumps(steps_brief, ensure_ascii=False)

        prompt = f"""
{system}

APP: {app}
GOAL: {goal}
ORIGINAL_USER_QUESTION:
{user_prompt}

RAW_INTERACTION_STEPS_JSON:
{steps_brief_json}

Now output the JSON with the "steps" array only:
"""

        raw = self._call(prompt)
        cleaned = extract_and_clean_json(raw)
        data = json.loads(cleaned)
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            raise RuntimeError("summarize_tutorial_steps: 'steps' is not a list")
        return steps


# =========================
# 3. Browser Wrapper + Auto Login Click
# =========================

class UIBrowser:
    def __init__(self, screenshot_root: str = SCREENSHOT_ROOT):
        self.screenshot_root = screenshot_root
        self.driver: Optional[webdriver.Chrome] = None
        self.last_clickable_map: Dict[int, Any] = {}
        self.last_input_map: Dict[int, Any] = {}

    def __enter__(self):
        options = webdriver.ChromeOptions()
        os.makedirs(AUT_PROFILE_DIR, exist_ok=True)
        options.add_argument(f"--user-data-dir={AUT_PROFILE_DIR}")
        options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()

    def open(self, url: str):
        print(f"[Browser] GET {url}")
        self.driver.get(url)

    def observe(self, max_clickables: int = 40, max_inputs: int = 10) -> Dict[str, Any]:
        """
        Inspect the current page: URL, title, body text, visible clickables and inputs.

        IMPORTANT:
        - Login / Sign up / Get started buttons are treated as HIGH PRIORITY
          and always included in clickables if they exist, even if they are small.
        """
        d = self.driver

        url = d.current_url
        title = ""
        try:
            title = d.title or ""
        except Exception:
            pass

        body_text = ""
        try:
            body_el = d.find_element(By.TAG_NAME, "body")
            body_text = body_el.text or ""
            if len(body_text) > 1500:
                body_text = body_text[:1500]
        except Exception:
            pass

        # Collect clickables
        clickables = []
        click_map = {}
        try:
            elems = d.find_elements(By.XPATH, "//button | //*[@role='button'] | //a")

            priority = []  # login / signup / get started
            others = []

            login_like_words = [
                "log in", "login", "sign in", "signin",
                "sign up", "signup", "get started", "start for free",
                "open notion", "get notion free",
                "进入", "开始使用", "免费试用", "登录", "登入", "注册"
            ]

            for idx, el in enumerate(elems):
                try:
                    if not el.is_displayed():
                        continue
                    text = (el.text or "").strip()
                    if not text:
                        aria = el.get_attribute("aria-label") or ""
                        text = aria.strip()
                    if not text:
                        continue
                    rect = el.rect or {}
                    area = rect.get("width", 0) * rect.get("height", 0)
                    low = text.lower()
                    item = (area, el, text, el.tag_name)
                    if any(w in low for w in login_like_words):
                        priority.append(item)
                    else:
                        others.append(item)
                except StaleElementReferenceException:
                    continue

            priority.sort(key=lambda x: x[0], reverse=True)
            others.sort(key=lambda x: x[0], reverse=True)

            tmp = (priority + others)[:max_clickables]

            for new_id, (_, el, text, tag) in enumerate(tmp):
                clickables.append({
                    "id": new_id,
                    "text": text[:80],
                    "tag": tag,
                })
                click_map[new_id] = el
        except Exception as e:
            print(f"[Browser] Error collecting clickables: {e}")

        # Collect inputs
        inputs = []
        input_map = {}
        try:
            elems = d.find_elements(By.XPATH, "//input | //textarea")
            tmp = []
            for idx, el in enumerate(elems):
                try:
                    if not el.is_displayed():
                        continue
                    placeholder = (el.get_attribute("placeholder") or "").strip()
                    if not placeholder:
                        continue
                    rect = el.rect or {}
                    area = rect.get("width", 0) * rect.get("height", 0)
                    tmp.append((area, el, placeholder, el.tag_name))
                except StaleElementReferenceException:
                    continue

            tmp.sort(key=lambda x: x[0], reverse=True)
            tmp = tmp[:max_inputs]

            for new_id, (_, el, placeholder, tag) in enumerate(tmp):
                inputs.append({
                    "id": new_id,
                    "placeholder": placeholder[:80],
                    "tag": tag,
                })
                input_map[new_id] = el
        except Exception as e:
            print(f"[Browser] Error collecting inputs: {e}")

        self.last_clickable_map = click_map
        self.last_input_map = input_map

        state = {
            "url": url,
            "title": title,
            "body_text": body_text,
            "clickables": clickables,
            "inputs": inputs,
        }
        return state

    def do_click(self, click_id: int):
        el = self.last_clickable_map.get(click_id)
        if not el:
            raise RuntimeError(f"clickable id={click_id} not found (page may have refreshed)")
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center', inline:'center'});", el
            )
            time.sleep(0.2)
            el.click()
        except WebDriverException:
            self.driver.execute_script("arguments[0].click();", el)

    def do_type(self, input_id: int, text: str):
        el = self.last_input_map.get(input_id)
        if not el:
            raise RuntimeError(f"input id={input_id} not found (page may have refreshed)")
        el.clear()
        el.send_keys(text)

    def screenshot(self, app: str, goal: str, step_index: int, label: str):
        safe_name = f"{app}_{goal}".replace(" ", "_")
        dir_path = os.path.join(self.screenshot_root, safe_name)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"step_{step_index:02d}_{label}.png"
        path = os.path.join(dir_path, filename)
        self.driver.save_screenshot(path)
        print(f"[Screenshot] {path}")


def auto_click_login_or_open(browser: UIBrowser) -> bool:
    """
    Fallback: directly scan the DOM for login / signup / 'Get started' buttons
    and click one via JS.
    """
    d = browser.driver

    login_like_words = [
        "log in", "login", "sign in", "signin",
        "sign up", "signup", "get started", "start for free",
        "open notion", "get notion free",
        "进入", "开始使用", "免费试用", "登录", "登入", "注册"
    ]

    xpath = (
        "//*[self::a or self::button or @role='button' or "
        "contains(@class,'button') or contains(@class,'Button') or contains(@class,'btn')]"
    )

    try:
        elems = d.find_elements(By.XPATH, xpath)
    except Exception as e:
        print(f"  >> auto_click_login_or_open: error finding elements: {e}")
        return False

    for el in elems:
        try:
            if not el.is_displayed():
                continue
            text = (el.text or el.get_attribute("aria-label") or "").strip()
            if not text:
                continue
            low = text.lower()
            if any(w in low for w in login_like_words):
                print(f"  >> auto_click_login_or_open: auto-click login/open-like button: '{text}'")
                try:
                    d.execute_script(
                        "arguments[0].scrollIntoView({block:'center', inline:'center'});",
                        el,
                    )
                    time.sleep(0.2)
                    el.click()
                except WebDriverException:
                    d.execute_script("arguments[0].click();", el)
                return True
        except StaleElementReferenceException:
            continue
        except Exception as e:
            print(f"  >> auto_click_login_or_open: error handling element: {e}")
            continue

    return False


# =========================
# 4. Tutorial HTML Generator
# =========================

def generate_tutorial_html(
    app: str,
    goal: str,
    user_prompt: str,
    history: List[Dict[str, Any]],
    start_url: str,
):
    """
    Generate tutorial.html based on history + screenshots.
    """
    safe_name = f"{app}_{goal}".replace(" ", "_")
    folder = os.path.join(SCREENSHOT_ROOT, safe_name)
    os.makedirs(folder, exist_ok=True)

    # 1) Save meta.json
    meta = {
        "app": app,
        "goal": goal,
        "start_url": start_url,
        "user_prompt": user_prompt,
        "history": history,
    }
    meta_path = os.path.join(folder, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Meta] Saved to {meta_path}")

    # 2) Collect screenshots
    screenshots: Dict[int, str] = {}
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".png"):
            continue
        if not fn.startswith("step_"):
            continue
        parts = fn.split("_")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        if idx not in screenshots:
            screenshots[idx] = fn

    # 3) Denoise history
    history_sorted = sorted(history, key=lambda x: x.get("step_index", 0))

    noisy_actions = {
        "sleep",
        "finish_overruled",
        "wait_for_user_skipped",
        "click_invalid_id_skipped",
        "auto_click_login_or_open_from_invalid_click",
        "auto_click_login_or_open_pre",
        "auto_click_login_or_open",
        "auth_wait_for_user_auto",
    }

    filtered_steps: List[Dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for h in history_sorted:
        idx = h.get("step_index")
        if idx is None:
            continue

        action = h.get("action", "")
        sig = h.get("signature")

        if h.get("error"):
            continue

        if action in noisy_actions:
            continue

        if action != "finish":
            if sig and sig in seen_signatures:
                continue
            if sig:
                seen_signatures.add(sig)

        filtered_steps.append(h)

    # 4) Ask Gemini for summary
    if not filtered_steps:
        print("[Tutorial] No usable steps; generating a simple one-step tutorial.")
        steps_for_tutorial = [{
            "title": "Navigate inside the app",
            "description": "Once you are logged in, follow the on-screen instructions in the app to complete your task.",
            "source_indices": [],
        }]
    else:
        try:
            gemini = GeminiClient()
            steps_for_tutorial = gemini.summarize_tutorial_steps(
                app=app,
                goal=goal,
                user_prompt=user_prompt,
                filtered_steps=filtered_steps,
            )

            # Remove accidental "open website" steps
            filtered_again: List[Dict[str, Any]] = []
            url_domain = start_url.lower().replace("https://", "").replace("http://", "").split("/")[0]
            for s in steps_for_tutorial:
                desc = (s.get("description") or "").lower()
                if any(kw in desc for kw in ["open ", "go to ", "visit "]):
                    if url_domain in desc or app.lower() in desc:
                        continue
                filtered_again.append(s)
            if filtered_again:
                steps_for_tutorial = filtered_again

        except Exception as e:
            print(f"[Tutorial] summarize_tutorial_steps failed, falling back to simple step list: {e}")
            steps_for_tutorial = []
            for h in filtered_steps:
                idx = h.get("step_index")
                action = h.get("action", "")
                target_text = h.get("target_text") or ""
                placeholder = h.get("input_placeholder") or ""
                reason = h.get("reason", "")

                desc = reason or "Follow this step inside the app."

                if action in ("auth_wait_for_user_auto", "wait_for_user"):
                    title = "Complete any login or verification"
                    if not reason:
                        desc = (
                            "If a login or verification screen appears, finish signing in or "
                            "verifying your account, then return to the app."
                        )
                elif action == "click":
                    if target_text:
                        title = f'Click "{target_text}"'
                    else:
                        title = "Click the relevant button or link"
                elif action == "type":
                    if placeholder:
                        title = f'Fill in the "{placeholder}" field'
                    else:
                        title = "Enter the required text"
                elif action == "finish":
                    title = "Check that the result looks correct"
                    if not reason:
                        desc = "Confirm that you have reached the final page and the task is completed."
                else:
                    title = "Continue in the app"

                steps_for_tutorial.append({
                    "title": title,
                    "description": desc,
                    "source_indices": [idx] if idx is not None else [],
                })

    # 5) Build HTML
    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang='en'>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='utf-8' />")
    goal_title = goal.replace("_", " ")
    title_txt = f"{app} - {goal_title} Tutorial"
    html_parts.append(f"<title>{title_txt}</title>")
    html_parts.append("""
<style>
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
  max-width: 900px;
  margin: 20px auto;
  line-height: 1.6;
}
h1, h2 {
  color: #222;
}
.step {
  margin: 24px 0;
  padding: 16px 18px;
  border-radius: 10px;
  border: 1px solid #eee;
  background: #fafafa;
}
.step img {
  max-width: 100%;
  border-radius: 6px;
  border: 1px solid #ddd;
  margin-top: 8px;
}
.step-index {
  font-weight: bold;
  color: #555;
}
.step-desc {
  margin-top: 6px;
}
.code {
  font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
  background: #f5f5f5;
  padding: 2px 4px;
  border-radius: 4px;
}
</style>
""")
    html_parts.append("</head>")
    html_parts.append("<body>")

    html_parts.append(f"<h1>{app}: {goal_title} Tutorial</h1>")
    html_parts.append("<p><strong>Original question:</strong></p>")
    html_parts.append(f"<p>{user_prompt}</p>")
    html_parts.append("<hr />")

    # Step 1
    html_parts.append("<div class='step'>")
    html_parts.append("<div class='step-index'>Step 1: Open the website</div>")
    html_parts.append(
        "<div class='step-desc'>"
        f"Open <a href='{start_url}' target='_blank' class='code'>{start_url}</a> "
        f"in your browser and sign in to your {app} account if needed."
        "</div>"
    )
    initial_img = screenshots.get(0)
    if initial_img:
        html_parts.append(
            f"<img src='{initial_img}' alt='Step 1 screenshot' />"
        )
    html_parts.append("</div>")

    # Step 2+
    for display_offset, step in enumerate(steps_for_tutorial, start=2):
        title = step.get("title") or f"Step {display_offset}"
        desc = step.get("description") or "Follow this step inside the app."
        src_indices = step.get("source_indices") or []

        html_parts.append("<div class='step'>")
        html_parts.append(f"<div class='step-index'>Step {display_offset}: {title}</div>")
        html_parts.append(f"<div class='step-desc'>{desc}</div>")

        # Prefer the LARGEST source_index (latest state) that has a screenshot
        img_name = None
        int_indices = [i for i in src_indices if isinstance(i, int)]
        for idx in sorted(int_indices, reverse=True):
            if idx in screenshots:
                img_name = screenshots[idx]
                break

        # Fallback: if none of the source_indices have a screenshot, use the latest screenshot
        if img_name is None and screenshots:
            max_idx = max(screenshots.keys())
            img_name = screenshots[max_idx]

        if img_name:
            html_parts.append(
                f"<img src='{img_name}' alt='Step {display_offset} screenshot' />"
            )

        html_parts.append("</div>")

    html_parts.append("</body></html>")

    html = "\n".join(html_parts)
    out_path = os.path.join(folder, "tutorial.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Tutorial] Generated tutorial: {out_path}")


# =========================
# 5. Main Agent
# =========================

def run_interactive_agent():
    print("Interactive Gemini-driven UI Agent")
    print("You can ask for any website operation!")

    user_prompt = input("Enter your question:\n> ")

    # 1) Decide app / goal / start_url
    try:
        gemini = GeminiClient()
    except RuntimeError as e:
        print("[Config Error]", e)
        return

    print("\n[Gemini] Figuring out which app this is, and the start URL...")
    try:
        task_info = gemini.decide_task_entry(user_prompt)
    except Exception as e:
        print("[Gemini Error] Could not decide start URL:", e)
        return

    app = task_info["app"]
    goal = task_info["goal"]
    start_url = task_info["start_url"]

    # Cleanup previous data for this app+goal
    safe_name = f"{app}_{goal}".replace(" ", "_")
    folder = os.path.join(SCREENSHOT_ROOT, safe_name)
    if os.path.exists(folder):
        print(f"[Cleanup] Removing previous data at {folder} ...")
        shutil.rmtree(folder, ignore_errors=True)

    print(f"\n[Gemini] Parsed task: app={app}, goal={goal}, start_url={start_url}")

    history: List[Dict[str, Any]] = []
    last_signature: Optional[str] = None  # kept, but no longer used for screenshot de-dup
    ever_saw_auth_form: bool = False
    auth_page_keys_waited: set[str] = set()
    login_fallback_tried_signatures: set[str] = set()

    with UIBrowser() as browser:
        browser.open(start_url)

        # Initial page (step 0)
        state0 = browser.observe()
        last_signature = compute_state_signature(state0)
        browser.screenshot(app, goal, 0, "initial")

        for step_index in range(1, MAX_STEPS + 1):
            print(f"\n===== Loop Step {step_index} =====")

            state = browser.observe()
            print(f"[State] url={state['url']}, title={state['title']}")
            print(f"[State] {len(state['clickables'])} clickables, {len(state['inputs'])} inputs")

            current_sig = compute_state_signature(state)
            auth_like_now = is_auth_form_like(state)
            auth_page_key = compute_auth_page_key(state)

            # 1) If this looks like auth/login/signup page and we haven't paused here yet → auto pause
            if auth_like_now and auth_page_key not in auth_page_keys_waited:
                ever_saw_auth_form = True

                # ALWAYS screenshot auth page
                browser.screenshot(app, goal, step_index, "auth_wait_before")

                input(
                    "  >> Detected a possible login / signup / verification page.\n"
                    "  >> Please complete the login/sign-up in the browser, then press Enter here to continue..."
                )

                auth_page_keys_waited.add(auth_page_key)

                history.append({
                    "step_index": step_index,
                    "action": "auth_wait_for_user_auto",
                    "reason": "Executor detected an auth/login/signup form and paused for user.",
                    "url": state["url"],
                    "title": state["title"],
                    "signature": current_sig,
                })

                time.sleep(0.7)
                continue

            # 2) making sure it won't stop at marketing page.
            if (
                not ever_saw_auth_form
                and not auth_like_now
                and not state.get("inputs")
                and current_sig not in login_fallback_tried_signatures
            ):
                clicked = auto_click_login_or_open(browser)
                if clicked:
                    login_fallback_tried_signatures.add(current_sig)
                    history.append({
                        "step_index": step_index,
                        "action": "auto_click_login_or_open_pre",
                        "reason": "Executor auto-clicked a login/open-like button on a marketing page before calling Gemini.",
                        "url": state["url"],
                        "title": state["title"],
                        "signature": current_sig,
                    })
                    time.sleep(0.7)
                    continue  # observe the new page in the next loop

            # 3) Let Gemini decide next action
            try:
                decision = gemini.decide_next_action(app, goal, user_prompt, state, history)
            except Exception as e:
                print("[Gemini Error] Failed to decide next action:", e)
                break

            print(
                f"[Decision] action={decision.action}, target_kind={decision.target_kind}, "
                f"target_id={decision.target_id}, text='{decision.text}', "
                f"seconds={decision.seconds}, reason={decision.reason}"
            )

            # Handle 'finish'
            if decision.action == "finish":
                history.append({
                    "step_index": step_index,
                    "action": "finish",
                    "reason": decision.reason,
                    "url": state["url"],
                    "title": state["title"],
                    "signature": current_sig,
                })
                # Still take a final screenshot for the finished state
                browser.screenshot(app, goal, step_index, "finish")
                print("[Agent] Got 'finish' from model; ending task.")
                break

            # Handle 'wait_for_user'
            if decision.action == "wait_for_user":
                # ALWAYS screenshot before waiting
                browser.screenshot(app, goal, step_index, decision.action + "_before")

                if is_auth_form_like(state):
                    input(
                        "  >> Detected a possible login / signup / verification form.\n"
                        "  >> Please complete it in the browser, then press Enter here to continue..."
                    )
                    history.append({
                        "step_index": step_index,
                        "action": "wait_for_user",
                        "reason": decision.reason,
                        "url": state["url"],
                        "title": state["title"],
                        "signature": current_sig,
                    })
                else:
                    clicked = auto_click_login_or_open(browser)
                    if clicked:
                        history.append({
                            "step_index": step_index,
                            "action": "auto_click_login_or_open",
                            "reason": "Gemini chose wait_for_user on a marketing page; executor auto-clicked a login/open button.",
                            "url": state["url"],
                            "title": state["title"],
                            "signature": current_sig,
                        })
                    else:
                        print("  >> Current page is not an auth form, and no obvious login/open button was found. Skipping this wait_for_user and letting Gemini think again.")
                        history.append({
                            "step_index": step_index,
                            "action": "wait_for_user_skipped",
                            "reason": decision.reason,
                            "url": state["url"],
                            "title": state["title"],
                            "signature": current_sig,
                        })

                time.sleep(0.7)
                continue

            # actions: click / type / sleep
            try:
                # ALWAYS screenshot before executing action
                browser.screenshot(app, goal, step_index, decision.action + "_before")

                target_text = None
                input_placeholder = None

                if decision.action == "click":
                    if decision.target_kind != "clickable" or decision.target_id is None:
                        raise RuntimeError("Gemini returned 'click' but target is invalid.")

                    if decision.target_id not in browser.last_clickable_map:
                        print(f"  >> Gemini chose clickable id={decision.target_id}, but it does not exist. Trying login/open fallback...")
                        clicked = auto_click_login_or_open(browser)
                        if clicked:
                            history.append({
                                "step_index": step_index,
                                "action": "auto_click_login_or_open_from_invalid_click",
                                "reason": f"Invalid clickable id={decision.target_id}, executor auto-clicked a login/open-like button.",
                                "url": state["url"],
                                "title": state["title"],
                                "signature": current_sig,
                            })
                        else:
                            print("  >> Fallback could not find a suitable button. Skipping this step.")
                            history.append({
                                "step_index": step_index,
                                "action": "click_invalid_id_skipped",
                                "reason": decision.reason,
                                "url": state["url"],
                                "title": state["title"],
                                "signature": current_sig,
                            })
                        time.sleep(0.7)
                        continue

                    for c in state["clickables"]:
                        if c["id"] == decision.target_id:
                            target_text = c.get("text")
                            break

                    browser.do_click(decision.target_id)

                elif decision.action == "type":
                    if decision.target_kind != "input" or decision.target_id is None:
                        raise RuntimeError("Gemini returned 'type' but target is invalid.")

                    for i2 in state["inputs"]:
                        if i2["id"] == decision.target_id:
                            input_placeholder = i2.get("placeholder")
                            break

                    browser.do_type(decision.target_id, decision.text)

                elif decision.action == "sleep":
                    sec = decision.seconds if decision.seconds > 0 else 1.0
                    print(f"  -> sleep {sec} seconds")
                    time.sleep(sec)

                else:
                    print(f"  !! Unknown action: {decision.action}, skipping this step.")

                history_entry = {
                    "step_index": step_index,
                    "action": decision.action,
                    "target_kind": decision.target_kind,
                    "target_id": decision.target_id,
                    "text": decision.text,
                    "seconds": decision.seconds,
                    "reason": decision.reason,
                    "url": state["url"],
                    "title": state["title"],
                    "signature": current_sig,
                }
                if target_text:
                    history_entry["target_text"] = target_text
                if input_placeholder:
                    history_entry["input_placeholder"] = input_placeholder

                history.append(history_entry)

            except TimeoutException:
                print("  !! Timeout: element operation timed out; recording failure and continuing.")
                history.append({
                    "step_index": step_index,
                    "action": decision.action,
                    "error": "TimeoutException",
                    "signature": current_sig,
                })
            except Exception as e:
                print(f"  !! Error: {e} (this step failed; continuing with next step)")
                history.append({
                    "step_index": step_index,
                    "action": decision.action,
                    "error": str(e),
                    "signature": current_sig,
                })

            time.sleep(0.7)

        else:
            print(f"\n[Agent] Reached max steps ({MAX_STEPS}); stopping automatically.")

    # generate tutorial
    generate_tutorial_html(app, goal, user_prompt, history, start_url)
    print(f"\nDone! Screenshots and tutorial are saved under {SCREENSHOT_ROOT}/{app}_{goal}/")


if __name__ == "__main__":
    run_interactive_agent()
