"""OpenAI-compatible judge client configured with JUDGE_* environment variables."""
import os
import random
import time

import requests


def chat_url(base_url):
    """Accept either an API base URL or a complete chat-completions URL."""
    url = base_url.rstrip('/')
    return url if url.endswith('/chat/completions') else url + '/chat/completions'


def safe_post(session, url, **kwargs):
    for attempt in range(5):
        try:
            response = session.post(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt + random.random())


def call_chat_api(conversation):
    """Return assistant text, or raise after bounded retries; never score failures."""
    api_key = os.environ.get('JUDGE_API_KEY')
    if not api_key:
        raise RuntimeError('Set JUDGE_API_KEY before running the judge.')
    url = chat_url(os.environ.get('JUDGE_BASE_URL', 'https://api.openai.com/v1'))
    with requests.Session() as session:
        session.headers.update({'Authorization': f'Bearer {api_key}'})
        response = safe_post(session, url, json={
            'model': os.environ.get('JUDGE_MODEL', 'gpt-4o'),
            'messages': conversation,
            'stream': False,
        }, timeout=(10, 120))
        result = response.json()
    try:
        content = result['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('Judge response is missing assistant content.') from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Judge returned empty assistant content.')
    return content.strip()
