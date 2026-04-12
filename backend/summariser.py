import os
import json
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(
    BASE_DIR, 'phi3_model', 'cpu_and_mobile',
    'cpu-int4-rtn-block-32-acc-level-4'
)

model = None
tokenizer = None
_model_ready = False

def load_model():
    global model, tokenizer, _model_ready
    if not os.path.exists(MODEL_DIR):
        print(f'Model directory not found: {MODEL_DIR}')
        return False
    try:
        import onnxruntime_genai as og
        print('Loading Phi-3...')
        model = og.Model(MODEL_DIR)
        tokenizer = og.Tokenizer(model)
        _model_ready = True
        print('Phi-3 ready')
        return True
    except Exception as e:
        print(f'Phi-3 load failed: {e}')
        return False

load_model()

PROMPT = """<|system|>
You are a meeting assistant. Read the transcript and return ONLY a JSON object with:
- "summary": 2-3 sentence overview of the meeting
- "decisions": list of key decisions made
- "actions": list of action items, include owner name if mentioned
Return only the JSON. No other text.<|end|>
<|user|>
Transcript:
{transcript}<|end|>
<|assistant|>"""

def _run_phi3(transcript: str) -> dict:
    import onnxruntime_genai as og
    prompt = PROMPT.format(transcript=transcript[:1500])
    encoded = tokenizer.encode(prompt)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=300, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(encoded)
    output_tokens = []
    count = 0
    while not generator.is_done() and count < 50:
        generator.generate_next_token()
        token = int(generator.get_next_tokens()[0])
        if token == 0:
            break
        output_tokens.append(token)
        count += 1
    del generator
    if not output_tokens:
        return None
    output_text = tokenizer.decode(output_tokens)
    if '<|assistant|>' in output_text:
        output_text = output_text.split('<|assistant|>')[-1].strip()
    return _parse_json(output_text)

def _parse_json(text: str) -> dict:
    # Try direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Try extracting JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # Try fixing trailing commas
    try:
        cleaned = re.sub(r',\s*}', '}', text)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        return json.loads(cleaned)
    except Exception:
        pass
    return None

def _fallback(transcript: str) -> dict:
    sentences = [
        s.strip() for s in
        transcript.replace('\n', '. ').split('.')
        if len(s.strip()) > 10
    ]
    summary = '. '.join(sentences[:3]) + '.' if sentences else 'Meeting processed.'
    decision_kw = ['decided', 'agreed', 'confirmed', 'approved', 'will', 'going to']
    action_kw = ['will', 'should', 'need to', 'must', 'follow up', 'by end', 'by friday', 'action']
    decisions = [s for s in sentences if any(k in s.lower() for k in decision_kw)][:5]
    actions = [s for s in sentences if any(k in s.lower() for k in action_kw)][:5]
    return {
        'summary': summary,
        'decisions': decisions or ['See transcript for details'],
        'actions': actions or ['Review transcript for action items']
    }

def summarise_transcript(transcript: str) -> dict:
    if not transcript or len(transcript.strip()) < 10:
        return {
            'summary': 'No transcript provided.',
            'decisions': [],
            'actions': []
        }
    if _model_ready:
        try:
            result = _run_phi3(transcript)
            if result:
                return result
        except Exception as e:
            print(f'Phi-3 inference error: {e}. Using fallback.')
    print('Using fallback summariser')
    return _fallback(transcript)
