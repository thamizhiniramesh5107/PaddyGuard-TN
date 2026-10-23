from __future__ import annotations
import csv, hashlib, json, os, secrets, sqlite3, tempfile, uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
def configured_path(key: str, default: str | Path) -> Path:
    value = Path(os.getenv(key, str(default))).expanduser()
    return value if value.is_absolute() else (ROOT / value).resolve()

DB = configured_path('DATABASE_PATH', ROOT / 'paddyguard.sqlite3')
MODEL = configured_path('MODEL_PATH', ROOT / 'artifacts/paddyguard.pt')
ADVISORY = configured_path('ADVISORY_CSV', ROOT / 'paddyguard_advisory_data_starter.csv')
MODEL_URL = os.getenv('MODEL_URL', '').strip()
MODEL_SHA256 = os.getenv('MODEL_SHA256', '').strip().lower()
MODEL_DOWNLOAD_TOKEN = os.getenv('MODEL_DOWNLOAD_TOKEN', '').strip()
THRESHOLD = float(os.getenv('ABSTAIN_THRESHOLD', '0.55'))
SUPABASE_URL = os.getenv('SUPABASE_URL', '').strip()
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY', '').strip()
SUPABASE_TABLE = os.getenv('SUPABASE_TABLE', 'paddyguard_predictions').strip()
APP_ENV = os.getenv('APP_ENV', 'development').strip().lower()
HISTORY_ADMIN_TOKEN = os.getenv('HISTORY_ADMIN_TOKEN', '')
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)
CLASSES = json.loads((ROOT / 'class_map.json').read_text(encoding='utf-8'))
OFFICIAL = json.loads((ROOT / 'official_advisories.json').read_text(encoding='utf-8'))
app = FastAPI(title='PaddyGuard-TN API', version='1.0.0')
_cors_origins = [origin.strip() for origin in os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=['GET','POST','DELETE','OPTIONS'], allow_headers=['*'])
_model = None
_device = 'cpu'
_supabase = None

def supabase_client():
    global _supabase
    if not USE_SUPABASE: return None
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase

def init_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as c:
        c.execute('''CREATE TABLE IF NOT EXISTS predictions (
          id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT NOT NULL,
          class_name TEXT, confidence REAL NOT NULL, abstained INTEGER NOT NULL,
          affected_pct REAL NOT NULL, severity TEXT NOT NULL, language TEXT NOT NULL,
          advisory_json TEXT NOT NULL)''')
        c.commit()

def advisories() -> dict[str, dict[str, str]]:
    result = {}
    if ADVISORY.exists():
        with ADVISORY.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f): result[row['class_name']] = row
    return result

def connect():
    return sqlite3.connect(DB)

def ensure_model_file() -> bool:
    """Fetch the ignored model artifact only from an HTTPS URL and verify its SHA-256."""
    if MODEL.is_file():
        if MODEL_SHA256 and hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError(f'Model checksum mismatch for {MODEL}')
        return True
    if not MODEL_URL:
        return False
    if not MODEL_URL.lower().startswith('https://'):
        raise RuntimeError('MODEL_URL must use HTTPS')
    if len(MODEL_SHA256) != 64:
        raise RuntimeError('Set MODEL_SHA256 to the 64-character SHA-256 of the model artifact')
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    headers = {'User-Agent': 'PaddyGuard-TN/1.0'}
    if MODEL_DOWNLOAD_TOKEN:
        headers['Authorization'] = f'Bearer {MODEL_DOWNLOAD_TOKEN}'
        headers['Accept'] = 'application/octet-stream'
    request = Request(MODEL_URL, headers=headers)
    temporary = None
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=90) as response:
            declared_size = int(response.headers.get('Content-Length', '0') or 0)
            if declared_size > 100 * 1024 * 1024:
                raise RuntimeError('Model artifact exceeds the 100 MB safety limit')
            with tempfile.NamedTemporaryFile(dir=MODEL.parent, prefix='.paddyguard-', suffix='.part', delete=False) as target:
                temporary = Path(target.name)
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > 100 * 1024 * 1024:
                        raise RuntimeError('Model artifact exceeds the 100 MB safety limit')
                    digest.update(block)
                    target.write(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise RuntimeError('Downloaded model checksum does not match MODEL_SHA256')
        os.replace(temporary, MODEL)
        temporary = None
        return True
    finally:
        if temporary and temporary.exists():
            temporary.unlink()

def history(limit=20):
    if USE_SUPABASE:
        rows = supabase_client().table(SUPABASE_TABLE).select('*').order('created_at', desc=True).limit(limit).execute().data or []
        for row in rows:
            row['abstained'] = bool(row.get('abstained'))
            row['advisory'] = row.get('advisory') or row.get('advisory_json')
            if isinstance(row['advisory'], str): row['advisory'] = json.loads(row['advisory'])
        return rows
    with connect() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute('SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
    return [dict(r) | {'abstained': bool(r['abstained']), 'advisory': json.loads(r['advisory_json'])} for r in rows]

def severity(image: Image.Image):
    # Approximate color segmentation bounded by the dominant leaf-shaped region.
    import cv2
    rgb = np.asarray(image.convert('RGB').resize((480, 480)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = ((hsv[:,:,0] >= 20) & (hsv[:,:,0] <= 100) & (hsv[:,:,1] >= 24) & (hsv[:,:,2] >= 30)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros(green.shape, np.uint8)
    if contours:
        cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    # Brown/black lesion color estimate inside the dominant green-leaf silhouette.
    lesion = (((hsv[:,:,0] <= 25) | (hsv[:,:,0] >= 170)) & (hsv[:,:,1] >= 35) & (hsv[:,:,2] >= 25) & (hsv[:,:,2] <= 220))
    leaf_count = int(region.sum())
    pct = round(float((lesion & (region > 0)).sum()) * 100 / leaf_count, 1) if leaf_count else 0.0
    label = 'Low' if pct < 8 else ('Moderate' if pct < 22 else 'High')
    return pct, label

def load_model():
    global _model, _device
    if _model is not None: return _model
    try:
        if not ensure_model_file(): return None
        import torch
        torch.set_num_threads(max(1, int(os.getenv('TORCH_NUM_THREADS', '4'))))
        state = torch.load(MODEL, map_location='cpu', weights_only=False)
        names = state['classes']
        if state.get('architecture') == 'mobilenet_v3_small':
            from torchvision.models import mobilenet_v3_small
            net = mobilenet_v3_small(weights=None)
            net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, len(names))
        else:
            from torchvision.models import efficientnet_b0
            net = efficientnet_b0(weights=None)
            net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, len(names))
        net.load_state_dict(state['model']); net.eval()
        _device = 'cpu'; _model = (net, names, torch, int(state.get('input_size', 224))); return _model
    except Exception as e:
        print(f'Model could not be loaded: {e}')
        return None

def advice_for(name: str, language: str):
    official = OFFICIAL.get(name)
    if official:
        return {'label':official['label_en' if language == 'en' else 'label_ta'],
          'symptoms':official['symptoms_en' if language == 'en' else 'symptoms_ta'],
          'guidance':official['guidance_en' if language == 'en' else 'guidance_ta'],
          'chemical':'Do not use this image result alone to choose a pesticide. Follow current local label and extension guidance.' if language == 'en' else 'பூச்சிக்கொல்லியைத் தேர்ந்தெடுக்க இந்தப் பட முடிவை மட்டும் பயன்படுத்த வேண்டாம். தற்போதைய உள்ளூர் லேபிள் மற்றும் வேளாண் வழிகாட்டுதலைப் பின்பற்றவும்.',
          'sources':official['sources'],'notes':'Bilingual summary extracted from the linked TNAU pages; verify current field conditions and advice.'}
    row = advisories().get(name)
    if row:
        return {'label':'Synthetic starter fallback — verify before use' if language == 'en' else 'செயற்கை தொடக்கத் தரவு — பயன்படுத்தும் முன் சரிபார்க்கவும்',
          'symptoms': row.get('symptoms_en' if language == 'en' else 'symptoms_ta',''),
          'guidance': row.get('immediate_nonchemical_guidance_en' if language == 'en' else 'immediate_nonchemical_guidance_ta',''),
          'chemical': row.get('chemical_guidance_en' if language == 'en' else 'chemical_guidance_ta',''),
          'sources': [{'name':n.strip(),'url':u.strip()} for n,u in zip(row.get('source_name','').split(';'),row.get('source_url','').split(';')) if u.strip()],
          'notes': row.get('notes','')}
    return {'label':'Synthetic starter recommendation' if language == 'en' else 'செயற்கை தொடக்கப் பரிந்துரை',
      'symptoms':'No verified advisory is available for this label.' if language == 'en' else 'இந்த வகுப்பிற்கு சரிபார்க்கப்பட்ட ஆலோசனை இல்லை.',
      'guidance':'Retake a clear image and consult a local agricultural extension officer.' if language == 'en' else 'தெளிவான படத்தை மீண்டும் எடுத்து உள்ளூர் வேளாண் அலுவலரை அணுகவும்.',
      'chemical':'Do not apply pesticides based on this automated result.' if language == 'en' else 'இந்த தானியங்கி முடிவை மட்டும் வைத்து பூச்சிக்கொல்லி பயன்படுத்த வேண்டாம்.', 'sources':[], 'notes':'Synthetic fallback; not official guidance.'}

@app.get('/api/health')
def health():
    ready = load_model() is not None
    return {'status':'ok','model_ready':ready,'model_path':str(MODEL),'database':'supabase' if USE_SUPABASE else 'sqlite'}

@app.get('/healthz')
def healthz():
    return {'status':'ok'}

@app.get('/readyz')
def readyz():
    if load_model() is None:
        raise HTTPException(503, 'Model is unavailable or failed checksum/initialization')
    return {'status':'ready','model':'ready'}

@app.get('/api/classes')
def classes():
    data=advisories()
    return [{'name':k,'display_en':v['en'],'display_ta':v['ta'],'type':v['type'],'count':0} for k,v in CLASSES.items()]

@app.get('/api/stats')
def stats():
    items=history(1000)
    return {'total':len(items),'abstentions':sum(x['abstained'] for x in items),'class_counts':{k:sum(x['class_name']==k and not x['abstained'] for x in items) for k in CLASSES},'metrics':json.loads((ROOT / os.getenv('METRICS_PATH','artifacts/metrics.json')).read_text()) if (ROOT / os.getenv('METRICS_PATH','artifacts/metrics.json')).exists() else None}

@app.get('/api/history')
def get_history(limit: int=20): return history(max(1,min(100,limit)))

@app.delete('/api/history')
def clear_history(admin_token: str | None = Header(default=None, alias='X-Admin-Token')):
    if APP_ENV == 'production':
        if not HISTORY_ADMIN_TOKEN or not admin_token or not secrets.compare_digest(admin_token, HISTORY_ADMIN_TOKEN):
            raise HTTPException(403, 'A valid history admin token is required')
    if USE_SUPABASE:
        supabase_client().table(SUPABASE_TABLE).delete().neq('id', '').execute()
        return {'deleted':True}
    with connect() as c: c.execute('DELETE FROM predictions'); c.commit()
    return {'deleted':True}

@app.post('/api/predict')
async def predict(file: UploadFile=File(...), language: str=Form('en')):
    if language not in ('en','ta'): raise HTTPException(400,'language must be en or ta')
    raw=await file.read(12*1024*1024+1)
    if len(raw)>12*1024*1024: raise HTTPException(413,'Image must be under 12 MB')
    try:
        import io
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        if min(image.size)<96: raise HTTPException(400,'Image is too small; use at least 96 px')
    except (UnidentifiedImageError, OSError): raise HTTPException(400,'Upload a valid image file')
    pct, sev=severity(image)
    loaded=load_model(); name=None; conf=0.0; abstained=True; probabilities=[]
    if not loaded:
        raise HTTPException(503, 'The classifier is not ready. Please retry later.')
    if loaded:
        import torch
        from torchvision import transforms
        net,names,torch,input_size=loaded
        x=transforms.Compose([transforms.Resize((input_size,input_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(image).unsqueeze(0)
        with torch.inference_mode(): probs=torch.softmax(net(x)[0],dim=0).numpy()
        idx=int(np.argmax(probs)); conf=float(probs[idx]); name=names[idx]; abstained=conf<THRESHOLD
        probabilities=sorted([{'class_name':n,'confidence':round(float(p),4)} for n,p in zip(names,probs)],key=lambda x:x['confidence'],reverse=True)[:3]
    if abstained: name=None
    advisory=advice_for(name,language) if name else None
    advisory_en=advice_for(name,'en') if name else None
    advisory_ta=advice_for(name,'ta') if name else None
    extension=Path(file.filename or '').suffix.lower()
    if extension not in ('.jpg','.jpeg','.png','.webp'): extension='.img'
    item={'id':str(uuid.uuid4()),'created_at':datetime.now(timezone.utc).isoformat(),'filename':f"leaf-{uuid.uuid4().hex[:8]}{extension}",'class_name':name,'confidence':round(conf,4),'abstained':abstained,'affected_pct':pct,'severity':sev,'language':language,'advisory':advisory}
    if USE_SUPABASE:
        supabase_client().table(SUPABASE_TABLE).insert({**item, 'advisory': advisory}).execute()
    else:
        with connect() as c:
            c.execute('INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?)',(item['id'],item['created_at'],item['filename'],item['class_name'],item['confidence'],int(item['abstained']),pct,sev,language,json.dumps(advisory,ensure_ascii=False))); c.commit()
    return item | {'advisory_en':advisory_en,'advisory_ta':advisory_ta,'top3':probabilities,'model_ready':loaded is not None,'message':('No trained model is available. Add verified labeled leaf images and train before classification.' if loaded is None else ('Low confidence; please upload a clearer leaf photo.' if abstained else ''))}

@app.on_event('startup')
def startup():
    if not USE_SUPABASE: init_db()
    if MODEL.exists() or MODEL_URL:
        # Warm up and validate the exact artifact before readiness health checks pass.
        load_model()

if (ROOT / 'frontend/dist').exists(): app.mount('/',StaticFiles(directory=ROOT / 'frontend/dist',html=True),name='frontend')
