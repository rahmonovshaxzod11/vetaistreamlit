"""
VetAI — Streamlit App
=====================
Ishga tushirish:
    streamlit run app.py

O'rnatish:
    pip install streamlit torch torchvision pillow youtubesearchpython
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import time

# ─────────────────────────────────────────────
#  PAGE CONFIG — eng birinchi bo'lishi shart!
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VetAI — Hayvon Salomatligi",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── RESET ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c0a !important;
    color: #e8f5e9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #080c0a !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }

/* ── SCROLLBAR ──────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0f1a14; }
::-webkit-scrollbar-thumb { background: #2d5a3d; border-radius: 2px; }

/* ── FILE UPLOADER ──────────────────────── */
[data-testid="stFileUploader"] {
    background: #0f1a14 !important;
    border: 2px dashed rgba(74,222,128,0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(74,222,128,0.5) !important;
    background: rgba(74,222,128,0.03) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #4ade80 !important;
}

/* ── BUTTONS ────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    letter-spacing: -0.2px !important;
    box-shadow: 0 4px 20px rgba(34,197,94,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(34,197,94,0.35) !important;
}

/* ── PROGRESS BAR ───────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #22c55e, #4ade80) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 4px !important;
}

/* ── TABS ───────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1a14 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(74,222,128,0.08) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: #1a2e22 !important;
    color: #4ade80 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px !important;
}

/* ── SPINNER ────────────────────────────── */
.stSpinner > div {
    border-top-color: #4ade80 !important;
}

/* ── IMAGE ──────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(74,222,128,0.15) !important;
}

/* ── DIVIDER ────────────────────────────── */
hr {
    border-color: rgba(74,222,128,0.1) !important;
    margin: 28px 0 !important;
}

/* ── COLUMNS GAP ────────────────────────── */
[data-testid="column"] {
    padding: 0 8px !important;
}

/* ── METRIC ─────────────────────────────── */
[data-testid="stMetric"] {
    background: #0f1a14 !important;
    border: 1px solid rgba(74,222,128,0.1) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #4ade80 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KASALLIK MA'LUMOTLAR BAZASI
# ─────────────────────────────────────────────
DISEASE_DB = {
    "Dental Disease in Dog": {
        "uz": "Tish kasalligi — It", "animal": "🐕", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Og'iz hidi", "Ovqatdan bosh tortish", "Tish tushishi", "Qizil milk"],
        "action": "Professional tish tozalash uchun veterinarga boring. Anesteziya ostida tish tashlanishi mumkin.",
        "prevention": "Kunlik tish yuvish, maxsus tish chaynagichlari va quruq dental ozuqa bering.",
        "youtube": "dog dental disease treatment vet guide",
    },
    "Distemper in Dog": {
        "uz": "Çuma (Distemper) — It", "animal": "🐕", "risk": "HIGH", "risk_uz": "YUQORI",
        "color": "#dc2626", "bg": "rgba(220,38,38,0.08)",
        "symptoms": ["Isitma", "Burun oqishi", "Ko'z yoshi", "Yo'tal", "Talvasa"],
        "action": "SHOSHILINCH! Davo yo'q — simptomatik davolash. Kasal itni zudlik bilan izolyatsiya qiling.",
        "prevention": "Har yili CDV (DHPP) emlash majburiy. 6-8-12 haftalik bolakaylarga vaksinatsiya.",
        "youtube": "distemper in dogs symptoms treatment recovery",
    },
    "Eye Infection in Dog": {
        "uz": "Ko'z infeksiyasi — It", "animal": "🐕", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Ko'z qizarishi", "Ko'z yoshi", "Yiring", "Qovoq shishi"],
        "action": "Steril ko'z yuvish eritmasi bilan yuving. Antibiotikli ko'z tomizg'isi uchun veterinarga boring.",
        "prevention": "Ko'zni begona jismlardan saqlang, muntazam ko'z tekshiruvi o'tkazing.",
        "youtube": "dog eye infection conjunctivitis treatment vet",
    },
    "Fungal Infection in Dog": {
        "uz": "Qo'ziqorin infeksiyasi — It", "animal": "🐕", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Teri toshmasi", "Qalinlashgan teri", "Qichish", "Yoqimsiz hid"],
        "action": "Antifungal shampun (Ketoconazole 2%) va og'iz antifungal dori kerak. Veterinar retsepti zarur.",
        "prevention": "Quruq, toza muhit saqlang. Namlik va immunitetni pasaytiruvchi holatlardan saqlang.",
        "youtube": "fungal infection dog skin yeast treatment",
    },
    "Hot Spots in Dog": {
        "uz": "Issiq dog'lar (Pyoderma) — It", "animal": "🐕", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Qizil, nam yara", "Qichish", "Jun to'kilishi", "Shish"],
        "action": "Soha junini qirqing, antiseptik bilan tozalang. Antibiotikli krem va Elizabeth yoqasi kerak.",
        "prevention": "Terini quruq ushlab turing. Tirnash sababini (burgalar, allergen) davolang.",
        "youtube": "hot spots in dogs treatment home remedy vet",
    },
    "Kennel Cough in Dog": {
        "uz": "Kennel yo'tali — It", "animal": "🐕", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Qattiq yo'tal", "G'o'ng'illash", "Burun oqishi", "Letargiya"],
        "action": "Dam oldiring, boshqa itlardan izolyatsiya qiling. Og'ir holda antibiotik buyuriladi.",
        "prevention": "Bordetella vaksinasi. Ko'p itlar bo'lgan joylarda xavf yuqori.",
        "youtube": "kennel cough in dogs home treatment recovery",
    },
    "Mange in Dog": {
        "uz": "Manj (Qo'tir) — It", "animal": "🐕", "risk": "HIGH", "risk_uz": "YUQORI",
        "color": "#dc2626", "bg": "rgba(220,38,38,0.08)",
        "symptoms": ["Jun to'kilishi", "Qattiq qichish", "Teri qorayishi", "Yara"],
        "action": "IZOLYATSIYA ZARUR! Teri biopsiyasi uchun veterinar ko'rigi zarur. Ivermectin bilan davolash.",
        "prevention": "Kasal hayvonlar bilan kontaktni oldini oling. Uy va to'shakni dezinfektsiya qiling.",
        "youtube": "mange in dogs treatment ivermectin demodectic",
    },
    "Parvovirus in Dog": {
        "uz": "Parvovirus — It", "animal": "🐕", "risk": "HIGH", "risk_uz": "YUQORI",
        "color": "#dc2626", "bg": "rgba(220,38,38,0.08)",
        "symptoms": ["Qon aralash ich ketish", "Qusish", "Letargiya", "Ishtaha yo'qligi"],
        "action": "HAYOT XAVFI! Darhol veterinarga. Stasionar davolash, IV suyuqlik, antibiotik majburiy.",
        "prevention": "DHPP vaksinatsiyasi qat'iy. 6 haftalikdan boshlab, har yili takrorlash.",
        "youtube": "parvovirus in dogs treatment survival guide",
    },
    "Skin Allergy in Dog": {
        "uz": "Teri allergiyasi — It", "animal": "🐕", "risk": "LOW", "risk_uz": "PAST",
        "color": "#16a34a", "bg": "rgba(22,163,74,0.08)",
        "symptoms": ["Qichish", "Qizarish", "Toshma", "Quloq infeksiyasi"],
        "action": "Allergeni aniqlash uchun veterinar sinovlari. Antihistamin yoki ozuqa o'zgartirish kerak.",
        "prevention": "Allergenlardan (chang, ba'zi ozuqalar) uzoq turing. Muntazam cho'miltirib turing.",
        "youtube": "dog skin allergy treatment itching relief atopy",
    },
    "Tick Infestation in Dog": {
        "uz": "Kana infestatsiyasi — It", "animal": "🐕", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Ko'rinadigan kanalar", "Bezovtalik", "Isitma", "Letargiya"],
        "action": "Kanani pinset bilan tekis tortib oling. Borrelia tekshiruvi uchun veterinarga boring.",
        "prevention": "Har oyda kana dori (NexGard, Frontline). O't-o'lanlar orasida yurmaslik.",
        "youtube": "tick removal dog treatment lyme disease prevention",
    },
    "Worm Infection in Dog": {
        "uz": "Qurt kasalligi — It", "animal": "🐕", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Ich kelishi", "Qusish", "Vazn yo'qotish", "Axlatda qurtlar"],
        "action": "Darhol veterinarga murojaat. Antiparazitar dori (Milbemax, Drontal) kerak.",
        "prevention": "Har 3 oyda gijja dori bering. Xom go'sht bermang.",
        "youtube": "worm infection in dogs deworming treatment roundworm",
    },
    "Dental Disease in Cat": {
        "uz": "Tish kasalligi — Mushuk", "animal": "🐱", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Og'iz hidi", "Ovqatdan bosh tortish", "Oqish", "Tish tushishi"],
        "action": "Professional tish tozalash uchun veterinarga boring. NSAIDlar og'riqni kamaytiradi.",
        "prevention": "Kunlik tish yuvish, maxsus dental qo'shimchalar va quruq ozuqa.",
        "youtube": "cat dental disease tooth resorption treatment vet",
    },
    "Ear Mites in Cat": {
        "uz": "Quloq kanasi — Mushuk", "animal": "🐱", "risk": "LOW", "risk_uz": "PAST",
        "color": "#16a34a", "bg": "rgba(22,163,74,0.08)",
        "symptoms": ["Quloq qashish", "Qoramtir axlat", "Bosh silkitish", "Quloq hidi"],
        "action": "Quloq tomizg'isi (Otodex, Milbemite) va quloqni tozalash kerak.",
        "prevention": "Muntazam quloq tekshiruvi. Ko'chada yurgan mushukda xavf yuqori.",
        "youtube": "ear mites in cats treatment home remedy otodectes",
    },
    "Eye Infection in Cat": {
        "uz": "Ko'z infeksiyasi — Mushuk", "animal": "🐱", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Ko'z yoshi", "Yiring", "Qizarish", "Ko'z ochib bo'lmaslik"],
        "action": "Antibiotikli ko'z tomizg'isi uchun veterinarga boring. Herpesvirus uchun antiviral ham kerak.",
        "prevention": "Yangi mushukni karantin qiling. Barcha mushuklar FVRCP vaksinatsiyasi.",
        "youtube": "cat eye infection conjunctivitis herpesvirus treatment",
    },
    "Feline Leukemia": {
        "uz": "Mushuk leykemiyasi (FeLV)", "animal": "🐱", "risk": "HIGH", "risk_uz": "YUQORI",
        "color": "#dc2626", "bg": "rgba(220,38,38,0.08)",
        "symptoms": ["Letargiya", "Vazn yo'qotish", "Isitma", "Teri infeksiyalari"],
        "action": "DARHOL veterinarga! Davolash yo'q, hayot sifatini oshirish muhim. Izolyatsiya qiling.",
        "prevention": "FeLV vaksinatsiyasi majburiy. Mushukni faqat ichkarida saqlang.",
        "youtube": "feline leukemia FeLV management care prognosis",
    },
    "Feline Panleukopenia": {
        "uz": "Mushuk panleykopenyasi (FPV)", "animal": "🐱", "risk": "HIGH", "risk_uz": "YUQORI",
        "color": "#dc2626", "bg": "rgba(220,38,38,0.08)",
        "symptoms": ["Qusish", "Ich ketish", "Yuqori isitma", "Xavfli letargiya"],
        "action": "HAYOT XAVFI! Darhol veterinarga. Stasionar davolash, IV suyuqlik, immunoglobulin kerak.",
        "prevention": "FVRCP vaksinatsiyasi qat'iy. 8-16 haftalikda emlash.",
        "youtube": "feline panleukopenia treatment recovery survival",
    },
    "Fungal Infection in Cat": {
        "uz": "Qo'ziqorin infeksiyasi — Mushuk", "animal": "🐱", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Jun to'kilishi", "Teri qichishi", "Qazish", "Dog'lar"],
        "action": "Antifungal dori (Itraconazole yoki Terbinafine) uchun veterinarga boring. 6-8 hafta davolash.",
        "prevention": "Namlikni kamaytiring, immunitetni kuchaytirib turing.",
        "youtube": "fungal infection cat skin ringworm treatment antifungal",
    },
    "Ringworm in Cat": {
        "uz": "Qo'ng'iroq qurt (Dermatofitoz) — Mushuk", "animal": "🐱", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Doira shaklidagi dog'lar", "Jun to'kilishi", "Teri qichishi"],
        "action": "Antifungal shampun (Miconazole) va og'iz dori. Uy dezinfektsiyasi — ODAMLARGA YUQADI!",
        "prevention": "Yangi hayvonlarni karantin qiling. Immunitetni kuchaytirib turing.",
        "youtube": "ringworm in cats treatment antifungal shampoo vet",
    },
    "Scabies in Cat": {
        "uz": "Qo'tir (Scabies) — Mushuk", "animal": "🐱", "risk": "HIGH", "risk_uz": "YUQORI",
        "color": "#dc2626", "bg": "rgba(220,38,38,0.08)",
        "symptoms": ["Kuchli qichish", "Jun to'kilishi", "Teri yalang'ochligi", "Yara"],
        "action": "IZOLYATSIYA! Ivermectin yoki Selamectin bilan davolash. ODAMLARGA YUQADI!",
        "prevention": "Kasal hayvonlar bilan kontaktni to'xtatib, uy va to'shakni dezinfektsiya qiling.",
        "youtube": "scabies in cats notoedric mange treatment vet",
    },
    "Skin Allergy in Cat": {
        "uz": "Teri allergiyasi — Mushuk", "animal": "🐱", "risk": "LOW", "risk_uz": "PAST",
        "color": "#16a34a", "bg": "rgba(22,163,74,0.08)",
        "symptoms": ["Qichish", "Toshma", "Jun yutish", "Teri qizarishi"],
        "action": "Veterinar allergen testini o'tkazsin. Ozuqa allergiyasi bo'lsa — gidrolizat ozuqaga o'ting.",
        "prevention": "Yangi ozuqa, yostiq, detergentlardan keyin kuzating.",
        "youtube": "cat skin allergy treatment food allergy itching",
    },
    "Urinary Tract Infection in Cat": {
        "uz": "Siydik yo'li infeksiyasi — Mushuk", "animal": "🐱", "risk": "HIGH", "risk_uz": "YUQORI",
        "color": "#dc2626", "bg": "rgba(220,38,38,0.08)",
        "symptoms": ["Peshob qila olmaslik", "Qon aralash siydik", "Tez-tez siydik", "Qorin og'rig'i"],
        "action": "SHOSHILINCH! Qila olmasa — hayot xavfi. Darhol veterinarga. Antibiotik va tekshirish kerak.",
        "prevention": "Ko'proq suv ichirish, nam ozuqa bering. Siydik qumiga e'tibor bering.",
        "youtube": "cat urinary tract infection FLUTD blockage treatment",
    },
    "Worm Infection in Cat": {
        "uz": "Qurt kasalligi — Mushuk", "animal": "🐱", "risk": "MEDIUM", "risk_uz": "O'RTA",
        "color": "#d97706", "bg": "rgba(217,119,6,0.08)",
        "symptoms": ["Ich kelishi", "Qusish", "Vazn yo'qotish", "Dam bo'lish"],
        "action": "Gijja dorisi (Milbemax, Profender) bering. 2 hafta o'tib takrorlash kerak.",
        "prevention": "Har 3 oyda profilaktik gijja dori. Xom go'sht bermang.",
        "youtube": "worm infection cats deworming tapeworm roundworm treatment",
    },
}

CLASS_NAMES = sorted(DISEASE_DB.keys())

# ─────────────────────────────────────────────
#  MODEL YUKLASH
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))
        net.load_state_dict(torch.load("best_model.pth", map_location=device))
        net = net.to(device)
        net.eval()
        return net, device, None
    except Exception as e:
        return None, None, str(e)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(img: Image.Image, model, device):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], probs.cpu().tolist()

def get_youtube_url(query):
    return f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"

def try_youtube_embed(query):
    try:
        from youtubesearchpython import VideosSearch
        s = VideosSearch(query, limit=4)
        results = s.result().get("result", [])
        return results
    except:
        return []

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: rgba(15,26,20,0.8);
    border-bottom: 1px solid rgba(74,222,128,0.1);
    padding: 18px 0;
    margin: -80px -80px 40px -80px;
    backdrop-filter: blur(12px);
">
    <div style="max-width: 1200px; margin: 0 auto; padding: 0 40px;
                display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="
                width: 42px; height: 42px; border-radius: 10px;
                background: rgba(74,222,128,0.1);
                border: 1px solid rgba(74,222,128,0.2);
                display: flex; align-items: center; justify-content: center;
                font-size: 20px;
            ">🌿</div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:800;
                            color:#f0fdf4; letter-spacing:-0.5px;">VetAI</div>
                <div style="font-size:10px; color:#4ade80; letter-spacing:2px;
                            text-transform:uppercase; font-weight:600;">Hayvon Salomatligi</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:24px;">
            <span style="font-size:12px; color:#6b7280; font-weight:500;">ResNet18 Model</span>
            <div style="
                background: rgba(74,222,128,0.08);
                border: 1px solid rgba(74,222,128,0.2);
                color: #4ade80; padding: 6px 14px; border-radius: 20px;
                font-size: 12px; font-weight: 700;
            ">
                <span style="animation: pulse 2s infinite; display:inline-block;">●</span>
                22 Kasallik
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL YUKLASH
# ─────────────────────────────────────────────
with st.spinner(""):
    model, device, model_error = load_model()

if model_error:
    st.markdown(f"""
    <div style="background:rgba(220,38,38,0.08); border:1px solid rgba(220,38,38,0.3);
                border-radius:14px; padding:20px 24px; margin-bottom:24px; color:#fca5a5;">
        <div style="font-weight:700; font-size:16px; margin-bottom:8px;">❌ Model yuklanmadi</div>
        <div style="font-size:13px; line-height:1.7;">{model_error}</div>
        <div style="margin-top:10px; font-size:12px; color:#f87171;">
            <code style="background:rgba(0,0,0,0.3); padding:2px 8px; border-radius:4px;">
            best_model.pth</code> faylini app.py bilan bir papkaga qo'ying
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO SECTION
# ─────────────────────────────────────────────
col_hero, col_upload = st.columns([1.1, 1], gap="large")

with col_hero:
    st.markdown("""
    <div style="padding: 20px 0 40px; animation: slideIn 0.6s ease;">
        <div style="
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.2);
            color: #4ade80; padding: 6px 14px; border-radius: 20px;
            font-size: 12px; font-weight: 600; margin-bottom: 24px;
            letter-spacing: 0.3px;
        ">
            <span style="width:7px; height:7px; border-radius:50%; background:#4ade80;
                         display:inline-block; animation:pulse 2s infinite;"></span>
            AI · PyTorch · Real-vaqt tashxis
        </div>

        <h1 style="
            font-family: 'Syne', sans-serif;
            font-size: 52px; font-weight: 900;
            line-height: 1.05; margin: 0 0 20px;
            color: #f0fdf4; letter-spacing: -2px;
        ">
            Hayvonlaringiz<br>
            <span style="color: #4ade80; font-style: italic;">kasalligini</span><br>
            aniqlaymiz
        </h1>

        <p style="
            font-size: 16px; color: #6b7280; line-height: 1.7;
            max-width: 460px; margin: 0 0 36px;
        ">
            Rasm yuklang — ResNet18 model darhol kasallikni aniqlaydi,
            xavf darajasini baholaydi va davolash maslahatlarini beradi.
        </p>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
    """ + "".join([f"""
            <div style="border-left: 2px solid rgba(74,222,128,0.2); padding-left: 14px;">
                <div style="font-family:'Syne',sans-serif; font-size:24px; font-weight:800;
                            color:#f0fdf4; letter-spacing:-1px;">{n}</div>
                <div style="font-size:10px; color:#6b7280; margin-top:2px; font-weight:500;">{l}</div>
            </div>
    """ for n, l in [("22","Kasallik"), ("ResNet18","Model"), ("98%","Aniqlik"), ("24/7","Mavjud")]]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_upload:
    st.markdown("""
    <div style="
        background: #0f1a14;
        border: 1px solid rgba(74,222,128,0.1);
        border-radius: 20px;
        padding: 28px;
        margin-top: 20px;
    ">
        <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:700;
                    color:#f0fdf4; margin-bottom:6px;">📷 Rasm Yuklang</div>
        <div style="font-size:13px; color:#6b7280; margin-bottom:20px;">
            Ta'sirlangan joyning aniq rasmini yuklang
        </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "heic", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded and model:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        diagnose_btn = st.button("🔍 Tashxis Olish", key="diagnose")
    else:
        diagnose_btn = False

# ─────────────────────────────────────────────
#  TASHXIS
# ─────────────────────────────────────────────
if diagnose_btn and uploaded and model:
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner("ResNet18 tahlil qilmoqda..."):
        time.sleep(0.5)
        img = Image.open(uploaded).convert("RGB")
        pred_class, probs = predict(img, model, device)

    info = DISEASE_DB[pred_class]
    confidence = max(probs)
    top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]

    # ── ASOSIY NATIJA ────────────────────────
    st.markdown(f"""
    <div style="
        background: #0f1a14;
        border: 1px solid rgba(74,222,128,0.1);
        border-left: 4px solid {info['color']};
        border-radius: 16px; padding: 28px;
        margin-bottom: 20px;
        animation: slideIn 0.5s ease;
    ">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;
                    flex-wrap:wrap; gap:16px;">
            <div>
                <div style="font-size:11px; color:#6b7280; font-weight:600;
                            letter-spacing:0.5px; text-transform:uppercase; margin-bottom:8px;">
                    Aniqlanган kasallik
                </div>
                <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
                            color:#f0fdf4; letter-spacing:-0.5px; line-height:1.1;">
                    {info['animal']} {info['uz']}
                </div>
                <div style="font-size:13px; color:#6b7280; margin-top:6px;">{pred_class}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:#6b7280; font-weight:600;
                            letter-spacing:0.5px; text-transform:uppercase; margin-bottom:4px;">
                    Ishonch
                </div>
                <div style="font-family:'Syne',sans-serif; font-size:48px; font-weight:900;
                            color:{info['color']}; letter-spacing:-2px; line-height:1;">
                    {confidence*100:.1f}%
                </div>
            </div>
        </div>

        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:20px;">
            <div style="
                background:{info['bg']}; border:1.5px solid {info['color']};
                color:{info['color']}; padding:8px 18px; border-radius:20px;
                font-size:13px; font-weight:700;
            ">
                {'🔴' if info['risk']=='HIGH' else '🟡' if info['risk']=='MEDIUM' else '🟢'}
                Xavf: {info['risk_uz']}
            </div>
            <div style="
                background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                color:#9ca3af; padding:8px 18px; border-radius:20px; font-size:13px; font-weight:600;
            ">
                🌿 ResNet18 · PyTorch
            </div>
        </div>

        <div style="margin-top:22px;">
            <div style="font-size:11px; color:#6b7280; font-weight:600;
                        letter-spacing:0.5px; text-transform:uppercase; margin-bottom:10px;">
                Asosiy belgilar
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:8px;">
                {''.join([f"""<span style="background:rgba(74,222,128,0.07); border:1px solid rgba(74,222,128,0.15);
                color:#4ade80; padding:5px 14px; border-radius:20px; font-size:12px; font-weight:600;">{s}</span>"""
                for s in info['symptoms']])}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["💊 Davolash Ma'lumoti", "📊 Top 5 Natija", "📺 YouTube Videolar"])

    # TAB 1: INFO
    with tab1:
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown(f"""
            <div style="background:#0f1a14; border:1px solid rgba(74,222,128,0.1);
                        border-radius:14px; padding:22px; height:100%;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
                    <div style="width:36px; height:36px; border-radius:10px;
                                background:rgba(74,222,128,0.1); display:flex;
                                align-items:center; justify-content:center; font-size:18px;">💊</div>
                    <span style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                                 color:#4ade80;">Nima Qilish Kerak?</span>
                </div>
                <p style="font-size:14px; color:#9ca3af; line-height:1.7; margin:0;">
                    {info['action']}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div style="background:#0f1a14; border:1px solid rgba(74,222,128,0.1);
                        border-radius:14px; padding:22px; height:100%;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
                    <div style="width:36px; height:36px; border-radius:10px;
                                background:rgba(96,165,250,0.1); display:flex;
                                align-items:center; justify-content:center; font-size:18px;">🛡️</div>
                    <span style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                                 color:#60a5fa;">Oldini Olish</span>
                </div>
                <p style="font-size:14px; color:#9ca3af; line-height:1.7; margin:0;">
                    {info['prevention']}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # DISCLAIMER
        st.markdown("""
        <div style="background:rgba(217,119,6,0.06); border:1px solid rgba(217,119,6,0.15);
                    border-radius:10px; padding:12px 16px; font-size:12px; color:#d97706;
                    line-height:1.6; margin-top:16px;">
            ⚠️ Bu AI tomonidan berilgan dastlabki tashxis. Professional veterinar ko'rigini almashtira olmaydi.
        </div>
        """, unsafe_allow_html=True)

    # TAB 2: TOP 5
    with tab2:
        st.markdown("""
        <div style="background:#0f1a14; border:1px solid rgba(74,222,128,0.1);
                    border-radius:14px; padding:24px;">
            <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700;
                        color:#f0fdf4; margin-bottom:20px;">📊 Eng Ehtimoliy Natijalar</div>
        """, unsafe_allow_html=True)

        for rank, (idx, prob) in enumerate(top5):
            cls_name = CLASS_NAMES[idx]
            d = DISEASE_DB[cls_name]
            is_top = rank == 0
            bar_color = "#4ade80" if is_top else "#374151"
            text_color = "#f0fdf4" if is_top else "#9ca3af"
            font_weight = "800" if is_top else "400"
            pct = prob * 100

            st.markdown(f"""
            <div style="margin-bottom:18px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <div style="width:24px; height:24px; border-radius:50%; display:flex;
                                    align-items:center; justify-content:center; font-size:11px;
                                    font-weight:800; background:{'#4ade80' if is_top else '#1f2d25'};
                                    color:{'#000' if is_top else '#6b7280'};">
                            {'✓' if is_top else rank+1}
                        </div>
                        <span style="font-size:14px; font-weight:{font_weight}; color:{text_color};">
                            {d['animal']} {d['uz']}
                        </span>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-size:11px; font-weight:700; padding:3px 10px;
                                     border-radius:10px; background:{d['bg']}; color:{d['color']};">
                            {d['risk_uz']}
                        </span>
                        <span style="font-size:15px; font-weight:800; color:{'#4ade80' if is_top else '#6b7280'};">
                            {pct:.1f}%
                        </span>
                    </div>
                </div>
                <div style="height:5px; background:rgba(255,255,255,0.05); border-radius:3px; overflow:hidden;">
                    <div style="height:100%; width:{pct:.1f}%; background:{bar_color};
                                border-radius:3px; transition:width 1s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 3: YOUTUBE
    with tab3:
        yt_url = get_youtube_url(info['youtube'])

        st.markdown(f"""
        <div style="background:rgba(255,0,0,0.05); border:1px solid rgba(255,0,0,0.12);
                    border-radius:12px; padding:16px 20px; margin-bottom:20px;
                    display:flex; align-items:center; gap:14px;">
            <span style="font-size:24px;">📺</span>
            <div>
                <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:#f0fdf4;">
                    Davolash Videolari
                </div>
                <div style="font-size:12px; color:#6b7280; margin-top:2px;">
                    "{info['uz']}" bo'yicha videolar
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        videos = try_youtube_embed(info['youtube'])

        if videos:
            cols = st.columns(2, gap="medium")
            for i, v in enumerate(videos[:4]):
                with cols[i % 2]:
                    thumb = (v.get("thumbnails") or [{}])[0].get("url", "")
                    title = v.get("title", "")
                    channel = v.get("channel", {}).get("name", "")
                    duration = v.get("duration", "")
                    views = v.get("viewCount", {}).get("short", "")
                    url = v.get("link", "")

                    st.markdown(f"""
                    <a href="{url}" target="_blank" style="text-decoration:none;">
                        <div style="
                            background:#0f1a14; border:1px solid rgba(255,255,255,0.06);
                            border-radius:12px; overflow:hidden; margin-bottom:12px;
                            transition:all 0.2s; cursor:pointer;
                        " onmouseover="this.style.borderColor='rgba(255,255,255,0.15)'"
                           onmouseout="this.style.borderColor='rgba(255,255,255,0.06)'">
                            <div style="position:relative; padding-top:56.25%; background:#0a0f0d;">
                                {'<img src="'+thumb+'" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;">' if thumb else '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:36px;color:#374151;">▶</div>'}
                                <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                                            width:44px;height:44px;background:rgba(255,255,255,0.9);
                                            border-radius:50%;display:flex;align-items:center;
                                            justify-content:center;font-size:14px;color:#000;font-weight:700;">▶</div>
                                {f'<div style="position:absolute;bottom:6px;right:6px;background:rgba(0,0,0,0.8);color:#fff;font-size:11px;padding:2px 6px;border-radius:4px;font-weight:700;">{duration}</div>' if duration else ''}
                            </div>
                            <div style="padding:12px 14px;">
                                <div style="font-size:13px;font-weight:600;color:#e5e7eb;
                                            line-height:1.4;margin-bottom:6px;
                                            display:-webkit-box;-webkit-line-clamp:2;
                                            -webkit-box-orient:vertical;overflow:hidden;">
                                    {title}
                                </div>
                                <div style="font-size:11px;color:#6b7280;">
                                    {channel}{f' · {views} ko'+"'"+'rish' if views else ''}
                                </div>
                            </div>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center; padding:40px 0;">
                <div style="font-size:40px; margin-bottom:12px;">📺</div>
                <div style="color:#6b7280; font-size:14px; margin-bottom:20px;">
                    Videolar yuklanmadi yoki youtubesearchpython o'rnatilmagan
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <a href="{yt_url}" target="_blank" style="text-decoration:none;">
            <div style="
                text-align:center; padding:14px; background:rgba(255,0,0,0.07);
                border:1px solid rgba(255,0,0,0.15); color:#f87171;
                border-radius:10px; font-size:14px; font-weight:600; cursor:pointer;
                margin-top:8px;
            ">
                📺 YouTube'da Ko'proq Videolar →
            </div>
        </a>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL YUKLANMAGAN
# ─────────────────────────────────────────────
elif not model and not uploaded:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#374151;">
        <div style="font-size:48px; margin-bottom:16px;">🔬</div>
        <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:700;
                    color:#4b5563; margin-bottom:8px;">
            Rasm yuklang va tashxis oling
        </div>
        <div style="font-size:14px; color:#374151;">
            Yuqoridagi maydondan hayvon rasmini yuklang
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FEATURES SECTION
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
            color:#f0fdf4; text-align:center; margin-bottom:28px; letter-spacing:-0.5px;">
    Qanday ishlaydi?
</div>
""", unsafe_allow_html=True)

feat_cols = st.columns(4, gap="medium")
features = [
    ("📸", "Vizual Tahlil", "Rasm yuklang — AI vizual belgilarni aniqlaydi", "#4ade80"),
    ("🤖", "ResNet18 Model", "PyTorch bilan train qilingan 22 ta kasallik uchun maxsus model", "#60a5fa"),
    ("⚡", "Darhol Natija", "Soniyalar ichida aniq tashxis va simptomlar ro'yxati", "#f59e0b"),
    ("📺", "YouTube", "Kasallikka oid real shifokor videolarini avtomatik topadi", "#f472b6"),
]
for col, (ic, t, d, c) in zip(feat_cols, features):
    with col:
        st.markdown(f"""
        <div style="background:#0f1a14; border:1px solid rgba(255,255,255,0.05);
                    border-radius:16px; padding:22px 18px; text-align:center; height:100%;">
            <div style="width:52px; height:52px; border-radius:14px; margin:0 auto 14px;
                        background:{c}18; border:1px solid {c}33;
                        display:flex; align-items:center; justify-content:center; font-size:26px;">
                {ic}
            </div>
            <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700;
                        color:{c}; margin-bottom:8px;">{t}</div>
            <div style="font-size:12px; color:#6b7280; line-height:1.6;">{d}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    border-top: 1px solid rgba(74,222,128,0.08);
    padding: 24px 0; margin-top: 60px;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 12px;
">
    <span style="font-family:'Syne',sans-serif; font-weight:800; color:#4ade80; font-size:16px;">
        🌿 VetAI
    </span>
    <span style="font-size:12px; color:#374151;">
        © 2025 · ResNet18 · PyTorch · Streamlit · 22 ta kasallik
    </span>
    <span style="font-size:12px; color:#374151;">
        ⚠️ Faqat dastlabki tashxis — veterinar o'rnini bosmaydi
    </span>
</div>
""", unsafe_allow_html=True)
