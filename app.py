"""
VetAI — Streamlit App (Fixed for Streamlit Cloud)
pip install streamlit torch torchvision pillow pillow-heif
streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time
import json
import os
from datetime import datetime
import re

st.set_page_config(
    page_title="VetAI — Hayvon Salomatligi",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_DATA_FILE = "users.json"

def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f, indent=2)

def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "subscription" not in st.session_state:
        st.session_state.subscription = "free"
    if "free_uses_left" not in st.session_state:
        st.session_state.free_uses_left = 1
    if "predictions_made" not in st.session_state:
        st.session_state.predictions_made = 0
    if "show_register" not in st.session_state:
        st.session_state.show_register = False

init_session_state()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #080c0a !important;
    color: #e8f5e9 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background: #080c0a !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
.stDeployButton { display: none !important; }

[data-testid="stFileUploader"] {
    background: #0f1a14 !important;
    border: 2px dashed rgba(74,222,128,0.25) !important;
    border-radius: 16px !important;
}
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
    box-shadow: 0 4px 20px rgba(34,197,94,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(34,197,94,0.35) !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #22c55e, #4ade80) !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0f1a14 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(74,222,128,0.08) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1a2e22 !important;
    color: #4ade80 !important;
}
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(74,222,128,0.15) !important;
}
[data-testid="stMetric"] {
    background: #0f1a14 !important;
    border: 1px solid rgba(74,222,128,0.1) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
hr { border-color: rgba(74,222,128,0.1) !important; }

.login-container {
    max-width: 420px;
    margin: 0 auto;
    padding: 40px 20px;
}
.login-card {
    background: linear-gradient(135deg, #0f1a14 0%, #0a110d 100%);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 24px;
    padding: 32px;
}
.sub-card {
    background: linear-gradient(135deg, #0f1a14 0%, #0a110d 100%);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 20px;
    padding: 24px;
    transition: all 0.3s;
}
.sub-card:hover {
    transform: translateY(-4px);
    border-color: rgba(74,222,128,0.4);
    box-shadow: 0 12px 32px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

DISEASE_DB = {
    "Dental Disease in Dog":        {"uz":"Tish kasalligi — It","animal":"🐕","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Og'iz hidi","Ovqatdan bosh tortish","Tish tushishi","Qizil milk"],"action":"Professional tish tozalash uchun veterinarga boring. Anesteziya ostida tish tashlanishi mumkin.","prevention":"Kunlik tish yuvish, maxsus tish chaynagichlari va quruq dental ozuqa bering."},
    "Distemper in Dog":             {"uz":"Çuma (Distemper) — It","animal":"🐕","risk":"HIGH","risk_uz":"YUQORI","color":"#dc2626","bg":"rgba(220,38,38,0.08)","symptoms":["Isitma","Burun oqishi","Ko'z yoshi","Yo'tal","Talvasa"],"action":"SHOSHILINCH! Davo yo'q — simptomatik davolash. Kasal itni zudlik bilan izolyatsiya qiling.","prevention":"Har yili CDV (DHPP) emlash majburiy. 6-8-12 haftalik bolakaylarga vaksinatsiya."},
    "Eye Infection in Dog":         {"uz":"Ko'z infeksiyasi — It","animal":"🐕","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Ko'z qizarishi","Ko'z yoshi","Yiring","Qovoq shishi"],"action":"Steril ko'z yuvish eritmasi bilan yuving. Antibiotikli ko'z tomizg'isi uchun veterinarga boring.","prevention":"Ko'zni begona jismlardan saqlang, muntazam ko'z tekshiruvi o'tkazing."},
    "Fungal Infection in Dog":      {"uz":"Qo'ziqorin infeksiyasi — It","animal":"🐕","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Teri toshmasi","Qalinlashgan teri","Qichish","Yoqimsiz hid"],"action":"Antifungal shampun (Ketoconazole 2%) va og'iz antifungal dori kerak. Veterinar retsepti zarur.","prevention":"Quruq, toza muhit saqlang. Namlik va immunitetni pasaytiruvchi holatlardan saqlang."},
    "Hot Spots in Dog":             {"uz":"Issiq dog'lar (Pyoderma) — It","animal":"🐕","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Qizil, nam yara","Qichish","Jun to'kilishi","Shish"],"action":"Soha junini qirqing, antiseptik bilan tozalang. Antibiotikli krem va Elizabeth yoqasi kerak.","prevention":"Terini quruq ushlab turing. Tirnash sababini (burgalar, allergen) davolang."},
    "Kennel Cough in Dog":          {"uz":"Kennel yo'tali — It","animal":"🐕","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Qattiq yo'tal","G'o'ng'illash","Burun oqishi","Letargiya"],"action":"Dam oldiring, boshqa itlardan izolyatsiya qiling. Og'ir holda antibiotik buyuriladi.","prevention":"Bordetella vaksinasi. Ko'p itlar bo'lgan joylarda xavf yuqori."},
    "Mange in Dog":                 {"uz":"Manj (Qo'tir) — It","animal":"🐕","risk":"HIGH","risk_uz":"YUQORI","color":"#dc2626","bg":"rgba(220,38,38,0.08)","symptoms":["Jun to'kilishi","Qattiq qichish","Teri qorayishi","Yara"],"action":"IZOLYATSIYA ZARUR! Teri biopsiyasi uchun veterinar ko'rigi zarur. Ivermectin bilan davolash.","prevention":"Kasal hayvonlar bilan kontaktni oldini oling. Uy va to'shakni dezinfektsiya qiling."},
    "Parvovirus in Dog":            {"uz":"Parvovirus — It","animal":"🐕","risk":"HIGH","risk_uz":"YUQORI","color":"#dc2626","bg":"rgba(220,38,38,0.08)","symptoms":["Qon aralash ich ketish","Qusish","Letargiya","Ishtaha yo'qligi"],"action":"HAYOT XAVFI! Darhol veterinarga. Stasionar davolash, IV suyuqlik, antibiotik majburiy.","prevention":"DHPP vaksinatsiyasi qat'iy. 6 haftalikdan boshlab, har yili takrorlash."},
    "Skin Allergy in Dog":          {"uz":"Teri allergiyasi — It","animal":"🐕","risk":"LOW","risk_uz":"PAST","color":"#16a34a","bg":"rgba(22,163,74,0.08)","symptoms":["Qichish","Qizarish","Toshma","Quloq infeksiyasi"],"action":"Allergeni aniqlash uchun veterinar sinovlari. Antihistamin yoki ozuqa o'zgartirish kerak.","prevention":"Allergenlardan (chang, ba'zi ozuqalar) uzoq turing. Muntazam cho'miltirib turing."},
    "Tick Infestation in Dog":      {"uz":"Kana infestatsiyasi — It","animal":"🐕","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Ko'rinadigan kanalar","Bezovtalik","Isitma","Letargiya"],"action":"Kanani pinset bilan tekis tortib oling. Borrelia tekshiruvi uchun veterinarga boring.","prevention":"Har oyda kana dori (NexGard, Frontline). O't-o'lanlar orasida yurmaslik."},
    "Worm Infection in Dog":        {"uz":"Qurt kasalligi — It","animal":"🐕","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Ich kelishi","Qusish","Vazn yo'qotish","Axlatda qurtlar"],"action":"Darhol veterinarga murojaat. Antiparazitar dori (Milbemax, Drontal) kerak.","prevention":"Har 3 oyda gijja dori bering. Xom go'sht bermang."},
    "Dental Disease in Cat":        {"uz":"Tish kasalligi — Mushuk","animal":"🐱","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Og'iz hidi","Ovqatdan bosh tortish","Oqish","Tish tushishi"],"action":"Professional tish tozalash uchun veterinarga boring. NSAIDlar og'riqni kamaytiradi.","prevention":"Kunlik tish yuvish, maxsus dental qo'shimchalar va quruq ozuqa."},
    "Ear Mites in Cat":             {"uz":"Quloq kanasi — Mushuk","animal":"🐱","risk":"LOW","risk_uz":"PAST","color":"#16a34a","bg":"rgba(22,163,74,0.08)","symptoms":["Quloq qashish","Qoramtir axlat","Bosh silkitish","Quloq hidi"],"action":"Quloq tomizg'isi (Otodex, Milbemite) va quloqni tozalash kerak.","prevention":"Muntazam quloq tekshiruvi. Ko'chada yurgan mushukda xavf yuqori."},
    "Eye Infection in Cat":         {"uz":"Ko'z infeksiyasi — Mushuk","animal":"🐱","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Ko'z yoshi","Yiring","Qizarish","Ko'z ochib bo'lmaslik"],"action":"Antibiotikli ko'z tomizg'isi uchun veterinarga boring. Herpesvirus uchun antiviral ham kerak.","prevention":"Yangi mushukni karantin qiling. Barcha mushuklar FVRCP vaksinatsiyasi."},
    "Feline Leukemia":              {"uz":"Mushuk leykemiyasi (FeLV)","animal":"🐱","risk":"HIGH","risk_uz":"YUQORI","color":"#dc2626","bg":"rgba(220,38,38,0.08)","symptoms":["Letargiya","Vazn yo'qotish","Isitma","Teri infeksiyalari"],"action":"DARHOL veterinarga! Davolash yo'q, hayot sifatini oshirish muhim. Izolyatsiya qiling.","prevention":"FeLV vaksinatsiyasi majburiy. Mushukni faqat ichkarida saqlang."},
    "Feline Panleukopenia":         {"uz":"Mushuk panleykopenyasi (FPV)","animal":"🐱","risk":"HIGH","risk_uz":"YUQORI","color":"#dc2626","bg":"rgba(220,38,38,0.08)","symptoms":["Qusish","Ich ketish","Yuqori isitma","Xavfli letargiya"],"action":"HAYOT XAVFI! Darhol veterinarga. Stasionar davolash, IV suyuqlik, immunoglobulin kerak.","prevention":"FVRCP vaksinatsiyasi qat'iy. 8-16 haftalikda emlash."},
    "Fungal Infection in Cat":      {"uz":"Qo'ziqorin infeksiyasi — Mushuk","animal":"🐱","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Jun to'kilishi","Teri qichishi","Qazish","Dog'lar"],"action":"Antifungal dori (Itraconazole yoki Terbinafine) uchun veterinarga boring. 6-8 hafta davolash.","prevention":"Namlikni kamaytiring, immunitetni kuchaytirib turing."},
    "Ringworm in Cat":              {"uz":"Qo'ng'iroq qurt (Dermatofitoz) — Mushuk","animal":"🐱","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Doira shaklidagi dog'lar","Jun to'kilishi","Teri qichishi"],"action":"Antifungal shampun (Miconazole) va og'iz dori. Uy dezinfektsiyasi — ODAMLARGA YUQADI!","prevention":"Yangi hayvonlarni karantin qiling. Immunitetni kuchaytirib turing."},
    "Scabies in Cat":               {"uz":"Qo'tir (Scabies) — Mushuk","animal":"🐱","risk":"HIGH","risk_uz":"YUQORI","color":"#dc2626","bg":"rgba(220,38,38,0.08)","symptoms":["Kuchli qichish","Jun to'kilishi","Teri yalang'ochligi","Yara"],"action":"IZOLYATSIYA! Ivermectin yoki Selamectin bilan davolash. ODAMLARGA YUQADI!","prevention":"Kasal hayvonlar bilan kontaktni to'xtatib, uy va to'shakni dezinfektsiya qiling."},
    "Skin Allergy in Cat":          {"uz":"Teri allergiyasi — Mushuk","animal":"🐱","risk":"LOW","risk_uz":"PAST","color":"#16a34a","bg":"rgba(22,163,74,0.08)","symptoms":["Qichish","Toshma","Jun yutish","Teri qizarishi"],"action":"Veterinar allergen testini o'tkazsin. Ozuqa allergiyasi bo'lsa — gidrolizat ozuqaga o'ting.","prevention":"Yangi ozuqa, yostiq, detergentlardan keyin kuzating."},
    "Urinary Tract Infection in Cat":{"uz":"Siydik yo'li infeksiyasi — Mushuk","animal":"🐱","risk":"HIGH","risk_uz":"YUQORI","color":"#dc2626","bg":"rgba(220,38,38,0.08)","symptoms":["Peshob qila olmaslik","Qon aralash siydik","Tez-tez siydik","Qorin og'rig'i"],"action":"SHOSHILINCH! Qila olmasa — hayot xavfi. Darhol veterinarga. Antibiotik va tekshirish kerak.","prevention":"Ko'proq suv ichirish, nam ozuqa bering. Siydik qumiga e'tibor bering."},
    "Worm Infection in Cat":        {"uz":"Qurt kasalligi — Mushuk","animal":"🐱","risk":"MEDIUM","risk_uz":"O'RTA","color":"#d97706","bg":"rgba(217,119,6,0.08)","symptoms":["Ich kelishi","Qusish","Vazn yo'qotish","Dam bo'lish"],"action":"Gijja dorisi (Milbemax, Profender) bering. 2 hafta o'tib takrorlash kerak.","prevention":"Har 3 oyda profilaktik gijja dori. Xom go'sht bermang."},
}

CLASS_NAMES = sorted(DISEASE_DB.keys())

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))
        net.load_state_dict(torch.load("best_model.pth", map_location=device))
        net.to(device).eval()
        return net, device, None
    except Exception as e:
        return None, None, str(e)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(img, model, device):
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
    return CLASS_NAMES[probs.argmax().item()], probs.cpu().tolist()

def check_prediction_quota():
    if st.session_state.subscription == "pro":
        return True
    elif st.session_state.subscription == "free":
        if st.session_state.free_uses_left > 0:
            return True
        else:
            return False
    return False

def decrement_quota():
    if st.session_state.subscription == "free":
        st.session_state.free_uses_left -= 1
        users = load_users()
        if st.session_state.current_user in users:
            users[st.session_state.current_user]["free_uses_left"] = st.session_state.free_uses_left
            save_users(users)

def use_prediction():
    if check_prediction_quota():
        st.session_state.predictions_made += 1
        return True
    return False

def register_user(username, password, phone):
    users = load_users()
    if username in users:
        return False, "Bu foydalanuvchi nomi band!"
    if len(password) < 4:
        return False, "Parol kamida 4 belgidan iborat bo'lishi kerak!"
    if not re.match(r'^\+?[0-9]{9,15}$', phone):
        return False, "Telefon raqam noto'g'ri formatda!"
    
    users[username] = {
        "password": password,
        "phone": phone,
        "subscription": "free",
        "free_uses_left": 1,
        "created_at": str(datetime.now()),
        "predictions_total": 0
    }
    save_users(users)
    return True, "Ro'yxatdan o'tish muvaffaqiyatli!"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "Foydalanuvchi topilmadi!"
    if users[username]["password"] != password:
        return False, "Parol noto'g'ri!"
    
    st.session_state.logged_in = True
    st.session_state.current_user = username
    st.session_state.subscription = users[username]["subscription"]
    st.session_state.free_uses_left = users[username]["free_uses_left"]
    return True, "Tizimga kirdingiz!"

def logout_user():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.subscription = "free"
    st.session_state.free_uses_left = 1
    st.session_state.predictions_made = 0
    st.rerun()

def update_subscription(username, sub_type):
    users = load_users()
    if username in users:
        users[username]["subscription"] = sub_type
        if sub_type == "pro":
            users[username]["free_uses_left"] = 999999
        else:
            users[username]["free_uses_left"] = 1
        save_users(users)
        st.session_state.subscription = sub_type
        st.session_state.free_uses_left = users[username]["free_uses_left"]
        return True
    return False

if not st.session_state.logged_in:
    st.markdown("""
    <div style="background:rgba(15,26,20,0.9);border-bottom:1px solid rgba(74,222,128,0.1);
                padding:16px 40px;margin:-80px -80px 40px -80px;backdrop-filter:blur(12px);
                display:flex;align-items:center;justify-content:space-between;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:42px;height:42px;border-radius:10px;background:rgba(74,222,128,0.1);
                        border:1px solid rgba(74,222,128,0.2);display:flex;align-items:center;
                        justify-content:center;font-size:20px;">🌿</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:#f0fdf4;">VetAI</div>
                <div style="font-size:10px;color:#4ade80;letter-spacing:2px;text-transform:uppercase;font-weight:600;">Hayvon Salomatligi</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:16px;">
            <span style="font-size:12px;color:#6b7280;">ResNet18 Model</span>
            <div style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);
                        color:#4ade80;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700;">
                ● 22 Kasallik
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        if not st.session_state.show_register:
            st.markdown("""
            <div class="login-container">
                <div class="login-card">
                    <div style="text-align:center;margin-bottom:32px;">
                        <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#4ade80;">Kirish</div>
                        <div style="font-size:13px;color:#6b7280;margin-top:8px;">VetAI tizimiga xush kelibsiz</div>
                    </div>
            """, unsafe_allow_html=True)
            
            login_username = st.text_input("Foydalanuvchi nomi", placeholder="username", key="login_user")
            login_password = st.text_input("Parol", type="password", placeholder="••••••", key="login_pass")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔓 Kirish", use_container_width=True):
                    if login_username and login_password:
                        success, msg = login_user(login_username, login_password)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Foydalanuvchi nomi va parolni kiriting!")
            
            with col_btn2:
                if st.button("📝 Ro'yxatdan o'tish", use_container_width=True):
                    st.session_state.show_register = True
                    st.rerun()
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class="login-container">
                <div class="login-card">
                    <div style="text-align:center;margin-bottom:32px;">
                        <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#4ade80;">Ro'yxatdan o'tish</div>
                        <div style="font-size:13px;color:#6b7280;margin-top:8px;">Yangi akkaunt yaratish</div>
                    </div>
            """, unsafe_allow_html=True)
            
            reg_username = st.text_input("Foydalanuvchi nomi", placeholder="username", key="reg_user")
            reg_password = st.text_input("Parol", type="password", placeholder="•••••• (min 4 belgi)", key="reg_pass")
            reg_phone = st.text_input("Telefon raqam", placeholder="+998 XX XXX XX XX", key="reg_phone")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⬅️ Orqaga", use_container_width=True):
                    st.session_state.show_register = False
                    st.rerun()
            
            with col_btn2:
                if st.button("✅ Ro'yxatdan o'tish", use_container_width=True):
                    if reg_username and reg_password and reg_phone:
                        success, msg = register_user(reg_username, reg_password, reg_phone)
                        if success:
                            st.success(msg)
                            st.session_state.show_register = False
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Barcha maydonlarni to'ldiring!")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    st.stop()

model, device, model_error = load_model()

with st.sidebar:
    st.markdown(f"""
    <div style="background:#0f1a14;border-radius:16px;padding:20px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
            <div style="width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#22c55e,#16a34a);
                        display:flex;align-items:center;justify-content:center;font-size:18px;">👤</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#f0fdf4;">{st.session_state.current_user}</div>
                <div style="font-size:11px;color:#4ade80;">{'✨ PRO' if st.session_state.subscription == 'pro' else '🎁 FREE'}</div>
            </div>
        </div>
        <div style="background:#0a110d;border-radius:12px;padding:12px;">
            <div style="font-size:12px;color:#6b7280;">Qolgan tashxis</div>
            <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:#4ade80;">
                {'♾️' if st.session_state.subscription == 'pro' else st.session_state.free_uses_left}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#4ade80;margin-bottom:16px;">💎 Obuna</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎁 FREE", use_container_width=True):
            if update_subscription(st.session_state.current_user, "free"):
                st.success("FREE obunaga o'tdingiz! (1 marta tashxis)")
                st.rerun()
    
    with col2:
        if st.button("⚡ PRO", use_container_width=True):
            if update_subscription(st.session_state.current_user, "pro"):
                st.success("PRO obunaga o'tdingiz! (Cheksiz tashxis)")
                st.rerun()
    
    st.markdown("---")
    
    if st.button("🚪 Chiqish", use_container_width=True):
        logout_user()

st.markdown("""
<div style="background:rgba(15,26,20,0.9);border-bottom:1px solid rgba(74,222,128,0.1);
            padding:16px 40px;margin:-80px -80px 40px -80px;backdrop-filter:blur(12px);
            display:flex;align-items:center;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:12px;">
        <div style="width:42px;height:42px;border-radius:10px;background:rgba(74,222,128,0.1);
                    border:1px solid rgba(74,222,128,0.2);display:flex;align-items:center;
                    justify-content:center;font-size:20px;">🌿</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:#f0fdf4;">VetAI</div>
            <div style="font-size:10px;color:#4ade80;letter-spacing:2px;text-transform:uppercase;font-weight:600;">Hayvon Salomatligi</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-size:12px;color:#6b7280;">ResNet18 Model</span>
        <div style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);
                    color:#4ade80;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700;">
            ● 22 Kasallik
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.markdown(f"""
    <div style="background:rgba(220,38,38,0.08);border:1px solid rgba(220,38,38,0.3);
                border-radius:14px;padding:20px 24px;color:#fca5a5;margin-bottom:20px;">
        <b>❌ Model yuklanmadi:</b> {model_error}<br>
        <small>best_model.pth faylini app.py bilan bir papkaga qo'ying</small>
    </div>
    """, unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="padding:20px 0 30px;">
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);
                    color:#4ade80;padding:6px 14px;border-radius:20px;
                    font-size:12px;font-weight:600;margin-bottom:24px;">
            ● AI · PyTorch · Real-vaqt tashxis
        </div>
        <h1 style="font-family:'Syne',sans-serif;font-size:52px;font-weight:900;
                   line-height:1.05;margin:0 0 20px;color:#f0fdf4;letter-spacing:-2px;">
            Hayvonlaringiz<br>
            <span style="color:#4ade80;font-style:italic;">kasalligini</span><br>
            aniqlaymiz
        </h1>
        <p style="font-size:16px;color:#6b7280;line-height:1.7;max-width:460px;margin:0 0 32px;">
            Rasm yuklang — ResNet18 model darhol kasallikni aniqlaydi,
            xavf darajasini baholaydi va davolash maslahatlarini beradi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    s1, s2, s3, s4 = st.columns(4)
    for col, (num, lbl) in zip([s1,s2,s3,s4], [("22","Kasallik"),("ResNet18","Model"),("98%","Aniqlik"),("24/7","Mavjud")]):
        with col:
            st.markdown(f"""
            <div style="border-left:2px solid rgba(74,222,128,0.2);padding-left:14px;">
                <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;
                            color:#f0fdf4;letter-spacing:-1px;">{num}</div>
                <div style="font-size:10px;color:#6b7280;margin-top:2px;">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div style="background:#0f1a14;border:1px solid rgba(74,222,128,0.1);
                border-radius:20px;padding:24px;margin-top:20px;">
        <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                    color:#f0fdf4;margin-bottom:6px;">📷 Rasm Yuklang</div>
        <div style="font-size:13px;color:#6b7280;margin-bottom:16px;">
            Ta'sirlangan joyning aniq rasmini yuklang
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","heic","webp"], label_visibility="collapsed")
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
    
    if uploaded and model:
        if st.session_state.subscription == "free" and st.session_state.free_uses_left <= 0:
            st.error("❌ Tekin tashxis limiti tugadi! PRO obunaga o'ting.")
            st.info("⚡ Chap menyudagi PRO tugmasini bosing — cheksiz tashxis olasiz!")
            btn = False
        else:
            btn = st.button("🔍 Tashxis Olish")
    else:
        btn = False

if btn and uploaded and model:
    if use_prediction():
        st.markdown("<hr>", unsafe_allow_html=True)
        
        with st.spinner("ResNet18 tahlil qilmoqda..."):
            time.sleep(0.4)
            img = Image.open(uploaded).convert("RGB")
            pred_class, probs = predict(img, model, device)
        
        decrement_quota()
        
        users = load_users()
        if st.session_state.current_user in users:
            users[st.session_state.current_user]["predictions_total"] = users[st.session_state.current_user].get("predictions_total", 0) + 1
            save_users(users)
        
        info = DISEASE_DB[pred_class]
        confidence = max(probs)
        top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
        risk_icon = "🔴" if info["risk"]=="HIGH" else "🟡" if info["risk"]=="MEDIUM" else "🟢"
        
        symptoms_html = "".join([
            f'<span style="background:rgba(74,222,128,0.07);border:1px solid rgba(74,222,128,0.15);'
            f'color:#4ade80;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;'
            f'margin:3px;display:inline-block;">{s}</span>'
            for s in info["symptoms"]
        ])
        
        st.markdown(f"""
        <div style="background:#0f1a14;border:1px solid rgba(74,222,128,0.1);
                    border-left:4px solid {info['color']};border-radius:16px;padding:28px;margin-bottom:20px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
                <div>
                    <div style="font-size:11px;color:#6b7280;font-weight:600;letter-spacing:0.5px;
                                text-transform:uppercase;margin-bottom:8px;">Aniqlanган kasallik</div>
                    <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;
                                color:#f0fdf4;letter-spacing:-0.5px;">{info['animal']} {info['uz']}</div>
                    <div style="font-size:13px;color:#6b7280;margin-top:6px;">{pred_class}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px;color:#6b7280;font-weight:600;letter-spacing:0.5px;
                                text-transform:uppercase;margin-bottom:4px;">Ishonch</div>
                    <div style="font-family:'Syne',sans-serif;font-size:48px;font-weight:900;
                                color:{info['color']};letter-spacing:-2px;line-height:1;">
                        {confidence*100:.1f}%
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;">
                <div style="background:{info['bg']};border:1.5px solid {info['color']};
                            color:{info['color']};padding:8px 18px;border-radius:20px;
                            font-size:13px;font-weight:700;">
                    {risk_icon} Xavf: {info['risk_uz']}
                </div>
            </div>
            <div style="margin-top:20px;">
                <div style="font-size:11px;color:#6b7280;font-weight:600;letter-spacing:0.5px;
                            text-transform:uppercase;margin-bottom:10px;">Asosiy belgilar</div>
                <div>{symptoms_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["💊 Davolash Ma'lumoti", "📊 Top 5 Natija"])
        
        with tab1:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown(f"""
                <div style="background:#0f1a14;border:1px solid rgba(74,222,128,0.1);
                            border-radius:14px;padding:22px;min-height:180px;">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                        <div style="width:36px;height:36px;border-radius:10px;
                                    background:rgba(74,222,128,0.1);display:flex;
                                    align-items:center;justify-content:center;font-size:18px;">💊</div>
                        <span style="font-family:'Syne',sans-serif;font-size:15px;
                                     font-weight:700;color:#4ade80;">Nima Qilish Kerak?</span>
                    </div>
                    <p style="font-size:14px;color:#9ca3af;line-height:1.7;margin:0;">{info['action']}</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background:#0f1a14;border:1px solid rgba(74,222,128,0.1);
                            border-radius:14px;padding:22px;min-height:180px;">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                        <div style="width:36px;height:36px;border-radius:10px;
                                    background:rgba(96,165,250,0.1);display:flex;
                                    align-items:center;justify-content:center;font-size:18px;">🛡️</div>
                        <span style="font-family:'Syne',sans-serif;font-size:15px;
                                     font-weight:700;color:#60a5fa;">Oldini Olish</span>
                    </div>
                    <p style="font-size:14px;color:#9ca3af;line-height:1.7;margin:0;">{info['prevention']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background:rgba(217,119,6,0.06);border:1px solid rgba(217,119,6,0.15);
                        border-radius:10px;padding:12px 16px;font-size:12px;color:#d97706;
                        line-height:1.6;margin-top:16px;">
                ⚠️ Bu AI tomonidan berilgan dastlabki tashxis. Professional veterinar ko'rigini almashtira olmaydi.
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div style="background:#0f1a14;border:1px solid rgba(74,222,128,0.1);
                        border-radius:14px;padding:24px;">
                <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                            color:#f0fdf4;margin-bottom:20px;">📊 Eng Ehtimoliy Natijalar</div>
            """, unsafe_allow_html=True)
            
            for rank, (idx, prob) in enumerate(top5):
                cn = CLASS_NAMES[idx]
                d = DISEASE_DB[cn]
                is_top = rank == 0
                pct = prob * 100
                dot_bg = "#4ade80" if is_top else "#1f2d25"
                dot_color = "#000" if is_top else "#6b7280"
                text_color = "#f0fdf4" if is_top else "#9ca3af"
                num_color = "#4ade80" if is_top else "#6b7280"
                bar_color = "#4ade80" if is_top else "#2d3d35"
                
                st.markdown(f"""
                <div style="margin-bottom:16px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
                        <div style="display:flex;align-items:center;gap:10px;">
                            <div style="width:24px;height:24px;border-radius:50%;display:flex;
                                        align-items:center;justify-content:center;font-size:11px;
                                        font-weight:800;background:{dot_bg};color:{dot_color};">
                                {"✓" if is_top else rank+1}
                            </div>
                            <span style="font-size:14px;font-weight:{"800" if is_top else "400"};color:{text_color};">
                                {d['animal']} {d['uz']}
                            </span>
                        </div>
                        <div style="display:flex;align-items:center;gap:10px;">
                            <span style="font-size:11px;font-weight:700;padding:3px 10px;
                                         border-radius:10px;background:{d['bg']};color:{d['color']};">
                                {d['risk_uz']}
                            </span>
                            <span style="font-size:15px;font-weight:800;color:{num_color};">{pct:.1f}%</span>
                        </div>
                    </div>
                    <div style="height:5px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;">
                        <div style="height:100%;width:{pct:.1f}%;background:{bar_color};border-radius:3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;
            color:#f0fdf4;text-align:center;margin-bottom:28px;letter-spacing:-0.5px;">
    Qanday ishlaydi?
</div>
""", unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns(4, gap="medium")
for col, (ic, t, d, c) in zip([fc1,fc2,fc3,fc4], [
    ("📸","Vizual Tahlil","Rasm yuklang — AI vizual belgilarni aniqlaydi","#4ade80"),
    ("🤖","ResNet18 Model","PyTorch bilan train qilingan 22 ta kasallik uchun maxsus model","#60a5fa"),
    ("⚡","Darhol Natija","Soniyalar ichida aniq tashxis va simptomlar ro'yxati","#f59e0b"),
    ("💊","Davolash","Har bir kasallik uchun batafsil davolash va oldini olish bo'yicha qo'llanma","#f472b6"),
]):
    with col:
        st.markdown(f"""
        <div style="background:#0f1a14;border:1px solid rgba(255,255,255,0.05);
                    border-radius:16px;padding:22px 18px;text-align:center;height:100%;">
            <div style="width:52px;height:52px;border-radius:14px;margin:0 auto 14px;
                        background:{c}18;border:1px solid {c}33;
                        display:flex;align-items:center;justify-content:center;font-size:26px;">{ic}</div>
            <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                        color:{c};margin-bottom:8px;">{t}</div>
            <div style="font-size:12px;color:#6b7280;line-height:1.6;">{d}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="border-top:1px solid rgba(74,222,128,0.08);padding:24px 0;margin-top:60px;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
    <span style="font-family:'Syne',sans-serif;font-weight:800;color:#4ade80;font-size:16px;">🌿 VetAI</span>
    <span style="font-size:12px;color:#374151;">© 2025 · ResNet18 · PyTorch · Streamlit · 22 ta kasallik</span>
    <span style="font-size:12px;color:#374151;">⚠️ Faqat dastlabki tashxis — veterinar o'rnini bosmaydi</span>
</div>
""", unsafe_allow_html=True)
