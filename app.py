import streamlit as st
import pandas as pd
import random
import requests
import hashlib
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

# ═══════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="Netflix AI",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════
# USER DB
# ═══════════════════════════════════════════════════
USERS_FILE     = "users.json"
WATCHLIST_FILE = "watchlists.json"

def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}

def save_json(p, d):
    json.dump(d, open(p, "w"), indent=2)

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(u, pw, em):
    users = load_json(USERS_FILE)
    if u in users:
        return False, "Username already exists."
    users[u] = {"password": hash_pw(pw), "email": em, "avatar": u[0].upper()}
    save_json(USERS_FILE, users)
    return True, "Account created!"

def login_user(u, pw):
    users = load_json(USERS_FILE)
    if u not in users:
        return False, "User not found."
    if users[u]["password"] != hash_pw(pw):
        return False, "Wrong password."
    return True, users[u]

def get_watchlist(u):
    return load_json(WATCHLIST_FILE).get(u, [])

def add_to_watchlist(u, movie):
    wl = load_json(WATCHLIST_FILE)
    wl.setdefault(u, [])
    if movie["title"] not in [m["title"] for m in wl[u]]:
        wl[u].append(movie)
        save_json(WATCHLIST_FILE, wl)
        return True
    return False

def remove_from_watchlist(u, title):
    wl = load_json(WATCHLIST_FILE)
    if u in wl:
        wl[u] = [m for m in wl[u] if m["title"] != title]
        save_json(WATCHLIST_FILE, wl)

# ═══════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════
for k, v in {
    "logged_in": False, "username": "", "avatar": "U",
    "page": "home", "auth_mode": "login",
    "trailer_url": "", "show_trailer": False,
    "featured": None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════
# MASTER CSS
# ═══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;700&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

:root{
  --red:#E50914;
  --red2:#f40612;
  --black:#141414;
  --dark:#181818;
  --card:#1f1f1f;
  --green:#46D369;
  --muted:#808080;
  --blue:#0071eb;
  --fn:'DM Sans','Helvetica Neue',Arial,sans-serif;
  --fd:'Bebas Neue',sans-serif;
}

html,body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stVerticalBlock"]{
  background:var(--black) !important;
  font-family:var(--fn) !important;
  color:#fff !important;
}
[data-testid="block-container"]{padding:0 !important;max-width:100% !important;}
[data-testid="stHeader"]{display:none !important;}
section[data-testid="stSidebar"]{display:none !important;}
footer,#MainMenu{display:none !important;}

::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--black);}
::-webkit-scrollbar-thumb{background:#333;border-radius:3px;}

/* ─── NAVBAR ─── */
.nf-nav{
  position:fixed;top:0;left:0;right:0;z-index:9000;height:70px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 60px;
  background:linear-gradient(180deg,rgba(0,0,0,.97) 0%,transparent 100%);
  transition:background .4s;
}
.nf-nav.opaque{background:var(--black) !important;}
.nf-logo{font-family:var(--fd);font-size:32px;color:var(--red);letter-spacing:2px;user-select:none;}
.nf-links{display:flex;gap:20px;}
.nf-links a{color:#e5e5e5;text-decoration:none;font-size:13.5px;transition:color .2s;}
.nf-links a:hover,.nf-links a.cur{color:#fff;font-weight:500;}
.nf-right{display:flex;align-items:center;gap:14px;}
.nf-ico{color:#ddd;font-size:18px;cursor:pointer;transition:opacity .2s;}
.nf-ico:hover{opacity:.7;}
.nf-uname{color:#e5e5e5;font-size:13px;font-weight:500;}
.nf-av{
  width:32px;height:32px;border-radius:6px;
  background:var(--red);display:flex;align-items:center;
  justify-content:center;font-size:13px;font-weight:700;color:#fff;cursor:pointer;
}

/* ─── HERO ─── */
.nf-hero{position:relative;width:100%;height:100vh;min-height:660px;overflow:hidden;}
.nf-hero-bg{
  position:absolute;inset:0;background-size:cover;background-position:center 15%;
  animation:hzoom 16s ease-in-out infinite alternate;
}
@keyframes hzoom{from{transform:scale(1)}to{transform:scale(1.06)}}
.nf-hero-fog{
  position:absolute;inset:0;
  background:
    linear-gradient(90deg,rgba(0,0,0,.93) 0%,rgba(0,0,0,.55) 45%,rgba(0,0,0,.08) 75%,transparent 100%),
    linear-gradient(0deg,rgba(20,20,20,1) 0%,rgba(20,20,20,0) 42%),
    linear-gradient(180deg,rgba(0,0,0,.55) 0%,transparent 20%);
}
.nf-hero-body{
  position:absolute;bottom:26%;left:60px;max-width:520px;
  animation:fup .85s ease both;
}
@keyframes fup{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
.nf-badge{display:inline-flex;align-items:center;gap:9px;margin-bottom:14px;}
.nf-badge-n{
  width:26px;height:26px;border-radius:3px;background:var(--red);
  display:flex;align-items:center;justify-content:center;
  font-family:var(--fd);font-size:16px;color:#fff;
}
.nf-badge-txt{font-size:11px;font-weight:700;letter-spacing:4px;color:#b3b3b3;text-transform:uppercase;}
.nf-htitle{
  font-size:clamp(42px,5vw,76px);font-weight:700;line-height:1.0;
  color:#fff;margin-bottom:16px;
  text-shadow:0 2px 22px rgba(0,0,0,.8);letter-spacing:-.5px;
}
.nf-hmeta{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:14px;}
.nf-hmatch{color:var(--green);font-size:15px;font-weight:700;}
.nf-hyear{color:#aaa;font-size:13px;}
.nf-hgenre{
  border:1px solid rgba(255,255,255,.28);border-radius:3px;
  padding:2px 8px;font-size:11px;color:#bbb;
}
.nf-hdesc{
  font-size:15px;font-weight:300;line-height:1.7;
  color:rgba(255,255,255,.82);margin-bottom:24px;
}
.nf-hbtns{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.nf-bplay{
  display:inline-flex;align-items:center;gap:8px;
  padding:11px 28px;background:#fff;color:#000;
  font-size:16px;font-weight:700;border:none;border-radius:4px;
  cursor:pointer;transition:background .15s;
  box-shadow:0 4px 18px rgba(0,0,0,.5);font-family:var(--fn);
}
.nf-bplay:hover{background:rgba(255,255,255,.82);}
.nf-binfo{
  display:inline-flex;align-items:center;gap:8px;
  padding:11px 24px;background:rgba(109,109,110,.65);color:#fff;
  font-size:16px;font-weight:600;border:none;border-radius:4px;
  cursor:pointer;backdrop-filter:blur(8px);font-family:var(--fn);
  transition:background .15s;
}
.nf-binfo:hover{background:rgba(109,109,110,.45);}
.nf-badd{
  display:inline-flex;align-items:center;gap:7px;
  padding:11px 20px;background:transparent;color:#fff;
  font-size:15px;font-weight:600;
  border:1.5px solid rgba(255,255,255,.55);border-radius:4px;
  cursor:pointer;font-family:var(--fn);transition:all .2s;
}
.nf-badd:hover{background:rgba(255,255,255,.1);border-color:#fff;}
.nf-hscore{
  position:absolute;bottom:26%;right:60px;
  border-left:4px solid rgba(255,255,255,.6);padding-left:12px;
}
.nf-hscore-n{font-size:28px;font-weight:700;color:#fff;}
.nf-hscore-l{font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:1px;margin-top:2px;}

/* ─── ROWS ─── */
.nf-row{padding:0 60px;margin-bottom:52px;}
.nf-row-hdr{display:flex;align-items:center;gap:14px;margin-bottom:14px;}
.nf-row-title{font-size:19px;font-weight:600;color:#e5e5e5;}
.nf-row-more{font-size:12px;font-weight:600;color:var(--green);opacity:0;transition:opacity .22s;cursor:pointer;}
.nf-row:hover .nf-row-more{opacity:1;}
.nf-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;}

/* ─── MOVIE CARD ─── */
.nf-card{
  position:relative;border-radius:4px;overflow:visible;
  cursor:pointer;z-index:1;
  transition:transform .32s cubic-bezier(.25,.46,.45,.94),z-index 0s .32s;
}
.nf-card:hover{
  transform:scale(1.3) translateY(-10px);z-index:200;
  transition:transform .32s cubic-bezier(.25,.46,.45,.94),z-index 0s;
}
.nf-card:first-child:hover{transform:scale(1.3) translate(11%,-10px);}
.nf-card:last-child:hover{transform:scale(1.3) translate(-11%,-10px);}
.nf-card img{
  width:100%;aspect-ratio:2/3;object-fit:cover;
  display:block;border-radius:4px;background:#1a1a1a;
}
.nf-card-ov{
  position:absolute;inset:0;border-radius:4px;
  background:linear-gradient(0deg,rgba(0,0,0,.96) 0%,rgba(0,0,0,.35) 50%,transparent 72%);
  opacity:0;transition:opacity .24s;
  display:flex;flex-direction:column;justify-content:flex-end;
  padding:10px;box-shadow:0 14px 44px rgba(0,0,0,.8);
}
.nf-card:hover .nf-card-ov{opacity:1;}
.nf-cbtns{display:flex;align-items:center;gap:5px;margin-bottom:7px;}
.nf-cb{
  width:26px;height:26px;border-radius:50%;
  border:1.5px solid rgba(255,255,255,.5);
  background:rgba(20,20,20,.9);
  display:flex;align-items:center;justify-content:center;
  font-size:9px;cursor:pointer;color:#fff;
  transition:border-color .15s,background .15s;
}
.nf-cb.pl{background:#fff;border-color:#fff;color:#000;font-weight:900;}
.nf-cb:hover{border-color:#fff;background:rgba(255,255,255,.18);}
.nf-cb.pl:hover{background:rgba(255,255,255,.84);}
.nf-ctitle{font-size:11px;font-weight:700;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px;}
.nf-cmeta{display:flex;align-items:center;gap:6px;}
.nf-cmatch{color:var(--green);font-size:10px;font-weight:700;}
.nf-crat{color:#ccc;font-size:10px;}
.nf-cgen{font-size:9px;color:#777;margin-top:2px;}

/* ─── REC SECTION ─── */
.nf-rec{
  padding:54px 60px 46px;
  background:linear-gradient(180deg,#0b0b0b 0%,#0e0e0e 100%);
  border-top:1px solid rgba(255,255,255,.05);
}
.nf-rec-h{font-size:26px;font-weight:700;color:#fff;margin-bottom:5px;letter-spacing:-.2px;}
.nf-rec-s{font-size:14px;color:#555;margin-bottom:28px;}
.nf-res-hdr{display:flex;align-items:center;gap:12px;margin-bottom:20px;}
.nf-res-tag{
  font-size:10px;font-weight:700;letter-spacing:2px;
  padding:5px 13px;border-radius:3px;text-transform:uppercase;
  background:var(--red);color:#fff;
}
.nf-res-label{font-size:16px;font-weight:500;color:#ccc;}
.nf-res-label em{color:#fff;font-style:normal;}
.nf-res-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;}
.nf-rcard{
  border-radius:8px;overflow:hidden;background:var(--card);
  border:1px solid rgba(255,255,255,.05);
  transition:transform .28s cubic-bezier(.25,.46,.45,.94),box-shadow .28s,border-color .28s;
  cursor:pointer;
}
.nf-rcard:hover{
  transform:translateY(-7px) scale(1.02);
  box-shadow:0 20px 52px rgba(0,0,0,.8);
  border-color:rgba(255,255,255,.12);
}
.nf-rcard img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:#1a1a1a;}
.nf-rbody{padding:12px 14px 15px;}
.nf-rtitle{font-size:13px;font-weight:700;color:#fff;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.nf-ryear{font-size:11px;color:var(--green);font-weight:600;margin-bottom:5px;}
.nf-rreason{font-size:11px;color:#666;line-height:1.5;}
.nf-rgenre{display:inline-block;margin-top:7px;font-size:10px;padding:2px 7px;border:1px solid rgba(255,255,255,.1);border-radius:3px;color:#666;}
.nf-ract{display:flex;gap:5px;margin-top:10px;}
.nf-ra{flex:1;padding:7px 3px;border:none;border-radius:4px;font-size:10px;font-weight:700;cursor:pointer;font-family:var(--fn);transition:all .18s;}
.nf-ra.play{background:#fff;color:#000;}
.nf-ra.play:hover{background:rgba(255,255,255,.85);}
.nf-ra.save{background:rgba(255,255,255,.07);color:#bbb;border:1px solid rgba(255,255,255,.1);}
.nf-ra.save:hover{background:rgba(255,255,255,.14);color:#fff;}
.nf-ra.tmdb{background:rgba(1,180,228,.08);color:#01b4e4;border:1px solid rgba(1,180,228,.18);}
.nf-ra.tmdb:hover{background:rgba(1,180,228,.18);}

/* ─── WATCHLIST ─── */
.nf-wl-hero{padding:108px 60px 28px;}
.nf-wl-title{font-size:34px;font-weight:700;color:#fff;margin-bottom:4px;}
.nf-wl-count{color:#555;font-size:13px;}
.nf-wl-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;padding:24px 60px 56px;}
.nf-wcard{border-radius:8px;overflow:hidden;background:var(--card);border:1px solid rgba(255,255,255,.05);transition:transform .26s,box-shadow .26s;}
.nf-wcard:hover{transform:translateY(-5px);box-shadow:0 14px 36px rgba(0,0,0,.7);}
.nf-wcard img{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:#1a1a1a;}
.nf-wcbody{padding:10px 13px 13px;}
.nf-wctitle{font-size:12px;font-weight:700;color:#fff;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.nf-wcyear{font-size:11px;color:var(--green);margin-bottom:8px;}
.nf-wl-empty{text-align:center;padding:80px 40px;color:#444;}
.nf-wl-empty-icon{font-size:52px;margin-bottom:16px;}
.nf-wl-empty-title{font-size:20px;font-weight:700;color:#888;margin-bottom:8px;}

/* ─── PROFILE ─── */
.nf-prof-hero{
  padding:115px 60px 40px;
  background:linear-gradient(135deg,rgba(229,9,20,.07) 0%,transparent 55%);
  border-bottom:1px solid rgba(255,255,255,.05);
}
.nf-prof-row{display:flex;align-items:center;gap:22px;margin-bottom:6px;}
.nf-prof-av{
  width:80px;height:80px;border-radius:12px;
  background:linear-gradient(135deg,var(--red),#9b0710);
  display:flex;align-items:center;justify-content:center;
  font-size:36px;font-weight:700;color:#fff;
  box-shadow:0 6px 20px rgba(229,9,20,.3);
}
.nf-prof-name{font-size:28px;font-weight:700;color:#fff;}
.nf-prof-email{color:#444;font-size:14px;margin-top:3px;}
.nf-prof-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:28px 60px;}
.nf-stat{background:var(--card);border:1px solid rgba(255,255,255,.04);border-radius:10px;padding:20px 24px;}
.nf-stat-n{font-size:32px;font-weight:700;color:var(--red);}
.nf-stat-l{font-size:12px;color:#444;margin-top:4px;}

/* ─── TRAILER ─── */
.nf-trailer{
  position:fixed;inset:0;background:rgba(0,0,0,.95);
  z-index:99999;display:flex;align-items:center;
  justify-content:center;flex-direction:column;gap:14px;padding:20px;
}
.nf-trailer iframe{width:100%;max-width:920px;aspect-ratio:16/9;border:none;border-radius:8px;}

/* ═══════════════════════════════════════════
   AUTH PAGE — Pixel-perfect Netflix clone
═══════════════════════════════════════════ */
.nf-auth-page{
  position:fixed;inset:0;z-index:99000;
  background:linear-gradient(rgba(0,0,0,.6),rgba(0,0,0,.6)),
    url('https://assets.nflxext.com/ffe/siteui/vlv3/9134db96-10d6-4a64-a619-a21da22f8999/a449fabb-05e4-4c8a-b062-b0bec7d03085/IN-en-20240311-popsignuptwoweeks-perspective_alpha_website_large.jpg')
    center/cover no-repeat fixed;
  display:flex;flex-direction:column;
  font-family:var(--fn);
}
.nf-auth-nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:26px 60px 20px;
}
.nf-auth-nav-logo{font-family:var(--fd);font-size:36px;color:var(--red);letter-spacing:2px;}
.nf-auth-center{
  flex:1;display:flex;align-items:center;justify-content:center;padding:0 20px 40px;
}
.nf-auth-card{
  background:rgba(0,0,0,.82);border-radius:4px;
  padding:60px 68px 50px;width:100%;max-width:450px;min-height:480px;
}
.nf-auth-title{
  font-size:32px;font-weight:700;color:#fff;
  margin-bottom:28px;letter-spacing:-.3px;
  font-family:var(--fn);
}
/* Netflix-style inputs inside the card */
.nf-auth-input{
  width:100%;height:56px;
  background:#333;border:none;border-bottom:2px solid transparent;
  border-radius:4px;
  padding:16px 16px 0;
  color:#fff;font-size:16px;font-family:var(--fn);
  outline:none;transition:background .2s,border-color .2s;
  margin-bottom:16px;display:block;
}
.nf-auth-input::placeholder{color:#8c8c8c;font-size:14px;}
.nf-auth-input:focus{background:#454545;border-bottom-color:var(--red);}
.nf-auth-btn{
  width:100%;height:50px;
  background:var(--red);color:#fff;border:none;border-radius:4px;
  font-size:16px;font-weight:700;cursor:pointer;
  font-family:var(--fn);margin-top:8px;margin-bottom:6px;
  transition:background .15s;letter-spacing:.3px;
}
.nf-auth-btn:hover{background:var(--red2);}
.nf-auth-or{
  display:flex;align-items:center;gap:10px;
  color:#737373;font-size:13px;margin:14px 0;
}
.nf-auth-or::before,.nf-auth-or::after{
  content:'';flex:1;height:1px;background:#404040;
}
.nf-auth-switch{
  color:#737373;font-size:15px;margin-top:16px;
}
.nf-auth-switch span{
  color:#fff;cursor:pointer;
}
.nf-auth-switch span:hover{text-decoration:underline;}
.nf-auth-recaptcha{
  color:#8c8c8c;font-size:12px;margin-top:16px;line-height:1.5;
}
.nf-auth-recaptcha a{color:#0071eb;text-decoration:none;}
.nf-auth-err{
  background:rgba(255,160,0,.14);
  border:2px solid #e87c03;
  border-radius:4px;padding:14px 16px;
  color:#e87c03;font-size:14px;
  font-weight:500;margin-bottom:14px;
}
.nf-auth-ok{
  background:rgba(70,211,105,.12);
  border:1px solid rgba(70,211,105,.4);
  border-radius:4px;padding:14px 16px;
  color:var(--green);font-size:14px;margin-bottom:14px;
}

/* ─── STREAMLIT OVERRIDES ─── */
/* Normal page inputs */
div[data-baseweb="select"]>div{
  background:#1e1e1e !important;border:1.5px solid #2e2e2e !important;
  border-radius:5px !important;color:#fff !important;font-family:var(--fn) !important;
}
div[data-baseweb="select"]>div:hover,
div[data-baseweb="select"]>div:focus-within{border-color:#555 !important;}
div[data-baseweb="select"] span{color:#fff !important;}
div[data-baseweb="popover"]{background:#1e1e1e !important;border:1px solid #2e2e2e !important;}
li[role="option"]{background:#1e1e1e !important;color:#ddd !important;}
li[role="option"]:hover{background:#2a2a2a !important;}
div[data-baseweb="input"]>div{
  background:#1e1e1e !important;border:1.5px solid #2e2e2e !important;border-radius:5px !important;
}
div[data-baseweb="input"]>div:focus-within{border-color:#555 !important;}
div[data-baseweb="input"] input{color:#fff !important;font-family:var(--fn) !important;}
div[data-baseweb="input"] input::placeholder{color:#444 !important;}
div.stSelectbox>label,div.stTextInput>label{
  color:#444 !important;font-size:10px !important;font-weight:700 !important;
  letter-spacing:2px !important;text-transform:uppercase !important;
}
div.stButton>button{
  background:var(--red) !important;color:#fff !important;
  font-size:15px !important;font-weight:700 !important;
  border:none !important;border-radius:5px !important;
  padding:12px 32px !important;width:100% !important;
  font-family:var(--fn) !important;letter-spacing:.3px !important;
  transition:background .15s,transform .1s !important;
}
div.stButton>button:hover{background:var(--red2) !important;transform:translateY(-1px) !important;}
div.stRadio>div{flex-direction:row !important;gap:8px !important;background:transparent !important;}
div.stRadio>div label{
  background:#1e1e1e !important;border:1.5px solid #2e2e2e !important;
  border-radius:4px !important;padding:8px 18px !important;
  color:#666 !important;font-size:13px !important;font-weight:600 !important;
  font-family:var(--fn) !important;cursor:pointer !important;transition:all .2s !important;
}
div.stRadio>div label:hover{border-color:#555 !important;color:#ccc !important;}
div.stSpinner>div{color:var(--red) !important;}
p,li{color:#ddd !important;}
h1,h2,h3{color:#fff !important;}
.stAlert{background:#1a1a1a !important;border:1px solid #222 !important;}

.nf-divider{height:1px;background:linear-gradient(90deg,transparent,#1e1e1e 30%,#1e1e1e 70%,transparent);margin:0 0 50px;}

/* ─── FOOTER ─── */
.nf-footer{padding:48px 60px 42px;border-top:1px solid rgba(255,255,255,.04);}
.nf-footer a{color:#505050;text-decoration:none;font-size:12px;margin-right:30px;margin-bottom:10px;display:inline-block;transition:color .2s;}
.nf-footer a:hover{color:#999;}
.nf-footer-copy{font-size:12px;color:#333;margin-top:14px;}

/* fallback */
.no-img{width:100%;aspect-ratio:2/3;background:linear-gradient(160deg,#181818,#1f1f1f);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;color:#2a2a2a;font-size:22px;border-radius:4px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# VALIDATE SECRETS
# ═══════════════════════════════════════════════════
missing = []
try:    _ = st.secrets["TMDB_API_KEY"]
except: missing.append("TMDB_API_KEY")
try:    _ = st.secrets["ANTHROPIC_API_KEY"]
except: missing.append("ANTHROPIC_API_KEY")
if missing:
    st.error("Missing keys in .streamlit/secrets.toml: " + ", ".join(missing))
    st.code('TMDB_API_KEY = "your_key"\nANTHROPIC_API_KEY = "your_key"', language="toml")
    st.stop()

# ═══════════════════════════════════════════════════
# DATA & ML
# ═══════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movies.csv")
    except FileNotFoundError:
        st.error("movies.csv not found.")
        st.stop()
    df["overview"] = df["overview"].fillna("")
    df["genres"]   = df["genres"].fillna("")
    df["combined"] = df["overview"] + " " + df["genres"]
    if "id" not in df.columns and "movie_id" in df.columns:
        df["id"] = df["movie_id"]
    return df

movies = load_data()

@st.cache_data
def build_sim(c):
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    return cosine_similarity(tfidf.fit_transform(c))

cosine_sim = build_sim(movies["combined"])

@st.cache_data
def get_genres(s):
    gs = set()
    for g in s.dropna(): gs.update(str(g).split())
    return sorted([g for g in gs if len(g) > 2])

all_genres = get_genres(movies["genres"])

# ═══════════════════════════════════════════════════
# TMDB
# ═══════════════════════════════════════════════════
IMG = "https://image.tmdb.org/t/p"
BLANK = "https://placehold.co/300x450/181818/2a2a2a?text=No+Image"

@st.cache_data(ttl=86400)
def tmdb_get(mid):
    try:
        k = st.secrets["TMDB_API_KEY"]
        r = requests.get(IMG.replace("/t/p","") + "://api.themoviedb.org/3/movie/" + str(int(mid)) + "?api_key=" + k, timeout=6)
        if r.status_code != 200: return BLANK, BLANK, 0.0, ""
        d = r.json()
        p  = d.get("poster_path");   b = d.get("backdrop_path")
        po = IMG + "/w500"    + p if p else BLANK
        ba = IMG + "/original" + b if b else po
        url= "https://www.themoviedb.org/movie/" + str(int(mid))
        return po, ba, round(float(d.get("vote_average", 0)), 1), url
    except: return BLANK, BLANK, 0.0, ""

@st.cache_data(ttl=86400)
def tmdb_search(title, year=""):
    try:
        k = st.secrets["TMDB_API_KEY"]
        q = requests.utils.quote(str(title))
        u = "https://api.themoviedb.org/3/search/movie?api_key=" + k + "&query=" + q
        if year: u += "&year=" + str(year)
        r = requests.get(u, timeout=6); r.raise_for_status()
        res = r.json().get("results", [])
        if res:
            hit = res[0]; p = hit.get("poster_path"); mid = hit.get("id", 0)
            po  = IMG + "/w500" + p if p else BLANK
            tu  = "https://www.themoviedb.org/movie/" + str(mid) if mid else ""
            return po, tu, int(mid)
    except: pass
    return BLANK, "", 0

@st.cache_data(ttl=3600)
def tmdb_trailer(mid):
    try:
        k = st.secrets["TMDB_API_KEY"]
        r = requests.get("https://api.themoviedb.org/3/movie/" + str(int(mid)) + "/videos?api_key=" + k, timeout=6)
        r.raise_for_status()
        for v in r.json().get("results", []):
            if v.get("site") == "YouTube" and "Trailer" in v.get("type", ""):
                return "https://www.youtube.com/embed/" + v["key"] + "?autoplay=1&rel=0"
        for v in r.json().get("results", []):
            if v.get("site") == "YouTube":
                return "https://www.youtube.com/embed/" + v["key"] + "?autoplay=1&rel=0"
    except: pass
    return ""

@st.cache_data(ttl=86400)
def tmdb_trending():
    try:
        k = st.secrets["TMDB_API_KEY"]
        r = requests.get("https://api.themoviedb.org/3/trending/movie/week?api_key=" + k, timeout=6)
        r.raise_for_status()
        res = [x for x in r.json().get("results", []) if x.get("backdrop_path")]
        if res:
            pick = random.choice(res[:8])
            return IMG + "/original" + pick["backdrop_path"], int(pick.get("id", 0))
    except: pass
    return "", 0

# ═══════════════════════════════════════════════════
# RECOMMENDATION ENGINES
# ═══════════════════════════════════════════════════
def rec_tfidf(title, n=5):
    m = movies[movies["title"] == title]
    if m.empty: return []
    idx = m.index[0]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    out = []
    for i, sc in scores:
        row = movies.iloc[i]
        po, _, _, tu = tmdb_get(int(row["id"]))
        out.append({
            "title":   str(row["title"]),
            "poster":  po,
            "year":    str(row.get("release_date", ""))[:4],
            "reason":  str(int(sc*100)) + "% content similarity",
            "genre":   str(row.get("genres","")).split()[0] if str(row.get("genres","")).strip() else "",
            "tmdb_id": int(row["id"]),
            "tmdb_url": tu
        })
    return out

@st.cache_data(ttl=600)
def rec_ai(title, mood, genre, n=5):
    try:
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        prompt = (
            "You are a professional movie recommendation engine.\n"
            "Movie: \"" + title + "\" | Mood: " + mood + " | Genre: " + genre + "\n"
            "Return exactly " + str(n) + " picks as JSON array only (no markdown):\n"
            '[{"title":"Name","year":2021,"reason":"Short reason under 12 words","genre":"Genre"}]'
        )
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=700,
            messages=[{"role":"user","content":prompt}]
        )
        text = msg.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"): text = text[4:]
            text = text.strip()
        recs = json.loads(text)
        out  = []
        for rec in recs:
            t = str(rec.get("title",""))
            y = str(rec.get("year",""))
            po, tu, tid = tmdb_search(t, y)
            out.append({
                "title":   t, "year": y,
                "reason":  str(rec.get("reason","")),
                "genre":   str(rec.get("genre","")),
                "poster":  po, "tmdb_id": tid, "tmdb_url": tu
            })
        return out
    except anthropic.AuthenticationError:
        st.error("Invalid Anthropic API key.")
        return []
    except Exception as e:
        st.error("Error: " + str(e))
        return []

# ═══════════════════════════════════════════════════
# FEATURED MOVIE
# ═══════════════════════════════════════════════════
if st.session_state.featured is None:
    pool = movies
    if "vote_count" in movies.columns:
        pool = movies[movies["vote_count"] > 200]
    pool = pool.sort_values("vote_average", ascending=False).head(30)
    row  = pool.sample(1).iloc[0]
    po, ba, rat, tu = tmdb_get(int(row["id"]))
    if "placehold" in ba or ba == po:
        tb, _ = tmdb_trending()
        if tb: ba = tb
    st.session_state.featured = {
        "title":   str(row["title"]),
        "overview": str(row.get("overview",""))[:190],
        "poster":  po, "backdrop": ba,
        "rating":  rat, "genres": str(row.get("genres","")),
        "year":    str(row.get("release_date",""))[:4],
        "tmdb_id": int(row["id"]), "tmdb_url": tu
    }

feat = st.session_state.featured

# ═══════════════════════════════════════════════════════════════
#  AUTH PAGE  — rendered BEFORE navbar so it covers everything
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "auth":
    mode     = st.session_state.auth_mode
    is_login = (mode == "login")
    title_txt = "Sign In" if is_login else "Sign Up"

    # Full-screen overlay background
    st.markdown(
        '<div class="nf-auth-page">'
        '<div class="nf-auth-nav">'
        '<div class="nf-auth-nav-logo">NETFLIX AI</div>'
        '</div>'
        '<div class="nf-auth-center">'
        '<div class="nf-auth-card">'
        '<div class="nf-auth-title">' + title_txt + '</div>',
        unsafe_allow_html=True
    )

    if is_login:
        uname = st.text_input(" ", key="li_u", placeholder="Username")
        passw = st.text_input("  ", key="li_p", placeholder="Password", type="password")

        # Override Streamlit inputs to look Netflix-style inside auth card
        st.markdown("""
        <style>
        div[data-testid="stTextInput"]>div>div>input{
            background:#333 !important;
            border:none !important;
            border-bottom:2px solid transparent !important;
            border-radius:4px !important;
            padding:28px 16px 10px !important;
            height:56px !important;
            color:#fff !important;
            font-size:16px !important;
            font-family:'DM Sans',sans-serif !important;
            outline:none !important;
            width:100% !important;
            transition:background .2s !important;
        }
        div[data-testid="stTextInput"]>div>div>input:focus{
            background:#454545 !important;
            border-bottom-color:#e50914 !important;
        }
        div[data-testid="stTextInput"]>div>div>input::placeholder{color:#8c8c8c !important;}
        div[data-testid="stTextInput"]>div{
            background:transparent !important;
            border:none !important;
            box-shadow:none !important;
        }
        div[data-testid="stTextInput"] label{
            display:none !important;
        }
        div.stButton>button{
            background:#e50914 !important;color:#fff !important;
            height:50px !important;border-radius:4px !important;
            font-size:16px !important;font-weight:700 !important;
            border:none !important;width:100% !important;
            margin-top:8px !important;
        }
        div.stButton>button:hover{background:#f40612 !important;}
        </style>""", unsafe_allow_html=True)

        if st.button("Sign In", key="do_li"):
            if uname and passw:
                ok, res = login_user(uname, passw)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username  = uname
                    st.session_state.avatar    = res.get("avatar", uname[0].upper())
                    st.session_state.page      = "home"
                    st.rerun()
                else:
                    st.markdown('<div class="nf-auth-err">&#9888; ' + res + '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="nf-auth-err">&#9888; Please enter your username and password.</div>', unsafe_allow_html=True)

        st.markdown('<div class="nf-auth-switch">New to Netflix AI? <span>', unsafe_allow_html=True)
        if st.button("Sign up now.", key="go_reg"):
            st.session_state.auth_mode = "register"; st.rerun()
        st.markdown('</span></div>', unsafe_allow_html=True)

    else:
        uname = st.text_input(" ", key="rg_u", placeholder="Username")
        email = st.text_input("  ", key="rg_e", placeholder="Email")
        passw = st.text_input("   ", key="rg_p", placeholder="Password", type="password")

        st.markdown("""
        <style>
        div[data-testid="stTextInput"]>div>div>input{
            background:#333 !important;border:none !important;
            border-bottom:2px solid transparent !important;
            border-radius:4px !important;
            padding:28px 16px 10px !important;height:56px !important;
            color:#fff !important;font-size:16px !important;
            font-family:'DM Sans',sans-serif !important;
            outline:none !important;width:100% !important;
            transition:background .2s !important;
        }
        div[data-testid="stTextInput"]>div>div>input:focus{background:#454545 !important;border-bottom-color:#e50914 !important;}
        div[data-testid="stTextInput"]>div>div>input::placeholder{color:#8c8c8c !important;}
        div[data-testid="stTextInput"]>div{background:transparent !important;border:none !important;box-shadow:none !important;}
        div[data-testid="stTextInput"] label{display:none !important;}
        div.stButton>button{
            background:#e50914 !important;color:#fff !important;
            height:50px !important;border-radius:4px !important;
            font-size:16px !important;font-weight:700 !important;
            border:none !important;width:100% !important;margin-top:8px !important;
        }
        div.stButton>button:hover{background:#f40612 !important;}
        </style>""", unsafe_allow_html=True)

        if st.button("Create Account", key="do_reg"):
            if uname and email and passw:
                ok, msg = register_user(uname, passw, email)
                if ok:
                    st.markdown('<div class="nf-auth-ok">&#9989; ' + msg + '</div>', unsafe_allow_html=True)
                    st.session_state.auth_mode = "login"; st.rerun()
                else:
                    st.markdown('<div class="nf-auth-err">&#9888; ' + msg + '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="nf-auth-err">&#9888; Please fill in all fields.</div>', unsafe_allow_html=True)

        st.markdown('<div class="nf-auth-switch">Already have an account? <span>', unsafe_allow_html=True)
        if st.button("Sign in.", key="go_li"):
            st.session_state.auth_mode = "login"; st.rerun()
        st.markdown('</span></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="nf-auth-recaptcha">'
        'This page is protected by Google reCAPTCHA to ensure you\'re not a bot. '
        '<a href="#">Learn more.</a>'
        '</div>'
        '</div></div></div>',
        unsafe_allow_html=True
    )
    st.stop()

# ═══════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════
li = st.session_state.logged_in
rhtml = ""
if li:
    rhtml = (
        '<span class="nf-uname">Hi, ' + st.session_state.username + '</span>'
        '<div class="nf-av">' + st.session_state.avatar + '</div>'
    )

st.markdown(
    '<div class="nf-nav">'
    '<div class="nf-logo">NETFLIX AI</div>'
    '<div class="nf-links">'
    '<a href="#" class="cur">Home</a>'
    '<a href="#">Movies</a>'
    '<a href="#">TV Shows</a>'
    '<a href="#">New &amp; Popular</a>'
    + ('<a href="#">My List</a>' if li else '') +
    '</div>'
    '<div class="nf-right">'
    '<div class="nf-ico">&#128269;</div>'
    '<div class="nf-ico">&#128276;</div>'
    + rhtml +
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# Nav action buttons
bc = st.columns([8, 1, 1, 1, 1])
if li:
    with bc[1]:
        if st.button("🏠", key="nb_h", help="Home"):
            st.session_state.page = "home"; st.rerun()
    with bc[2]:
        if st.button("📋", key="nb_w", help="My List"):
            st.session_state.page = "watchlist"; st.rerun()
    with bc[3]:
        if st.button("👤", key="nb_p", help="Profile"):
            st.session_state.page = "profile"; st.rerun()
    with bc[4]:
        if st.button("🚪", key="nb_o", help="Sign Out"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.page      = "home"
            st.rerun()
else:
    with bc[4]:
        if st.button("Sign In", key="nb_i"):
            st.session_state.page      = "auth"
            st.session_state.auth_mode = "login"
            st.rerun()

# ═══════════════════════════════════════════════════
# TRAILER MODAL
# ═══════════════════════════════════════════════════
if st.session_state.show_trailer and st.session_state.trailer_url:
    st.markdown(
        '<div class="nf-trailer">'
        '<iframe src="' + st.session_state.trailer_url + '"'
        ' allow="autoplay; encrypted-media; fullscreen" allowfullscreen></iframe>'
        '</div>',
        unsafe_allow_html=True
    )
    if st.button("✕  Close Trailer", key="cl_t"):
        st.session_state.show_trailer = False
        st.session_state.trailer_url  = ""
        st.rerun()
    st.stop()

# ═══════════════════════════════════════════════════
# PAGE ROUTER
# ═══════════════════════════════════════════════════
page = st.session_state.page

# ──────────────────────────────────
# WATCHLIST
# ──────────────────────────────────
if page == "watchlist":
    if not li:
        st.session_state.page = "auth"; st.rerun()
    wl = get_watchlist(st.session_state.username)
    st.markdown(
        '<div class="nf-wl-hero">'
        '<div class="nf-wl-title">My List</div>'
        '<div class="nf-wl-count">' + str(len(wl)) + ' title' + ('s' if len(wl)!=1 else '') + ' saved</div>'
        '</div>',
        unsafe_allow_html=True
    )
    if not wl:
        st.markdown(
            '<div class="nf-wl-empty">'
            '<div class="nf-wl-empty-icon">&#127909;</div>'
            '<div class="nf-wl-empty-title">Your list is empty</div>'
            '<div style="font-size:14px;color:#333;">Browse and tap + to save movies here.</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        cols = st.columns(5)
        for i, mv in enumerate(wl):
            with cols[i % 5]:
                src   = mv.get("poster", BLANK)
                mtit  = mv.get("title",  "")
                myr   = mv.get("year",   "")
                murl  = mv.get("tmdb_url","")
                st.markdown(
                    '<div class="nf-wcard">'
                    '<img src="' + src + '" alt="' + mtit + '"'
                    ' loading="lazy" onerror="this.src=\'' + BLANK + '\'"/>'
                    '<div class="nf-wcbody">'
                    '<div class="nf-wctitle">' + mtit + '</div>'
                    '<div class="nf-wcyear">'  + myr  + '</div>'
                    '</div></div>',
                    unsafe_allow_html=True
                )
                ca, cb = st.columns(2)
                with ca:
                    if murl:
                        st.markdown(
                            '<a href="' + murl + '" target="_blank"'
                            ' style="display:block;text-align:center;padding:7px;'
                            'background:rgba(1,180,228,.08);color:#01b4e4;'
                            'border:1px solid rgba(1,180,228,.18);border-radius:4px;'
                            'font-size:10px;font-weight:700;text-decoration:none;">TMDB &#8599;</a>',
                            unsafe_allow_html=True
                        )
                with cb:
                    if st.button("Remove", key="rm_" + str(i)):
                        remove_from_watchlist(st.session_state.username, mtit)
                        st.rerun()
    if st.button("&#8592; Back", key="wl_bk"):
        st.session_state.page = "home"; st.rerun()

# ──────────────────────────────────
# PROFILE
# ──────────────────────────────────
elif page == "profile":
    if not li:
        st.session_state.page = "auth"; st.rerun()
    ud  = load_json(USERS_FILE).get(st.session_state.username, {})
    wlc = len(get_watchlist(st.session_state.username))
    st.markdown(
        '<div class="nf-prof-hero">'
        '<div class="nf-prof-row">'
        '<div class="nf-prof-av">' + st.session_state.avatar + '</div>'
        '<div>'
        '<div class="nf-prof-name">' + st.session_state.username + '</div>'
        '<div class="nf-prof-email">' + ud.get("email","") + '</div>'
        '</div></div></div>'
        '<div class="nf-prof-stats">'
        '<div class="nf-stat"><div class="nf-stat-n">' + str(wlc) + '</div><div class="nf-stat-l">Movies Saved</div></div>'
        '<div class="nf-stat"><div class="nf-stat-n">ML</div><div class="nf-stat-l">TF-IDF · Cosine Similarity</div></div>'
        '<div class="nf-stat"><div class="nf-stat-n">AI</div><div class="nf-stat-l">Smart Recommendations</div></div>'
        '</div>',
        unsafe_allow_html=True
    )
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        if st.button("My List", key="p_wl"):
            st.session_state.page = "watchlist"; st.rerun()
    with pc2:
        if st.button("Home",    key="p_h"):
            st.session_state.page = "home"; st.rerun()
    with pc3:
        if st.button("Sign Out",key="p_o"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.page      = "home"
            st.rerun()

# ──────────────────────────────────
# HOME
# ──────────────────────────────────
else:
    # ── HERO ──
    match = random.randint(90, 99)
    gtags = [g for g in feat["genres"].split()[:3] if len(g) > 2]
    ghtml = "".join(['<span class="nf-hgenre">' + g + '</span>' for g in gtags])
    yr    = feat["year"] or ""
    rat   = feat["rating"]
    desc  = feat["overview"][:185] + ("…" if len(feat["overview"]) > 185 else "")

    score = ""
    if rat and float(rat) > 0:
        score = (
            '<div class="nf-hscore">'
            '<div class="nf-hscore-n">' + str(rat) + '</div>'
            '<div class="nf-hscore-l">User Score</div>'
            '</div>'
        )

    add_btn = '<button class="nf-badd">&#65291;&nbsp; My List</button>' if li else ""

    st.markdown(
        '<div class="nf-hero">'
        '<div class="nf-hero-bg" style="background-image:url(\'' + feat["backdrop"] + '\');"></div>'
        '<div class="nf-hero-fog"></div>'
        '<div class="nf-hero-body">'
        '<div class="nf-badge">'
        '<div class="nf-badge-n">N</div>'
        '<span class="nf-badge-txt">Featured Film</span>'
        '</div>'
        '<div class="nf-htitle">' + feat["title"] + '</div>'
        '<div class="nf-hmeta">'
        '<span class="nf-hmatch">' + str(match) + '% Match</span>'
        + ('<span class="nf-hyear">' + yr + '</span>' if yr else "")
        + ghtml +
        '</div>'
        '<div class="nf-hdesc">' + desc + '</div>'
        '<div class="nf-hbtns">'
        '<button class="nf-bplay">&#9654;&nbsp; Play Trailer</button>'
        '<button class="nf-binfo">&#9432;&nbsp; More Info</button>'
        + add_btn +
        '</div>'
        '</div>'
        + score +
        '</div>',
        unsafe_allow_html=True
    )

    # hero action buttons
    hc = st.columns([1, 1, 1, 5])
    with hc[0]:
        if st.button("▶ Trailer", key="h_tr"):
            t = tmdb_trailer(feat["tmdb_id"])
            if t:
                st.session_state.trailer_url  = t
                st.session_state.show_trailer = True
                st.rerun()
            else:
                st.toast("No trailer available.", icon="⚠️")
    with hc[1]:
        if li:
            if st.button("＋ List", key="h_wl"):
                added = add_to_watchlist(st.session_state.username, {
                    "title": feat["title"], "poster": feat["poster"],
                    "year":  feat["year"],  "tmdb_url": feat["tmdb_url"]
                })
                st.toast("Added to My List!" if added else "Already in My List.")
    with hc[2]:
        if feat.get("tmdb_url"):
            st.markdown(
                '<a href="' + feat["tmdb_url"] + '" target="_blank"'
                ' style="display:inline-block;margin-top:4px;padding:8px 14px;'
                'background:rgba(1,180,228,.08);color:#01b4e4;'
                'border:1px solid rgba(1,180,228,.18);border-radius:5px;'
                'font-size:12px;font-weight:600;text-decoration:none;">TMDB &#8599;</a>',
                unsafe_allow_html=True
            )
    with hc[3]:
        if not li:
            if st.button("Sign in to save movies →", key="h_si"):
                st.session_state.page = "auth"; st.rerun()

    # ── ROW HELPER ──
    def render_row(label, df_s):
        cards = ""
        for row in df_s.itertuples():
            po, _, _, tu = tmdb_get(int(row.id))
            rat2   = str(round(row.vote_average, 1)) if hasattr(row,"vote_average") and isinstance(row.vote_average, float) else ""
            genres = str(getattr(row,"genres",""))
            g1     = genres.split()[0] if genres.strip() else ""
            mp     = str(random.randint(73, 98))
            click  = 'onclick="window.open(\'' + tu + '\',\'_blank\')"' if tu else ""
            rhtml  = ('<span class="nf-crat">&#11088; ' + rat2 + '</span>') if rat2 else ""
            cards += (
                '<div class="nf-card" ' + click + '>'
                '<img src="' + po + '" alt="' + str(row.title) + '" loading="lazy"'
                ' onerror="this.outerHTML=\'<div class=no-img>&#127909;</div>\'"/>'
                '<div class="nf-card-ov">'
                '<div class="nf-cbtns">'
                '<div class="nf-cb pl">&#9654;</div>'
                '<div class="nf-cb">&#65291;</div>'
                '<div class="nf-cb">&#128077;</div>'
                '<div class="nf-cb" style="margin-left:auto;">&#8964;</div>'
                '</div>'
                '<div class="nf-ctitle">' + str(row.title) + '</div>'
                '<div class="nf-cmeta">'
                '<span class="nf-cmatch">' + mp + '% Match</span>' + rhtml +
                '</div>'
                '<div class="nf-cgen">' + g1 + '</div>'
                '</div></div>'
            )
        st.markdown(
            '<div class="nf-row">'
            '<div class="nf-row-hdr">'
            '<span class="nf-row-title">' + label + '</span>'
            '<span class="nf-row-more">Explore All &rsaquo;</span>'
            '</div>'
            '<div class="nf-grid">' + cards + '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── CONTENT ROWS ──
    base = movies
    if "vote_count" in movies.columns:
        base = movies[movies["vote_count"] > 100]

    render_row("Trending Now",  base.sort_values("vote_average", ascending=False).head(5))
    render_row("Top Rated",     movies.sort_values("vote_average", ascending=False).iloc[5:10])

    if all_genres:
        pg = random.choice([g for g in all_genres if len(g) > 3][:12])
        gm = movies[movies["genres"].str.contains(pg, na=False)].head(5)
        if len(gm) >= 3:
            render_row("Best of " + pg, gm)

    render_row("New Arrivals", movies.sample(min(5, len(movies))))

    st.markdown('<div class="nf-divider"></div>', unsafe_allow_html=True)

    # ── SMART RECOMMENDATIONS ──
    st.markdown(
        '<div class="nf-rec">'
        '<div class="nf-rec-h">Smart Recommendations</div>'
        '<div class="nf-rec-s">Select a movie and mood — our AI finds your perfect next watch.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div style="padding:0 60px 22px;background:#0e0e0e;">', unsafe_allow_html=True)

        eng = st.radio("Engine", ["🤖 AI", "📊 TF-IDF"], horizontal=True, key="eng")
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            srch = st.text_input("Search Movie", placeholder="e.g. The Dark Knight", key="srch")
        with c2:
            mood = st.selectbox("Your Mood", [
                "Exciting & Thrilling","Light & Fun",
                "Emotional & Deep","Dark & Scary",
                "Feel-good & Uplifting","Mind-bending",
                "Romantic","Artistic & Indie"
            ], key="mood")
        with c3:
            gp = st.selectbox("Genre", ["Any"] + all_genres, key="gp")

        flt = movies.copy()
        if srch: flt = flt[flt["title"].str.contains(srch, case=False, na=False)]
        if gp != "Any": flt = flt[flt["genres"].str.contains(gp, na=False)]
        if flt.empty:   flt = movies

        sc, sb = st.columns([3, 1])
        with sc:
            sel = st.selectbox("Select a Movie", flt["title"].values, key="sel")
        with sb:
            st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
            go = st.button("Find Movies →", key="go")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RESULTS ──
    if go:
        use_ai = "AI" in eng
        badge  = "AI Pick" if use_ai else "Content Match"

        st.markdown('<div style="padding:0 60px 60px;background:#0e0e0e;">', unsafe_allow_html=True)

        if use_ai:
            with st.spinner("Finding your perfect movies…"):
                recs = rec_ai(sel, mood, gp if gp != "Any" else "any")
            if not recs:
                st.warning("Falling back to content-based.")
                recs  = rec_tfidf(sel)
                badge = "Content Match"
        else:
            with st.spinner("Analysing content…"):
                recs = rec_tfidf(sel)

        if recs:
            st.markdown(
                '<div class="nf-res-hdr">'
                '<div class="nf-res-tag">' + badge + '</div>'
                '<div class="nf-res-label">Because you liked <em>' + sel + '</em></div>'
                '</div>',
                unsafe_allow_html=True
            )
            rc = st.columns(5)
            for i, rec in enumerate(recs):
                with rc[i]:
                    rp  = rec.get("poster",   BLANK)
                    rt  = rec.get("title",    "")
                    ry  = rec.get("year",     "")
                    rrs = rec.get("reason",   "")
                    rg  = rec.get("genre",    "")
                    rtu = rec.get("tmdb_url", "")
                    rid = rec.get("tmdb_id",  0)

                    yr_h  = '<div class="nf-ryear">' + ry + '</div>' if ry else ""
                    gen_h = '<div class="nf-rgenre">' + rg + '</div>' if rg else ""

                    st.markdown(
                        '<div class="nf-rcard">'
                        '<img src="' + rp + '" alt="' + rt + '" loading="lazy"'
                        ' onerror="this.outerHTML=\'<div class=no-img>&#127909;</div>\'"/>'
                        '<div class="nf-rbody">'
                        '<div class="nf-rtitle">' + rt + '</div>'
                        + yr_h +
                        '<div class="nf-rreason">' + rrs + '</div>'
                        + gen_h +
                        '</div></div>',
                        unsafe_allow_html=True
                    )

                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("▶", key="tr_"+str(i), help="Trailer"):
                            if rid:
                                t = tmdb_trailer(rid)
                                if t:
                                    st.session_state.trailer_url  = t
                                    st.session_state.show_trailer = True
                                    st.rerun()
                                else:
                                    st.toast("No trailer.", icon="⚠️")
                    with b2:
                        if li:
                            if st.button("＋", key="wl_"+str(i), help="Save"):
                                added = add_to_watchlist(st.session_state.username, {
                                    "title": rt, "poster": rp,
                                    "year": ry, "tmdb_url": rtu
                                })
                                st.toast("Saved!" if added else "Already saved.")
                    with b3:
                        if rtu:
                            st.markdown(
                                '<a href="' + rtu + '" target="_blank"'
                                ' style="display:block;text-align:center;padding:6px 2px;'
                                'background:rgba(1,180,228,.07);color:#01b4e4;'
                                'border:1px solid rgba(1,180,228,.15);border-radius:4px;'
                                'font-size:10px;font-weight:700;text-decoration:none;margin-top:4px;">'
                                'TMDB&#8599;</a>',
                                unsafe_allow_html=True
                            )
        else:
            st.markdown(
                '<div style="text-align:center;padding:50px;color:#333;font-size:15px;">'
                'No results found. Try a different movie.</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown(
        '<div class="nf-footer">'
        '<div>'
        '<a href="#">Help Centre</a>'
        '<a href="#">Terms of Use</a>'
        '<a href="#">Privacy</a>'
        '<a href="#">Cookie Preferences</a>'
        '<a href="#">Corporate Information</a>'
        '<a href="#">Contact Us</a>'
        '</div>'
        '<div class="nf-footer-copy">'
        '&#169; 2024 Netflix AI &nbsp;&middot;&nbsp; '
        'Built by Alka Rani &nbsp;&middot;&nbsp; '
        'Python &nbsp;&#183;&nbsp; Streamlit &nbsp;&#183;&nbsp; '
        'Machine Learning &nbsp;&#183;&nbsp; NLP &nbsp;&#183;&nbsp; TMDB API'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
