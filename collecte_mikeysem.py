"""Collecte des donnees brutes Genius pour Mikeysem (meme source que LRFAF)."""
import re, json, time, requests
from bs4 import BeautifulSoup
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
S=requests.Session(); S.headers.update({"User-Agent":UA})
OUT=".cache_lex/mikeysem_raw.json"
ARTIST_ID=3152412

def api(path, **params):
    r=S.get(f"https://genius.com/api{path}", params=params, timeout=30)
    r.raise_for_status(); return r.json()["response"]

def scrape_lyrics(url):
    h=S.get(url, timeout=30).text
    soup=BeautifulSoup(h.replace('<br/>','\n').replace('<br>','\n'),"html.parser")
    divs=soup.find_all("div", attrs={"data-lyrics-container":"true"})
    if not divs: divs=soup.find_all("div", class_=re.compile(r"Lyrics__Container"))
    return "\n".join(d.get_text() for d in divs) if divs else None

songs=[]; page=1
while True:
    r=api(f"/artists/{ARTIST_ID}/songs", per_page=50, page=page, sort="popularity")
    songs+=r["songs"]
    if not r.get("next_page"): break
    page+=1; time.sleep(1)
print(f"{len(songs)} titres listes pour l'artiste {ARTIST_ID}")

out=[]
for s in songs:
    d=api(f"/songs/{s['id']}")["song"]; time.sleep(1.2)
    lyr=scrape_lyrics(d["url"]); time.sleep(1.2)
    rd=d.get("release_date_components") or {}
    out.append({
      "song_id": d["id"],
      "title": d["title"],
      "full_title": d["full_title"],
      "primary_artist": d["primary_artist"]["name"],
      "primary_artist_id": d["primary_artist"]["id"],
      "featured_artists": [a["name"] for a in d.get("featured_artists",[])],
      "url": d["url"],
      "language": d.get("language"),
      "release_date": d.get("release_date"),
      "year": rd.get("year"),
      "pageviews": (d.get("stats") or {}).get("pageviews"),
      "contributors": (d.get("stats") or {}).get("contributors"),
      "album": (d.get("album") or {}).get("name") if d.get("album") else None,
      "lyrics_raw": lyr,
    })
    print(f"  ok {d['title'][:40]:40} | {rd.get('year')} | pv={(d.get('stats') or {}).get('pageviews')} | contrib={(d.get('stats') or {}).get('contributors')} | lyrics={'oui' if lyr else 'NON'}")
json.dump(out, open(OUT,"w"), ensure_ascii=False, indent=1)
print("ecrit ->", OUT)
