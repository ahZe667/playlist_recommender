
from flask import Flask, request, render_template_string
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random, json

DATASET_PATH   = Path("dataset.csv")
SAMPLE_SIZE    = 30
MIN_RATE       = 10
POPULAR_W      = 0.2          
ALL_TAG        = "__ALL__"
NEG_PENALTY    = 0.3
CANDIDATE_POOL = 6000
DIVERSITY_W    = 0.20
EUC_W          = 0.5   
app = Flask(__name__)

def load_and_dedupe(path):
    df = pd.read_csv(path).drop(columns=["album_name"], errors="ignore")
    df["tn"] = df.track_name.str.lower().str.strip()
    df["an"] = df.artists.str.lower().str.strip()
    agg = {"popularity":"mean", "track_id":lambda x: x.sample(1).iloc[0]}
    for c in df.columns:
        if c not in ("track_name","artists","popularity","track_id","tn","an","track_genre"):
            agg[c] = "first"
    grp  = df.groupby(["tn","an"], as_index=False).agg(agg)
    meta = df[["tn","an","track_name","artists","track_genre"]].drop_duplicates(["tn","an"])
    out  = grp.merge(meta, on=["tn","an"], how="left") \
              .drop(columns=["tn","an"]) \
              .drop_duplicates(["track_name","artists"])
    return out.reset_index(drop=True)
def build_feature_vectors(df, scaler=None):
    num    = ["duration_ms","danceability","energy","loudness","speechiness",
              "acousticness","instrumentalness","liveness","valence","tempo"]
    dfc    = df.copy()
    scaler = scaler or StandardScaler()
    Z      = scaler.fit_transform(dfc[num])
    scaled = pd.DataFrame(Z, columns=[f"{c}_z" for c in num], index=dfc.index)
    k      = dfc.key.astype(int)
    keydf  = pd.DataFrame({
        "key_sin": np.sin(2*np.pi*k/12),
        "key_cos": np.cos(2*np.pi*k/12)
    }, index=dfc.index)
    mode   = dfc["mode"].astype(int).to_frame("mode")
    tsig   = dfc.time_signature.astype(int).clip(3,7)
    ts_oh  = pd.get_dummies(tsig, prefix="tsig") \
                  .reindex(columns=[f"tsig_{i}" for i in range(3,8)], fill_value=0)
    feat   = pd.concat([scaled, keydf, mode, ts_oh], axis=1).reset_index(drop=True)
    return pd.concat([dfc[["track_id","artists","track_name"]].reset_index(drop=True),
                      feat], axis=1), scaler
def _l2(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.where(n<1e-6,1,n)).astype(np.float32)
def _wmean(X, w):
    return (X * w[:, None]).sum(0, keepdims=True) / w.sum()
def _cluster(X, w):
    n = X.shape[0]
    if n < 2:
        return _l2(_wmean(X, w))
    kmax = min(6, int(np.ceil(np.sqrt(n))))
    best = (1, -1, None)
    for k in range(2, kmax+1):
        lbl = KMeans(k, n_init="auto", random_state=0).fit_predict(X)
        s   = silhouette_score(X, lbl)
        if s > best[1] + 0.05:
            best = (k, s, lbl)
    if best[0] == 1:
        return _l2(_wmean(X, w))
    cents = np.vstack([_wmean(X[best[2]==i], w[best[2]==i]) for i in range(best[0])])
    return _l2(cents)

full_df, scaler = load_and_dedupe(DATASET_PATH), None
feat_df, _      = build_feature_vectors(full_df, scaler)
FEATURE_COLS    = [c for c in feat_df.columns if c not in ("track_id","track_name","artists")]
X_GLOBAL        = _l2(feat_df[FEATURE_COLS].to_numpy(np.float32))
ID2IDX          = {tid: i for i, tid in enumerate(feat_df.track_id)}
genres          = sorted(full_df[full_df.popularity>60].track_genre.dropna().unique())

STYLE = "<style>body{font-family:Arial,sans-serif;background:#f0f0f0;}.container{max-width:700px;margin:30px auto;background:#fff;padding:20px;border-radius:8px;box-shadow:0 0 10px rgba(0,0,0,0.1);}h1{text-align:center;}table{width:100%;border-collapse:collapse;margin-top:20px;}th,td{padding:8px;}thead{background:#009879;color:#fff;}tbody tr:nth-child(even){background:#f3f3f3;}input[type=number]{width:50px;}button{margin-top:10px;padding:8px 12px;background:#009879;color:#fff;border:none;border-radius:4px;}button:hover{background:#007f65;}label{margin-right:10px;}.form-search{display:flex;gap:8px;margin-bottom:15px;}.form-search input[type=text]{flex:1;padding:6px;}</style>"
GENRE_TMPL = "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Genres</title>" + STYLE + "</head><body><div class='container'><h1>Select Your Favorite Genres</h1><form method='post'><div><input type='checkbox' name='genre' value='" + ALL_TAG + "' id='all'><label for='all'><strong>All</strong></label></div><hr>{% for g in genres %}<div><input type='checkbox' name='genre' value='{{g}}' id='g{{loop.index}}'><label for='g{{loop.index}}'>{{g}}</label></div>{% endfor %}<button type='submit'>Next</button></form></div></body></html>"
FORM_TMPL = "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Rate</title>" + STYLE + "</head><body><div class='container'><h1>Rate (1–10) or <em>Skip</em></h1><form method='post' action='/recommend'><div class='form-search'><input type='text' name='q' placeholder='Search'><button formaction='/search'>Search</button></div><table><thead><tr><th>Artist</th><th>Title</th><th>Rating</th><th>Skip</th></tr></thead><tbody>{% for i, tr in sample.iterrows() %}<tr><td>{{tr.artists}}</td><td>{{tr.track_name}}</td><td><input type='number' name='rating_{{i}}' min='1' max='10'></td><td><input type='checkbox' name='skip_{{i}}'></td><input type='hidden' name='track_id_{{i}}' value='{{tr.track_id}}'></tr>{% endfor %}</tbody></table><input type='hidden' name='count' value='{{sample.shape[0]}}'><input type='hidden' name='genres' value='{{genres|join(\"|\")}}'><input type='hidden' name='seen' value='{{seen|join(\",\")}}'><input type='hidden' name='rated_json' value='{{rated_json|tojson}}'><button type='submit'>Submit</button></form></div></body></html>"
RESULT_TMPL = "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Playlist</title>" + STYLE + "</head><body><div class='container'><h1>Your Playlist</h1><table><thead><tr><th>Artist</th><th>Title</th><th>Score</th></tr></thead><tbody>{% for r in recs.itertuples() %}<tr><td>{{r.artists}}</td><td>{{r.track_name}}</td><td>{{'%.2f' % r.final_score}}</td></tr>{% endfor %}</tbody></table><p style='text-align:center'><a href='/'>Start over</a></p></div></body></html>"

def collect_ratings(form, count):
    rates, skip = [], []
    for i in range(count):
        tid = form.get(f"track_id_{i}")
        if not tid: continue
        if form.get(f"skip_{i}"):
            skip.append(tid)
        else:
            r = form.get(f"rating_{i}")
            if r:
                rates.append({"track_id": tid, "rating": int(r)})
    return rates, skip, None
def recommend(feat_df, rate_df, top_n=40):
    id2i    = ID2IDX
    X       = X_GLOBAL
    ridx    = rate_df.track_id.map(id2i).to_numpy()
    c       = rate_df.rating.clip(1,10).astype(float).to_numpy() - 5.5
    pos_mask = c > 0
    neg_mask = c < 0
    pos_idx  = ridx[pos_mask]
    neg_idx  = ridx[neg_mask]
    if pos_idx.size:
        cent_p  = _cluster(X[pos_idx], c[pos_mask])
        sim_p   = (cent_p @ X.T).max(0)
    else:
        sim_p   = np.zeros(X.shape[0])
    if neg_idx.size:
        cent_n  = _l2(_wmean(X[neg_idx], -c[neg_mask]))
        sim_n   = (X @ cent_n.T).squeeze()
    else:
        sim_n   = np.zeros(X.shape[0])
    cos_part = sim_p - NEG_PENALTY * sim_n
    sim_e_p  = 1 - np.sqrt(np.clip(2 - 2*sim_p, 0, 2)) / 2
    sim_e_n  = 1 - np.sqrt(np.clip(2 - 2*sim_n, 0, 2)) / 2
    euc_part = sim_e_p - NEG_PENALTY * sim_e_n
    sims = (1 - EUC_W)*cos_part + EUC_W*euc_part
    sims[ridx] = -np.inf
    if np.isneginf(sims).all():
        return pd.DataFrame(columns=["track_id","score"])
    pool = min(CANDIDATE_POOL, sims.size)
    idx  = np.argpartition(-sims, pool-1)[:pool]
    c_sim= sims[idx]
    sel, vecs = [], []
    while len(sel) < top_n and idx.size:
        if not vecs:
            p = np.argmax(c_sim)
        else:
            V   = np.vstack(vecs)
            div = (X[idx] @ V.T).max(axis=1)
            p   = np.argmax((1-DIVERSITY_W)*c_sim - DIVERSITY_W*div)
        pick = int(idx[p])
        sel.append(pick)
        vecs.append(X[pick])
        mask = idx != pick
        idx, c_sim = idx[mask], c_sim[mask]
    return pd.DataFrame({"track_id": feat_df.track_id.values[sel], "score": sims[sel]})
@app.route("/", methods=["GET","POST"])
def choose_genre():
    if request.method=="POST":
        sel = request.form.getlist("genre") or []
        if ALL_TAG in sel:
            sel = genres
        if not sel:
            return "No genres selected", 400
        pool = full_df[full_df.track_genre.isin(sel)]
        sample = pool.groupby("artists", group_keys=False)\
                     .apply(lambda x: x.sample(min(len(x),2)))\
                     .sample(n=min(SAMPLE_SIZE,len(pool)), random_state=random.randint(0,9999))\
                     .reset_index(drop=True)
        return render_template_string(FORM_TMPL, sample=sample,
                                      genres=sel, seen=[], rated_json=[])
    return render_template_string(GENRE_TMPL, genres=genres)
@app.route("/search", methods=["POST"])
def search():
    q = request.form.get("q","").strip()
    if not q:
        return "No search query", 400
    cnt   = int(request.form.get("count","0"))
    seen  = set(request.form.get("seen","").split(",")) - {""}
    rated = json.loads(request.form.get("rated_json","[]"))
    rates, skip = collect_ratings(request.form, cnt)[:2]
    seen |= set(skip) | {r["track_id"] for r in rates}
    rated += rates
    mask = full_df.track_name.str.contains(q, case=False) | \
           full_df.artists.str.contains(q, case=False)
    df = full_df[mask & ~full_df.track_id.isin(seen)].sort_values("popularity",ascending=False)
    if df.empty:
        return f"No results found for “{q}”", 404
    sample = df.head(SAMPLE_SIZE).reset_index(drop=True)
    return render_template_string(FORM_TMPL, sample=sample,
                                  genres=request.form.get("genres","").split("|"),
                                  seen=list(seen), rated_json=rated)
@app.route("/recommend", methods=["POST"])
def rec_page():
    cnt    = int(request.form["count"])
    seen   = set(request.form.get("seen","").split(",")) - {""}
    rated  = json.loads(request.form.get("rated_json","[]"))
    rates, skip = collect_ratings(request.form, cnt)[:2]
    seen  |= set(skip) | {r["track_id"] for r in rates}
    rated += rates
    if skip:
        pool = full_df[~full_df.track_id.isin(seen)]
        if pool.empty:
            return "No more tracks available", 400
        sample = pool.groupby("artists", group_keys=False)\
                     .apply(lambda x: x.sample(min(len(x),2)))\
                     .sample(n=len(skip), random_state=random.randint(0,9999))\
                     .reset_index(drop=True)
        return render_template_string(FORM_TMPL, sample=sample,
                                      genres=request.form.get("genres","").split("|"),
                                      seen=list(seen|set(sample.track_id)),
                                      rated_json=rated)
    if len(rated) < MIN_RATE:
        return f"Please rate at least {MIN_RATE} tracks", 400
    recs = recommend(feat_df, pd.DataFrame(rated), top_n=40)
    meta = full_df[["track_id","track_name","artists","popularity","track_genre"]]\
            .drop_duplicates("track_id")
    recs = recs.merge(meta, on="track_id")
    sel_genres = set(request.form.get("genres","").split("|")) - {""}
    if sel_genres:
        allowed_ids = set(full_df[full_df.track_genre.isin(sel_genres)].track_id)
        recs = recs[recs.track_id.isin(allowed_ids)]
    recs["final_score"] = (1-POPULAR_W)*recs.score + POPULAR_W*(np.log1p(recs.popularity)/np.log(101))
    recs = recs.sort_values("final_score", ascending=False)
    return render_template_string(RESULT_TMPL, recs=recs)
if __name__ == "__main__":
    app.run(debug=True)