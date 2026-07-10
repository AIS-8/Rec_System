"""
Streamlit storefront for the Amazon-Electronics recommender.

Calls only the FastAPI backend — different sections are powered by different models:
  * Guest homepage           -> Popularity              (/popular)
  * Logged-in homepage       -> Two-tower + LightGBM     (/recommend/{user})
  * "Because you liked X"     -> cosine on embeddings     (/because-you-liked/{user})
  * Item page "Similar items" -> cosine on embeddings     (/similar/{item})
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
PLACEHOLDER = "https://via.placeholder.com/300x300.png?text=No+Image"

# friendly display names so the demo-user picker reads like a real sign-in
NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Riley", "Casey", "Jamie", "Morgan",
         "Priya", "Diego", "Mei", "Omar", "Nina", "Leo", "Zara", "Ivan", "Ada",
         "Kai", "Luca", "Noor"]

# a colour per model, so the "which model made this" chip is scannable
MODEL_COLOR = {
    "Popular right now": "#f59e0b",       # amber
    "Two-tower + LightGBM": "#6366f1",    # indigo
    "Cosine similarity": "#14b8a6",       # teal
    "Personalized search": "#ec4899",     # pink
    "Search (TF-IDF)": "#8b5cf6",         # violet
}

st.set_page_config(page_title="ElectroRec", page_icon="🛒", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
:root { --brand:#131921; --accent:#febd69; --ink:#0f1111; }
.stApp { background:#eaeded; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
/* keep the sidebar open + its toggle always reachable */
[data-testid="stSidebarCollapsedControl"] { visibility:visible !important; }
.block-container { padding-top:2.75rem; max-width:1400px; }
/* keep the ElectroRec banner clear of Streamlit's top header strip */
.topbar { margin-top:6px; }

/* top banner */
.topbar { background:linear-gradient(90deg,#131921,#232f3e); color:#fff;
  padding:14px 22px; border-radius:12px; display:flex; align-items:center;
  justify-content:space-between; margin-bottom:18px; box-shadow:0 2px 8px rgba(0,0,0,.15); }
.topbar .brand { font-size:24px; font-weight:800; letter-spacing:.3px; }
.topbar .brand span { color:var(--accent); }
.topbar .who { font-size:14px; opacity:.9; }

/* section header */
.sect { display:flex; align-items:baseline; gap:12px; margin:6px 0 2px; }
.sect h3 { margin:0; font-size:22px; font-weight:800; color:var(--ink); }
.chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px;
  font-weight:700; color:#fff; }
.sub { color:#565959; font-size:12.5px; margin-bottom:8px; }

/* product card */
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border:1px solid #e3e6e6 !important;
  border-radius:12px; padding:10px 12px 12px; transition:.15s; height:100%; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { box-shadow:0 6px 18px rgba(0,0,0,.12);
  transform:translateY(-2px); }
[data-testid="stImage"] img { height:170px; width:100%; object-fit:contain; background:#fff;
  border-radius:8px; }
.card-title { font-size:14px; font-weight:600; color:#0f1111; line-height:1.3; height:54px;
  overflow:hidden; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; margin:6px 0 4px; }
.price { font-size:17px; font-weight:800; color:#0f1111; }
.price b { font-size:12px; vertical-align:super; font-weight:700; }
.noprice { font-size:13px; color:#8a8f8f; font-style:italic; }
.rating { color:#f59e0b; font-size:13px; font-weight:700; }
.cat { color:#565959; font-size:12px; }

/* buttons */
.stButton>button { border-radius:999px; border:1px solid #d5d9d9; background:#f7fafa;
  font-weight:600; font-size:13px; }
.stButton>button:hover { background:var(--accent); border-color:var(--accent); color:#111; }
/* active (in cart / saved / liked) action buttons get an accent highlight */
.stButton>button[kind="primary"] { background:var(--accent) !important; border-color:#f0a63a !important;
  color:#111 !important; }
.topbar .bag { font-size:14px; font-weight:700; opacity:.95; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- API helpers
@st.cache_data(ttl=5)
def api_get(path, params=None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=20); r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error on {path}: {exc}"); return None


def api_fresh(path, params=None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=20); r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"API error on {path}: {exc}"); return None


def name_for(uid: int) -> str:
    return NAMES[uid % len(NAMES)]


# ----------------------------------------------------------------- state
st.session_state.setdefault("user", None)     # None = Guest
st.session_state.setdefault("item", None)      # None = home, else item_idx
st.session_state.setdefault("cart", {})        # item_idx -> item (session-only demo cart)
st.session_state.setdefault("wishlist", {})    # item_idx -> item
st.session_state.setdefault("likes", {})       # item_idx -> item
st.session_state.setdefault("view", None)      # None=feed, "cart", or "wishlist"


def open_item(i): st.session_state.item = i
def go_home(): st.session_state.item = None; st.session_state.view = None
def open_cart(): st.session_state.view = "cart"; st.session_state.item = None
def open_wishlist(): st.session_state.view = "wishlist"; st.session_state.item = None


def _toggle(store_key, item, added, removed):
    store = st.session_state[store_key]
    i = item["item_idx"]
    if i in store:
        del store[i]; st.toast(removed)
    else:
        store[i] = item; st.toast(added)


def toggle_cart(item): _toggle("cart", item, "🛒 Added to cart", "Removed from cart")
def toggle_wishlist(item): _toggle("wishlist", item, "❤️ Saved to wishlist", "Removed from wishlist")
def toggle_like(item): _toggle("likes", item, "👍 Liked", "Unliked")


# ----------------------------------------------------------------- rendering
def chip(model: str) -> str:
    color = MODEL_COLOR.get(model, "#6b7280")
    return f'<span class="chip" style="background:{color}">🔧 {model}</span>'


def card(item: dict, key: str):
    with st.container(border=True):
        st.image(item.get("image_url") or PLACEHOLDER, use_container_width=True)
        title = (item.get("title") or "(untitled)")[:90]
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
        bits = []
        if item.get("price") is not None:
            p = f"{item['price']:.2f}".split(".")
            bits.append(f'<span class="price"><b>$</b>{p[0]}<b>{p[1]}</b></span>')
        else:
            bits.append('<span class="noprice">Price unavailable</span>')
        if item.get("average_rating") is not None:
            bits.append(f'<span class="rating">★ {item["average_rating"]:.1f}</span>')
        st.markdown("&nbsp;&nbsp;".join(bits), unsafe_allow_html=True)
        if item.get("category"):
            st.markdown(f'<div class="cat">{item["category"]}</div>', unsafe_allow_html=True)
        if item.get("model"):
            st.markdown(chip(item["model"]), unsafe_allow_html=True)
        i = item["item_idx"]
        a1, a2, a3 = st.columns(3)
        a1.button("✅" if i in st.session_state.cart else "🛒", key=f"{key}_cart",
                  help="Add to cart", use_container_width=True,
                  type="primary" if i in st.session_state.cart else "secondary",
                  on_click=toggle_cart, args=(item,))
        a2.button("❤️" if i in st.session_state.wishlist else "🤍", key=f"{key}_wish",
                  help="Save to wishlist", use_container_width=True,
                  type="primary" if i in st.session_state.wishlist else "secondary",
                  on_click=toggle_wishlist, args=(item,))
        a3.button("👍", key=f"{key}_like", help="Like", use_container_width=True,
                  type="primary" if i in st.session_state.likes else "secondary",
                  on_click=toggle_like, args=(item,))
        st.button("View  ·  Similar", key=key, on_click=open_item,
                  args=(item["item_idx"],), use_container_width=True)


def grid(items, prefix, cols=5):
    if not items:
        st.info("No recommendations available."); return
    for start in range(0, len(items), cols):
        row, columns = items[start:start + cols], st.columns(cols)
        for j, (col, it) in enumerate(zip(columns, row)):
            with col:
                card(it, f"{prefix}_{start + j}")


def section(resp):
    st.markdown(
        f'<div class="sect"><h3>{resp.get("section","")}</h3>{chip(resp.get("model",""))}</div>'
        f'<div class="sub">served in {resp.get("latency_ms","?")} ms</div>',
        unsafe_allow_html=True)


# ----------------------------------------------------------------- data + sidebar (secondary info only)
demo_users = api_get("/users", {"limit": 60}) or []
labels = ["🚪  Guest (browsing)"] + [
    f"👤  {name_for(u['user_idx'])}  ·  #{u['user_idx']}" for u in demo_users]

st.sidebar.markdown("### 🛒 ElectroRec")
st.sidebar.caption("Amazon Electronics · real-time recommender")
m = api_fresh("/metrics")
if m:
    st.sidebar.metric("Avg API response", f"{m['avg_latency_ms']} ms",
                      help=f"{m['requests']} requests · p95 {m['p95_latency_ms']} ms")
st.sidebar.markdown("---")

# your bag (session-only demo cart / wishlist / likes)
st.sidebar.markdown("### 🛍️ Your bag")
_cart, _wish, _likes = st.session_state.cart, st.session_state.wishlist, st.session_state.likes
bc1, bc2, bc3 = st.sidebar.columns(3)
bc1.metric("Cart", len(_cart)); bc2.metric("Saved", len(_wish)); bc3.metric("Likes", len(_likes))
st.sidebar.button("🛒  Open cart", use_container_width=True, on_click=open_cart)
st.sidebar.button("❤️  Open wishlist", use_container_width=True, on_click=open_wishlist)
st.sidebar.markdown("---")
st.sidebar.caption("Guests see popular items. Sign in to get personalized picks "
                   "(Two-tower + LightGBM) and a *Because you liked* row from your history. "
                   "Item pages show embedding-similar products. Use the **search bar** for "
                   "text search — signed-in, results are personalized to you. Recs resample on reload.")

# ----------------------------------------------------------------- top banner + sign-in (always visible in the main page)
banner = st.empty()
search_col, signin_col = st.columns([3.3, 1.4])
with search_col:
    query = st.text_input(
        "Search", value="", label_visibility="collapsed",
        placeholder="🔎  Search products")
with signin_col:
    choice = st.selectbox("👤  Sign in as", labels, index=0)
    if choice.startswith("🚪"):
        st.session_state.user = None
        _stats = None
    else:
        u = demo_users[labels.index(choice) - 1]
        st.session_state.user = u["user_idx"]
        _stats = f"🧾 {u['n_history']} past · {u['n_recent']} recent"
    if st.button("🔄  Refresh recommendations", use_container_width=True):
        go_home(); st.rerun()
    if _stats:
        st.caption(_stats)

who = "Browsing as guest" if st.session_state.user is None else \
    f"Hi, {name_for(st.session_state.user)}  ·  signed in"
n_cart, n_wish = len(st.session_state.cart), len(st.session_state.wishlist)
banner.markdown(
    f'<div class="topbar"><div class="brand">🛒 Electro<span>Rec</span></div>'
    f'<div class="who">{who} &nbsp;·&nbsp; '
    f'<span class="bag">🛒 {n_cart} &nbsp;&nbsp; ❤️ {n_wish}</span></div></div>',
    unsafe_allow_html=True)

# quick-access Cart / Wishlist buttons — always visible on the main page
nav_c, nav_w, _navsp = st.columns([1.3, 1.3, 5.4])
nav_c.button(f"🛒  Cart ({n_cart})", key="nav_cart", on_click=open_cart, use_container_width=True)
nav_w.button(f"❤️  Wishlist ({n_wish})", key="nav_wish", on_click=open_wishlist, use_container_width=True)


# ----------------------------------------------------------------- item detail page
if st.session_state.item is not None:
    st.button("←  Back to home", on_click=go_home)
    it = api_get(f"/items/{st.session_state.item}")
    if it:
        left, right = st.columns([1, 1.6])
        with left:
            with st.container(border=True):
                st.image(it.get("image_url") or PLACEHOLDER, use_container_width=True)
        with right:
            st.markdown(f"## {it.get('title') or '(untitled)'}")
            if it.get("category"):
                st.markdown(f"**Category:** {it['category']}")
            if it.get("price") is not None:
                st.markdown(f"**Price:** <span class='price'>${it['price']:.2f}</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("**Price:** <span class='noprice'>Price unavailable</span>",
                            unsafe_allow_html=True)
            if it.get("average_rating") is not None:
                st.markdown(f"**Rating:** <span class='rating'>★ {it['average_rating']:.1f}</span>",
                            unsafe_allow_html=True)
            di = it["item_idx"]
            b1, b2, b3 = st.columns(3)
            b1.button("🛒  In cart ✓" if di in st.session_state.cart else "🛒  Add to cart",
                      key="detail_cart", use_container_width=True,
                      type="primary" if di in st.session_state.cart else "secondary",
                      on_click=toggle_cart, args=(it,))
            b2.button("❤️  Saved" if di in st.session_state.wishlist else "🤍  Wishlist",
                      key="detail_wish", use_container_width=True,
                      type="primary" if di in st.session_state.wishlist else "secondary",
                      on_click=toggle_wishlist, args=(it,))
            b3.button("👍  Liked" if di in st.session_state.likes else "👍  Like",
                      key="detail_like", use_container_width=True,
                      type="primary" if di in st.session_state.likes else "secondary",
                      on_click=toggle_like, args=(it,))
        st.markdown("---")
        sim = api_get(f"/similar/{st.session_state.item}", {"k": 10})
        if sim:
            section(sim)
            grid(sim["items"], "sim")
    st.stop()


# ----------------------------------------------------------------- cart / wishlist pages
if st.session_state.view in ("cart", "wishlist"):
    is_cart = st.session_state.view == "cart"
    store = st.session_state.cart if is_cart else st.session_state.wishlist
    st.button("←  Back to shopping", on_click=go_home)
    heading = "🛒 Your cart" if is_cart else "❤️ Your wishlist"
    st.markdown(f"### {heading} · {len(store)} item(s)")
    if is_cart and store:
        total = sum(v.get("price") or 0 for v in store.values())
        st.caption(f"Estimated total: ${total:,.2f}  ·  (demo — no real checkout)")
    if store:
        grid(list(store.values()), "cartview" if is_cart else "wishview")
    else:
        st.info("Nothing here yet — add items with the 🛒 / 🤍 buttons on any product.")
    st.stop()


# ----------------------------------------------------------------- search results
# When a query is typed, search replaces the feed. Signed-in users get a
# personalized re-rank (TF-IDF retrieval -> LightGBM); guests get text relevance.
if query and query.strip():
    params = {"q": query.strip(), "k": 10}
    if st.session_state.user is not None:
        params["user_idx"] = st.session_state.user
    res = api_fresh("/search", params)
    if res is not None:
        section(res)
        if st.session_state.user is None:
            st.caption("Text-relevance results (TF-IDF over titles). "
                       "Sign in to get these ranked *for you*.")
        else:
            st.caption(f"Ranked for {name_for(st.session_state.user)} — TF-IDF text relevance "
                       "blended with a LightGBM re-rank on your features. Sign out and search "
                       "the same term to see the non-personalized order.")
        if res["items"]:
            grid(res["items"], "search")
        else:
            st.info("No products matched your search — try a different term.")
    st.stop()


# ----------------------------------------------------------------- homepage
if st.session_state.user is None:
    st.markdown("### Trending on ElectroRec")
    st.caption("You're browsing as a guest — showing what's popular right now.")
    pop = api_fresh("/popular", {"k": 10})
    if pop:
        section(pop); grid(pop["items"], "pop")
else:
    uid = st.session_state.user
    rec = api_fresh(f"/recommend/{uid}", {"k": 10})
    if rec:
        section(rec); grid(rec["items"], "rec")
    st.markdown("---")
    bel = api_fresh(f"/because-you-liked/{uid}", {"k": 10})
    if bel and bel.get("items"):
        section(bel); grid(bel["items"], "bel")
