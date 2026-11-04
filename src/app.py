import os
import re
import random
import difflib
import warnings
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from fpdf import FPDF
import time

warnings.filterwarnings('ignore')

# --- 1. Base Directory & App Setup ---
BASE_DIR = "C:/Users/natha/Downloads/Product Recommendation System"

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# --- 2. Model & Data Paths ---
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/tf_idf_vectorizer.pkl")
INTENT_MODEL_PATH = os.path.join(BASE_DIR, "models/naive_bayes_intent.pkl")
RECOMMENDER_MODEL_PATH = os.path.join(BASE_DIR, "models/random_forest_recommender.pkl")
FEATURE_MATRIX_PATH = os.path.join(BASE_DIR, "data/processed/instacart_feature_matrix.csv")
PRODUCTS_PATH = os.path.join(BASE_DIR, "data/raw/instacart/products.csv")
DEPARTMENTS_PATH = os.path.join(BASE_DIR, "data/raw/instacart/departments.csv")
AISLES_PATH = os.path.join(BASE_DIR, "data/raw/instacart/aisles.csv")
APRIORI_PATH = os.path.join(BASE_DIR, "data/processed/apriori_rules.csv") # <-- Updated

# --- 3. Load Models & Data ---
print("Booting web server & loading ML models...")
vectorizer = joblib.load(VECTORIZER_PATH)
intent_model = joblib.load(INTENT_MODEL_PATH)
rf_model = joblib.load(RECOMMENDER_MODEL_PATH)
feature_matrix = pd.read_csv(FEATURE_MATRIX_PATH)
apriori_rules = pd.read_csv(APRIORI_PATH) # <-- Updated

products_df = pd.read_csv(PRODUCTS_PATH)
departments_df = pd.read_csv(DEPARTMENTS_PATH)
aisles_df = pd.read_csv(AISLES_PATH)

catalog_full = products_df.merge(departments_df, on='department_id', how='left')
catalog_full = catalog_full.merge(aisles_df, on='aisle_id', how='left')

product_dict = dict(zip(catalog_full['product_id'], catalog_full['product_name']))
product_name_to_id = {v: k for k, v in product_dict.items()} # <-- Added for reverse lookup
all_product_names = catalog_full['product_name'].dropna().tolist()
available_users = feature_matrix['user_id'].unique().tolist()

# Precompute top 100 best-sellers for fast trending feeds
top_100_items = feature_matrix.sort_values(by='prod_total_purchases', ascending=False)['product_id'].unique()[:100].tolist()

# --- 4. Helper Functions ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def format_product(pid):
    row = catalog_full[catalog_full['product_id'] == pid]
    if not row.empty:
        r = row.iloc[0]
        return {
            "product_id": int(r['product_id']),
            "product_name": str(r['product_name']),
            "department": str(r['department']),
            "aisle": str(r['aisle'])
        }
    return {
        "product_id": int(pid),
        "product_name": product_dict.get(int(pid), f"Product {pid}"),
        "department": "General",
        "aisle": "General"
    }

def get_basket_affinity_recs(cart_items, top_k=3):
    if not cart_items:
        return []
        
    results = []
    represented_depts = set()
    
    # 1. Map out the departments in the cart (Newest items get priority)
    cart_depts = []
    for pid in reversed(cart_items):
        dept = format_product(pid).get('department', 'General')
        if dept not in cart_depts:
            cart_depts.append(dept)
            
    # 2. GUARANTEED REPRESENTATION: Force 1 recommendation per department
    for target_dept in cart_depts:
        if len(results) >= top_k:
            break
            
        trigger_items = [pid for pid in cart_items if format_product(pid).get('department', 'General') == target_dept]
        best_cand_id = None
        highest_score = -1
        
        # A. Try to find a valid Machine Learning rule for this specific department
        for item_id in trigger_items:
            item_name = product_dict.get(item_id)
            if not item_name:
                continue
                
            matches = apriori_rules[apriori_rules['antecedents'].astype(str).str.contains(rf'(?i)\b{re.escape(item_name)}\b', regex=True, na=False)]
            
            for _, row in matches.iterrows():
                consequents = [c.strip() for c in str(row['consequents']).split(',')]
                for c in consequents:
                    cand_id = product_name_to_id.get(c)
                    
                    if cand_id and cand_id not in cart_items and cand_id not in results:
                        cand_dept = format_product(cand_id).get('department', 'General')
                        
                        # Apply a penalty if we are looking at an item from a department we already suggested
                        score = row['lift']
                        if cand_dept in represented_depts:
                            score *= 0.1 
                            
                        if score > highest_score:
                            highest_score = score
                            best_cand_id = cand_id
                            
        # B. If ML finds NOTHING for this department (e.g., niche alcohol), force a category-specific fallback
        if not best_cand_id:
            dept_subset = catalog_full[catalog_full['department'] == target_dept]['product_id'].tolist()
            valid_subset = [pid for pid in dept_subset if pid not in cart_items and pid not in results]
            
            if valid_subset:
                rng = random.Random(sum(cart_items)) # Keeps the fallback dynamic but stable for the current cart state
                best_cand_id = rng.choice(valid_subset)
                
        # Lock in the candidate for this department
        if best_cand_id:
            results.append(best_cand_id)
            represented_depts.add(format_product(best_cand_id).get('department', 'General'))

    # 3. Fill remaining slots globally (If cart only has 1 department, this fills slots 2 and 3)
    if len(results) < top_k:
        global_cands = {}
        
        # Gather all rules for the entire cart
        for item_id in cart_items:
            item_name = product_dict.get(item_id)
            if not item_name: continue
            
            matches = apriori_rules[apriori_rules['antecedents'].astype(str).str.contains(rf'(?i)\b{re.escape(item_name)}\b', regex=True, na=False)]
            for _, row in matches.iterrows():
                consequents = [c.strip() for c in str(row['consequents']).split(',')]
                for c in consequents:
                    cand_id = product_name_to_id.get(c)
                    if cand_id and cand_id not in cart_items and cand_id not in results:
                        global_cands[cand_id] = max(global_cands.get(cand_id, 0), row['lift'])
                        
        # Sort and append the highest ML rules available
        for cand_id, _ in sorted(global_cands.items(), key=lambda x: x[1], reverse=True):
            if len(results) >= top_k: break
            results.append(cand_id)

    # 4. Ultimate Safety Net
    if len(results) < top_k:
        generic_hubs = ['Banana', 'Bag of Organic Bananas', 'Organic Whole Milk']
        for pid in top_100_items:
            if pid not in cart_items and pid not in results and product_dict.get(pid) not in generic_hubs:
                results.append(pid)
            if len(results) >= top_k: 
                break

    return results[:top_k]


def get_personalized_recs(user_id, cart_items, top_k=4):
    """Generates recommendations strictly based on the user's past purchase profile (Random Forest)."""
    user_data = feature_matrix[feature_matrix['user_id'] == user_id].copy()
    
    if user_data.empty:
        return []

    # Pull the active affinity items so we don't duplicate them in the personalized panel
    basket_recs = get_basket_affinity_recs(cart_items, top_k=3)
    
    X_predict = user_data.drop(['user_id', 'product_id', 'target'], axis=1, errors='ignore')
    user_data['rf_score'] = rf_model.predict_proba(X_predict)[:, 1]
    
    top_items = user_data.sort_values(by='rf_score', ascending=False)
    
    # Filter out items already in the cart OR already showing in the Frequently Bought Together panel
    top_items = top_items[~top_items['product_id'].isin(cart_items)]
    top_items = top_items[~top_items['product_id'].isin(basket_recs)]
    
    # Take the top 10 historical favorites and shuffle them slightly as the cart grows
    rf_recs = top_items['product_id'].head(10).tolist()
    random.seed(len(cart_items))
    random.shuffle(rf_recs)
    
    return rf_recs[:top_k]

# --- 5. Support Bot Mapping & Responses ---
INTENT_RESPONSES = {
    'payment_and_receipts': (
        "We accept all major credit/debit cards, Apple Pay, Google Pay, and SNAP EBT."
    )
}

MODEL_INTENT_MAP = {
    'track_order': 'track_order',
    'cancel_order': 'cancel_or_modify_order',
    'change_order': 'cancel_or_modify_order',
    'check_cancellation_fee': 'cancel_or_modify_order',
    'check_payment_methods': 'payment_and_receipts',
    'payment_issue': 'payment_and_receipts',
}

def resolve_intents(raw_text, predicted_raw_class):
    text = raw_text.lower().strip()

    if any(k in text for k in ['receipt', 'invoice', 'bill', 'total amount']):
        return 'generate_receipt'

    if any(k in text for k in ['track', 'where is my', 'order status', 'eta', 'delivery status', 'package']):
        return 'track_order'

    if any(k in text for k in ['cancel', 'cancellation', 'change order', 'modify', 'remove item', 'add item to order', 'stop order']):
        return 'cancel_or_modify_order'

    if any(k in text for k in ['pay', 'payment', 'card', 'apple pay', 'charged', 'declined']):
        return 'payment_and_receipts'

    return MODEL_INTENT_MAP.get(predicted_raw_class, None)

def generate_pdf_receipt(cart_items, user_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Instacart Fresh Market - Receipt", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Order Summary", ln=True, align='C') # <-- Removed shopper number
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    total = 0
    for pid in cart_items:
        name = product_dict.get(int(pid), f"Product {pid}")
        name = name.encode('latin-1', 'replace').decode('latin-1')
        
        rnd = random.Random(pid)
        price = round(rnd.uniform(2.99, 14.99), 2)
        total += price
        
        display_name = (name[:60] + '...') if len(name) > 60 else name
        
        pdf.cell(150, 10, txt=f"{display_name}")
        pdf.cell(40, 10, txt=f"${price:.2f}", ln=True, align='R')
        
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(150, 10, txt="TOTAL:")
    pdf.cell(40, 10, txt=f"${total:.2f}", ln=True, align='R')
    
    safe_user = str(user_id).replace(" ", "_")
    filename = f"receipt_{safe_user}_{int(time.time())}.pdf"
    
    filepath = os.path.join(BASE_DIR, "static", filename)
    pdf.output(filepath, 'F')
    
    print(f"[SYSTEM] Generated PDF Receipt at: {filepath}")
    return filename

# --- 6. API Endpoints ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/init', methods=['GET'])
def api_init():
    user_id = random.choice(available_users)
    departments = sorted(catalog_full['department'].dropna().unique().tolist())
    trending = [format_product(pid) for pid in random.sample(top_100_items, 8)]
    return jsonify({
        "user_id": int(user_id),
        "departments": departments,
        "trending": trending
    })

@app.route('/api/search', methods=['GET'])
def api_search():
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify([])

    direct = catalog_full[catalog_full['product_name'].str.lower().str.contains(query, na=False)]
    if not direct.empty:
        results = direct.head(10)['product_id'].tolist()
    else:
        close_names = difflib.get_close_matches(query, all_product_names, n=10, cutoff=0.5)
        results = catalog_full[catalog_full['product_name'].isin(close_names)]['product_id'].tolist()

    return jsonify([format_product(pid) for pid in results])

@app.route('/api/department', methods=['GET'])
def api_department():
    dept_name = request.args.get('name', '').strip()
    subset = catalog_full[catalog_full['department'].str.lower() == dept_name.lower()]
    if subset.empty:
        return jsonify([])
    sample_size = min(8, len(subset))
    sampled = subset.sample(sample_size)['product_id'].tolist()
    return jsonify([format_product(pid) for pid in sampled])

@app.route('/api/cart_updates', methods=['POST'])
def api_cart_updates():
    data = request.json or {}
    cart = [int(x) for x in data.get('cart', [])]
    
    # Strictly enforce the 10-item limit
    if len(cart) > 10:
        cart = cart[:10]
        
    user_id = int(data.get('user_id', available_users[0]))

    # Run the recommendation engines on the limited cart
    basket_complements = [format_product(pid) for pid in get_basket_affinity_recs(cart, top_k=3)]
    personalized = [format_product(pid) for pid in get_personalized_recs(user_id, cart, top_k=4)]

    return jsonify({
        "basket_complements": basket_complements,
        "personalized": personalized,
        "direct_cross_sells": basket_complements,
        "enforced_cart": cart
    })

@app.route('/download/receipt/<filename>')
def download_receipt(filename):
    filepath = os.path.join(BASE_DIR, "static", filename)
    if not os.path.exists(filepath):
        return f"File not found on server at {filepath}", 404
    return send_file(filepath, as_attachment=False)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json or {}
    raw_message = data.get('message', '').strip()
    
    cart = [int(x) for x in data.get('cart', [])]
    last_order = [int(x) for x in data.get('last_order', [])]
    user_id = data.get('user_id', 'Guest')

    if not raw_message:
        return jsonify({"reply": "How can I help you today?"})

    cleaned = clean_text(raw_message)
    vec = vectorizer.transform([cleaned])
    raw_pred = intent_model.predict(vec)[0]

    resolved_intent = resolve_intents(raw_message, raw_pred)
    action = None # Used to trigger Javascript events in the frontend

    if resolved_intent == 'generate_receipt':
        if not last_order:
            reply = "You haven't placed an order yet. Please complete a checkout first to generate a receipt."
        else:
            pdf_filename = generate_pdf_receipt(last_order, user_id)
            reply = f"Here is the receipt for your recent purchase: <br><br><a href='/download/receipt/{pdf_filename}' target='_blank' style='color: #108910; font-weight: bold; text-decoration: underline;'>Download PDF Receipt</a>"
            
    elif resolved_intent == 'track_order':
        if not last_order:
            reply = "You don't have any active orders to track right now. Place an order to see its status."
        else:
            reply = """
            <div class="tracking-card">
                <div class="tracking-header">Live Order Status</div>
                <div class="tracking-step completed">
                    <div class="step-icon">✓</div>
                    <div class="step-text">Order Confirmed</div>
                </div>
                <div class="tracking-step active">
                    <div class="step-icon"></div>
                    <div class="step-text">Shopper is Packing</div>
                </div>
                <div class="tracking-step">
                    <div class="step-icon"></div>
                    <div class="step-text">Out for Delivery</div>
                </div>
                <div class="eta-box">Est. Arrival: 45 Minutes</div>
            </div>
            """
            
    elif resolved_intent == 'cancel_or_modify_order':
        if not last_order:
            reply = "You don't have any active orders to cancel."
        else:
            reply = "I have successfully cancelled your active order. Your items have been cleared."
            action = "cancel_active_order" # Sends the command to script.js

    elif resolved_intent in INTENT_RESPONSES:
        reply = INTENT_RESPONSES[resolved_intent]
    else:
        reply = (
            "I'm specialized in order tracking, cancellations & modifications, and payment or receipt questions. "
            "For anything else, I can connect you with a live customer care representative."
        )

    return jsonify({
        "reply": reply,
        "intent": resolved_intent,
        "action": action
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)