let currentUserId = null;
let shoppingCart = [];
let lastOrder = [];
let lastOrderDetails = [];

// DOM Elements
const ordersDrawer = document.getElementById('ordersDrawer');
const openOrdersBtn = document.getElementById('openOrdersBtn');
const closeOrdersBtn = document.getElementById('closeOrdersBtn');
const ordersItemsList = document.getElementById('ordersItemsList');
const cancelOrderBtn = document.getElementById('cancelOrderBtn');
const productsGrid = document.getElementById('productsGrid');
const categoriesBar = document.getElementById('categoriesBar');
const personalizedRecs = document.getElementById('personalizedRecs');
const basketComplements = document.getElementById('basketComplements');
const cartCount = document.getElementById('cartCount');
const sectionTitle = document.getElementById('sectionTitle');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const cartDrawer = document.getElementById('cartDrawer');
const cartItemsList = document.getElementById('cartItemsList');
const openCartBtn = document.getElementById('openCartBtn');
const closeCartBtn = document.getElementById('closeCartBtn');
const toast = document.getElementById('toast');
const customNotification = document.getElementById('custom-notification');

// New Drawer Elements
const chatDrawer = document.getElementById('chatDrawer');
const openChatBtn = document.getElementById('openChatBtn');
const closeChatBtn = document.getElementById('closeChatBtn');
const drawerOverlay = document.getElementById('drawerOverlay');

// Chat Elements
const chatBox = document.getElementById('chatBox');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');

// Initialize Storefront Session
window.addEventListener('DOMContentLoaded', async () => {
    const res = await fetch('/api/init');
    const data = await res.json();
    currentUserId = data.user_id;

    // Render Department Filter Pills
    data.departments.forEach(dept => {
        const pill = document.createElement('button');
        pill.className = 'cat-pill';
        pill.innerText = dept.charAt(0).toUpperCase() + dept.slice(1);
        pill.onclick = () => filterDepartment(dept, pill);
        categoriesBar.appendChild(pill);
    });

    // Render Initial Popular Catalog
    renderProducts(data.trending);
    refreshRecommendations();
});

// Render Product Cards
function renderProducts(products) {
    productsGrid.innerHTML = '';
    if (!products || products.length === 0) {
        productsGrid.innerHTML = '<p class="empty-state">No products found.</p>';
        return;
    }
    products.forEach(p => {
        const card = document.createElement('div');
        card.className = 'product-card';
        card.innerHTML = `
            <div>
                <h4>${p.product_name}</h4>
                <div class="tag">${p.department} &bull; ${p.aisle}</div>
            </div>
            <button class="add-btn" onclick="addToCart(${p.product_id}, '${p.product_name.replace(/'/g, "\\'")}')">
                + Add to Cart
            </button>
        `;
        productsGrid.appendChild(card);
    });
}

// Load Popular Items
async function loadTrending() {
    sectionTitle.innerText = "Popular Items";
    document.querySelectorAll('.cat-pill').forEach(b => b.classList.remove('active'));
    categoriesBar.firstElementChild.classList.add('active');
    const res = await fetch('/api/init');
    const data = await res.json();
    renderProducts(data.trending);
}

// Filter by Department
async function filterDepartment(deptName, pillElement) {
    sectionTitle.innerText = `${deptName.charAt(0).toUpperCase() + deptName.slice(1)} Products`;
    document.querySelectorAll('.cat-pill').forEach(b => b.classList.remove('active'));
    pillElement.classList.add('active');

    const res = await fetch(`/api/department?name=${encodeURIComponent(deptName)}`);
    const data = await res.json();
    renderProducts(data);
}

// Search Handler
async function performSearch() {
    const q = searchInput.value.trim();
    if (!q) return;
    sectionTitle.innerText = `Search results for "${q}"`;
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    renderProducts(data);
}

searchBtn.onclick = performSearch;
searchInput.onkeyup = (e) => { if (e.key === 'Enter') performSearch(); };

// Add Item to Cart
async function addToCart(productId, productName) {
    // Check the correct array name
    if (shoppingCart.length >= 10) {
        alert("You have reached the maximum cart limit of 10 items.");
        return; // Halts execution before updating the cart
    }
    
    shoppingCart.push({ id: productId, name: productName });
    updateCartCount();

    // Extract just the IDs for the backend
    const cartIds = shoppingCart.map(item => item.id);

    const res = await fetch('/api/cart_updates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: currentUserId, cart: cartIds })
    });
    const data = await res.json();

    updateRecSidebars(data);

    let toastMsg = `Added ${productName} to cart.`;
    if (data.direct_cross_sells && data.direct_cross_sells.length > 0) {
        toastMsg += ` Frequently bought with: ${data.direct_cross_sells[0].product_name}`;
    }
    showToast(toastMsg);
}

// Update Cart Count Badge
function updateCartCount() {
    cartCount.innerText = shoppingCart.length;
}

// Remove from Cart
function removeFromCart(index) {
    shoppingCart.splice(index, 1);
    updateCartCount();
    renderCartDrawer();
    refreshRecommendations();
}

// Fetch Recommendations
async function refreshRecommendations() {
    const cartIds = shoppingCart.map(item => item.id);
    const res = await fetch('/api/cart_updates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: currentUserId, cart: cartIds })
    });
    const data = await res.json();
    updateRecSidebars(data);
}

// Render Sidebar Lists
function updateRecSidebars(data) {
    personalizedRecs.innerHTML = '';
    if (data.personalized && data.personalized.length > 0) {
        data.personalized.forEach(p => {
            const item = document.createElement('div');
            item.className = 'rec-item';
            item.innerHTML = `
                <span>${p.product_name}</span>
                <button class="add-btn" onclick="addToCart(${p.product_id}, '${p.product_name.replace(/'/g, "\\'")}')">+</button>
            `;
            personalizedRecs.appendChild(item);
        });
    } else {
        personalizedRecs.innerHTML = '<p class="empty-state">No recommendations yet.</p>';
    }

    basketComplements.innerHTML = '';
    if (data.basket_complements && data.basket_complements.length > 0) {
        data.basket_complements.forEach(p => {
            const item = document.createElement('div');
            item.className = 'rec-item';
            item.innerHTML = `
                <span>${p.product_name}</span>
                <button class="add-btn" onclick="addToCart(${p.product_id}, '${p.product_name.replace(/'/g, "\\'")}')">+</button>
            `;
            basketComplements.appendChild(item);
        });
    } else {
        basketComplements.innerHTML = '<p class="empty-state">Add items to view pairings.</p>';
    }
}

// Cancel the active order
function cancelOrder() {
    if (confirm("Are you sure you want to cancel this order?")) {
        lastOrder = [];
        lastOrderDetails = [];
        renderOrdersDrawer();
        alert("Your order has been cancelled successfully.");
        closeDrawers();
    }
}

// Render the Orders Drawer
async function renderOrdersDrawer() {
    ordersItemsList.innerHTML = '';
    
    if (lastOrderDetails.length === 0) {
        ordersItemsList.innerHTML = '<p class="empty-state">You have no active orders.</p>';
        cancelOrderBtn.style.display = 'none';
        return;
    }

    cancelOrderBtn.style.display = 'block';

    // Inject the visual tracker at the top of the orders list
    const trackingHtml = `
        <div class="tracking-card" style="margin-bottom: 1.5rem;">
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
        <h4 style="margin-bottom: 0.75rem; font-size: 0.9rem; color: var(--text-main);">Order Items:</h4>
    `;
    
    const trackerContainer = document.createElement('div');
    trackerContainer.innerHTML = trackingHtml;
    ordersItemsList.appendChild(trackerContainer);

    // List the actual items beneath the tracker
    lastOrderDetails.forEach((item) => {
        const entry = document.createElement('div');
        entry.className = 'cart-entry';
        entry.innerHTML = `<span>${item.name}</span>`;
        ordersItemsList.appendChild(entry);
    });
}

// Drawer Handlers (Cart, Chat, & Orders)
openCartBtn.onclick = () => {
    renderCartDrawer();
    cartDrawer.classList.add('open');
    drawerOverlay.style.display = 'block';
};

openChatBtn.onclick = () => {
    chatDrawer.classList.add('open');
    drawerOverlay.style.display = 'block';
};

openOrdersBtn.onclick = () => {
    renderOrdersDrawer();
    ordersDrawer.classList.add('open');
    drawerOverlay.style.display = 'block';
};

function closeDrawers() {
    cartDrawer.classList.remove('open');
    chatDrawer.classList.remove('open');
    ordersDrawer.classList.remove('open');
    drawerOverlay.style.display = 'none';
}

closeCartBtn.onclick = closeDrawers;
closeChatBtn.onclick = closeDrawers;
closeOrdersBtn.onclick = closeDrawers;
drawerOverlay.onclick = closeDrawers;

async function renderCartDrawer() {
    cartItemsList.innerHTML = '';
    if (shoppingCart.length === 0) {
        cartItemsList.innerHTML = '<p class="empty-state">Your cart is empty.</p>';
        return;
    }

    shoppingCart.forEach((item, idx) => {
        const entry = document.createElement('div');
        entry.className = 'cart-entry';
        entry.innerHTML = `
            <span>${item.name}</span>
            <button class="remove-btn" onclick="removeFromCart(${idx})">&times; Remove</button>
        `;
        cartItemsList.appendChild(entry);
    });
}

// Native Notification Handlers
function showOrderSuccess() {
    customNotification.classList.remove('hidden');
    
    // Automatically hide after 4 seconds
    setTimeout(() => {
        closeNotification();
    }, 4000);
}

function closeNotification() {
    customNotification.classList.add('hidden');
}

function checkout() {
    if (shoppingCart.length === 0) return;
    
    // Trigger the new custom UI dropdown instead of alert()
    showOrderSuccess();
    
    // Save IDs for the backend receipt generator, and full objects for the UI
    lastOrder = shoppingCart.map(item => item.id); 
    lastOrderDetails = [...shoppingCart];
    
    shoppingCart = [];
    updateCartCount();
    closeDrawers();
    refreshRecommendations();
}

function showToast(msg) {
    toast.innerText = msg;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 4000);
}

// Support Chat
async function sendChatMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    chatBox.innerHTML += `<div class="user-msg">${msg}</div>`;
    chatInput.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;

    const cartIds = shoppingCart.map(item => item.id);

    const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            message: msg,
            cart: cartIds,
            last_order: lastOrder,       
            user_id: currentUserId
        })
    });
    const data = await res.json();
    
    // Render the bot's text reply
    chatBox.innerHTML += `<div class="bot-msg">${data.reply}</div>`;
    
    // Listen for backend action commands
    if (data.action === 'cancel_active_order') {
        lastOrder = [];
        lastOrderDetails = [];
        renderOrdersDrawer(); // Instantly clears the graph and items from the UI
    }
    
    chatBox.scrollTop = chatBox.scrollHeight;
}

chatSendBtn.onclick = sendChatMessage;
chatInput.onkeyup = (e) => { if (e.key === 'Enter') sendChatMessage(); };