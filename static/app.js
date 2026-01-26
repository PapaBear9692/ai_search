const form = document.getElementById("searchForm");
const queryInput = document.getElementById("queryInput");
const loading = document.getElementById("loading");
const alertBox = document.getElementById("alertBox");

const resultsPanel = document.getElementById("resultsPanel");
const resultsList = document.getElementById("resultsList");
const countBadge = document.getElementById("countBadge");

function setLoading(on) {
  if (on) loading.classList.remove("d-none");
  else loading.classList.add("d-none");
}

function showError(msg) {
  alertBox.textContent = msg;
  alertBox.classList.remove("d-none");
}

function clearError() {
  alertBox.classList.add("d-none");
  alertBox.textContent = "";
}

function escapeHtml(str) {
  return (str || "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[m]));
}

function safeUrl(url) {
  if (!url) return "";
  try {
    return new URL(url).href;
  } catch {
    try {
      return new URL("https://" + url).href;
    } catch {
      return "";
    }
  }
}

function firstOrDash(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return "—";
  return arr[0];
}

function renderEmpty(message = "No results yet. Try a search.") {
  resultsList.innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  resultsPanel.classList.remove("d-none");
  countBadge.textContent = "0";
}

function vendorCard(v) {
  const name = escapeHtml(v.business_name || "—");

  const websiteUrl = safeUrl(v.website || "");
  const sourceUrl = safeUrl(v.source_page || "");

  const email = firstOrDash(v.emails);
  const phone = firstOrDash(v.phone_numbers);

  const address = Array.isArray(v.addresses) && v.addresses.length
    ? escapeHtml(v.addresses[0])
    : "—";

  const websiteHtml = websiteUrl
    ? `<a href="${escapeHtml(websiteUrl)}" target="_blank" rel="noopener">${escapeHtml(websiteUrl)}</a>`
    : "—";

  const emailHtml = (email !== "—")
    ? `<a href="mailto:${escapeHtml(email)}">${escapeHtml(email)}</a>`
    : "—";

  const phoneHtml = (phone !== "—")
    ? `<a href="tel:${escapeHtml(phone)}">${escapeHtml(phone)}</a>`
    : "—";

  const sourceHtml = sourceUrl
    ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener">open page</a>`
    : "—";

  return `
    <div class="vendor-card">
      <div class="vendor-title">
        <span>${name}</span>
        <span class="small text-muted">${sourceHtml}</span>
      </div>

      <div class="vendor-meta">
        <div class="kv"><div class="k">Website</div><div class="v">${websiteHtml}</div></div>
        <div class="kv"><div class="k">Email</div><div class="v">${emailHtml}</div></div>
        <div class="kv"><div class="k">Phone</div><div class="v">${phoneHtml}</div></div>
        <div class="kv"><div class="k">Location</div><div class="v">${address}</div></div>
      </div>
    </div>
  `;
}

function renderVendors(vendors) {
  resultsList.innerHTML = "";
  resultsPanel.classList.remove("d-none");

  const list = Array.isArray(vendors) ? vendors : [];
  countBadge.textContent = String(list.length);

  if (list.length === 0) {
    renderEmpty("No vendors found. Try a different query.");
    return;
  }

  for (const v of list) {
    resultsList.insertAdjacentHTML("beforeend", vendorCard(v));
  }
}

async function runSearch(query) {
  clearError();
  setLoading(true);
  resultsPanel.classList.add("d-none");
  resultsList.innerHTML = "";

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });

    const data = await res.json().catch(() => null);
    if (!res.ok || !data || data.ok !== true) {
      throw new Error((data && data.error) ? data.error : `Request failed (${res.status})`);
    }

    renderVendors(data.vendors);

  } catch (e) {
    showError(e.message || "Something went wrong.");
    renderEmpty("Error occurred. Try again.");
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (!q) {
    showError("Type something to search.");
    return;
  }
  runSearch(q);
});

// initial state
renderEmpty();
