// ── Toast ──────────────────────────────────────────────────────────────
function showToast(msg, type = "success") {
    const t = document.getElementById("toast");
    if (!t) return;
    t.textContent = (type === "success" ? "✅ " : "❌ ") + msg;
    t.className = `toast ${type} show`;
    setTimeout(() => t.classList.remove("show"), 3500);
}

// ── Loader ─────────────────────────────────────────────────────────────
function showLoader(msg = "Procesando...") {
    const el = document.getElementById("loader");
    if (!el) return;
    el.querySelector(".loader-text").textContent = msg;
    el.classList.add("active");
}

function hideLoader() {
    const el = document.getElementById("loader");
    if (el) el.classList.remove("active");
}

// ── Upload area drag & drop ────────────────────────────────────────────
function initUploadArea(areaId, inputId, onFiles) {
    const area  = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    if (!area || !input) return;

    area.addEventListener("click", () => input.click());

    area.addEventListener("dragover", e => {
        e.preventDefault();
        area.classList.add("dragover");
    });

    area.addEventListener("dragleave", () => area.classList.remove("dragover"));

    area.addEventListener("drop", e => {
        e.preventDefault();
        area.classList.remove("dragover");
        onFiles(e.dataTransfer.files);
    });

    input.addEventListener("change", () => onFiles(input.files));
}

// ── Preview de imagen ──────────────────────────────────────────────────
function previewImage(file, imgId, placeholderId) {
    const reader = new FileReader();
    reader.onload = e => {
        const img = document.getElementById(imgId);
        const ph  = document.getElementById(placeholderId);
        if (img) { img.src = e.target.result; img.style.display = "block"; }
        if (ph)  ph.style.display = "none";
    };
    reader.readAsDataURL(file);
}

// ── Marcar nav link activo ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    const links = document.querySelectorAll(".navbar-links a");
    links.forEach(link => {
        if (link.href === location.href) link.classList.add("active");
    });
});
