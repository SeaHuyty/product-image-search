const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const fileInput = document.getElementById("file");
const pickButton = document.getElementById("pick-image");
const clearButton = document.getElementById("clear-image");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("preview-img");
const previewName = document.getElementById("preview-name");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

let latestRequest = 0;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function errorMessage(data) {
  // FastAPI sends a string for our own errors, a list for validation errors
  if (data && typeof data.detail === "string") return data.detail;
  return "Something went wrong. Please try again.";
}

function renderResults(results) {
  const cards = results.map((item) => {
    const card = document.createElement("article");
    card.className = "card";

    const img = document.createElement("img");
    img.src = item.image_url;
    img.alt = item.name;
    img.loading = "lazy";

    const body = document.createElement("div");
    body.className = "card-body";

    const name = document.createElement("h2");
    name.textContent = item.name;

    const description = document.createElement("p");
    description.textContent = item.description;

    body.append(name, description);
    card.append(img, body);
    return card;
  });
  resultsEl.replaceChildren(...cards);
}

async function search(request) {
  const id = ++latestRequest;
  document.body.classList.add("searched");
  resultsEl.replaceChildren();
  setStatus("Searching…");

  try {
    const response = await request();
    const data = await response.json().catch(() => null);
    if (id !== latestRequest) return; // a newer search replaced this one

    if (!response.ok) throw new Error(errorMessage(data));
    if (!data.results.length) {
      setStatus("No results found.");
      return;
    }
    setStatus("");
    renderResults(data.results);
  } catch (err) {
    if (id !== latestRequest) return;
    const offline = err instanceof TypeError;
    setStatus(offline ? "Could not reach the server." : err.message, true);
  }
}

function searchText(query) {
  search(() =>
    fetch("/api/search/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    })
  );
}

function searchImage(file) {
  const body = new FormData();
  body.append("file", file);
  search(() => fetch("/api/search/image", { method: "POST", body }));
}

function showImage(file) {
  URL.revokeObjectURL(previewImg.src);
  previewImg.src = URL.createObjectURL(file);
  previewName.textContent = file.name;
  preview.hidden = false;
  form.classList.add("has-image");
  queryInput.value = "";
}

function clearImage() {
  URL.revokeObjectURL(previewImg.src);
  previewImg.removeAttribute("src");
  preview.hidden = true;
  form.classList.remove("has-image");
  fileInput.value = "";
  queryInput.focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }
  searchText(query);
});

pickButton.addEventListener("click", () => fileInput.click());
clearButton.addEventListener("click", clearImage);

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    setStatus("File must be an image.", true);
  } else if (file.size > MAX_UPLOAD_BYTES) {
    setStatus("Image is larger than 5 MB.", true);
  } else {
    showImage(file);
    searchImage(file);
    return;
  }
  document.body.classList.add("searched");
  fileInput.value = "";
});
