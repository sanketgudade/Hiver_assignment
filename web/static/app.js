/**
 * Hiver AI Support Agent - Frontend Application
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const customerInput = document.getElementById("customer-input");
  const charCounter = document.getElementById("char-counter");
  const analyzeBtn = document.getElementById("analyze-btn");
  const errorCard = document.getElementById("error-card");
  const errorMessage = document.getElementById("error-message");
  const loadingContainer = document.getElementById("loading-container");
  const loadingText = document.getElementById("loading-text");
  const resultsSection = document.getElementById("results-section");

  // Result card elements
  const intentNameEl = document.getElementById("intent-name");
  const confidenceBadgeEl = document.getElementById("confidence-badge");
  const classifierReasonEl = document.getElementById("classifier-reason");
  const routingBadgeEl = document.getElementById("routing-badge");
  const escalateTargetEl = document.getElementById("escalate-target");
  const routingReasonEl = document.getElementById("routing-reason");
  const retrievedListEl = document.getElementById("retrieved-list");
  const draftReplyTextEl = document.getElementById("draft-reply-text");
  const replyCounterEl = document.getElementById("reply-counter");

  // Preset buttons
  const presetButtons = document.querySelectorAll(".preset-btn");

  // State
  let isFirstRequest = true;
  let isAnalyzing = false;

  // Initialize character counter
  updateCharCounter();

  // Event Listeners
  customerInput.addEventListener("input", updateCharCounter);

  // Ctrl + Enter shortcut
  customerInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleAnalyze();
    }
  });

  // Analyze button click
  analyzeBtn.addEventListener("click", () => {
    handleAnalyze();
  });

  // Preset buttons click
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text") || btn.textContent.trim();
      customerInput.value = text;
      updateCharCounter();
      handleAnalyze();
    });
  });

  /**
   * Update character counter for the textarea
   */
  function updateCharCounter() {
    const length = customerInput.value.length;
    charCounter.textContent = `${length} / 280`;
    if (length > 280) {
      charCounter.classList.add("counter-warning");
    } else {
      charCounter.classList.remove("counter-warning");
    }
  }

  /**
   * Main submit/analyze handler
   */
  async function handleAnalyze() {
    if (isAnalyzing) return;
    const rawText = customerInput.value;
    const trimmedText = rawText.trim();

    // Validation
    if (!trimmedText) {
      showError("Please enter a customer message.");
      customerInput.focus();
      return;
    }

    isAnalyzing = true;
    clearError();
    setLoading(true);

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: trimmedText }),
      });

      if (!response.ok) {
        let errDetail = `Server returned status ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson && errJson.detail) {
            errDetail = errJson.detail;
          }
        } catch {
          // Fallback to default message
        }
        throw new Error(errDetail);
      }

      const data = await response.json();
      renderResults(data);
      isFirstRequest = false;
    } catch (err) {
      const msg = err && err.message ? err.message : "Server is unavailable. Please check that the server is running.";
      showError(`Analysis failed: ${msg}`);
      resultsSection.classList.add("hidden");
    } finally {
      isAnalyzing = false;
      setLoading(false);
    }
  }

  /**
   * Set loading UI state
   */
  function setLoading(isLoading) {
    if (isLoading) {
      analyzeBtn.disabled = true;
      customerInput.disabled = true;
      presetButtons.forEach((b) => (b.disabled = true));

      if (isFirstRequest) {
        loadingText.textContent = "Loading model… first request may take 15 seconds";
      } else {
        loadingText.textContent = "Analyzing...";
      }

      loadingContainer.classList.remove("hidden");
    } else {
      analyzeBtn.disabled = false;
      customerInput.disabled = false;
      presetButtons.forEach((b) => (b.disabled = false));
      loadingContainer.classList.add("hidden");
    }
  }

  /**
   * Show error card
   */
  function showError(msg) {
    errorMessage.textContent = msg;
    errorCard.classList.remove("hidden");
  }

  /**
   * Clear error card
   */
  function clearError() {
    errorCard.classList.add("hidden");
    errorMessage.textContent = "";
  }

  /**
   * Render all result sections safely using textContent
   */
  function renderResults(data) {
    clearError();
    renderIntent(data);
    renderRouting(data);
    renderRetrieved(data);
    renderReply(data);

    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /**
   * Render Intent Card
   */
  function renderIntent(data) {
    intentNameEl.textContent = data.intent || "--";

    if (typeof data.confidence === "number") {
      const pct = Math.round(data.confidence * 100);
      confidenceBadgeEl.textContent = `${pct}%`;
    } else {
      confidenceBadgeEl.textContent = "--%";
    }

    classifierReasonEl.textContent = data.classifier_reason || "No explanation provided.";
  }

  /**
   * Render Routing Card
   */
  function renderRouting(data) {
    const routing = data.routing || {};
    const decision = routing.decision || "--";
    const escalateTo = routing.escalate_to || (decision === "auto_handle" ? "ai" : "human");
    const reason = routing.reason || "No routing reason provided.";

    routingBadgeEl.textContent = decision;
    routingBadgeEl.className = "badge decision-badge";

    if (decision === "auto_handle") {
      routingBadgeEl.classList.add("decision-auto");
    } else if (decision === "escalate") {
      routingBadgeEl.classList.add("decision-escalate");
    } else {
      routingBadgeEl.classList.add("decision-other");
    }

    if (escalateTargetEl) {
      escalateTargetEl.textContent = escalateTo;
      escalateTargetEl.className = "target-badge";
      if (escalateTo === "ai") {
        escalateTargetEl.classList.add("target-ai");
      } else {
        escalateTargetEl.classList.add("target-escalate");
      }
    }

    routingReasonEl.textContent = reason;
  }

  /**
   * Render Retrieved Examples Card safely
   */
  function renderRetrieved(data) {
    // Clear previous items safely
    retrievedListEl.replaceChildren();

    const retrieved = Array.isArray(data.retrieved) ? data.retrieved : [];
    const itemsToShow = retrieved.slice(0, 3);

    if (itemsToShow.length === 0) {
      const emptyP = document.createElement("p");
      emptyP.className = "empty-retrieved-message";
      emptyP.textContent = "No similar examples were retrieved.";
      retrievedListEl.appendChild(emptyP);
      return;
    }

    itemsToShow.forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "retrieved-item";

      const header = document.createElement("div");
      header.className = "retrieved-item-header";

      const tag = document.createElement("span");
      tag.className = "example-tag";
      tag.textContent = `Example ${index + 1}`;

      const score = document.createElement("span");
      score.className = "score-badge";
      const scoreVal = typeof item.score === "number" ? item.score.toFixed(3) : item.score;
      score.textContent = `Score: ${scoreVal}`;

      header.appendChild(tag);
      header.appendChild(score);

      // Customer row
      const custRow = document.createElement("div");
      custRow.className = "retrieved-row";
      const custLabel = document.createElement("span");
      custLabel.className = "retrieved-row-label";
      custLabel.textContent = "Customer:";
      const custText = document.createElement("p");
      custText.className = "retrieved-text";
      custText.textContent = item.customer || "--";
      custRow.appendChild(custLabel);
      custRow.appendChild(custText);

      // Brand reply row
      const brandRow = document.createElement("div");
      brandRow.className = "retrieved-row";
      const brandLabel = document.createElement("span");
      brandLabel.className = "retrieved-row-label";
      brandLabel.textContent = "Brand Reply:";
      const brandText = document.createElement("p");
      brandText.className = "retrieved-text";
      brandText.textContent = item.reply || "--";
      brandRow.appendChild(brandLabel);
      brandRow.appendChild(brandText);

      card.appendChild(header);
      card.appendChild(custRow);
      card.appendChild(brandRow);

      retrievedListEl.appendChild(card);
    });
  }

  /**
   * Render Draft Reply Card
   */
  function renderReply(data) {
    const reply = data.reply || "";
    draftReplyTextEl.textContent = reply;

    const replyLen = reply.length;
    replyCounterEl.textContent = `Character count: ${replyLen} / 280`;

    if (replyLen > 280) {
      replyCounterEl.classList.add("reply-warning");
    } else {
      replyCounterEl.classList.remove("reply-warning");
    }
  }
});
