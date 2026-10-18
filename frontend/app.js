/* ==========================================================================
   SERENE-AI · app.js
   Sections: config · helpers · API · UI shell · health · options ·
             chat · habit coach · music · writing · clustering · boot
   No libraries. All text from users or from the server is inserted with
   textContent / text nodes (never innerHTML), so it cannot inject HTML.
   ========================================================================== */
(() => {
  "use strict";

  /* ---------------------------------------------------------------- config */
  // Empty string = same origin as the page (Flask serves this frontend).
  // Set window.SERENE_CONFIG.apiBaseUrl in index.html if the API lives elsewhere.
  const API_BASE_URL = String((window.SERENE_CONFIG && window.SERENE_CONFIG.apiBaseUrl) || "").replace(/\/+$/, "");
  const REQUEST_TIMEOUT_MS = 15000;
  const HEALTH_INTERVAL_MS = 30000;
  const CHAT_MAX_CHARS = 1000;
  const MAX_MINUTES = 1440;

  const VIEW_TITLES = {
    dashboard: "Dashboard",
    chat: "AI Wellness Chat",
    habit: "Habit Coach",
    music: "Music Therapy",
    writing: "Writing Therapy",
    clusters: "Habit Clustering",
    about: "About & Status",
  };

  // Request field name -> select element id (field names match the backend exactly).
  const HABIT_FIELDS = {
    exercise_frequency: "habit-exercise",
    social_media_hours: "habit-social",
    stress_level: "habit-stress",
    mindfulness_frequency: "habit-mindful",
  };

  // Friendly text for mood values the backend currently offers. Unknown values still work.
  const MUSIC_MOOD_INFO = {
    relaxation: { title: "Relaxation", hint: "Slow down and unwind" },
    motivation: { title: "Motivation", hint: "Find some energy" },
    fun: { title: "Fun", hint: "Lighten the mood" },
  };
  const WRITING_MOOD_INFO = {
    good: { title: "Good", hint: "Things feel positive" },
    neutral: { title: "Neutral", hint: "Somewhere in the middle" },
    bad: { title: "Not so good", hint: "It has been a hard day" },
  };

  const HEALTH_LABELS = {
    chatbot_responses: "Chatbot replies",
    habits: "Habit data",
    music_data: "Music library list",
    music_folder: "Music folder",
  };

  const state = {
    options: null,
    optionsError: null,
    health: null,
    connection: "checking",
    lastHabitPayload: null,
    lastMusicMood: null,
    lastWritingMood: null,
  };
  const busy = new Set();

  /* --------------------------------------------------------------- helpers */
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const SVG_NS = "http://www.w3.org/2000/svg";

  /** Create an element. Strings in `children` become text nodes (safe). */
  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat()) {
      if (child !== null && child !== undefined && child !== false) node.append(child);
    }
    return node;
  }

  function icon(name, extraClass = "") {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", ("icon " + extraClass).trim());
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", "#i-" + name);
    svg.append(use);
    return svg;
  }

  const isNonEmptyString = (value) => typeof value === "string" && value.trim() !== "";
  const capitalize = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : text);

  function setHidden(node, hidden) { node.hidden = hidden; }

  function showFieldError(errorNode, message, controls = []) {
    errorNode.textContent = message || "";
    errorNode.hidden = !message;
    controls.forEach((control) => {
      if (message) control.setAttribute("aria-invalid", "true");
      else control.removeAttribute("aria-invalid");
    });
  }

  /* ------------------------------------------------------------------- API */
  class ApiError extends Error {
    constructor(message, { kind = "http", status = 0, payload = null } = {}) {
      super(message);
      this.name = "ApiError";
      this.kind = kind;       // "network" | "timeout" | "invalid" | "http"
      this.status = status;
      this.payload = payload;
    }
  }

  /**
   * One place for every backend call.
   * Returns parsed JSON (an object) or throws ApiError.
   */
  async function apiRequest(path, { method = "GET", body } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    let response;
    try {
      response = await fetch(API_BASE_URL + path, {
        method,
        headers: body !== undefined ? { "Content-Type": "application/json", Accept: "application/json" } : { Accept: "application/json" },
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } catch (err) {
      throw new ApiError(err && err.name === "AbortError" ? "timeout" : "network", {
        kind: err && err.name === "AbortError" ? "timeout" : "network",
      });
    } finally {
      clearTimeout(timer);
    }

    let data = null;
    try {
      const text = await response.text();
      data = text ? JSON.parse(text) : null;
    } catch (_) {
      data = null; // not JSON; handled below
    }

    if (!response.ok) {
      const serverMessage = data && isNonEmptyString(data.error) ? data.error : `The server returned an error (${response.status}).`;
      throw new ApiError(serverMessage, { kind: "http", status: response.status, payload: data });
    }
    if (data === null || typeof data !== "object" || Array.isArray(data)) {
      throw new ApiError("invalid", { kind: "invalid", status: response.status });
    }
    return data;
  }

  /** Friendly text for any error thrown by apiRequest. */
  function describeError(err) {
    if (err instanceof ApiError) {
      if (err.kind === "network") return "We can't reach the SERENE-AI server. Please check that it is running and try again.";
      if (err.kind === "timeout") return "The server took too long to answer. Please try again.";
      if (err.kind === "invalid") return "The server sent a reply we could not understand. Please try again.";
      return err.message;
    }
    return "Something went wrong. Please try again.";
  }

  const unexpected = () => new ApiError("invalid", { kind: "invalid" });

  /**
   * Run `task` only if no other task with the same key is running (prevents
   * double submits) and keep the given buttons disabled/loading meanwhile.
   */
  async function exclusive(key, buttons, task) {
    if (busy.has(key)) return;
    busy.add(key);
    buttons.forEach((button) => {
      button.disabled = true;
      button.setAttribute("aria-busy", "true");
      button.classList.add("is-loading");
    });
    try {
      await task();
    } finally {
      busy.delete(key);
      buttons.forEach((button) => {
        button.disabled = false;
        button.removeAttribute("aria-busy");
        button.classList.remove("is-loading");
      });
      syncMoodButtons();
    }
  }

  /* -------------------------------------------------------------- toasts */
  function toast(message, type = "info", duration = 5000) {
    const box = $("#toasts");
    const close = el("button", { type: "button", "aria-label": "Dismiss notification" }, icon("close"));
    const node = el("div", { class: "toast toast--" + type }, el("p", { text: message }), close);
    const remove = () => node.remove();
    close.addEventListener("click", remove);
    box.append(node);
    if (duration > 0) setTimeout(remove, duration);
  }

  /* ---------------------------------------------------------------- shell */
  const sidebar = $("#sidebar");
  const overlay = $("#nav-overlay");
  const openBtn = $("#nav-open");
  const closeBtn = $("#nav-close");
  const mobileQuery = window.matchMedia("(max-width: 900px)");

  function openDrawer() {
    document.body.classList.add("nav-open");
    overlay.hidden = false;
    openBtn.setAttribute("aria-expanded", "true");
    const first = $(".nav__link", sidebar);
    if (first) first.focus();
  }

  function closeDrawer({ restoreFocus = false } = {}) {
    if (!document.body.classList.contains("nav-open")) return;
    document.body.classList.remove("nav-open");
    overlay.hidden = true;
    openBtn.setAttribute("aria-expanded", "false");
    if (restoreFocus) openBtn.focus();
  }

  openBtn.addEventListener("click", openDrawer);
  closeBtn.addEventListener("click", () => closeDrawer({ restoreFocus: true }));
  overlay.addEventListener("click", () => closeDrawer({ restoreFocus: true }));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer({ restoreFocus: true });
  });
  mobileQuery.addEventListener("change", (event) => { if (!event.matches) closeDrawer(); });

  function showView(name, { moveFocus = true } = {}) {
    const view = VIEW_TITLES[name] ? name : "dashboard";
    $$("[data-view]").forEach((section) => { section.hidden = section.dataset.view !== view; });
    $$("[data-nav]").forEach((link) => {
      if (link.dataset.nav === view) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    document.title = `${VIEW_TITLES[view]} · SERENE-AI`;
    closeDrawer();
    window.scrollTo(0, 0);
    if (moveFocus) {
      const heading = $(`#view-${view} h2`);
      if (heading) heading.focus({ preventScroll: true });
    }
    if (view === "chat") setTimeout(() => $("#chat-input").focus({ preventScroll: true }), 0);
  }

  function routeFromHash(moveFocus) {
    const name = (window.location.hash || "#dashboard").slice(1);
    showView(name, { moveFocus });
  }
  window.addEventListener("hashchange", () => routeFromHash(true));

  /* --------------------------------------------------------------- health */
  const pill = $("#status-pill");
  const pillText = $("#status-text");
  const PILL_TEXT = { checking: "Connecting…", ok: "Connected", degraded: "Partly available", offline: "Offline" };

  function setConnection(next) {
    const previous = state.connection;
    state.connection = next;
    pill.dataset.state = next;
    pillText.textContent = PILL_TEXT[next];
    const title = {
      checking: "Checking the connection to the SERENE-AI server…",
      ok: "Connected to the SERENE-AI server. Click to check again.",
      degraded: "The server is running but some parts are unavailable. Open About & Status for details.",
      offline: "Can't reach the server. Click to try again.",
    }[next];
    pill.title = title;
    pill.setAttribute("aria-label", `Server status: ${PILL_TEXT[next]}. Activate to check again.`);

    if (previous !== next && previous !== "checking") {
      if (next === "offline") toast("Lost connection to the SERENE-AI server.", "error");
      else if (previous === "offline") toast("Back online.", "success", 3000);
    }
  }

  function renderHealthDetails() {
    $("#about-base").textContent = API_BASE_URL || window.location.origin;
    $("#about-status").textContent = state.connection === "offline" ? "Can't reach the server" : PILL_TEXT[state.connection];
    const list = $("#about-checks");
    list.replaceChildren();
    if (!state.health) return;
    for (const [key, value] of Object.entries(state.health.checks)) {
      const ok = value === "ok";
      list.append(
        el("li", { "data-ok": String(ok) },
          el("span", { class: "dot", "aria-hidden": "true" }),
          el("span", { text: `${HEALTH_LABELS[key] || key}: ${ok ? "ready" : "not available"}` }))
      );
    }
  }

  async function checkHealth() {
    if (busy.has("health")) return;
    busy.add("health");
    if (state.connection !== "ok" && state.connection !== "degraded") setConnection("checking");
    try {
      const data = await apiRequest("/health");
      const checks = data.checks && typeof data.checks === "object" && !Array.isArray(data.checks) ? data.checks : {};
      state.health = { status: data.status, checks };
      setConnection(data.status === "ok" ? "ok" : "degraded");
      if (!state.options) loadOptions();
    } catch (err) {
      state.health = null;
      setConnection("offline");
    } finally {
      busy.delete("health");
      renderHealthDetails();
    }
  }

  pill.addEventListener("click", () => { checkHealth(); });
  $("#about-recheck").addEventListener("click", () => { checkHealth(); loadOptions(); });

  /* -------------------------------------------------------------- options */
  const optionBanners = ["#habit-options-error", "#music-options-error", "#writing-options-error"];

  function fillSelect(select, values, placeholder) {
    const previous = select.value;
    select.replaceChildren(el("option", { value: "", text: placeholder }));
    values.forEach((value) => select.append(el("option", { value, text: value })));
    if (values.includes(previous)) select.value = previous;
    select.disabled = values.length === 0;
  }

  function renderMoodCards(container, name, values, info) {
    const previous = $("input:checked", container);
    const keep = previous ? previous.value : null;
    container.replaceChildren();
    if (values.length === 0) {
      const loading = !state.options && !state.optionsError;
      container.append(el("p", { class: "note", text: loading ? "Loading moods…" : "No moods are available right now." }));
      return;
    }
    values.forEach((value, index) => {
      const meta = info[value] || {};
      const input = el("input", { type: "radio", name, value, id: `${name}-${index}`, class: "sr-only", checked: value === keep });
      input.addEventListener("change", () => {
        const errorNode = $(name === "music-mood" ? "#music-error" : "#writing-error");
        showFieldError(errorNode, "");
        syncMoodButtons();
      });
      container.append(
        el("label", { class: "mood-card" },
          input,
          el("span", { class: "mood-card__body" },
            el("span", { class: "mood-card__title", text: meta.title || capitalize(value) }),
            meta.hint ? el("span", { class: "mood-card__hint", text: meta.hint }) : null),
          el("span", { class: "mood-card__check", "aria-hidden": "true" }, icon("check")))
      );
    });
  }

  function syncMoodButtons() {
    const musicReady = Boolean($("input[name='music-mood']:checked")) && !busy.has("music");
    const writingReady = Boolean($("input[name='writing-mood']:checked")) && !busy.has("writing");
    $("#music-submit").disabled = !musicReady;
    $("#writing-submit").disabled = !writingReady;
  }

  function stringList(value) {
    return Array.isArray(value) ? value.filter(isNonEmptyString) : [];
  }

  function renderOptions() {
    const options = state.options;
    const failed = Boolean(state.optionsError);

    optionBanners.forEach((selector) => {
      const banner = $(selector);
      banner.hidden = !failed;
      if (failed) $("p", banner).textContent = state.optionsError;
    });

    for (const [field, id] of Object.entries(HABIT_FIELDS)) {
      fillSelect($("#" + id), options ? stringList(options[field]) : [], failed ? "Options unavailable" : "Loading…");
      if (options && stringList(options[field]).length) $("#" + id).options[0].text = "Select…";
    }
    renderMoodCards($("#music-moods"), "music-mood", options ? stringList(options.music_moods) : [], MUSIC_MOOD_INFO);
    renderMoodCards($("#writing-moods"), "writing-mood", options ? stringList(options.writing_moods) : [], WRITING_MOOD_INFO);
    if (options && stringList(options.music_moods).length === 0) {
      $("#music-moods").replaceChildren(el("p", { class: "note", text: "The music library isn't set up on the server yet, so there are no moods to choose from." }));
    }
    syncMoodButtons();
  }

  async function loadOptions() {
    if (busy.has("options")) return;
    busy.add("options");
    try {
      const data = await apiRequest("/options");
      const missing = [...Object.keys(HABIT_FIELDS), "writing_moods"].filter((key) => !Array.isArray(data[key]));
      if (missing.length) throw unexpected();
      state.options = data;
      state.optionsError = null;
    } catch (err) {
      state.options = null;
      state.optionsError = describeError(err);
    } finally {
      busy.delete("options");
      renderOptions();
    }
  }

  $$("[data-retry-options]").forEach((button) => button.addEventListener("click", () => { checkHealth(); loadOptions(); }));

  /* ----------------------------------------------------------------- chat */
  const chatLog = $("#chat-log");
  const chatForm = $("#chat-form");
  const chatInput = $("#chat-input");
  const chatSend = $("#chat-send");
  const chatError = $("#chat-error");

  const WELCOME = "Hi, I'm glad you're here. How are you feeling today? You can write as much or as little as you like.";

  function scrollChatToEnd() { chatLog.scrollTop = chatLog.scrollHeight; }

  function addMessage(role, text, { support = false } = {}) {
    const isUser = role === "user";
    const bubbleChildren = [];
    if (support) bubbleChildren.push(el("span", { class: "msg__label", text: "Support message" }));
    bubbleChildren.push(el("span", { class: "msg__text", text }));
    const node = el("div", { class: `msg msg--${isUser ? "user" : "ai"}${support ? " msg--support" : ""}` },
      el("span", { class: "msg__avatar", "aria-hidden": "true" }, isUser ? "You" : icon("leaf")),
      el("div", { class: "msg__bubble" },
        el("span", { class: "sr-only", text: isUser ? "You said: " : "SERENE-AI said: " }),
        ...bubbleChildren));
    chatLog.append(node);
    scrollChatToEnd();
    return node;
  }

  function addTyping() {
    const node = el("div", { class: "msg msg--ai", "aria-label": "SERENE-AI is typing" },
      el("span", { class: "msg__avatar", "aria-hidden": "true" }, icon("leaf")),
      el("div", { class: "msg__bubble" },
        el("span", { class: "typing", "aria-hidden": "true" }, el("span"), el("span"), el("span")),
        el("span", { class: "sr-only", text: "SERENE-AI is typing…" })));
    chatLog.append(node);
    scrollChatToEnd();
    return node;
  }

  function addChatError(message, originalText) {
    const retry = el("button", { class: "btn btn--secondary btn--small", type: "button" }, icon("refresh"), "Try again");
    const node = el("div", { class: "msg msg--error", role: "alert" },
      el("div", { class: "msg__bubble" },
        icon("alert"),
        el("span", { text: message }),
        retry));
    retry.addEventListener("click", () => {
      if (busy.has("chat")) return;
      node.remove();
      sendChat(originalText, { isRetry: true });
    });
    chatLog.append(node);
    scrollChatToEnd();
  }

  function sendChat(text, { isRetry = false } = {}) {
    return exclusive("chat", [chatSend], async () => {
      if (!isRetry) addMessage("user", text);
      const typing = addTyping();
      try {
        const data = await apiRequest("/chatbot", { method: "POST", body: { input: text } });
        if (!isNonEmptyString(data.response)) throw unexpected();
        typing.remove();
        addMessage("ai", data.response, { support: data.crisis === true });
      } catch (err) {
        typing.remove();
        addChatError(describeError(err), text);
      }
    });
  }

  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (busy.has("chat")) return;
    const text = chatInput.value.trim();
    if (!text) {
      showFieldError(chatError, "Please write a message first.", [chatInput]);
      chatInput.focus();
      return;
    }
    if (text.length > CHAT_MAX_CHARS) {
      showFieldError(chatError, `Please keep your message under ${CHAT_MAX_CHARS} characters.`, [chatInput]);
      return;
    }
    showFieldError(chatError, "", [chatInput]);
    chatInput.value = "";
    autoGrow();
    sendChat(text);
    chatInput.focus();
  });

  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });
  chatInput.addEventListener("input", () => {
    autoGrow();
    if (chatInput.value.trim()) showFieldError(chatError, "", [chatInput]);
  });

  function autoGrow() {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px";
  }

  function resetChat() {
    chatLog.replaceChildren();
    addMessage("ai", WELCOME);
  }
  $("#chat-clear").addEventListener("click", () => {
    if (busy.has("chat")) return;
    resetChat();
    toast("Conversation cleared.", "info", 2500);
    chatInput.focus();
  });

  /* ---------------------------------------------------------- habit coach */
  const habitForm = $("#habit-form");
  const habitResult = $("#habit-result");
  const habitSubmit = $("#habit-submit");

  function habitPlaceholder() {
    habitResult.replaceChildren(
      el("div", { class: "placeholder" }, icon("sprout", "icon--xl"), el("p", { text: "Your suggestion will appear here." }))
    );
  }

  function renderLoading(container, message) {
    container.replaceChildren(el("div", { class: "loading-card", role: "status" }, el("span", { class: "spinner", "aria-hidden": "true" }), el("span", { text: message })));
  }

  function renderErrorCard(container, message, onRetry) {
    const children = [icon("alert"), el("h3", { text: "That didn't work" }), el("p", { text: message })];
    if (onRetry) {
      const retry = el("button", { class: "btn btn--secondary btn--small", type: "button" }, icon("refresh"), "Try again");
      retry.addEventListener("click", onRetry);
      children.push(retry);
    }
    container.replaceChildren(el("div", { class: "state-card state-card--error", role: "alert" }, ...children));
  }

  function renderInfoCard(container, title, message) {
    container.replaceChildren(el("div", { class: "state-card state-card--info", role: "status" }, el("h3", { text: title }), el("p", { text: message })));
  }

  function readHabitForm() {
    const payload = {};
    let firstInvalid = null;
    for (const [field, id] of Object.entries(HABIT_FIELDS)) {
      const select = $("#" + id);
      const errorNode = $(`#${id}-error`);
      if (!select.value) {
        showFieldError(errorNode, "Please choose an option.", [select]);
        if (!firstInvalid) firstInvalid = select;
      } else {
        showFieldError(errorNode, "", [select]);
        payload[field] = select.value;
      }
    }
    if (firstInvalid) { firstInvalid.focus(); return null; }
    return payload;
  }

  function requestHabit(payload) {
    return exclusive("habit", [habitSubmit], async () => {
      state.lastHabitPayload = payload;
      renderLoading(habitResult, "Finding a habit for you…");
      try {
        const data = await apiRequest("/recommend_habit", { method: "POST", body: payload });
        if (!isNonEmptyString(data.recommended_habit)) throw unexpected();
        const tags = [];
        if (isNonEmptyString(data.method)) tags.push(el("span", { class: "tag", text: data.method === "rule_based" ? "Rule-based suggestion" : data.method }));
        tags.push(el("span", { class: "tag tag--soft", text: "General wellness idea" }));
        habitResult.replaceChildren(
          el("article", { class: "result-card" },
            el("p", { class: "eyebrow", text: "Suggested habit" }),
            el("h3", { text: data.recommended_habit }),
            isNonEmptyString(data.reason) ? el("p", { class: "reason", text: data.reason }) : null,
            el("div", { class: "tags" }, ...tags),
            el("p", { class: "note", text: "This is a general suggestion, not medical advice. Take what helps and leave the rest." }))
        );
      } catch (err) {
        renderErrorCard(habitResult, describeError(err), () => { if (state.lastHabitPayload) requestHabit(state.lastHabitPayload); });
      }
    });
  }

  habitForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (busy.has("habit")) return;
    const payload = readHabitForm();
    if (payload) requestHabit(payload);
  });
  habitForm.addEventListener("reset", () => {
    Object.values(HABIT_FIELDS).forEach((id) => showFieldError($(`#${id}-error`), "", [$("#" + id)]));
    state.lastHabitPayload = null;
    habitPlaceholder();
  });
  Object.values(HABIT_FIELDS).forEach((id) => {
    $("#" + id).addEventListener("change", (event) => { if (event.target.value) showFieldError($(`#${id}-error`), "", [event.target]); });
  });

  /* ---------------------------------------------------------------- music */
  const musicForm = $("#music-form");
  const musicResults = $("#music-results");
  const musicSubmit = $("#music-submit");
  let currentAudios = [];

  /** Turn the API's stream_url (relative or absolute) into a playable absolute URL. */
  function resolveStreamUrl(value) {
    if (!isNonEmptyString(value)) return null;
    try {
      const base = API_BASE_URL || window.location.origin;
      const url = new URL(value, base + "/");
      return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
    } catch (_) {
      return null;
    }
  }

  function buildTrack(item) {
    const title = item && isNonEmptyString(item.title) ? item.title : "Untitled track";
    const url = item ? resolveStreamUrl(item.stream_url) : null;
    const status = el("p", { class: "track__status", role: "status", hidden: true });
    const body = el("div", { class: "track__body" }, el("h3", { text: title }));
    const card = el("article", { class: "track" }, el("span", { class: "track__icon", "aria-hidden": "true" }, icon("music")), body);

    if (!url) {
      card.classList.add("is-error");
      status.hidden = false;
      status.textContent = "This track has no playable address.";
      body.append(status);
      return card;
    }

    const audio = el("audio", { controls: true, preload: "none", "aria-label": `Play ${title}` });
    audio.src = url;
    audio.addEventListener("play", () => currentAudios.forEach((other) => { if (other !== audio) other.pause(); }));
    audio.addEventListener("error", () => {
      card.classList.add("is-error");
      status.textContent = "This track couldn't be played. The file may be missing or not supported by your browser.";
      status.hidden = false;
    });
    currentAudios.push(audio);
    body.append(audio, status);
    return card;
  }

  function stopAllAudio() {
    currentAudios.forEach((audio) => { audio.pause(); audio.removeAttribute("src"); audio.load(); });
    currentAudios = [];
  }

  function requestMusic(mood) {
    return exclusive("music", [musicSubmit], async () => {
      state.lastMusicMood = mood;
      stopAllAudio();
      renderLoading(musicResults, "Finding music…");
      try {
        const data = await apiRequest("/music_recommendation", { method: "POST", body: { mood } });
        if (!Array.isArray(data.recommendations)) throw unexpected();
        if (data.recommendations.length === 0) {
          renderInfoCard(musicResults, "No songs yet", "There are no songs for this mood right now. Please try another mood.");
          return;
        }
        const label = (MUSIC_MOOD_INFO[mood] || {}).title || capitalize(mood);
        musicResults.replaceChildren(
          el("h3", { class: "results-title", text: `${label} playlist · ${data.recommendations.length} ${data.recommendations.length === 1 ? "track" : "tracks"}` }),
          el("div", { class: "track-grid" }, data.recommendations.map(buildTrack))
        );
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          renderInfoCard(musicResults, "Nothing to play yet", err.message);
        } else {
          renderErrorCard(musicResults, describeError(err), () => { if (state.lastMusicMood) requestMusic(state.lastMusicMood); });
        }
      }
    });
  }

  musicForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (busy.has("music")) return;
    const checked = $("input[name='music-mood']:checked");
    if (!checked) {
      showFieldError($("#music-error"), "Please choose a mood first.");
      return;
    }
    requestMusic(checked.value);
  });

  /* -------------------------------------------------------------- writing */
  const writingForm = $("#writing-form");
  const writingFeedback = $("#writing-feedback");
  const writingResult = $("#writing-result");
  const writingSubmit = $("#writing-submit");
  const writingNotes = $("#writing-notes");

  function requestPrompt(mood) {
    return exclusive("writing", [writingSubmit], async () => {
      state.lastWritingMood = mood;
      writingFeedback.replaceChildren();
      try {
        const data = await apiRequest("/ai_writing_therapist", { method: "POST", body: { mood } });
        if (!isNonEmptyString(data.prompt)) throw unexpected();
        $("#writing-prompt").textContent = data.prompt;
        writingResult.hidden = false;
        if (!writingNotes.value) writingNotes.focus({ preventScroll: false });
        else $("#writing-prompt").scrollIntoView({ block: "nearest" });
      } catch (err) {
        renderErrorCard(writingFeedback, describeError(err), () => { if (state.lastWritingMood) requestPrompt(state.lastWritingMood); });
      }
    });
  }

  writingForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (busy.has("writing")) return;
    const checked = $("input[name='writing-mood']:checked");
    if (!checked) {
      showFieldError($("#writing-error"), "Please choose how you are feeling first.");
      return;
    }
    requestPrompt(checked.value);
  });

  $("#writing-new").addEventListener("click", () => {
    const checked = $("input[name='writing-mood']:checked");
    const mood = checked ? checked.value : state.lastWritingMood;
    if (mood) requestPrompt(mood);
  });

  $("#writing-clear").addEventListener("click", () => {
    writingNotes.value = "";
    $("#writing-prompt").textContent = "";
    writingResult.hidden = true;
    writingFeedback.replaceChildren();
    $$("input[name='writing-mood']").forEach((input) => { input.checked = false; });
    state.lastWritingMood = null;
    syncMoodButtons();
    toast("Prompt and notes cleared.", "info", 2500);
    const first = $("input[name='writing-mood']");
    if (first) first.focus();
  });

  /* ----------------------------------------------------------- clustering */
  const clustersForm = $("#clusters-form");
  const clustersInput = $("#clusters-time");
  const clustersResults = $("#clusters-results");
  const clustersSubmit = $("#clusters-submit");
  const clustersError = $("#clusters-error");

  function buildCluster(rec, index) {
    const habits = rec && Array.isArray(rec.habits) ? rec.habits.filter(isNonEmptyString) : [];
    const label = rec && isNonEmptyString(rec.label) ? rec.label : `Group ${index + 1}`;
    const total = rec && Number.isFinite(rec.total_time) ? rec.total_time : null;
    return el("article", { class: "cluster-card" },
      el("div", { class: "cluster-card__head" },
        el("h3", { text: label }),
        total !== null ? el("span", { class: "tag tag--soft" }, icon("clock"), ` ${total} min in total`) : null),
      habits.length
        ? el("ul", {}, habits.map((habit) => el("li", {}, icon("check"), el("span", { text: habit }))))
        : el("p", { class: "note", text: "No habits listed for this group." }));
  }

  function requestClusters(minutes) {
    return exclusive("clusters", [clustersSubmit], async () => {
      renderLoading(clustersResults, "Looking for habits that fit…");
      try {
        const data = await apiRequest("/habit_clustering", { method: "POST", body: { time_available: minutes } });
        if (!Array.isArray(data.recommendations)) throw unexpected();
        if (data.recommendations.length === 0) {
          renderInfoCard(clustersResults, "No habits found", `No habits fit into ${minutes} minutes. Try a little more time.`);
          return;
        }
        clustersResults.replaceChildren(
          el("h3", { class: "results-title", text: `Habits that fit into ${minutes} minutes` }),
          el("div", { class: "cluster-grid" }, data.recommendations.map(buildCluster))
        );
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          renderInfoCard(clustersResults, "No habits fit", err.message);
        } else {
          renderErrorCard(clustersResults, describeError(err), () => requestClusters(minutes));
        }
      }
    });
  }

  clustersForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (busy.has("clusters")) return;
    const raw = clustersInput.value.trim();
    const minutes = Number(raw);
    if (!raw) {
      showFieldError(clustersError, "Please enter how many minutes you have.", [clustersInput]);
      clustersInput.focus();
      return;
    }
    if (!Number.isInteger(minutes) || minutes < 1 || minutes > MAX_MINUTES) {
      showFieldError(clustersError, `Please enter a whole number of minutes between 1 and ${MAX_MINUTES}.`, [clustersInput]);
      clustersInput.focus();
      return;
    }
    showFieldError(clustersError, "", [clustersInput]);
    requestClusters(minutes);
  });
  clustersInput.addEventListener("input", () => { if (clustersInput.value.trim()) showFieldError(clustersError, "", [clustersInput]); });
  $$(".chip[data-minutes]").forEach((chip) => chip.addEventListener("click", () => {
    clustersInput.value = chip.dataset.minutes;
    showFieldError(clustersError, "", [clustersInput]);
    clustersInput.focus();
  }));

  /* ----------------------------------------------------------------- boot */
  function boot() {
    resetChat();
    habitPlaceholder();
    renderOptions();          // shows "Loading…" placeholders until /options answers
    renderHealthDetails();
    routeFromHash(false);
    checkHealth();
    loadOptions();
    setInterval(() => { if (!document.hidden) checkHealth(); }, HEALTH_INTERVAL_MS);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) checkHealth(); });
  }

  boot();
})();
