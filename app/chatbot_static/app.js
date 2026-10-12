const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const clearButton = document.querySelector("#clear-button");
const conversation = document.querySelector("#conversation");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const errorMessage = document.querySelector("#error-message");
const characterCount = document.querySelector("#character-count");

let pending = false;
let activeRequest = null;
let conversationVersion = 0;

function updateInput() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 150)}px`;
  characterCount.textContent = `${questionInput.value.length} / 2000`;
}

function scrollToLatest() {
  conversation.scrollTop = conversation.scrollHeight;
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function addUserMessage(question) {
  const message = createElement("article", "message message-user");
  message.append(createElement("div", "user-bubble", question));
  messages.append(message);
  scrollToLatest();
}

function addAssistantShell() {
  const message = createElement("article", "message message-assistant");
  const head = createElement("div", "assistant-head");
  head.append(createElement("span", "assistant-avatar", "§"));
  head.append(createElement("span", "", "법령의 맥락"));
  const bubble = createElement("div", "assistant-bubble");
  const loading = createElement("div", "loading-dots");
  loading.setAttribute("role", "status");
  loading.setAttribute("aria-label", "근거를 검색하고 답변을 작성하는 중");
  for (let index = 0; index < 3; index += 1) loading.append(createElement("span"));
  bubble.append(loading);
  message.append(head, bubble);
  messages.append(message);
  scrollToLatest();
  return bubble;
}

function linkCitations(container, sourceIds) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement && !node.parentElement.closest("a, code, pre, button")) {
      textNodes.push(node);
    }
  }

  for (const node of textNodes) {
    const parts = node.textContent.split(/(\[S\d+\])/g);
    if (parts.length === 1) continue;
    const fragment = document.createDocumentFragment();
    for (const part of parts) {
      const match = /^\[(S\d+)\]$/.exec(part);
      if (!match || !sourceIds.has(match[1])) {
        fragment.append(document.createTextNode(part));
        continue;
      }
      const link = createElement("button", "citation-link", part);
      link.type = "button";
      link.setAttribute("aria-label", `출처 ${match[1]} 보기`);
      link.addEventListener("click", () => {
        const target = container.parentElement.querySelector(`[data-source-id="${match[1]}"]`);
        if (target) {
          target.open = true;
          target.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      });
      fragment.append(link);
    }
    node.replaceWith(fragment);
  }
}

function addAnswerText(container, answer, answerHtml, sourceIds) {
  const content = createElement("div", "answer-markdown");
  if (answerHtml) {
    // 서버에서 Markdown을 HTML로 변환하고 허용 태그만 남긴 결과다.
    content.innerHTML = answerHtml;
  } else {
    content.classList.add("plain-text");
    content.textContent = answer;
  }
  linkCitations(content, sourceIds);
  container.append(content);
}

function addSources(container, sources) {
  if (!sources.length) return;
  const sourceSection = createElement("section", "sources");
  sourceSection.append(createElement("h3", "sources-heading", `참고 근거  ${sources.length}건`));

  for (const source of sources) {
    const card = createElement("details", "source-card");
    card.dataset.sourceId = source.id;
    const summary = createElement("summary");
    summary.append(createElement("span", "source-id", source.id));
    summary.append(createElement("span", "source-title", source.title));
    summary.append(createElement("span", "source-chevron", "⌄"));
    const content = createElement("div", "source-content", source.content);
    if (source.law_name) content.append(createElement("small", "source-law", source.law_name));
    card.append(summary, content);
    sourceSection.append(card);
  }
  container.append(sourceSection);
}

function showAnswer(bubble, result) {
  bubble.replaceChildren();
  const sources = Array.isArray(result.sources) ? result.sources : [];
  addAnswerText(bubble, result.answer || "답변을 받지 못했습니다.", result.answer_html, new Set(sources.map((source) => source.id)));
  bubble.append(createElement("div", "answer-meta", `${result.search_method || "문서 검색"} · 아래 근거 원문을 펼쳐 확인할 수 있습니다.`));
  addSources(bubble, sources);
  scrollToLatest();
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

async function ask(question) {
  if (pending) return;
  pending = true;
  sendButton.disabled = true;
  errorMessage.hidden = true;
  welcome.hidden = true;
  addUserMessage(question);
  const bubble = addAssistantShell();
  const requestVersion = conversationVersion;
  activeRequest = new AbortController();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: activeRequest.signal,
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(typeof result.detail === "string" ? result.detail : "답변을 가져오지 못했습니다.");
    }
    if (requestVersion === conversationVersion) showAnswer(bubble, result);
  } catch (error) {
    if (error.name !== "AbortError" && requestVersion === conversationVersion) {
      bubble.replaceChildren(createElement("p", "answer-text", "답변을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요."));
      showError(error.message || "연결에 실패했습니다.");
      scrollToLatest();
    }
  } finally {
    if (requestVersion === conversationVersion) {
      pending = false;
      activeRequest = null;
      sendButton.disabled = false;
      questionInput.focus();
    }
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || pending) return;
  questionInput.value = "";
  updateInput();
  ask(question);
});

questionInput.addEventListener("input", updateInput);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

clearButton.addEventListener("click", () => {
  conversationVersion += 1;
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  pending = false;
  sendButton.disabled = false;
  messages.replaceChildren();
  welcome.hidden = false;
  errorMessage.hidden = true;
  questionInput.value = "";
  updateInput();
  questionInput.focus();
  scrollToLatest();
});

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    updateInput();
    questionInput.focus();
  });
}

updateInput();
