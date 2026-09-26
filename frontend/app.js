const API_BASE = window.location.port === "5500" ? "http://127.0.0.1:8000" : "";
const WORKSPACE_KEY = "clauselens.workspaceId";
const DOCUMENT_KEY = "clauselens.activeDocumentId";
const CONVERSATION_KEY = "clauselens.activeConversationId";

const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");
const documentList = document.getElementById("documentList");
const conversationList = document.getElementById("conversationList");
const conversationHeader = document.getElementById("conversationHeader");
const messageList = document.getElementById("messageList");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const composerHint = document.getElementById("composerHint");
const workspaceNotice = document.getElementById("workspaceNotice");
const compareModeButton = document.getElementById("compareModeButton");
const compareStatus = document.getElementById("compareStatus");

const state = {
    workspaceId: null,
    documents: [],
    conversations: [],
    activeDocumentId: null,
    activeConversationId: null,
    activeConversation: null,
    busy: false,
    compareMode: false,
    comparisonDocumentIds: new Set(),
};

async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || `Request failed (${response.status}).`);
    }
    return data;
}

async function initializeWorkspace() {
    setNotice("Connecting to your workspace...");
    try {
        const savedId = localStorage.getItem(WORKSPACE_KEY);
        let workspace = null;
        if (savedId) {
            try {
                workspace = await api(`/workspaces/${encodeURIComponent(savedId)}`);
            } catch (error) {
                if (!error.message.includes("Workspace not found")) throw error;
            }
        }
        if (!workspace) {
            workspace = await api("/workspaces", { method: "POST" });
            localStorage.setItem(WORKSPACE_KEY, workspace.workspace_id);
            localStorage.removeItem(DOCUMENT_KEY);
            localStorage.removeItem(CONVERSATION_KEY);
        }
        state.workspaceId = workspace.workspace_id;
        await refreshWorkspace();
        setNotice("");
    } catch (error) {
        setNotice(`Could not connect to ClauseLens: ${error.message}`, true);
    }
}

async function refreshWorkspace() {
    const [workspace, conversations] = await Promise.all([
        api(`/workspaces/${state.workspaceId}`),
        api(`/workspaces/${state.workspaceId}/conversations`),
    ]);
    state.documents = workspace.documents;
    state.conversations = conversations;

    const savedDocumentId = localStorage.getItem(DOCUMENT_KEY);
    if (!state.documents.some((item) => item.document_id === state.activeDocumentId)) {
        state.activeDocumentId = state.documents.some((item) => item.document_id === savedDocumentId)
            ? savedDocumentId
            : state.documents[0]?.document_id || null;
    }

    const savedConversationId = localStorage.getItem(CONVERSATION_KEY);
    if (!state.conversations.some((item) => item.conversation_id === state.activeConversationId)) {
        state.activeConversationId = state.conversations.some((item) => item.conversation_id === savedConversationId)
            ? savedConversationId
            : state.conversations[0]?.conversation_id || null;
    }

    if (!state.activeConversationId && state.activeDocumentId) {
        await createConversation(false);
    }

    await loadActiveConversation();
    renderWorkspace();
}

async function loadActiveConversation() {
    if (!state.activeConversationId) {
        state.activeConversation = null;
        return;
    }
    state.activeConversation = await api(
        `/workspaces/${state.workspaceId}/conversations/${state.activeConversationId}`
    );
    if (!state.activeDocumentId && state.activeConversation.active_document_id) {
        state.activeDocumentId = state.activeConversation.active_document_id;
    }
}

async function createConversation(render = true) {
    if (!state.activeDocumentId) return;
    const conversation = await api(`/workspaces/${state.workspaceId}/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active_document_id: state.activeDocumentId }),
    });
    state.activeConversationId = conversation.conversation_id;
    localStorage.setItem(CONVERSATION_KEY, state.activeConversationId);
    state.conversations.unshift(conversation);
    await loadActiveConversation();
    if (render) renderWorkspace();
}

uploadButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", uploadDocuments);

async function uploadDocuments() {
    const files = Array.from(fileInput.files || []);
    if (!files.length || !state.workspaceId) return;
    uploadButton.disabled = true;
    let completed = 0;
    try {
        for (const file of files) {
            if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
                throw new Error(`${file.name}: only PDF files are supported.`);
            }
            uploadStatus.textContent = `Adding ${file.name} (${completed + 1} of ${files.length})...`;
            const formData = new FormData();
            formData.append("file", file);
            const document = await api(`/workspaces/${state.workspaceId}/documents`, {
                method: "POST",
                body: formData,
            });
            completed += 1;
            if (!state.activeDocumentId) {
                state.activeDocumentId = document.document_id;
                localStorage.setItem(DOCUMENT_KEY, state.activeDocumentId);
            }
        }
        fileInput.value = "";
        uploadStatus.textContent = `${completed} document${completed === 1 ? "" : "s"} added.`;
        await refreshWorkspace();
        window.setTimeout(() => { uploadStatus.textContent = ""; }, 3500);
    } catch (error) {
        uploadStatus.textContent = error.message;
    } finally {
        uploadButton.disabled = false;
    }
}

document.getElementById("newConversationButton").addEventListener("click", async () => {
    try {
        await createConversation();
        questionInput.focus();
    } catch (error) {
        setNotice(error.message, true);
    }
});

function renderWorkspace() {
    renderDocuments();
    renderConversations();
    renderConversation();
    updateComposer();
}

function renderDocuments() {
    compareModeButton.disabled = state.documents.length < 2;
    compareModeButton.classList.toggle("active", state.compareMode);
    compareModeButton.title = state.compareMode ? "Exit document comparison" : "Compare documents";
    compareModeButton.setAttribute("aria-label", compareModeButton.title);
    if (!state.documents.length) {
        documentList.innerHTML = '<p class="empty-side">No documents yet.</p>';
        return;
    }
    documentList.innerHTML = state.documents.map((item) => `
        <button class="document-row ${item.document_id === state.activeDocumentId && !state.compareMode ? "selected" : ""} ${state.comparisonDocumentIds.has(item.document_id) ? "comparison-selected" : ""}"
            data-document-id="${escapeHtml(item.document_id)}" type="button">
            <span class="file-mark" aria-hidden="true">${state.compareMode && state.comparisonDocumentIds.has(item.document_id) ? "✓" : "PDF"}</span>
            <span class="row-copy">
                <span class="row-title">${escapeHtml(item.filename)}</span>
                <span class="row-meta">${item.page_count} page${item.page_count === 1 ? "" : "s"} · ${item.evidence_count} clauses</span>
            </span>
        </button>
    `).join("");
}

function renderConversations() {
    if (!state.conversations.length) {
        conversationList.innerHTML = '<p class="empty-side">No conversations yet.</p>';
        return;
    }
    conversationList.innerHTML = state.conversations.map((item) => `
        <button class="conversation-row ${item.conversation_id === state.activeConversationId ? "selected" : ""}"
            data-conversation-id="${escapeHtml(item.conversation_id)}" type="button">
            <span class="conversation-dot" aria-hidden="true"></span>
            <span class="row-copy">
                <span class="row-title">${escapeHtml(item.title || "New conversation")}</span>
                <span class="row-meta">${item.message_count} messages</span>
            </span>
        </button>
    `).join("");
}

function renderConversation() {
    const activeDocument = state.documents.find((item) => item.document_id === state.activeDocumentId);
    const title = state.activeConversation?.title || "New conversation";
    conversationHeader.innerHTML = `
        <div class="header-copy">
            <p class="header-kicker">DOCUMENT WORKSPACE</p>
            <h1>${escapeHtml(title)}</h1>
        </div>
        <div class="active-document">
            <span class="active-doc-label">Answering from</span>
            <span class="active-doc-name">${escapeHtml(activeDocument?.filename || "No document selected")}</span>
        </div>
    `;

    if (!state.activeDocumentId) {
        messageList.innerHTML = `
            <div class="empty-state">
                <span class="empty-state-mark" aria-hidden="true">PDF</span>
                <h2>Add a document to begin</h2>
                <p>Upload one or more PDFs. ClauseLens will keep each source distinct and link answers to the original clauses.</p>
                <button type="button" class="primary-action" data-action="upload">Add PDF documents</button>
            </div>
        `;
        return;
    }

    const messages = state.activeConversation?.messages || [];
    if (!messages.length) {
        messageList.innerHTML = `
            <div class="empty-state compact-empty">
                <span class="empty-state-mark" aria-hidden="true">C</span>
                <h2>What would you like to understand?</h2>
                <p>Ask about a clause, obligation, deadline, or practical next step in <strong>${escapeHtml(activeDocument?.filename || "this document")}</strong>.</p>
            </div>
        `;
        return;
    }

    messageList.innerHTML = messages.map(renderMessage).join("");
    messageList.scrollTop = messageList.scrollHeight;
}

function renderMessage(message) {
    if (message.role === "user") {
        const document = state.documents.find((item) => item.document_id === message.document_id);
        return `
            <article class="message user-message">
                <div class="message-label">You <span>${escapeHtml(document?.filename || "")}</span></div>
                <div class="user-content">${escapeHtml(message.content)}</div>
            </article>
        `;
    }

    if (message.comparison_findings?.length) {
        return renderComparisonMessage(message);
    }

    const status = (message.answer_status || "INSUFFICIENT_EVIDENCE").toLowerCase();
    const evidence = message.evidence || [];
    const sources = evidence.length ? `
        <section class="message-section source-section">
            <h3>Source evidence <span>${evidence.length}</span></h3>
            <div class="source-list">${evidence.map((item) => `
                <button class="source-button" type="button"
                    data-chunk-id="${escapeHtml(item.chunk_id)}"
                    data-document-id="${escapeHtml(item.document_id || message.document_id || "")}">
                    <span class="source-meta">Section ${escapeHtml(item.section || "N/A")} · Page ${item.page_number}</span>
                    <span class="source-action">View original text <span aria-hidden="true">→</span></span>
                </button>
            `).join("")}</div>
            <div class="source-viewer"></div>
        </section>
    ` : '<p class="no-evidence">No supporting clauses were identified in this document.</p>';

    const missing = (message.missing_information || []).length ? `
        <section class="message-section missing-section">
            <h3>What the document does not establish</h3>
            <ul>${message.missing_information.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </section>
    ` : "";
    const nextSteps = (message.follow_up_questions || []).length ? `
        <section class="message-section next-section">
            <h3>Useful next steps</h3>
            <ul>${message.follow_up_questions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </section>
    ` : "";
    const document = state.documents.find((item) => item.document_id === message.document_id);

    return `
        <article class="message assistant-message">
            <div class="message-label assistant-label"><span class="assistant-mark" aria-hidden="true">C</span> ClauseLens <span>${escapeHtml(document?.filename || "")}</span></div>
            <div class="answer-status ${escapeHtml(status)}">${escapeHtml((message.answer_status || "INSUFFICIENT EVIDENCE").replaceAll("_", " "))}</div>
            <p class="answer-content">${escapeHtml(message.content)}</p>
            ${message.explanation ? `<p class="explanation-content">${escapeHtml(message.explanation)}</p>` : ""}
            ${sources}${missing}${nextSteps}
        </article>
    `;
}

function renderComparisonMessage(message) {
    const status = (message.answer_status || "INSUFFICIENT_EVIDENCE").toLowerCase();
    const findings = message.comparison_findings.map((finding) => {
        const findingStatus = (finding.answer_status || "INSUFFICIENT_EVIDENCE").toLowerCase();
        const sourceButtons = (finding.evidence || []).map((item) => `
            <button class="source-button" type="button"
                data-chunk-id="${escapeHtml(item.chunk_id)}"
                data-document-id="${escapeHtml(finding.document_id)}">
                <span class="source-meta">Section ${escapeHtml(item.section || "N/A")} · Page ${item.page_number}</span>
                <span class="source-action">View original text <span aria-hidden="true">→</span></span>
            </button>
        `).join("");
        const missing = finding.missing_information?.length
            ? `<ul>${finding.missing_information.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
            : "";
        return `
            <section class="comparison-finding source-section">
                <div class="finding-heading">
                    <strong>${escapeHtml(finding.filename)}</strong>
                    <span class="answer-status ${escapeHtml(findingStatus)}">${escapeHtml((finding.answer_status || "INSUFFICIENT EVIDENCE").replaceAll("_", " "))}</span>
                </div>
                <p class="answer-content">${escapeHtml(finding.answer)}</p>
                <p class="explanation-content">${escapeHtml(finding.explanation)}</p>
                ${sourceButtons ? `<div class="source-list">${sourceButtons}</div>` : '<p class="no-evidence">No supporting evidence was validated for this document.</p>'}
                <div class="source-viewer"></div>
                ${missing ? `<div class="missing-section">${missing}</div>` : ""}
            </section>
        `;
    }).join("");
    return `
        <article class="message assistant-message">
            <div class="message-label assistant-label"><span class="assistant-mark" aria-hidden="true">C</span> ClauseLens <span>Document comparison</span></div>
            <div class="answer-status ${escapeHtml(status)}">${escapeHtml((message.answer_status || "INSUFFICIENT EVIDENCE").replaceAll("_", " "))}</div>
            <p class="answer-content">${escapeHtml(message.content)}</p>
            ${message.explanation ? `<p class="explanation-content">${escapeHtml(message.explanation)}</p>` : ""}
            <div class="comparison-findings">${findings}</div>
        </article>
    `;
}

function updateComposer() {
    const selectedCount = state.comparisonDocumentIds.size;
    const compareReady = state.compareMode && selectedCount >= 2 && selectedCount <= 4;
    const enabled = Boolean((state.compareMode ? compareReady : state.activeDocumentId) && state.activeConversationId && !state.busy);
    questionInput.disabled = !enabled;
    sendButton.disabled = !enabled || !questionInput.value.trim();
    sendButton.innerHTML = state.compareMode ? '<span aria-hidden="true">⇄</span> Compare' : '<span aria-hidden="true">↑</span> Send';
    composerHint.textContent = state.compareMode
        ? (selectedCount > 4 ? "Choose no more than four documents." : `Select 2–4 documents to compare (${selectedCount} selected).`)
        : state.activeDocumentId
        ? "Answers use this document's evidence. Conversation context only helps resolve follow-up references."
        : "Select a document to get started.";
}

compareModeButton.addEventListener("click", () => {
    state.compareMode = !state.compareMode;
    state.comparisonDocumentIds.clear();
    compareStatus.textContent = state.compareMode ? "Select 2–4 documents to compare." : "";
    renderWorkspace();
});

documentList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-document-id]");
    if (!button) return;
    if (state.compareMode) {
        const id = button.dataset.documentId;
        if (state.comparisonDocumentIds.has(id)) state.comparisonDocumentIds.delete(id);
        else state.comparisonDocumentIds.add(id);
        compareStatus.textContent = `${state.comparisonDocumentIds.size} selected · Choose 2–4 documents.`;
        renderWorkspace();
        return;
    }
    state.activeDocumentId = button.dataset.documentId;
    localStorage.setItem(DOCUMENT_KEY, state.activeDocumentId);
    renderWorkspace();
    questionInput.focus();
});

conversationList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-conversation-id]");
    if (!button) return;
    state.activeConversationId = button.dataset.conversationId;
    state.compareMode = false;
    state.comparisonDocumentIds.clear();
    localStorage.setItem(CONVERSATION_KEY, state.activeConversationId);
    try {
        await loadActiveConversation();
        if (state.activeConversation?.active_document_id) {
            state.activeDocumentId = state.activeConversation.active_document_id;
            localStorage.setItem(DOCUMENT_KEY, state.activeDocumentId);
        }
        renderWorkspace();
    } catch (error) {
        setNotice(error.message, true);
    }
});

messageList.addEventListener("click", async (event) => {
    const uploadAction = event.target.closest('[data-action="upload"]');
    if (uploadAction) {
        fileInput.click();
        return;
    }
    const sourceButton = event.target.closest(".source-button");
    if (!sourceButton) return;
    const viewer = sourceButton.closest(".source-section").querySelector(".source-viewer");
    viewer.textContent = "Loading original source...";
    try {
        const evidence = await api(`/documents/${encodeURIComponent(sourceButton.dataset.documentId)}/evidence`);
        const chunk = evidence.find((item) => item.chunk_id === sourceButton.dataset.chunkId);
        if (!chunk) throw new Error("The cited source is no longer available.");
        viewer.innerHTML = `
            <div class="original-source">
                <div class="original-source-head"><span>Original document</span><span>Section ${escapeHtml(chunk.section || "N/A")} · Page ${chunk.page_number}</span></div>
                <p>${escapeHtml(chunk.text)}</p>
            </div>
        `;
    } catch (error) {
        viewer.textContent = error.message;
    }
});

composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question || !state.activeConversationId || !state.activeDocumentId || state.busy) return;

    state.busy = true;
    const previousText = question;
    questionInput.value = "";
    updateComposer();
    messageList.insertAdjacentHTML("beforeend", `
        <article class="message user-message"><div class="message-label">You</div><div class="user-content">${escapeHtml(previousText)}</div></article>
        <div class="thinking-row"><span class="spinner"></span><span>Checking the document evidence...</span></div>
    `);
    messageList.scrollTop = messageList.scrollHeight;
    try {
        const endpoint = state.compareMode
            ? `/workspaces/${state.workspaceId}/conversations/${state.activeConversationId}/comparisons`
            : `/workspaces/${state.workspaceId}/conversations/${state.activeConversationId}/questions`;
        const payload = state.compareMode
            ? { question: previousText, document_ids: Array.from(state.comparisonDocumentIds) }
            : { question: previousText, active_document_id: state.activeDocumentId };
        await api(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        await Promise.all([loadActiveConversation(), refreshConversationList()]);
        renderConversations();
        renderConversation();
    } catch (error) {
        const thinking = messageList.querySelector(".thinking-row");
        if (thinking) thinking.remove();
        messageList.insertAdjacentHTML("beforeend", `<p class="request-error">${escapeHtml(error.message)}</p>`);
        questionInput.value = previousText;
    } finally {
        state.busy = false;
        updateComposer();
        questionInput.focus();
    }
});

async function refreshConversationList() {
    state.conversations = await api(`/workspaces/${state.workspaceId}/conversations`);
}

questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
    updateComposer();
});
questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        composer.requestSubmit();
    }
});

function setNotice(message, isError = false) {
    workspaceNotice.textContent = message;
    workspaceNotice.classList.toggle("error", isError);
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    })[char]);
}

initializeWorkspace();
