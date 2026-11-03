/**
 * Valentius AI - Frontend Application Script
 * Stateful SQLite Chat Client with ChatGPT-style Sidebar
 */

(function () {
  'use strict';

  // --- State ---
  let currentConversationId = null;
  let conversations = [];
  let isGenerating = false;

  // --- DOM Elements ---
  const sidebar = document.getElementById('sidebar');
  const sidebarBackdrop = document.getElementById('sidebarBackdrop');
  const mobileToggleBtn = document.getElementById('mobileToggleBtn');
  const closeSidebarBtn = document.getElementById('closeSidebarBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const headerNewChatBtn = document.getElementById('headerNewChatBtn');
  const conversationsList = document.getElementById('conversationsList');
  const historyCount = document.getElementById('historyCount');
  const activeChatTitle = document.getElementById('activeChatTitle');

  // Settings DOM Elements
  const settingsBtn = document.getElementById('settingsBtn');
  const settingsModalOverlay = document.getElementById('settingsModalOverlay');
  const closeSettingsModalXBtn = document.getElementById('closeSettingsModalXBtn');
  const closeSettingsModalBtn = document.getElementById('closeSettingsModalBtn');
  const clearAllConversationsBtn = document.getElementById('clearAllConversationsBtn');
  const resetMemoriesBtn = document.getElementById('resetMemoriesBtn');

  const chatMain = document.getElementById('chatMain');
  const messagesList = document.getElementById('messagesList');
  const welcomeBanner = document.getElementById('welcomeBanner');
  const loadingIndicator = document.getElementById('loadingIndicator');

  const messageInput = document.getElementById('messageInput');
  const sendBtn = document.getElementById('sendBtn');

  // Determine API base URL
  const API_BASE_URL = window.location.origin.includes('5000')
    ? ''
    : (window.location.protocol === 'file:' ? 'http://127.0.0.1:5000' : 'http://127.0.0.1:5000');

  function generateUUID() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return 'conv-' + Date.now() + '-' + Math.random().toString(36).substring(2, 9);
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      chatMain.scrollTop = chatMain.scrollHeight;
    });
  }

  function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + 'px';
  }

  function updateSendButtonState() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isGenerating;
  }

  // --- Mobile Sidebar Controls ---
  function openMobileSidebar() {
    sidebar.classList.add('open');
    sidebarBackdrop.classList.add('active');
  }

  function closeMobileSidebar() {
    sidebar.classList.remove('open');
    sidebarBackdrop.classList.remove('active');
  }

  // --- Markdown Rendering ---
  function renderContent(text) {
    if (!text) return '';

    if (typeof marked !== 'undefined' && marked.parse) {
      try {
        marked.setOptions({ gfm: true, breaks: true });
        let html = marked.parse(text);
        return enhanceCodeBlocks(html);
      } catch (e) {
        console.warn('Marked parse error, using fallback:', e);
      }
    }

    // Fallback parser
    let safe = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    safe = safe.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (m, lang, code) => {
      return `<div class="code-wrapper"><div class="code-header"><span>${lang || 'code'}</span><button class="copy-code-btn" type="button">Copy</button></div><pre><code>${code.trim()}</code></pre></div>`;
    });
    safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
    safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    return safe.split(/\n\s*\n/).map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
  }

  function enhanceCodeBlocks(html) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const preBlocks = doc.querySelectorAll('pre');

    preBlocks.forEach(pre => {
      const code = pre.querySelector('code');
      const langClass = code ? Array.from(code.classList).find(c => c.startsWith('language-')) : '';
      const lang = langClass ? langClass.replace('language-', '') : 'code';

      const header = doc.createElement('div');
      header.className = 'code-header';
      header.innerHTML = `
        <span>${lang}</span>
        <button class="copy-code-btn" type="button">Copy</button>
      `;

      pre.parentNode.insertBefore(header, pre);
      const wrapper = doc.createElement('div');
      wrapper.className = 'code-wrapper';
      header.parentNode.insertBefore(wrapper, header);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    });

    return doc.body.innerHTML;
  }

  // --- DOM Message Injection ---
  function appendMessage(role, text, isError = false) {
    welcomeBanner.style.display = 'none';

    const row = document.createElement('div');
    row.className = `message-row ${role === 'user' ? 'user' : 'ai'}`;

    const bubble = document.createElement('div');
    bubble.className = `message-bubble ${isError ? 'error-bubble' : ''}`;

    if (role === 'user') {
      bubble.textContent = text;
    } else {
      bubble.innerHTML = renderContent(text);
    }

    row.appendChild(bubble);
    messagesList.appendChild(row);

    // Code block copy buttons
    bubble.querySelectorAll('.copy-code-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const pre = btn.closest('.code-wrapper').querySelector('pre code');
        if (pre) {
          navigator.clipboard.writeText(pre.innerText).then(() => {
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
          });
        }
      });
    });

    // Apply highlight.js if available
    if (typeof hljs !== 'undefined') {
      row.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    }

    scrollToBottom();
  }

  // --- 1. Sidebar Conversations Management ---
  async function fetchConversations() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/conversations`);
      if (!res.ok) return;
      const data = await res.json();
      conversations = data.conversations || [];
      renderSidebar();
    } catch (e) {
      console.error('Failed to load conversations:', e);
    }
  }

  function renderSidebar() {
    conversationsList.innerHTML = '';
    historyCount.textContent = conversations.length;

    if (conversations.length === 0) {
      const emptyEl = document.createElement('div');
      emptyEl.className = 'history-empty';
      emptyEl.textContent = 'No saved conversations.';
      conversationsList.appendChild(emptyEl);
      return;
    }

    conversations.forEach(conv => {
      const item = document.createElement('div');
      item.className = `history-item ${conv.conversation_id === currentConversationId ? 'active' : ''}`;
      item.setAttribute('data-id', conv.conversation_id);

      const titleSpan = document.createElement('span');
      titleSpan.className = 'history-item-title';
      titleSpan.textContent = conv.title || 'New Conversation';
      titleSpan.title = conv.title || 'New Conversation';
      item.appendChild(titleSpan);

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'history-item-delete';
      deleteBtn.title = 'Delete chat';
      deleteBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      `;
      deleteBtn.addEventListener('click', (e) => handleDeleteConversation(conv.conversation_id, e));
      item.appendChild(deleteBtn);

      item.addEventListener('click', () => selectConversation(conv.conversation_id));
      conversationsList.appendChild(item);
    });
  }

  async function selectConversation(conversationId) {
    if (isGenerating) return;
    currentConversationId = conversationId;
    closeMobileSidebar();

    try {
      const res = await fetch(`${API_BASE_URL}/api/conversations/${conversationId}`);
      if (!res.ok) {
        startNewChat();
        return;
      }

      const data = await res.json();
      const conv = data.conversation;
      const messages = data.messages || [];

      activeChatTitle.textContent = conv.title || 'Conversation';
      messagesList.innerHTML = '';

      if (messages.length === 0) {
        welcomeBanner.style.display = 'block';
      } else {
        welcomeBanner.style.display = 'none';
        messages.forEach(m => {
          appendMessage(m.role === 'model' ? 'ai' : 'user', m.content);
        });
      }

      renderSidebar();
      scrollToBottom();
      messageInput.focus();

    } catch (e) {
      console.error('Failed to load conversation messages:', e);
    }
  }

  async function handleDeleteConversation(conversationId, event) {
    if (event) event.stopPropagation();

    try {
      const res = await fetch(`${API_BASE_URL}/api/conversations/${conversationId}`, {
        method: 'DELETE'
      });

      if (res.ok) {
        // Remove locally
        conversations = conversations.filter(c => c.conversation_id !== conversationId);

        if (currentConversationId === conversationId) {
          if (conversations.length > 0) {
            selectConversation(conversations[0].conversation_id);
          } else {
            startNewChat();
          }
        } else {
          renderSidebar();
        }
      }
    } catch (e) {
      console.error('Failed to delete conversation:', e);
    }
  }

  // --- New Chat Functionality ---
  async function startNewChat() {
    if (isGenerating) return;

    try {
      const res = await fetch(`${API_BASE_URL}/api/conversations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Conversation' })
      });

      if (res.ok) {
        const data = await res.json();
        const conv = data.conversation;
        currentConversationId = conv.conversation_id;
        conversations.unshift(conv);
      } else {
        currentConversationId = generateUUID();
      }
    } catch (e) {
      currentConversationId = generateUUID();
    }

    messagesList.innerHTML = '';
    welcomeBanner.style.display = 'block';
    loadingIndicator.style.display = 'none';
    activeChatTitle.textContent = 'New Conversation';
    messageInput.value = '';
    adjustTextareaHeight();
    updateSendButtonState();
    renderSidebar();
    closeMobileSidebar();
    messageInput.focus();
  }

  // --- Message Sending & Backend Communication ---
  async function handleSendMessage() {
    const text = messageInput.value.trim();
    if (!text || isGenerating) return;

    if (!currentConversationId) {
      currentConversationId = generateUUID();
    }

    // Append user message on the right
    appendMessage('user', text);

    messageInput.value = '';
    adjustTextareaHeight();
    updateSendButtonState();

    // Show loading indicator
    isGenerating = true;
    loadingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          conversation_id: currentConversationId
        })
      });

      const data = await response.json();

      if (!response.ok || data.status === 'error') {
        const errorMsg = data.error || 'Failed to generate response.';
        appendMessage('ai', `⚠️ **Error:** ${errorMsg}`, true);
        return;
      }

      // Update state from server response
      if (data.conversation_id) {
        currentConversationId = data.conversation_id;
      }
      if (data.title) {
        activeChatTitle.textContent = data.title;
      }

      // Append AI response on the left
      appendMessage('ai', data.response);

      // Refresh sidebar list to reflect new title & updated timestamp
      fetchConversations();

    } catch (err) {
      console.error('Network error:', err);
      appendMessage(
        'ai',
        '⚠️ **Connection Error:** Unable to reach the Valentius AI server. Ensure `python chatbot.py` is running on port 5000.',
        true
      );
    } finally {
      isGenerating = false;
      loadingIndicator.style.display = 'none';
      updateSendButtonState();
      scrollToBottom();
      messageInput.focus();
    }
  }

  // --- Settings Modal Handlers (Unobtrusive) ---
  function openSettingsModal() {
    if (settingsModalOverlay) {
      settingsModalOverlay.style.display = 'flex';
    }
  }

  function closeSettingsModal() {
    if (settingsModalOverlay) {
      settingsModalOverlay.style.display = 'none';
    }
  }

  async function handleResetMemories() {
    if (!confirm('Are you sure you want to reset all automatically learned profile memories?')) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/memories`, { method: 'DELETE' });
      if (res.ok) {
        alert('Learned memories have been reset.');
        closeSettingsModal();
      }
    } catch (e) {
      console.error('Failed to reset memories:', e);
    }
  }

  async function handleClearAllConversations() {
    if (!confirm('Are you sure you want to delete all saved conversations? This cannot be undone.')) return;
    try {
      for (const conv of conversations) {
        await fetch(`${API_BASE_URL}/api/conversations/${conv.conversation_id}`, { method: 'DELETE' });
      }
      conversations = [];
      closeSettingsModal();
      startNewChat();
    } catch (e) {
      console.error('Failed to clear conversations:', e);
    }
  }

  // --- App Initialization ---
  async function init() {
    // Event listeners
    messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    messageInput.addEventListener('input', () => {
      adjustTextareaHeight();
      updateSendButtonState();
    });

    sendBtn.addEventListener('click', handleSendMessage);
    newChatBtn.addEventListener('click', startNewChat);
    headerNewChatBtn.addEventListener('click', startNewChat);

    mobileToggleBtn.addEventListener('click', openMobileSidebar);
    closeSidebarBtn.addEventListener('click', closeMobileSidebar);
    sidebarBackdrop.addEventListener('click', closeMobileSidebar);

    // Settings Modal Listeners
    if (settingsBtn) settingsBtn.addEventListener('click', openSettingsModal);
    if (closeSettingsModalXBtn) closeSettingsModalXBtn.addEventListener('click', closeSettingsModal);
    if (closeSettingsModalBtn) closeSettingsModalBtn.addEventListener('click', closeSettingsModal);
    if (settingsModalOverlay) {
      settingsModalOverlay.addEventListener('click', (e) => {
        if (e.target === settingsModalOverlay) closeSettingsModal();
      });
    }
    if (resetMemoriesBtn) resetMemoriesBtn.addEventListener('click', handleResetMemories);
    if (clearAllConversationsBtn) clearAllConversationsBtn.addEventListener('click', handleClearAllConversations);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && settingsModalOverlay && settingsModalOverlay.style.display !== 'none') {
        closeSettingsModal();
      }
    });

    // Initial load: fetch saved conversations from SQLite backend
    await fetchConversations();

    if (conversations.length > 0) {
      // Load the most recently active conversation
      selectConversation(conversations[0].conversation_id);
    } else {
      startNewChat();
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
