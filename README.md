# Valentius AI - Stateful Conversational Chatbot with Persistent Long-Term Memory

Valentius AI is an intelligent chatbot powered by the **Google Gemini API** (`google-genai` SDK) and a **Flask** backend. It features genuine multi-turn conversational memory, a 10-message sliding-window context management strategy, persistent **SQLite** storage, a modern **ChatGPT-style responsive sidebar**, and **persistent cross-conversation long-term user memory**.

---

## 🌟 Key Features

- **Automatic Background Long-Term Memory**:
  - Automatically learns and remembers user facts from natural chat (name, department, programming skills, tech preferences like Python for backend, explanation preferences, and active projects).
  - Operates completely in the background without requiring any manual memory panel, variable configuration, or remember buttons.
  - Updates existing facts smoothly when the user offers corrections (e.g., "Actually, my name is Alex").
  - Facts persist across new chats, conversation deletions, browser reloads, and backend server restarts.
  - Injects a structured user profile into Gemini's context while preserving strict conversation message isolation.
  - Filters out casual chit-chat, questions, and sensitive data (passwords, API keys, secrets).
- **Persistent SQLite Storage**:
  - Automatically initializes `conversations.db` on startup with `conversations`, `messages`, and `memories` tables.
  - Stores unique conversation IDs, titles, timestamps, and full user/model turns.
  - Uses foreign keys with `ON DELETE CASCADE` for clean, safe message deletions.
  - Parameterized SQL queries preventing SQL injection vulnerabilities.
- **Minimal, Professional Left Sidebar**:
  - Valentius AI branding with gold accent icon.
  - `+ New Chat` button for immediate fresh session creation.
  - `Chat History` section with a live conversation counter.
  - Clickable, auto-truncating conversation titles with active conversation highlights.
  - Individual conversation delete buttons (`🗑`).
  - Unobtrusive Settings button near the bottom.
  - Clean layout with no unnecessary decorative badges or empty gaps.
  - Mobile responsive slide-out drawer with backdrop blur and hamburger toggle (`☰`).
- **Genuine Multi-Turn Memory & Sliding Window**:
  - Remembers earlier context (user name, topics, technical details) across turns.
  - Limits active prompt context to the latest 10 messages (5 user-model exchanges).
  - Guarantees the prompt context begins with a `user` turn for Gemini compliance.
- **Defined Bot Identity & System Instruction**:
  - Persona: *Valentius AI* — a friendly, accurate assistant for programming, technology, and learning.
- **Automatic Chrome Launch**:
  - Automatically opens Google Chrome to `http://127.0.0.1:5000` when the backend is ready.
  - Single-trigger guard avoids duplicate tabs during Flask debug reloads.

---

## 🛠 Technology Stack

- **Backend**: Python 3.12+ / Flask 3.1+
- **Database**: SQLite3 (`conversations.db` with `conversations`, `messages`, and `memories`)
- **AI SDK**: Google GenAI SDK (`google-genai`)
- **AI Model**: Google Gemini (`gemini-3.5-flash-lite`)
- **Frontend**: HTML5, CSS3 (Modern Dark Theme with Gold Accents), Vanilla JavaScript (ES6+)
- **Testing**: Python `unittest` suite (23 comprehensive automated tests)

---

## 📁 Project Structure

```
AI-Chatbot/
│
├── frontend/
│   ├── index.html        # ChatGPT-style sidebar & chat interface
│   ├── style.css         # Responsive dark glassmorphism theme with gold accents
│   └── script.js         # Sidebar management, automatic memory sync, and chat streaming
│
├── chatbot.py            # Flask REST API, SQLite storage, Memory & Gemini integration
├── test_chatbot.py       # Automated test suite (23 comprehensive tests)
├── conversations.db      # SQLite database (auto-generated)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables with Gemini API key (git-ignored)
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/conversations` | Returns all saved conversations, newest first |
| `POST` | `/api/conversations` | Creates a new conversation row in SQLite |
| `GET` | `/api/conversations/<id>` | Retrieves complete message history for the selected conversation |
| `DELETE` | `/api/conversations/<id>` | Deletes the selected conversation and its messages |
| `POST` | `/api/chat` | Extracts user facts, sends message to Gemini, and persists turns to SQLite |
| `GET` | `/api/memories` | Returns all saved persistent user memories |
| `POST` | `/api/memories` | Saves or updates a user memory key-value pair |
| `DELETE` | `/api/memories/<key>` | Deletes a specific memory by key |
| `DELETE` | `/api/memories` | Clears all persistent user memories |
| `GET` | `/api/health` | Health status and configuration check (reports conversations and memory counts) |
| `GET` | `/` | Serves the web frontend |

---

## 🚀 How to Run the Application

### 1. Activate the Virtual Environment
In PowerShell inside `c:\Users\Sivaraj\Desktop\AI-Chatbot`:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Configure Your API Key
Ensure your Google Gemini API key is set in `.env`:
```env
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
MAX_CONTEXT_MESSAGES=10
FLASK_PORT=5000
FLASK_DEBUG=True
```

### 3. Start the Server
```powershell
python chatbot.py
```
- The backend starts on `http://127.0.0.1:5000`.
- Google Chrome automatically opens directly to the application.

---

## 🧪 Automated Testing

Execute the unit test suite verifying all 12 tests:
```powershell
.\venv\Scripts\python.exe -m unittest test_chatbot.py -v
```
All 12 automated tests run and pass in under 1 second.
