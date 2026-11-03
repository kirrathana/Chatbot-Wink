import os
import sys

# Configure UTF-8 encoding for safe output across Windows command prompts
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import time
import uuid
import sqlite3
import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Import Google GenAI SDK
from google import genai
from google.genai import types
from google.genai import errors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ValentiusAI")

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations.db")

# Defined System Instruction
SYSTEM_INSTRUCTION = (
    "You are Valentius AI, a friendly, intelligent, and helpful AI assistant. "
    "Your purpose is to assist users with programming, technology, learning, "
    "problem-solving, and general knowledge. Maintain a natural conversational tone. "
    "Use previous conversation context to understand follow-up questions. "
    "Give clear, accurate, and well-structured answers. "
    "If you do not know something, admit it instead of inventing information."
)


# ---------------------------------------------------------------------------
# SQLite Database Management
# ---------------------------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    """Returns a new SQLite connection with row_factory and foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initializes SQLite database and creates tables if they don't exist."""
    with get_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_key TEXT NOT NULL UNIQUE,
                memory_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(memory_key);
        """)
    logger.info("SQLite database initialized at: %s", DB_PATH)


# Initialize SQLite tables on startup
init_db()


class Conversation:
    """Represents a single conversation thread backed by SQLite."""

    def __init__(self, conversation_id: str, title: Optional[str] = None, created_at: Optional[str] = None, updated_at: Optional[str] = None):
        self.conversation_id: str = conversation_id
        self.title: str = title or "New Conversation"
        now = datetime.now(timezone.utc).isoformat()
        self.created_at: str = created_at or now
        self.updated_at: str = updated_at or self.created_at
        self.messages: List[Dict[str, str]] = []

        # Ensure conversation record exists in SQLite
        try:
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (self.conversation_id, self.title, self.created_at, self.updated_at)
                )
        except Exception as e:
            logger.error("Failed to ensure conversation row in SQLite: %s", e)

        self._load_messages()

    def _load_messages(self):
        """Loads messages from the SQLite database."""
        try:
            with get_db_connection() as conn:
                rows = conn.execute(
                    "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                    (self.conversation_id,)
                ).fetchall()
                self.messages = [
                    {"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
                    for r in rows
                ]
        except Exception as e:
            logger.error("Failed to load messages for conversation %s: %s", self.conversation_id, e)
            self.messages = []

    def add_message(self, role: str, content: str) -> Dict[str, str]:
        """Appends a message to database and in-memory list."""
        gemini_role = "model" if role in ("assistant", "model", "bot") else "user"
        now = datetime.now(timezone.utc).isoformat()
        msg = {
            "role": gemini_role,
            "content": content,
            "timestamp": now
        }

        # Auto-update title if it's the first user message
        if self.title in ("New Chat", "New Conversation") and gemini_role == "user":
            trimmed = content.strip().replace("\n", " ")
            self.title = (trimmed[:40] + "...") if len(trimmed) > 40 else trimmed

        self.updated_at = now

        try:
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (self.title, self.updated_at, self.conversation_id)
                )
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (self.conversation_id, gemini_role, content, now)
                )
        except Exception as e:
            logger.error("Failed to persist message to SQLite for conversation %s: %s", self.conversation_id, e)

        self.messages.append(msg)
        return msg

    def get_sliding_window_context(self, max_messages: int = MAX_CONTEXT_MESSAGES) -> List[Dict[str, str]]:
        """
        Retrieves active context using sliding window strategy.
        Keeps up to max_messages of the latest messages.
        Ensures window begins with a 'user' message.
        """
        if not self.messages:
            return []

        window = self.messages[-max_messages:]
        while window and window[0].get("role") != "user":
            window = window[1:]

        return window

    def summarize_context_hook(self) -> Optional[str]:
        if len(self.messages) > MAX_CONTEXT_MESSAGES:
            older = self.messages[:-MAX_CONTEXT_MESSAGES]
            return f"Summary of {len(older)} previous messages archived."
        return None

    def clear(self):
        """Clears all messages for this conversation."""
        try:
            with get_db_connection() as conn:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (self.conversation_id,))
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), self.conversation_id)
                )
        except Exception as e:
            logger.error("Failed to clear messages in SQLite: %s", e)
        self.messages = []
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_messages": len(self.messages),
            "active_context_messages": len(self.get_sliding_window_context())
        }


class ConversationsDictWrapper:
    """Provides dictionary-like interface around SQLite for backward compatibility with tests."""
    def __init__(self, manager: "ConversationManager"):
        self.manager = manager

    def clear(self):
        self.manager.clear_all()

    def __getitem__(self, conversation_id: str) -> Conversation:
        conv = self.manager.get(conversation_id)
        if not conv:
            raise KeyError(conversation_id)
        return conv

    def __contains__(self, conversation_id: str) -> bool:
        return self.manager.get(conversation_id) is not None

    def __len__(self) -> int:
        return len(self.manager.list_all())

    def values(self) -> List[Conversation]:
        return [self.manager.get(c["conversation_id"]) for c in self.manager.list_all()]

    def get(self, conversation_id: str, default=None):
        conv = self.manager.get(conversation_id)
        return conv if conv is not None else default


class ConversationManager:
    """Manages persistent conversations using SQLite."""

    def __init__(self, max_context_messages: int = MAX_CONTEXT_MESSAGES):
        self.max_context_messages = max_context_messages
        self.conversations = ConversationsDictWrapper(self)

    def create_conversation(self, title: Optional[str] = None, conversation_id: Optional[str] = None) -> Conversation:
        """Explicitly creates a new conversation row in SQLite."""
        cid = conversation_id.strip() if conversation_id and str(conversation_id).strip() else str(uuid.uuid4())
        t = title.strip() if title and str(title).strip() else "New Conversation"
        now = datetime.now(timezone.utc).isoformat()

        try:
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (cid, t, now, now)
                )
        except Exception as e:
            logger.error("Failed to insert conversation %s into SQLite: %s", cid, e)

        return Conversation(conversation_id=cid, title=t, created_at=now, updated_at=now)

    def get_or_create(self, conversation_id: Optional[str] = None) -> Conversation:
        """Retrieves an existing conversation or creates a new one in SQLite."""
        if conversation_id and str(conversation_id).strip():
            cid = str(conversation_id).strip()
            existing = self.get(cid)
            if existing:
                return existing
            return self.create_conversation(conversation_id=cid)
        return self.create_conversation()

    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Fetches a conversation and its messages from SQLite."""
        if not conversation_id:
            return None
        try:
            with get_db_connection() as conn:
                row = conn.execute(
                    "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                    (conversation_id,)
                ).fetchone()
                if row:
                    return Conversation(
                        conversation_id=row["id"],
                        title=row["title"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"]
                    )
        except Exception as e:
            logger.error("Error fetching conversation %s from SQLite: %s", conversation_id, e)
        return None

    def delete(self, conversation_id: str) -> bool:
        """Deletes a conversation and its cascaded messages from SQLite."""
        if not conversation_id:
            return False
        try:
            with get_db_connection() as conn:
                cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
                return cur.rowcount > 0
        except Exception as e:
            logger.error("Error deleting conversation %s from SQLite: %s", conversation_id, e)
            return False

    def list_all(self) -> List[dict]:
        """Returns all saved conversations from SQLite, newest updated first."""
        try:
            with get_db_connection() as conn:
                rows = conn.execute("""
                    SELECT c.id AS conversation_id, c.title, c.created_at, c.updated_at,
                           COUNT(m.id) AS total_messages
                    FROM conversations c
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                """).fetchall()
                return [
                    {
                        "conversation_id": r["conversation_id"],
                        "title": r["title"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "total_messages": r["total_messages"]
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error("Error listing conversations from SQLite: %s", e)
            return []

    def clear_all(self):
        """Wipes all conversations and messages from SQLite (used for test teardown)."""
        try:
            with get_db_connection() as conn:
                conn.execute("DELETE FROM messages")
                conn.execute("DELETE FROM conversations")
        except Exception as e:
            logger.error("Error clearing SQLite database: %s", e)


# Initialize global conversation manager
conversation_manager = ConversationManager(max_context_messages=MAX_CONTEXT_MESSAGES)


# ---------------------------------------------------------------------------
# Persistent Long-Term User Memory Management
# ---------------------------------------------------------------------------
def save_memory(memory_key: str, memory_value: str) -> dict:
    """
    Saves or updates a persistent memory key-value pair in SQLite.
    Returns the saved memory dictionary.
    """
    key = str(memory_key).strip()
    val = str(memory_value).strip()
    if not key or not val:
        raise ValueError("Memory key and value cannot be empty.")

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO memories (memory_key, memory_value, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_key) DO UPDATE SET
                memory_value = excluded.memory_value,
                updated_at = excluded.updated_at
        """, (key, val, now, now))

    return get_memory(key)


def get_memory(memory_key: str) -> Optional[dict]:
    """Retrieves a single memory fact by key from SQLite."""
    if not memory_key:
        return None
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, memory_key, memory_value, created_at, updated_at FROM memories WHERE memory_key = ?",
            (str(memory_key).strip(),)
        ).fetchone()
        if row:
            return {
                "id": row["id"],
                "memory_key": row["memory_key"],
                "memory_value": row["memory_value"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
    return None


def get_all_memories() -> List[dict]:
    """Retrieves all stored long-term memories from SQLite."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, memory_key, memory_value, created_at, updated_at FROM memories ORDER BY id ASC"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "memory_key": r["memory_key"],
                "memory_value": r["memory_value"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"]
            }
            for r in rows
        ]


def update_memory(memory_key: str, memory_value: str) -> bool:
    """Updates an existing memory value in SQLite. Returns True if updated."""
    key = str(memory_key).strip()
    val = str(memory_value).strip()
    if not key or not val:
        return False

    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cur = conn.execute(
            "UPDATE memories SET memory_value = ?, updated_at = ? WHERE memory_key = ?",
            (val, now, key)
        )
        return cur.rowcount > 0


def delete_memory(memory_key: str) -> bool:
    """Deletes a memory fact by key from SQLite. Returns True if deleted."""
    if not memory_key:
        return False
    with get_db_connection() as conn:
        cur = conn.execute(
            "DELETE FROM memories WHERE memory_key = ?",
            (str(memory_key).strip(),)
        )
        return cur.rowcount > 0


def clear_all_memories() -> bool:
    """Clears all stored memories from SQLite."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM memories")
    return True


def retrieve_relevant_memories(query: str = "") -> List[dict]:
    """
    Retrieves memories relevant to the user query for prompt augmentation.
    Returns all stored memories.
    """
    return get_all_memories()


# Sensitive keywords and high-entropy patterns that should NEVER be stored in long-term memory
SENSITIVE_PATTERNS = [
    re.compile(r'\b(password|passwd|secret|api[_-]?key|token|bearer|auth|private[_-]?key)\b', re.IGNORECASE),
    re.compile(r'\b[A-Za-z0-9_\-]{32,}\b')
]


def extract_memories_from_text(text: str) -> Dict[str, str]:
    """
    Analyzes user message text to automatically extract explicit personal facts
    (e.g., name, department, coding experience, explanation preference, current project).
    Ignores questions, casual greetings, and sensitive data (passwords, tokens).
    """
    memories: Dict[str, str] = {}
    if not text or not isinstance(text, str):
        return memories

    # Block extraction if message contains sensitive credentials or secrets
    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(text):
            return memories

    # Split into clause-level segments
    clauses = re.split(r'[.\n;!]+', text)
    for clause in clauses:
        clause = clause.strip()
        if not clause or clause.endswith('?'):
            continue

        lower = clause.lower()
        # Skip pure question phrases
        if any(lower.startswith(q) for q in (
            'what ', 'who ', 'where ', 'when ', 'why ', 'how ',
            'can you ', 'do you ', 'could you ', 'is ', 'are ', 'am '
        )):
            continue

        # 1. User Name & Corrections
        name_match = re.search(
            r'\b(?:my name is|call me|actually,? my name is|actually,? call me|my name was actually)\s+([A-Za-z][A-Za-z0-9_\'\-]{1,25})\b',
            clause, re.IGNORECASE
        )
        if not name_match:
            name_match2 = re.search(r'\b(?:i am|i\'m)\s+([a-zA-Z]{2,20})\b', clause, re.IGNORECASE)
            if name_match2:
                cand = name_match2.group(1)
                orig_match = re.search(r'\b(?:i am|i\'m)\s+(' + re.escape(cand) + r')\b', clause, re.IGNORECASE)
                cand_orig = orig_match.group(1) if orig_match else cand
                if cand_orig[0].isupper() and cand.lower() not in (
                    'a', 'an', 'the', 'studying', 'learning', 'working', 'looking',
                    'trying', 'very', 'not', 'just', 'doing', 'having', 'going',
                    'here', 'happy', 'excited', 'ready', 'new', 'busy', 'fine',
                    'good', 'student', 'developer', 'engineer', 'currently', 'interested',
                    'also', 'still', 'already', 'now', 'only', 'back', 'so', 'too'
                ):
                    name_match = name_match2
        if name_match:
            cand_name = name_match.group(1).strip()
            if cand_name.lower() not in ('here', 'happy', 'ready', 'fine', 'good', 'student', 'developer'):
                memories['user_name'] = cand_name

        # 2. Department / Field of Study & Corrections
        dept_match = re.search(
            r'\b(?:studying in(?: the)?|study in(?: the)?|in the|switched to(?: the)?)\s+([A-Za-z0-9\s&\-]{2,30}?)\s+(?:dept|department|engineering|branch)\b',
            clause, re.IGNORECASE
        )
        if not dept_match:
            dept_match = re.search(
                r'\bmy department is\s+([A-Za-z0-9\s&\-]{2,30}?)(?:\s+(?:dept|department))?\b',
                clause, re.IGNORECASE
            )
        if not dept_match:
            dept_match = re.search(
                r'\b(?:studying|major in|majoring in)\s+([A-Za-z0-9\s&\-]{2,25})\b',
                clause, re.IGNORECASE
            )
        if dept_match:
            dept = dept_match.group(1).strip()
            dept = re.sub(r'^(the|an?)\s+', '', dept, flags=re.IGNORECASE).strip()
            if dept and dept.lower() not in ('python', 'flask', 'java', 'coding', 'programming', 'college', 'school', 'university'):
                memories['department'] = dept

        # 3. Coding Experience / Knowledge
        if re.search(r'\b(?:don\'?t have|do not have|have no|zero|no)\s+(?:prior\s+)?(?:coding|programming)\s+experience\b', clause, re.IGNORECASE):
            memories['coding_experience'] = 'No prior coding experience (Beginner)'
        elif re.search(r'\b(?:beginner|novice|new to)\s+(?:in|at|to)?\s*(?:coding|programming|python)\b', clause, re.IGNORECASE):
            memories['coding_experience'] = 'Beginner'
        elif re.search(r'\b(?:intermediate|experienced|expert|senior)\s+(?:in|at)?\s*(?:coding|programming|python)\b', clause, re.IGNORECASE):
            match_exp = re.search(r'\b(intermediate|experienced|expert|senior)\b', clause, re.IGNORECASE)
            if match_exp:
                memories['coding_experience'] = match_exp.group(1).capitalize()

        # 4. Explicit Technology / Domain Preferences (e.g. "I prefer Python for backend development")
        pref_tech = re.search(
            r'\b(?:i prefer|my preference is|i like using)\s+([A-Za-z0-9#+.\s]{2,20}?)\s+for\s+([A-Za-z0-9\s\-_]{3,35})\b',
            clause, re.IGNORECASE
        )
        if pref_tech:
            tool = pref_tech.group(1).strip()
            area = pref_tech.group(2).strip()
            memories['preference'] = f"{tool} for {area}"
        else:
            fav_lang = re.search(r'\bmy favorite (?:programming )?language is\s+([A-Za-z0-9#+.]+)\b', clause, re.IGNORECASE)
            if fav_lang:
                memories['favorite_language'] = fav_lang.group(1).strip()

        # 5. Explanation Preferences
        pref_match = re.search(
            r'\b(?:i prefer|give me|keep explanations?|make explanations?|i like)\s+((?:[A-Za-z0-9\-]+\s+)*(?:explanations?|answers?|simple|concise|step by step|detailed|in depth|brief))\b',
            clause, re.IGNORECASE
        )
        if pref_match:
            memories['explanation_preference'] = pref_match.group(1).strip()
        elif re.search(r'\b(?:explain like i\'?m 5|eli5)\b', clause, re.IGNORECASE):
            memories['explanation_preference'] = 'Simple explanations (Explain Like I\'m 5)'
        elif re.search(r'\b(?:step by step|detailed explanation)\b', clause, re.IGNORECASE):
            memories['explanation_preference'] = 'Detailed, step-by-step explanations'

        # 6. Current Project
        proj_match = re.search(
            r'\b(?:working on|building|developing|my project is)\s+(?:an?\s+|the\s+)?([A-Za-z0-9\s\-_]{2,40}?(?:project|bot|app|application|website|system|tool))\b',
            clause, re.IGNORECASE
        )
        if proj_match:
            memories['current_project'] = proj_match.group(1).strip()

    return memories


def build_system_instruction_with_memories(base_instruction: str, memories: Optional[List[dict]] = None) -> str:
    """
    Augments the base system instruction with persistent long-term memories.
    Returns base_instruction unchanged when no memories exist, maintaining backward compatibility.
    """
    if memories is None:
        memories = get_all_memories()

    if not memories:
        return base_instruction

    memory_lines = [f"- {m['memory_key']}: {m['memory_value']}" for m in memories]
    memory_block = "\n".join(memory_lines)

    return (
        f"{base_instruction}\n\n"
        f"[PERSISTENT LONG-TERM USER PROFILE & MEMORIES]\n"
        f"The following are persistent facts you know about the user across conversations:\n"
        f"{memory_block}\n"
        f"Always tailor your responses, explanations, and tone according to these user facts and preferences."
    )


def get_gemini_client() -> genai.Client:
    """Validates and initializes the Google GenAI client."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "" or "your_gemini_api_key_here" in api_key:
        raise ValueError(
            "Gemini API key is not configured. Please set a valid GEMINI_API_KEY in your .env file."
        )
    return genai.Client(api_key=api_key.strip())


def generate_gemini_response(conversation_id: str, user_message: str) -> Tuple[str, Conversation, int]:
    """
    Orchestrates sending a message to the Gemini API with context history:
    1. Loads the conversation.
    2. Builds the sliding window context.
    3. Converts history to Google GenAI Content objects.
    4. Applies the defined system instruction.
    5. Dispatches request and saves user & assistant turns into memory.
    """
    conversation = conversation_manager.get_or_create(conversation_id)
    client = get_gemini_client()

    # Get active sliding-window context history (up to MAX_CONTEXT_MESSAGES)
    context_history = conversation.get_sliding_window_context(MAX_CONTEXT_MESSAGES)
    active_context_count = len(context_history)

    # Convert context history into google.genai types.Content objects
    contents: List[types.Content] = []
    for msg in context_history:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Append current user prompt
    new_user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)]
    )
    contents.append(new_user_content)

    # Retrieve persistent long-term memories and augment system instruction
    memories = retrieve_relevant_memories(user_message)
    effective_system_instruction = build_system_instruction_with_memories(SYSTEM_INSTRUCTION, memories)

    # Configure generation parameters including system instruction
    config = types.GenerateContentConfig(
        system_instruction=effective_system_instruction,
        temperature=0.7,
        max_output_tokens=2048,
    )

    logger.info(
        "Calling Gemini API [model=%s, conv_id=%s, context_turns=%d]",
        GEMINI_MODEL, conversation.conversation_id, active_context_count
    )

    # Execute API call
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )

    bot_reply = response.text or ""
    if not bot_reply.strip():
        # Handle cases where safety block or empty output occurs
        if response.candidates and response.candidates[0].finish_reason:
            bot_reply = f"[Response stopped: {response.candidates[0].finish_reason}]"
        else:
            bot_reply = "I apologize, but I could not generate a response at this moment."

    # Persist turns into backend conversation history
    conversation.add_message("user", user_message)
    conversation.add_message("model", bot_reply)

    return bot_reply, conversation, active_context_count


# ---------------------------------------------------------------------------
# Flask Application & Routing
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")


@app.after_request
def add_cors_headers(response):
    """Enable CORS so frontend can communicate smoothly across ports / live servers."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def serve_index():
    """Serves the chatbot web frontend."""
    return send_from_directory("frontend", "index.html")


@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    """Serves static files (CSS, JS, assets) from frontend directory."""
    return send_from_directory("frontend", path)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health status and configuration check."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    has_valid_key = bool(api_key and "your_gemini_api_key_here" not in api_key)

    return jsonify({
        "status": "healthy",
        "api_key_configured": has_valid_key,
        "model": GEMINI_MODEL,
        "max_context_messages": MAX_CONTEXT_MESSAGES,
        "active_conversations_count": len(conversation_manager.conversations),
        "memories_count": len(get_all_memories()),
        "system_instruction_configured": True
    }), 200


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat_endpoint():
    """
    Main chat endpoint.
    Expects JSON: { "message": "...", "conversation_id": "optional-uuid" }
    Returns JSON: { "status": "success", "response": "...", "conversation_id": "..." }
    """
    if request.method == "OPTIONS":
        return "", 204

    # Validate JSON content type
    if not request.is_json:
        return jsonify({
            "status": "error",
            "error": "Invalid request format. Content-Type must be application/json.",
            "code": "INVALID_JSON"
        }), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "status": "error",
            "error": "Malformed JSON payload.",
            "code": "MALFORMED_JSON"
        }), 400

    user_message = data.get("message")
    conversation_id = data.get("conversation_id")

    # Validate user message
    if not user_message or not isinstance(user_message, str) or not user_message.strip():
        return jsonify({
            "status": "error",
            "error": "Message is required and cannot be empty.",
            "code": "EMPTY_MESSAGE"
        }), 400

    # Ensure conversation ID exists or generate a fresh one
    if not conversation_id or not str(conversation_id).strip():
        conversation_id = str(uuid.uuid4())
    else:
        conversation_id = str(conversation_id).strip()

    clean_message = user_message.strip()

    # Automatically extract and persist long-term user facts
    try:
        extracted = extract_memories_from_text(clean_message)
        for k, v in extracted.items():
            save_memory(k, v)
            logger.info("Saved persistent memory: %s -> %s", k, v)
    except Exception as me:
        logger.error("Failed to extract or save memory: %s", me)

    try:
        reply_text, conversation, context_count = generate_gemini_response(
            conversation_id=conversation_id,
            user_message=clean_message
        )

        return jsonify({
            "status": "success",
            "response": reply_text,
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "active_context_messages": context_count,
            "total_messages": len(conversation.messages)
        }), 200

    except ValueError as ve:
        # Configuration error such as missing API key
        logger.warning("Configuration error: %s", ve)
        return jsonify({
            "status": "error",
            "error": str(ve),
            "code": "MISSING_API_KEY"
        }), 401

    except errors.ClientError as ce:
        # Invalid API key, model not found, bad request to Gemini
        logger.error("Gemini ClientError (%s): %s", ce.code, ce.message)
        status_code = ce.code if ce.code in (400, 401, 403, 404, 429) else 400
        return jsonify({
            "status": "error",
            "error": f"Gemini API Client Error: {ce.message}",
            "code": "GEMINI_CLIENT_ERROR"
        }), status_code

    except errors.ServerError as se:
        # Gemini internal server error or capacity issue
        logger.error("Gemini ServerError (%s): %s", se.code, se.message)
        return jsonify({
            "status": "error",
            "error": "Google Gemini servers are temporarily unavailable. Please retry in a few moments.",
            "code": "GEMINI_SERVER_ERROR"
        }), 502

    except errors.APIError as ae:
        # General Gemini API error
        logger.error("Gemini APIError (%s): %s", ae.code, ae.message)
        return jsonify({
            "status": "error",
            "error": f"Gemini API Error: {ae.message}",
            "code": "GEMINI_API_ERROR"
        }), ae.code if ae.code and 400 <= ae.code <= 599 else 500

    except TimeoutError:
        logger.error("Request to Gemini API timed out")
        return jsonify({
            "status": "error",
            "error": "The request to Gemini API timed out. Please try again.",
            "code": "REQUEST_TIMEOUT"
        }), 504

    except Exception as e:
        logger.exception("Unexpected exception in chat endpoint: %s", e)
        return jsonify({
            "status": "error",
            "error": f"An unexpected error occurred: {str(e)}",
            "code": "INTERNAL_SERVER_ERROR"
        }), 500


@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    """Lists all saved conversations from SQLite, newest first."""
    return jsonify({
        "status": "success",
        "conversations": conversation_manager.list_all()
    }), 200


@app.route("/api/conversations", methods=["POST", "OPTIONS"])
def create_conversation_endpoint():
    """Creates a new conversation in SQLite and returns it."""
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    title = data.get("title", "New Conversation")
    conversation_id = data.get("conversation_id")

    conv = conversation_manager.create_conversation(title=title, conversation_id=conversation_id)
    return jsonify({
        "status": "success",
        "conversation": conv.to_dict()
    }), 201


@app.route("/api/conversations/<conversation_id>", methods=["GET"])
def get_conversation_history(conversation_id):
    """Retrieves full conversation history and active context for a conversation."""
    conv = conversation_manager.get(conversation_id)
    if not conv:
        return jsonify({
            "status": "error",
            "error": f"Conversation '{conversation_id}' not found.",
            "code": "NOT_FOUND"
        }), 404

    return jsonify({
        "status": "success",
        "conversation": conv.to_dict(),
        "messages": conv.messages,
        "active_context": conv.get_sliding_window_context()
    }), 200


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation_endpoint(conversation_id):
    """Deletes or clears a conversation from backend memory."""
    deleted = conversation_manager.delete(conversation_id)
    if not deleted:
        return jsonify({
            "status": "error",
            "error": f"Conversation '{conversation_id}' does not exist.",
            "code": "NOT_FOUND"
        }), 404

    return jsonify({
        "status": "success",
        "message": f"Conversation '{conversation_id}' has been cleared.",
        "conversation_id": conversation_id
    }), 200


# ---------------------------------------------------------------------------
# Long-Term Memory API Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/memories", methods=["GET"])
def get_memories_endpoint():
    """Lists all saved persistent memories."""
    return jsonify({
        "status": "success",
        "memories": get_all_memories()
    }), 200


@app.route("/api/memories", methods=["POST", "OPTIONS"])
def create_or_update_memory_endpoint():
    """Explicitly saves or updates a persistent memory item."""
    if request.method == "OPTIONS":
        return "", 204

    if not request.is_json:
        return jsonify({
            "status": "error",
            "error": "Request body must be JSON.",
            "code": "INVALID_JSON"
        }), 400

    data = request.get_json(silent=True) or {}
    key = data.get("memory_key") or data.get("key")
    value = data.get("memory_value") or data.get("value")

    if not key or not isinstance(key, str) or not key.strip():
        return jsonify({
            "status": "error",
            "error": "Memory key is required and cannot be empty.",
            "code": "EMPTY_KEY"
        }), 400

    if not value or not isinstance(value, str) or not value.strip():
        return jsonify({
            "status": "error",
            "error": "Memory value is required and cannot be empty.",
            "code": "EMPTY_VALUE"
        }), 400

    mem = save_memory(key.strip(), value.strip())
    return jsonify({
        "status": "success",
        "memory": mem
    }), 201


@app.route("/api/memories/<memory_key>", methods=["DELETE"])
def delete_memory_endpoint(memory_key):
    """Deletes a specific persistent memory item."""
    deleted = delete_memory(memory_key)
    if not deleted:
        return jsonify({
            "status": "error",
            "error": f"Memory '{memory_key}' not found.",
            "code": "NOT_FOUND"
        }), 404

    return jsonify({
        "status": "success",
        "message": f"Memory '{memory_key}' deleted.",
        "memory_key": memory_key
    }), 200


@app.route("/api/memories", methods=["DELETE"])
def clear_all_memories_endpoint():
    """Wipes all persistent memories."""
    clear_all_memories()
    return jsonify({
        "status": "success",
        "message": "All memories cleared."
    }), 200


def open_chrome_when_ready(target_url: str):
    """
    Waits until the Flask application responds on /api/health,
    then automatically opens Google Chrome (falling back to default browser).
    """
    import urllib.request
    import webbrowser

    health_url = f"{target_url}/api/health"
    # Poll for server readiness (up to 6 seconds)
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.2)

    # Candidate Google Chrome executable paths
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]

    opened = False
    for path in chrome_paths:
        if os.path.isfile(path):
            try:
                webbrowser.register("google_chrome", None, webbrowser.BackgroundBrowser(path))
                webbrowser.get("google_chrome").open(target_url)
                opened = True
                break
            except Exception:
                pass

    if not opened:
        try:
            webbrowser.open(target_url)
        except Exception as e:
            logger.warning("Could not automatically launch browser: %s", e)


if __name__ == "__main__":
    import threading

    target_url = f"http://127.0.0.1:{FLASK_PORT}"

    # Only launch the browser once (in the primary process, preventing multiple tabs on debug reload)
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true" and os.environ.get("VALENTIUS_BROWSER_OPENED") != "1":
        os.environ["VALENTIUS_BROWSER_OPENED"] = "1"
        threading.Thread(target=open_chrome_when_ready, args=(target_url,), daemon=True).start()

    key_status = "Configured" if GEMINI_API_KEY and "your_gemini" not in GEMINI_API_KEY else "Missing/Placeholder in .env"

    # Avoid duplicate startup banners during debug reloads
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not FLASK_DEBUG:
        print("\n" + "=" * 65)
        print("  VALENTIUS AI - SERVER READY")
        print("=" * 65)
        print(f"  * Web Interface: {target_url}")
        print(f"  * AI Model: {GEMINI_MODEL}")
        print(f"  * Context Memory: Sliding window ({MAX_CONTEXT_MESSAGES} messages)")
        print(f"  * API Key: {key_status}")
        print("=" * 65 + "\n")

    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)
