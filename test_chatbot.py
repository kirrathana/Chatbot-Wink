"""
Comprehensive Test Suite for Valentius AI Chatbot
Covers all requirements from TEST 1 to TEST 8:
- Basic communication
- Multi-turn memory
- Earlier context preservation
- Sliding-window context management (10 messages max)
- Conversation isolation
- New chat initialization
- Error handling (empty messages, invalid JSON, missing API keys)
- Conversation persistence and cleanup
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import uuid

# Import backend components
from chatbot import (
    app,
    conversation_manager,
    Conversation,
    SYSTEM_INSTRUCTION,
    MAX_CONTEXT_MESSAGES,
    save_memory,
    get_memory,
    get_all_memories,
    update_memory,
    delete_memory,
    clear_all_memories,
    extract_memories_from_text,
    build_system_instruction_with_memories
)
from google.genai import types


class TestValentiusAIChatbot(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        # Reset conversation manager state for clean test isolation
        conversation_manager.conversations.clear()
        # Reset long-term memory state for clean test isolation
        clear_all_memories()

    # -------------------------------------------------------------------------
    # TEST 1: Health & Basic Communication Structure
    # -------------------------------------------------------------------------
    def test_01_health_endpoint(self):
        """Verify health check endpoint returns 200 and proper configuration."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["max_context_messages"], 10)
        self.assertTrue(data["system_instruction_configured"])

    @patch("chatbot.get_gemini_client")
    def test_01_basic_communication(self, mock_get_client):
        """TEST 1: Send a message and verify Gemini responds properly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello! I am Valentius AI, ready to assist you."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        conv_id = str(uuid.uuid4())
        payload = {"message": "Hello Valentius!", "conversation_id": conv_id}

        res = self.client.post("/api/chat", json=payload)
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["response"], "Hello! I am Valentius AI, ready to assist you.")
        self.assertEqual(data["conversation_id"], conv_id)
        self.assertEqual(data["total_messages"], 2)  # 1 user + 1 model

        # Verify system instruction and model were passed
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertIn("config", call_kwargs)
        self.assertEqual(call_kwargs["config"].system_instruction, SYSTEM_INSTRUCTION)

    # -------------------------------------------------------------------------
    # TEST 2: Multi-Turn Memory (Rahul Name Test)
    # -------------------------------------------------------------------------
    @patch("chatbot.get_gemini_client")
    def test_02_multi_turn_memory(self, mock_get_client):
        """
        TEST 2: Multi-turn memory
        User: My name is Rahul.
        User: What is my name?
        Expected: Earlier context with Rahul is passed into the Gemini API request.
        """
        mock_client = MagicMock()
        mock_response_1 = MagicMock()
        mock_response_1.text = "Nice to meet you, Rahul! How can I help you today?"
        mock_response_2 = MagicMock()
        mock_response_2.text = "Your name is Rahul."

        mock_client.models.generate_content.side_effect = [mock_response_1, mock_response_2]
        mock_get_client.return_value = mock_client

        conv_id = "test-rahul-conv"

        # Turn 1
        res1 = self.client.post("/api/chat", json={
            "message": "My name is Rahul.",
            "conversation_id": conv_id
        })
        self.assertEqual(res1.status_code, 200)

        # Turn 2
        res2 = self.client.post("/api/chat", json={
            "message": "What is my name?",
            "conversation_id": conv_id
        })
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()["response"], "Your name is Rahul.")

        # Check call args of second call to verify contents passed to Gemini
        second_call_contents = mock_client.models.generate_content.call_args_list[1][1]["contents"]
        
        # Turn 2 must include:
        # 1. user: My name is Rahul.
        # 2. model: Nice to meet you, Rahul!
        # 3. user: What is my name?
        self.assertEqual(len(second_call_contents), 3)
        self.assertEqual(second_call_contents[0].parts[0].text, "My name is Rahul.")
        self.assertEqual(second_call_contents[1].parts[0].text, "Nice to meet you, Rahul! How can I help you today?")
        self.assertEqual(second_call_contents[2].parts[0].text, "What is my name?")

    # -------------------------------------------------------------------------
    # TEST 3: Earlier Context (Python and Flask)
    # -------------------------------------------------------------------------
    @patch("chatbot.get_gemini_client")
    def test_03_earlier_context_technologies(self, mock_get_client):
        """
        TEST 3: Earlier context
        User: I am learning Python.
        User: I am also learning Flask.
        User: Which two technologies am I learning?
        Expected: All earlier turns are properly passed in context.
        """
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "You are learning Python and Flask."
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = mock_client

        conv_id = "test-tech-conv"

        # Turn 1
        self.client.post("/api/chat", json={"message": "I am learning Python.", "conversation_id": conv_id})
        # Turn 2
        self.client.post("/api/chat", json={"message": "I am also learning Flask.", "conversation_id": conv_id})
        # Turn 3
        res3 = self.client.post("/api/chat", json={
            "message": "Which two technologies am I learning?",
            "conversation_id": conv_id
        })
        self.assertEqual(res3.status_code, 200)

        # Inspect third call contents
        contents = mock_client.models.generate_content.call_args_list[2][1]["contents"]
        texts = [c.parts[0].text for c in contents]
        
        # Must contain Python and Flask context
        self.assertIn("I am learning Python.", texts)
        self.assertIn("I am also learning Flask.", texts)
        self.assertEqual(texts[-1], "Which two technologies am I learning?")

    # -------------------------------------------------------------------------
    # TEST 4: Sliding Window Context Management (Max 10 Messages)
    # -------------------------------------------------------------------------
    def test_04_sliding_window_context_management(self):
        """
        TEST 4: Context window management
        Send more than 10 messages and verify that only the latest 10 messages
        are included in the active context window, while older messages are pruned.
        """
        conv = Conversation("test-sliding-window")

        # Simulate 16 messages (8 turns: user 1..8, model 1..8)
        for i in range(1, 9):
            conv.add_message("user", f"User question {i}")
            conv.add_message("model", f"Model answer {i}")

        self.assertEqual(len(conv.messages), 16)

        # Retrieve active sliding window context
        active_context = conv.get_sliding_window_context(max_messages=10)

        # Context window must not exceed 10 messages
        self.assertEqual(len(active_context), 10)

        # Window must start with role 'user'
        self.assertEqual(active_context[0]["role"], "user")

        # Oldest message in window should be User question 4 (turns 4, 5, 6, 7, 8 = 10 messages)
        self.assertEqual(active_context[0]["content"], "User question 4")
        self.assertEqual(active_context[-1]["content"], "Model answer 8")

        # User question 1, 2, 3 must be excluded from active context
        active_texts = [m["content"] for m in active_context]
        self.assertNotIn("User question 1", active_texts)
        self.assertNotIn("User question 2", active_texts)
        self.assertNotIn("User question 3", active_texts)

    # -------------------------------------------------------------------------
    # TEST 5: Conversation Isolation
    # -------------------------------------------------------------------------
    @patch("chatbot.get_gemini_client")
    def test_05_conversation_isolation(self, mock_get_client):
        """
        TEST 5: Conversation isolation
        Create two conversations with different names and verify their histories do not mix.
        """
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        conv_alice = "conv-alice-123"
        conv_bob = "conv-bob-456"

        # Conversation 1: Alice
        mock_client.models.generate_content.return_value.text = "Hello Alice!"
        self.client.post("/api/chat", json={"message": "My name is Alice.", "conversation_id": conv_alice})

        # Conversation 2: Bob
        mock_client.models.generate_content.return_value.text = "Hello Bob!"
        self.client.post("/api/chat", json={"message": "My name is Bob.", "conversation_id": conv_bob})

        # Fetch histories from conversation manager
        alice_conv = conversation_manager.get(conv_alice)
        bob_conv = conversation_manager.get(conv_bob)

        self.assertIsNotNone(alice_conv)
        self.assertIsNotNone(bob_conv)

        alice_texts = [m["content"] for m in alice_conv.messages]
        bob_texts = [m["content"] for m in bob_conv.messages]

        self.assertIn("My name is Alice.", alice_texts)
        self.assertNotIn("My name is Bob.", alice_texts)

        self.assertIn("My name is Bob.", bob_texts)
        self.assertNotIn("My name is Alice.", bob_texts)

    # -------------------------------------------------------------------------
    # TEST 6: New Chat Empty History
    # -------------------------------------------------------------------------
    def test_06_new_chat_starts_empty(self):
        """TEST 6: Create a new conversation and verify it starts with an empty history."""
        new_id = "new-fresh-chat-" + str(uuid.uuid4())
        conv = conversation_manager.get_or_create(new_id)

        self.assertEqual(len(conv.messages), 0)
        self.assertEqual(conv.get_sliding_window_context(), [])
        self.assertEqual(conv.title, "New Conversation")

    # -------------------------------------------------------------------------
    # TEST 7: Error Handling
    # -------------------------------------------------------------------------
    def test_07_empty_message_error(self):
        """Verify empty message returns 400 Bad Request."""
        res = self.client.post("/api/chat", json={"message": "   ", "conversation_id": "test"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["code"], "EMPTY_MESSAGE")

    def test_07_invalid_json_error(self):
        """Verify non-JSON request returns 400 Bad Request."""
        res = self.client.post("/api/chat", data="not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data["code"], "INVALID_JSON")

    @patch("chatbot.os.getenv")
    def test_07_missing_api_key_error(self, mock_getenv):
        """Verify missing API key returns 401 Unauthorized with clear guidance."""
        mock_getenv.return_value = ""  # Simulate unset API key

        res = self.client.post("/api/chat", json={"message": "Hello", "conversation_id": "conv-test"})
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertEqual(data["code"], "MISSING_API_KEY")
        self.assertIn("GEMINI_API_KEY", data["error"])

    # -------------------------------------------------------------------------
    # TEST 8: Conversations List & Delete
    # -------------------------------------------------------------------------
    def test_08_conversation_list_and_delete(self):
        """Verify conversations listing and deletion endpoint functionality."""
        conv_id = "conv-delete-test"
        conv = conversation_manager.get_or_create(conv_id)
        conv.add_message("user", "Hello to delete")

        # Check list endpoint
        list_res = self.client.get("/api/conversations")
        self.assertEqual(list_res.status_code, 200)
        conv_ids = [c["conversation_id"] for c in list_res.get_json()["conversations"]]
        self.assertIn(conv_id, conv_ids)

        # Delete conversation
        del_res = self.client.delete(f"/api/conversations/{conv_id}")
        self.assertEqual(del_res.status_code, 200)

        # Verify it no longer exists in manager
        self.assertIsNone(conversation_manager.get(conv_id))

        # Second delete should return 404
        del_again = self.client.delete(f"/api/conversations/{conv_id}")
        self.assertEqual(del_again.status_code, 404)

    # -------------------------------------------------------------------------
    # TEST 9: SQLite Database Persistence & Explicit Endpoints
    # -------------------------------------------------------------------------
    def test_09_sqlite_persistence_and_endpoints(self):
        """Verify POST /api/conversations, GET /api/conversations/<id>, and SQLite persistence."""
        # 1. Create conversation via POST /api/conversations
        create_res = self.client.post("/api/conversations", json={"title": "Test AI Chat"})
        self.assertEqual(create_res.status_code, 201)
        data = create_res.get_json()
        self.assertEqual(data["status"], "success")
        conv_id = data["conversation"]["conversation_id"]
        self.assertEqual(data["conversation"]["title"], "Test AI Chat")

        # 2. Add messages to this conversation
        conv = conversation_manager.get(conv_id)
        self.assertIsNotNone(conv)
        conv.add_message("user", "Hello SQLite!")
        conv.add_message("model", "Hello! Messages are stored in SQLite.")

        # 3. Retrieve conversation via GET /api/conversations/<conversation_id>
        get_res = self.client.get(f"/api/conversations/{conv_id}")
        self.assertEqual(get_res.status_code, 200)
        get_data = get_res.get_json()
        self.assertEqual(len(get_data["messages"]), 2)
        self.assertEqual(get_data["messages"][0]["content"], "Hello SQLite!")
        self.assertEqual(get_data["messages"][1]["content"], "Hello! Messages are stored in SQLite.")

        # 4. Verify SQLite persistence by re-instantiating conversation directly from DB
        fresh_conv = Conversation(conv_id)
        self.assertEqual(len(fresh_conv.messages), 2)
        self.assertEqual(fresh_conv.messages[0]["content"], "Hello SQLite!")

        # 5. Clean up via DELETE /api/conversations/<conversation_id>
        del_res = self.client.delete(f"/api/conversations/{conv_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertIsNone(conversation_manager.get(conv_id))

    # -------------------------------------------------------------------------
    # TEST 10: Persistent Memory Save and Retrieval
    # -------------------------------------------------------------------------
    def test_10_save_and_retrieve_memory(self):
        """Verify saving and retrieving persistent user memories in SQLite."""
        m1 = save_memory("user_name", "Kirthu")
        self.assertEqual(m1["memory_key"], "user_name")
        self.assertEqual(m1["memory_value"], "Kirthu")

        m2 = save_memory("department", "AML")
        self.assertEqual(m2["memory_key"], "department")
        self.assertEqual(m2["memory_value"], "AML")

        # Retrieve specific memory
        fetched_name = get_memory("user_name")
        self.assertIsNotNone(fetched_name)
        self.assertEqual(fetched_name["memory_value"], "Kirthu")

        # Retrieve all memories
        all_memories = get_all_memories()
        self.assertEqual(len(all_memories), 2)
        keys = [m["memory_key"] for m in all_memories]
        self.assertIn("user_name", keys)
        self.assertIn("department", keys)

    # -------------------------------------------------------------------------
    # TEST 11: Memory Update and Deletion
    # -------------------------------------------------------------------------
    def test_11_update_and_delete_memory(self):
        """Verify updating existing memory values and deleting memories by key."""
        save_memory("coding_experience", "Beginner")
        self.assertEqual(get_memory("coding_experience")["memory_value"], "Beginner")

        # Update memory
        updated = update_memory("coding_experience", "Intermediate")
        self.assertTrue(updated)
        self.assertEqual(get_memory("coding_experience")["memory_value"], "Intermediate")

        # Delete memory
        deleted = delete_memory("coding_experience")
        self.assertTrue(deleted)
        self.assertIsNone(get_memory("coding_experience"))

        # Deleting non-existent key returns False
        self.assertFalse(delete_memory("non_existent_key"))

    # -------------------------------------------------------------------------
    # TEST 12: Automatic Memory Extraction
    # -------------------------------------------------------------------------
    def test_12_automatic_memory_extraction(self):
        """Verify extraction logic identifies user name, department, coding skills, and preferences."""
        # 1. Name & Department
        facts1 = extract_memories_from_text("My name is Kirthu. I am studying in the AML department.")
        self.assertEqual(facts1.get("user_name"), "Kirthu")
        self.assertEqual(facts1.get("department"), "AML")

        # 2. Coding experience and explanation preference
        facts2 = extract_memories_from_text("I don't have coding experience. I prefer simple explanations.")
        self.assertIn("coding_experience", facts2)
        self.assertEqual(facts2["coding_experience"], "No prior coding experience (Beginner)")
        self.assertIn("simple explanations", facts2.get("explanation_preference", ""))

        # 3. Alternative name phrase
        facts3 = extract_memories_from_text("Call me Alex.")
        self.assertEqual(facts3.get("user_name"), "Alex")

    # -------------------------------------------------------------------------
    # TEST 13: Memory Injection into System Instruction
    # -------------------------------------------------------------------------
    @patch("chatbot.get_gemini_client")
    def test_13_memory_injection_in_system_instruction(self, mock_get_client):
        """Verify persistent memories are injected into Gemini system instructions."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Hello Kirthu! How can I assist you with your AML studies?"
        mock_client.models.generate_content.return_value = mock_resp
        mock_get_client.return_value = mock_client

        # Seed persistent memories
        save_memory("user_name", "Kirthu")
        save_memory("department", "AML")

        conv_id = "test-memory-injection"
        res = self.client.post("/api/chat", json={
            "message": "Hello!",
            "conversation_id": conv_id
        })
        self.assertEqual(res.status_code, 200)

        # Inspect system instruction passed to Gemini
        call_kwargs = mock_client.models.generate_content.call_args[1]
        sys_inst = call_kwargs["config"].system_instruction
        self.assertIn("[PERSISTENT LONG-TERM USER PROFILE & MEMORIES]", sys_inst)
        self.assertIn("- user_name: Kirthu", sys_inst)
        self.assertIn("- department: AML", sys_inst)

    # -------------------------------------------------------------------------
    # TEST 14: Cross-Conversation Memory Retention
    # -------------------------------------------------------------------------
    @patch("chatbot.get_gemini_client")
    def test_14_cross_conversation_memory_retention(self, mock_get_client):
        """
        Verify that memories learned in Conversation 1 persist and are supplied
        in a completely fresh Conversation 2 (New Chat), while conversation
        message histories remain properly isolated.
        """
        mock_client = MagicMock()
        mock_resp_1 = MagicMock()
        mock_resp_1.text = "Nice to meet you, Kirthu! Glad to know you study AML."
        mock_resp_2 = MagicMock()
        mock_resp_2.text = "You are Kirthu, studying in the AML department."
        mock_client.models.generate_content.side_effect = [mock_resp_1, mock_resp_2]
        mock_get_client.return_value = mock_client

        conv_1 = "conversation-alpha"
        conv_2 = "conversation-beta-new-chat"

        # Turn 1: Conversation 1 introduces user
        res1 = self.client.post("/api/chat", json={
            "message": "My name is Kirthu. I am studying in the AML department.",
            "conversation_id": conv_1
        })
        self.assertEqual(res1.status_code, 200)

        # Verify memories were persisted to SQLite
        self.assertEqual(get_memory("user_name")["memory_value"], "Kirthu")
        self.assertEqual(get_memory("department")["memory_value"], "AML")

        # Turn 2: Conversation 2 (New Chat) asks who the user is
        res2 = self.client.post("/api/chat", json={
            "message": "What is my name and department?",
            "conversation_id": conv_2
        })
        self.assertEqual(res2.status_code, 200)

        # Check call arguments for Conversation 2
        conv2_call_kwargs = mock_client.models.generate_content.call_args_list[1][1]
        conv2_sys_inst = conv2_call_kwargs["config"].system_instruction

        # Conversation 2 received persistent user memory in its system instruction
        self.assertIn("- user_name: Kirthu", conv2_sys_inst)
        self.assertIn("- department: AML", conv2_sys_inst)

        # Conversation 2 contents only contain Conversation 2's user prompt (isolation preserved)
        conv2_contents = conv2_call_kwargs["contents"]
        self.assertEqual(len(conv2_contents), 1)
        self.assertEqual(conv2_contents[0].parts[0].text, "What is my name and department?")

    # -------------------------------------------------------------------------
    # TEST 15: Memory REST API Endpoints
    # -------------------------------------------------------------------------
    def test_15_memory_rest_api_endpoints(self):
        """Verify GET /api/memories, POST /api/memories, DELETE endpoints."""
        # 1. Create memory via POST
        res_post = self.client.post("/api/memories", json={
            "key": "user_name",
            "value": "Alex"
        })
        self.assertEqual(res_post.status_code, 201)
        data_post = res_post.get_json()
        self.assertEqual(data_post["status"], "success")
        self.assertEqual(data_post["memory"]["memory_key"], "user_name")
        self.assertEqual(data_post["memory"]["memory_value"], "Alex")

        # 2. List memories via GET
        res_get = self.client.get("/api/memories")
        self.assertEqual(res_get.status_code, 200)
        memories = res_get.get_json()["memories"]
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["memory_key"], "user_name")

        # 3. Delete specific memory via DELETE
        res_del = self.client.delete("/api/memories/user_name")
        self.assertEqual(res_del.status_code, 200)

        # Verify it's gone
        res_get2 = self.client.get("/api/memories")
        self.assertEqual(len(res_get2.get_json()["memories"]), 0)

        # 4. Deleting non-existent memory returns 404
        res_del_404 = self.client.delete("/api/memories/not_found_key")
        self.assertEqual(res_del_404.status_code, 404)

        # 5. Clear all memories via DELETE /api/memories
        save_memory("key1", "val1")
        save_memory("key2", "val2")
        self.assertEqual(len(get_all_memories()), 2)

        res_clear = self.client.delete("/api/memories")
        self.assertEqual(res_clear.status_code, 200)
        self.assertEqual(len(get_all_memories()), 0)

    # -------------------------------------------------------------------------
    # TEST 16: Ignore Questions and Sensitive Data
    # -------------------------------------------------------------------------
    def test_16_ignore_questions_and_sensitive_data(self):
        """Verify extraction ignores interrogative questions, casual chat, and sensitive tokens."""
        # Pure questions should not extract bogus facts
        self.assertEqual(extract_memories_from_text("What is my name?"), {})
        self.assertEqual(extract_memories_from_text("Who am I?"), {})
        self.assertEqual(extract_memories_from_text("Do I have any coding experience?"), {})
        self.assertEqual(extract_memories_from_text("Can you explain recursion?"), {})

        # Casual chat should not extract memories
        self.assertEqual(extract_memories_from_text("Hello Valentius!"), {})
        self.assertEqual(extract_memories_from_text("How are you doing today?"), {})

        # Sensitive credentials must be ignored
        sensitive_msg = "My password is supersecret123 and my api_key is secret-token-xyz"
        self.assertEqual(extract_memories_from_text(sensitive_msg), {})

    # -------------------------------------------------------------------------
    # TEST 17: Automatic Preference Extraction
    # -------------------------------------------------------------------------
    @patch("chatbot.get_gemini_client")
    def test_17_auto_preference_extraction(self, mock_get_client):
        """Verify explicit preferences like 'I prefer Python for backend development' are automatically saved."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "Python is fantastic for backend development!"
        mock_get_client.return_value = mock_client

        conv_id = "test-pref-conv"
        res = self.client.post("/api/chat", json={
            "message": "I prefer Python for backend development.",
            "conversation_id": conv_id
        })
        self.assertEqual(res.status_code, 200)

        # Verify preference was automatically saved in SQLite
        pref_mem = get_memory("preference")
        self.assertIsNotNone(pref_mem)
        self.assertEqual(pref_mem["memory_value"], "Python for backend development")

    # -------------------------------------------------------------------------
    # TEST 18: Memory Correction and Deduplication
    # -------------------------------------------------------------------------
    @patch("chatbot.get_gemini_client")
    def test_18_memory_correction_and_deduplication(self, mock_get_client):
        """Verify that corrections update existing memories without creating duplicates."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "Understood!"
        mock_get_client.return_value = mock_client

        conv_id = "test-correction-conv"

        # Turn 1: Initial name
        self.client.post("/api/chat", json={
            "message": "My name is Kirthu.",
            "conversation_id": conv_id
        })
        self.assertEqual(get_memory("user_name")["memory_value"], "Kirthu")
        self.assertEqual(len(get_all_memories()), 1)

        # Turn 2: User corrects name
        self.client.post("/api/chat", json={
            "message": "Actually, my name is Alex.",
            "conversation_id": conv_id
        })
        # Value must be updated to Alex
        self.assertEqual(get_memory("user_name")["memory_value"], "Alex")
        # Record count must remain exactly 1 (no duplicate rows)
        self.assertEqual(len(get_all_memories()), 1)

    # -------------------------------------------------------------------------
    # TEST 19: New Chat & Conversation Deletion Preserves Memory
    # -------------------------------------------------------------------------
    @patch("chatbot.get_gemini_client")
    def test_19_new_chat_and_deletion_preserves_memory(self, mock_get_client):
        """Verify creating New Chat or deleting a conversation never deletes long-term memories."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "Got it!"
        mock_get_client.return_value = mock_client

        conv_1 = "conv-to-delete-123"

        # Learn memory in Conversation 1
        self.client.post("/api/chat", json={
            "message": "My name is Kirthu.",
            "conversation_id": conv_1
        })
        self.assertEqual(get_memory("user_name")["memory_value"], "Kirthu")

        # 1. Create a fresh conversation (New Chat)
        create_res = self.client.post("/api/conversations", json={"title": "New Chat 2"})
        self.assertEqual(create_res.status_code, 201)
        conv_2 = create_res.get_json()["conversation"]["conversation_id"]

        # Memory is still intact
        self.assertEqual(get_memory("user_name")["memory_value"], "Kirthu")

        # 2. Delete Conversation 1
        del_res = self.client.delete(f"/api/conversations/{conv_1}")
        self.assertEqual(del_res.status_code, 200)

        # Verify Conversation 1 is gone
        self.assertIsNone(conversation_manager.get(conv_1))

        # Long-term memory MUST still be intact!
        self.assertIsNotNone(get_memory("user_name"))
        self.assertEqual(get_memory("user_name")["memory_value"], "Kirthu")

    # -------------------------------------------------------------------------
    # TEST 20: Memory Persistence Across Database Reconnection (Restart)
    # -------------------------------------------------------------------------
    def test_20_memory_persistence_across_db_reconnect(self):
        """Verify memories survive database connection teardown (simulating application restart)."""
        save_memory("favorite_framework", "Flask")
        save_memory("preferred_theme", "Dark Gold")

        # Open a completely fresh SQLite connection to simulate server restart
        import sqlite3
        from chatbot import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT memory_key, memory_value FROM memories ORDER BY id ASC").fetchall()
        saved = {r["memory_key"]: r["memory_value"] for r in rows}
        conn.close()

        self.assertIn("favorite_framework", saved)
        self.assertEqual(saved["favorite_framework"], "Flask")
        self.assertIn("preferred_theme", saved)
        self.assertEqual(saved["preferred_theme"], "Dark Gold")


if __name__ == "__main__":
    unittest.main()

