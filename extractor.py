import re
import json
import httpx
from typing import List, Dict, Any, Optional
from config import Config

class MemoryExtractor:
    SMALL_TALK_PATTERNS = [
        r"^(ok|okay|thanks|thank you|cool|great|nice|hello|hi|hey|good morning|good evening|bye|see ya|awesome|got it|sure|no problem)[\!\.\?]*$",
        r"^(how are you|what's up|how's it going|sounds good)[\!\.\?]*$"
    ]

    ENTITY_RULES = [
        (r"(?:use|using|run|running|installed|on)\s+([a-zA-Z0-9\s\.\+]+(?:linux|ubuntu|debian|fedora|arch|macos|windows|mac|ios|android))", "operating_system", "fact", "User uses {match}."),
        (r"(?:building|working on|developing|created|making)\s+a?\s*([a-zA-Z0-9\.\s\-\_]+?(?:app|project|website|service|backend|frontend|system|api|tool))", "project_name", "project", "User is building a {match}."),
        (r"(?:deadline|due date|due)\s+(?:is|moved to|postponed to|changed to)\s+([a-zA-Z0-9\s,]+)", "project_deadline", "project", "Project deadline is {match}."),
        (r"(?:prefer|favourite|favorite|like|love|use)\s+([a-zA-Z0-9\s\.\+]+?)\s+(?:database|db|postgres|postgresql|mongodb|mysql|sqlite|redis)", "database_preference", "preference", "User prefers {match}."),
        (r"(?:prefer|favourite|favorite|like|love)\s+([a-zA-Z0-9\s]+?)\s+(?:food|cuisine|dish|pizza|pasta|sushi|burger)", "favorite_food", "preference", "User's favorite food is {match}."),
        (r"(?:live in|located in|based in|from)\s+([a-zA-Z\s,]+)", "user_location", "fact", "User is located in {match}."),
        (r"(?:learning|studying|mastering)\s+([a-zA-Z0-9\s\+\#]+)", "learning_goal", "goal", "User is learning {match}.")
    ]

    def is_small_talk(self, message: str) -> bool:
        msg = message.strip().lower()
        return any(re.search(pat, msg) for pat in self.SMALL_TALK_PATTERNS)

    def extract(self, message: str) -> List[Dict[str, Any]]:
        clean_msg = message.strip()
        if self.is_small_talk(clean_msg):
            return []

        extracted = []
        msg_lower = clean_msg.lower()
        matched = False

        for pattern, topic_key, mem_type, template in self.ENTITY_RULES:
            match = re.search(pattern, msg_lower)
            if match:
                val = match.group(1).strip().title()
                extracted.append({
                    "text": template.format(match=val),
                    "type": mem_type,
                    "topic_key": topic_key,
                    "importance_score": 4.0 if mem_type in ["project", "goal"] else 3.0,
                    "source_message": clean_msg
                })
                matched = True

        if not matched and re.search(r"\b(i|my|we|our|me)\b", msg_lower):
            cleaned_text = re.sub(r"^(actually|by the way|so|anyway|btw|in fact)[,\s]*", "", clean_msg, flags=re.IGNORECASE)
            words = set(re.findall(r'\w+', cleaned_text.lower()))
            topic_key = "general"
            for kw in ["deadline", "database", "os", "app", "food", "city", "framework", "language"]:
                if kw in words:
                    topic_key = kw
                    break

            extracted.append({
                "text": cleaned_text if cleaned_text.endswith(".") else f"{cleaned_text}.",
                "type": "fact",
                "topic_key": topic_key,
                "importance_score": 3.0,
                "source_message": clean_msg
            })

        return extracted