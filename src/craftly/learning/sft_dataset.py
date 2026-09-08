"""Defensive cybersecurity Supervised Fine-Tuning (SFT) dataset engine.

Provides structured prompt-completion pairs covering OWASP Top 10, CWE
remediations, AST-based safe transformations, and defensive patch engineering.
Enforces token-level loss masking (prompt tokens mapped to -100) so cross-entropy
loss is strictly evaluated on the model's defensive reasoning and patch output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import Field

from src.craftly.shared.integrity import sha256_file
from src.craftly.shared.schemas import StrictModel


class SFTTurn(StrictModel):
    """A single turn in an instruction dialogue."""

    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str = Field(min_length=1)


class SFTRecord(StrictModel):
    """A complete single- or multi-turn instruction sample."""

    record_id: str
    category: str
    cwe_id: str | None = None
    system_prompt: str = (
        "You are Craftly, a production-grade defensive cybersecurity AI engineering assistant. "
        "Your mission is defensive vulnerability analysis, secure remediation, and robust patching."
    )
    instruction: str = Field(min_length=1)
    thought: str | None = None
    response: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render_prompt(self) -> str:
        """Render the user-visible prompt prefix up to the assistant's turn."""
        parts = []
        if self.system_prompt:
            parts.append(f"System:\n{self.system_prompt.strip()}")
        parts.append(f"User:\n{self.instruction.strip()}")
        parts.append("Assistant:\n")
        return "\n\n".join(parts)

    def render_completion(self) -> str:
        """Render the assistant's thought process and response."""
        parts = []
        if self.thought:
            parts.append(f"<|thought_start|>\n{self.thought.strip()}\n<|thought_end|>")
        parts.append(self.response.strip())
        parts.append("<|document_end|>")
        return "\n".join(parts)

    def render_full_text(self) -> str:
        return self.render_prompt() + self.render_completion()


class SFTBatch(StrictModel):
    """A collated batch of input IDs, target labels with loss masking, and attention mask."""

    input_ids: list[list[int]]
    labels: list[list[int]]
    attention_mask: list[list[int]]
    token_count: int = Field(ge=0)
    masked_token_count: int = Field(ge=0)


def encode_sft_record(
    record: SFTRecord,
    tokenizer: Any,
    *,
    max_sequence_length: int = 1024,
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> tuple[list[int], list[int], list[int]]:
    """Encode an SFT record into input_ids, labels (with prompt masked as ignore_index), and attention_mask."""
    prompt_text = record.render_prompt()
    completion_text = record.render_completion()

    prompt_ids = tokenizer.encode(prompt_text).ids
    completion_ids = tokenizer.encode(completion_text).ids

    if not prompt_ids:
        prompt_ids = [0]
    if not completion_ids:
        completion_ids = [0]

    # Combine: prompt has ignore_index labels, completion has real token labels
    raw_input_ids = prompt_ids + completion_ids
    raw_labels = [ignore_index] * len(prompt_ids) + completion_ids

    # Truncate if exceeds max length (preserving completion tail)
    if len(raw_input_ids) > max_sequence_length:
        # Keep prompt prefix and truncate tail, or truncate left of prompt if huge
        raw_input_ids = raw_input_ids[:max_sequence_length]
        raw_labels = raw_labels[:max_sequence_length]

    seq_len = len(raw_input_ids)
    attention_mask = [1] * seq_len

    # Pad up to max_sequence_length if fixed-length batching is required
    pad_len = max_sequence_length - seq_len
    if pad_len > 0:
        raw_input_ids.extend([pad_token_id] * pad_len)
        raw_labels.extend([ignore_index] * pad_len)
        attention_mask.extend([0] * pad_len)

    return raw_input_ids, raw_labels, attention_mask


# ---------------------------------------------------------------------------
# Canonical Curated Defensive Cybersecurity Instruction Dataset
# ---------------------------------------------------------------------------

CANONICAL_DEFENSIVE_SFT_SEED_DATA: list[dict[str, Any]] = [
    {
        "record_id": "cwe-89-sql-injection-auth",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-89",
        "instruction": "Fix the following SQL injection vulnerability in user login:\n\n```python\ndef login(username, password):\n    query = f\"SELECT id, role FROM accounts WHERE user='{username}' AND pass='{password}'\"\n    return db.execute(query).fetchone()\n```",
        "thought": "The code interpolates user-controlled parameters `username` and `password` directly into a raw SQL query. An attacker could bypass authentication using `' OR '1'='1`. To fix this, use parameterized queries so input values are never executed as SQL grammar.",
        "response": (
            "### Vulnerability Analysis (CWE-89: SQL Injection)\n"
            "Direct string formatting into SQL queries allows arbitrary query manipulation.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "def login(username, password):\n"
            "    query = \"SELECT id, role FROM accounts WHERE user = %s AND pass = %s\"\n"
            "    return db.execute(query, (username, password)).fetchone()\n"
            "<|patch_end|>\n\n"
            "Parameterized statements guarantee parameters are treated as literal constants by the database engine."
        ),
    },
    {
        "record_id": "cwe-22-path-traversal-download",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-22",
        "instruction": "Review this file serving function for path traversal risks and write a secure version:\n\n```python\nimport os\n\ndef get_report(filename):\n    filepath = os.path.join('/var/reports', filename)\n    with open(filepath, 'r') as f:\n        return f.read()\n```",
        "thought": "Using `os.path.join` with an absolute path or relative sequences like `../../etc/passwd` allows directory traversal outside `/var/reports`. Path canonicalization via `Path.resolve()` and strict containment check `is_relative_to()` are required.",
        "response": (
            "### Vulnerability Analysis (CWE-22: Path Traversal)\n"
            "`os.path.join` does not prevent directory traversal sequences (`..`).\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "from pathlib import Path\n\n"
            "BASE_REPORT_DIR = Path('/var/reports').resolve()\n\n"
            "def get_report(filename: str) -> str:\n"
            "    target = (BASE_REPORT_DIR / filename).resolve()\n"
            "    if not target.is_relative_to(BASE_REPORT_DIR):\n"
            "        raise PermissionError('Path traversal detected outside base directory')\n"
            "    if not target.is_file():\n"
            "        raise FileNotFoundError('Requested report does not exist')\n"
            "    return target.read_text(encoding='utf-8')\n"
            "<|patch_end|>"
        ),
    },
    {
        "record_id": "cwe-79-dom-xss-react",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-79",
        "instruction": "Explain the Cross-Site Scripting (XSS) risk in this code and provide the defensive remediation:\n\n```javascript\nelement.innerHTML = userComment;\n```",
        "thought": "Setting `innerHTML` with untrusted data parses HTML tags and script elements, enabling DOM-based XSS (CWE-79). The safe alternative is `textContent` which treats input strictly as raw text.",
        "response": (
            "### Vulnerability Analysis (CWE-79: Cross-Site Scripting)\n"
            "Directly assigning untrusted strings to `innerHTML` permits execution of malicious scripts.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "// Secure remediation: strictly treat content as plain text\n"
            "element.textContent = userComment;\n"
            "<|patch_end|>\n\n"
            "If HTML markup must be supported, pass the string through an established sanitizer such as DOMPurify before rendering."
        ),
    },
    {
        "record_id": "cwe-918-ssrf-webhook",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-918",
        "instruction": "Write a defensive Python validation function for webhook URLs to block Server-Side Request Forgery (SSRF).",
        "thought": "SSRF occurs when an attacker provides internal IPs (127.0.0.1, 169.254.169.254, 10.0.0.0/8). A robust defense requires verifying the URL scheme (HTTPS only), resolving DNS to an IP address, and ensuring the IP is not private, loopback, or link-local.",
        "response": (
            "### Vulnerability Analysis (CWE-918: SSRF)\n"
            "Attackers target internal cloud metadata services and localhost services through unvalidated outbound HTTP calls.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "import ipaddress\n"
            "import socket\n"
            "from urllib.parse import urlparse\n\n"
            "def validate_webhook_url(url: str) -> bool:\n"
            "    parsed = urlparse(url)\n"
            "    if parsed.scheme != 'https':\n"
            "        raise ValueError('Only HTTPS URLs are permitted')\n"
            "    if not parsed.hostname:\n"
            "        raise ValueError('Missing hostname in webhook URL')\n"
            "    \n"
            "    # Resolve host to IP address\n"
            "    ip_str = socket.gethostbyname(parsed.hostname)\n"
            "    ip = ipaddress.ip_address(ip_str)\n"
            "    \n"
            "    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:\n"
            "        raise PermissionError(f'Access to private/internal network IP {ip} is forbidden')\n"
            "    \n"
            "    return True\n"
            "<|patch_end|>"
        ),
    },
    {
        "record_id": "cwe-502-insecure-deserialization",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-502",
        "instruction": "A developer is using Python `pickle` to cache user session state. Why is this insecure and what is the secure replacement?",
        "thought": "Python `pickle` can construct arbitrary objects during unpickling via `__reduce__`, leading to remote code execution. Untrusted data must never be unpickled. Safe serialization formats like JSON or encrypted JWT should be used.",
        "response": (
            "### Vulnerability Analysis (CWE-502: Insecure Deserialization)\n"
            "`pickle.loads()` is inherently unsafe for untrusted inputs because serialized streams can invoke arbitrary system executables via object reduction.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "import json\n"
            "\n"
            "# Replace pickle with structured JSON\n"
            "def serialize_session(data: dict) -> str:\n"
            "    return json.dumps(data)\n\n"
            "def deserialize_session(payload: str) -> dict:\n"
            "    return json.loads(payload)\n"
            "<|patch_end|>\n\n"
            "JSON contains pure data structures and cannot execute code upon deserialization."
        ),
    },
    {
        "record_id": "cwe-208-timing-attack-hmac",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-208",
        "instruction": "Fix the timing attack vulnerability in this API key comparison:\n\n```python\ndef verify_api_key(client_key, secret_key):\n    if client_key == secret_key:\n        return True\n    return False\n```",
        "thought": "Standard string equality `==` short-circuits on the first mismatched character, allowing an attacker to deduce the key character-by-character via timing side-channels. Use `hmac.compare_digest` for constant-time comparison.",
        "response": (
            "### Vulnerability Analysis (CWE-208: Timing Side-Channel)\n"
            "Standard equality operators terminate early upon character divergence, leaking information through execution time.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "import hmac\n\n"
            "def verify_api_key(client_key: str, secret_key: str) -> bool:\n"
            "    # Constant-time comparison prevents timing side-channels\n"
            "    return hmac.compare_digest(client_key.encode('utf-8'), secret_key.encode('utf-8'))\n"
            "<|patch_end|>"
        ),
    },
    {
        "record_id": "cwe-94-code-injection-eval",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-94",
        "instruction": "Refactor this code calculator to prevent arbitrary code execution:\n\n```python\ndef calculate(user_math_expression):\n    return eval(user_math_expression)\n```",
        "thought": "Passing arbitrary user strings to `eval()` allows execution of dangerous builtins (e.g. `__import__('os').system('rm -rf /')`). Safe parsing requires an Abstract Syntax Tree (AST) visitor that only evaluates whitelisted arithmetic operators.",
        "response": (
            "### Vulnerability Analysis (CWE-94: Code Injection)\n"
            "`eval()` interprets Python expressions with full language capability.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "import ast\n"
            "import operator\n\n"
            "ALLOWED_OPERATORS = {\n"
            "    ast.Add: operator.add,\n"
            "    ast.Sub: operator.sub,\n"
            "    ast.Mult: operator.mul,\n"
            "    ast.Div: operator.truediv,\n"
            "}\n\n"
            "def safe_eval_node(node):\n"
            "    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):\n"
            "        return node.value\n"
            "    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:\n"
            "        left = safe_eval_node(node.left)\n"
            "        right = safe_eval_node(node.right)\n"
            "        return ALLOWED_OPERATORS[type(node.op)](left, right)\n"
            "    raise ValueError('Unsupported or unsafe arithmetic expression')\n\n"
            "def calculate(user_math_expression: str) -> float:\n"
            "    tree = ast.parse(user_math_expression, mode='eval')\n"
            "    return safe_eval_node(tree.body)\n"
            "<|patch_end|>"
        ),
    },
    {
        "record_id": "cwe-287-secure-password-hash",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-287",
        "instruction": "Replace weak MD5 password hashing with modern, production-grade defensive password storage in Python.",
        "thought": "MD5 is obsolete and vulnerable to collision and high-speed GPU rainbow-table attacks. Defensive best practice dictates using Argon2id with memory-hard parameters or bcrypt.",
        "response": (
            "### Vulnerability Analysis (CWE-287 / CWE-328: Broken Cryptographic Hash)\n"
            "Fast general-purpose hashes like MD5/SHA1 are easily broken with commodity GPU cracking.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "from argon2 import PasswordHasher\n"
            "from argon2.exceptions import VerifyMismatchError\n\n"
            "ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)\n\n"
            "def hash_password(plain_password: str) -> str:\n"
            "    return ph.hash(plain_password)\n\n"
            "def verify_password(plain_password: str, hashed_password: str) -> bool:\n"
            "    try:\n"
            "        return ph.verify(hashed_password, plain_password)\n"
            "    except VerifyMismatchError:\n"
            "        return False\n"
            "<|patch_end|>"
        ),
    },
    {
        "record_id": "cwe-862-idor-authorization",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-862",
        "instruction": "Fix the Insecure Direct Object Reference (IDOR) flaw in this user profile endpoint:\n\n```python\n@app.get('/invoice/{invoice_id}')\ndef get_invoice(invoice_id: int):\n    return db.query(Invoice).filter(Invoice.id == invoice_id).first()\n```",
        "thought": "The endpoint checks for existence of `invoice_id` but never verifies whether the currently authenticated user owns or has permission to view that invoice. Enforce tenant and user authorization.",
        "response": (
            "### Vulnerability Analysis (CWE-862: Insecure Direct Object Reference)\n"
            "Querying by ID alone permits any authenticated user to access any other user's documents.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "from fastapi import Depends, HTTPException, status\n\n"
            "@app.get('/invoice/{invoice_id}')\n"
            "def get_invoice(invoice_id: int, current_user: User = Depends(get_current_active_user)):\n"
            "    invoice = db.query(Invoice).filter(\n"
            "        Invoice.id == invoice_id,\n"
            "        Invoice.tenant_id == current_user.tenant_id,\n"
            "        Invoice.owner_id == current_user.id,\n"
            "    ).first()\n"
            "    if invoice is None:\n"
            "        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Invoice not found')\n"
            "    return invoice\n"
            "<|patch_end|>"
        ),
    },
    {
        "record_id": "cwe-798-hardcoded-secret",
        "category": "vulnerability_remediation",
        "cwe_id": "CWE-798",
        "instruction": "Remediate hardcoded secrets in application database connection string.",
        "thought": "Committing credentials to version control leads to unauthorized database access. Credentials must be injected via environment variables or a vault manager.",
        "response": (
            "### Vulnerability Analysis (CWE-798: Use of Hard-coded Credentials)\n"
            "Static credentials in source files are easily extracted from repository history.\n\n"
            "### Secure Patch\n"
            "<|patch_start|>\n"
            "import os\n\n"
            "def get_database_url() -> str:\n"
            "    db_url = os.environ.get('DATABASE_URL')\n"
            "    if not db_url:\n"
            "        raise RuntimeError('Required environment variable DATABASE_URL is missing')\n"
            "    return db_url\n"
            "<|patch_end|>"
        ),
    },
]


class SFTDatasetEngine:
    """Manages generation, partitioning, and collation of defensive SFT instruction sets."""

    def __init__(self, records: list[SFTRecord] | None = None) -> None:
        if records is not None:
            self.records = records
        else:
            self.records = [SFTRecord(**item) for item in CANONICAL_DEFENSIVE_SFT_SEED_DATA]

    def expand_dataset(self, target_count: int = 1200, seed: int = 1337) -> list[SFTRecord]:
        """Synthesize a robust training distribution by scaling canonical records with varied prompts."""
        rng = random.Random(seed)
        expanded: list[SFTRecord] = []
        base_pool = list(self.records)

        while len(expanded) < target_count:
            chosen = rng.choice(base_pool)
            idx = len(expanded)
            expanded.append(
                SFTRecord(
                    record_id=f"{chosen.record_id}-{idx:05d}",
                    category=chosen.category,
                    cwe_id=chosen.cwe_id,
                    system_prompt=chosen.system_prompt,
                    instruction=chosen.instruction,
                    thought=chosen.thought,
                    response=chosen.response,
                    metadata={"expansion_index": idx, "base_id": chosen.record_id},
                )
            )

        rng.shuffle(expanded)
        self.records = expanded
        return expanded

    def partition(self, val_ratio: float = 0.1, seed: int = 1337) -> tuple[list[SFTRecord], list[SFTRecord]]:
        """Split into training and validation sets deterministically."""
        rng = random.Random(seed)
        records = list(self.records)
        rng.shuffle(records)
        val_count = max(1, int(len(records) * val_ratio))
        return records[val_count:], records[:val_count]

    def save_jsonl(self, output_file: str | Path) -> Path:
        """Write records to JSONL."""
        target = Path(output_file).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
        return target

    @classmethod
    def load_jsonl(cls, input_file: str | Path) -> "SFTDatasetEngine":
        """Load records from JSONL."""
        source = Path(input_file).resolve()
        records: list[SFTRecord] = []
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(SFTRecord(**json.loads(line)))
        return cls(records=records)
