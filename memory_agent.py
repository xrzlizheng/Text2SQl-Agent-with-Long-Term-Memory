import os
import json
import time
import base64
import requests
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DENODO_API_HOST = os.getenv("DENODO_API_HOST", "http://localhost:8080")
DENODO_USERNAME = os.getenv("DENODO_USERNAME", "admin")
DENODO_PASSWORD = os.getenv("DENODO_PASSWORD", "admin")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

class Memory:
    """Base class for memory objects"""
    def __init__(self, id: Optional[int] = None, content: str = "", created_at: float = 0, 
                 source: str = "", metadata: Dict = None, embedding: List[float] = None):
        self.id = id
        self.content = content
        self.created_at = created_at
        self.source = source
        self.metadata = metadata or {}
        self.embedding = embedding


class PostgresMemoryStore:
    """PostgreSQL-based memory storage implementation with user isolation"""
    
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self.embedding_dim = 1536  # OpenAI embedding dimension
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables if they don't exist"""
        try:
            import psycopg2
            import psycopg2.extras
            
            # Connect to PostgreSQL
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            # Ensure vector extension is loaded
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()  # Commit extension creation
            
            # Check if memories table exists
            cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'memories')")
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                # Create memories table with user_id field
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at FLOAT NOT NULL,
                    source TEXT,
                    metadata JSONB DEFAULT '{}'::JSONB,
                    embedding vector(1536)
                );
                """)
                
                # Create standard index on embedding column (not ivfflat)
                cursor.execute("""
                CREATE INDEX memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops);
                """)
                
                # Create index on user_id
                cursor.execute("CREATE INDEX memories_user_id_idx ON memories(user_id);")
            
            # Check if conversation_summaries table exists
            cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'conversation_summaries')")
            summaries_exists = cursor.fetchone()[0]
            
            if not summaries_exists:
                # Create conversation_summaries table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Create index for user_id 
                cursor.execute("CREATE INDEX summaries_user_id_idx ON conversation_summaries(user_id);")
            
            # Check if recent_messages table exists
            cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'recent_messages')")
            messages_exists = cursor.fetchone()[0]
            
            if not messages_exists:
                # Create recent_messages table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS recent_messages (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Create index for user_id
                cursor.execute("CREATE INDEX messages_user_id_idx ON recent_messages(user_id);")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("Database initialized successfully")
        except ImportError:
            print("PostgreSQL drivers not installed. Using fallback JSON storage.")
        except Exception as e:
            print(f"Error initializing database: {e}")
            # Try to close connection if still open
            try:
                if 'conn' in locals() and conn:
                    conn.close()
            except:
                pass
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Ensure embedding has the correct dimensions"""
        if not embedding:
            return [0.0] * self.embedding_dim
        
        if len(embedding) == self.embedding_dim:
            return embedding
        
        if len(embedding) > self.embedding_dim:
            print(f"Warning: Truncating embedding from {len(embedding)} to {self.embedding_dim}")
            return embedding[:self.embedding_dim]
        
        print(f"Warning: Padding embedding from {len(embedding)} to {self.embedding_dim}")
        return embedding + [0.0] * (self.embedding_dim - len(embedding))
    
    def load_memories(self, user_id: str) -> List[Memory]:
        """Load memories for a specific user"""
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            cursor.execute(
                "SELECT id, content, created_at, source, metadata, embedding FROM memories WHERE user_id = %s",
                (user_id,)
            )
            rows = cursor.fetchall()
            
            memories = []
            for row in rows:
                memory = Memory(
                    id=row['id'],
                    content=row['content'],
                    created_at=row['created_at'],
                    source=row['source'],
                    metadata=row['metadata'],
                    embedding=row['embedding']
                )
                memories.append(memory)
            
            cursor.close()
            conn.close()
            
            return memories
        except Exception as e:
            print(f"Error loading memories: {e}")
            return []
    
    def save_memory(self, memory: Memory, user_id: str) -> int:
        """Save a memory to the database for a specific user"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            # Normalize embedding
            normalized_embedding = self._normalize_embedding(memory.embedding)
            
            if memory.id is None:
                # Insert new memory
                cursor.execute(
                    "INSERT INTO memories (user_id, content, created_at, source, metadata, embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (user_id, memory.content, memory.created_at, memory.source, json.dumps(memory.metadata), normalized_embedding)
                )
                memory.id = cursor.fetchone()[0]
            else:
                # Update existing memory
                cursor.execute(
                    "UPDATE memories SET content = %s, created_at = %s, source = %s, metadata = %s, embedding = %s WHERE id = %s AND user_id = %s",
                    (memory.content, memory.created_at, memory.source, json.dumps(memory.metadata), normalized_embedding, memory.id, user_id)
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return memory.id
        except Exception as e:
            print(f"Error saving memory: {e}")
            return -1
    
    def delete_memory(self, memory_id: int, user_id: str) -> bool:
        """Delete a memory for a specific user"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM memories WHERE id = %s AND user_id = %s", (memory_id, user_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
    
    def get_conversation_summary(self, user_id: str) -> str:
        """Get the latest conversation summary for a user"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT summary FROM conversation_summaries WHERE user_id = %s ORDER BY updated_at DESC LIMIT 1", 
                (user_id,)
            )
            row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return row[0] if row else ""
        except Exception as e:
            print(f"Error getting conversation summary: {e}")
            return ""
    
    def save_conversation_summary(self, summary: str, user_id: str) -> bool:
        """Save a conversation summary for a user"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO conversation_summaries (user_id, summary) VALUES (%s, %s)",
                (user_id, summary)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error saving conversation summary: {e}")
            return False
    
    def get_recent_messages(self, user_id: str, limit: int = 10) -> List[str]:
        """Get the most recent messages for a user"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT message FROM recent_messages WHERE user_id = %s ORDER BY created_at DESC LIMIT %s", 
                (user_id, limit)
            )
            rows = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [row[0] for row in rows]
        except Exception as e:
            print(f"Error getting recent messages: {e}")
            return []
    
    def save_message(self, message: str, user_id: str) -> bool:
        """Save a message for a user"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO recent_messages (user_id, message) VALUES (%s, %s)",
                (user_id, message)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error saving message: {e}")
            return False
    
    def find_similar_memories(self, embedding: List[float], user_id: str, top_k: int = 5) -> List[Tuple[Memory, float]]:
        """Find memories with similar embeddings for a specific user"""
        try:
            import psycopg2
            import psycopg2.extras
            
            # Normalize query embedding
            normalized_embedding = self._normalize_embedding(embedding)
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Using cosine distance for search
            cursor.execute("""
            SELECT id, content, created_at, source, metadata, embedding, 
                  1 - (embedding <=> %s::vector) AS similarity
            FROM memories
            WHERE user_id = %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """, (normalized_embedding, user_id, normalized_embedding, top_k))
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                memory = Memory(
                    id=row['id'],
                    content=row['content'],
                    created_at=row['created_at'],
                    source=row['source'],
                    metadata=row['metadata'],
                    embedding=row['embedding']
                )
                similarity = float(row['similarity'])
                results.append((memory, similarity))
            
            cursor.close()
            conn.close()
            
            return results
        except Exception as e:
            print(f"Error finding similar memories: {e}")
            return []
                        
class JSONMemoryStore:
    """JSON-based memory storage implementation (fallback)"""
    
    def __init__(self, file_path: str = "memory_db.json"):
        self.file_path = file_path
        self.summaries_file = "conversation_summaries.json"
        self.messages_file = "recent_messages.json"
    
    def load_memories(self) -> List[Memory]:
        """Load memories from JSON file"""
        try:
            if not os.path.exists(self.file_path):
                return []
                
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                
            memories = []
            for item in data:
                memory = Memory(
                    id=item.get("id"),
                    content=item["content"],
                    created_at=item["created_at"],
                    source=item.get("source", ""),
                    metadata=item.get("metadata", {}),
                    embedding=item.get("embedding")
                )
                memories.append(memory)
                
            return memories
        except Exception as e:
            print(f"Error loading memories from JSON: {e}")
            return []
    def save_memory(self, memory: Memory) -> int:
        """Save a memory to the database"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            # Ensure embedding is the correct size (1536 dimensions)
            if memory.embedding and len(memory.embedding) != 1536:
                print(f"Warning: Expected 1536 dimensions, got {len(memory.embedding)}. Truncating or padding.")
                # Truncate or pad the embedding to exactly 1536 dimensions
                if len(memory.embedding) > 1536:
                    memory.embedding = memory.embedding[:1536]
                else:
                    memory.embedding.extend([0.0] * (1536 - len(memory.embedding)))
            
            if memory.id is None:
                # Insert new memory with proper vector casting
                cursor.execute(
                    "INSERT INTO memories (content, created_at, source, metadata, embedding) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (memory.content, memory.created_at, memory.source, json.dumps(memory.metadata), memory.embedding)
                )
                memory.id = cursor.fetchone()[0]
            else:
                # Update existing memory
                cursor.execute(
                    "UPDATE memories SET content = %s, created_at = %s, source = %s, metadata = %s, embedding = %s WHERE id = %s",
                    (memory.content, memory.created_at, memory.source, json.dumps(memory.metadata), memory.embedding, memory.id)
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return memory.id
        except Exception as e:
            print(f"Error saving memory: {e}")
            return -1            
            conn.commit()
            cursor.close()
            conn.close()
            
            return memory.id
        except Exception as e:
            print(f"Error saving memory: {e}")
            return -1
            
    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory from the JSON file"""
        try:
            memories = self.load_memories()
            memories = [mem for mem in memories if mem.id != memory_id]
            
            # Save to file
            data = []
            for mem in memories:
                data.append({
                    "id": mem.id,
                    "content": mem.content,
                    "created_at": mem.created_at,
                    "source": mem.source,
                    "metadata": mem.metadata,
                    "embedding": mem.embedding
                })
                
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error deleting memory from JSON: {e}")
            return False
    
    def get_conversation_summary(self) -> str:
        """Get the latest conversation summary"""
        try:
            if not os.path.exists(self.summaries_file):
                return ""
                
            with open(self.summaries_file, 'r') as f:
                data = json.load(f)
                
            if not data:
                return ""
                
            # Return the latest summary
            return data[-1]["summary"]
        except Exception as e:
            print(f"Error getting conversation summary from JSON: {e}")
            return ""
    
    def save_conversation_summary(self, summary: str) -> bool:
        """Save a conversation summary to the JSON file"""
        try:
            summaries = []
            if os.path.exists(self.summaries_file):
                with open(self.summaries_file, 'r') as f:
                    summaries = json.load(f)
            
            summaries.append({
                "summary": summary,
                "timestamp": time.time()
            })
            
            with open(self.summaries_file, 'w') as f:
                json.dump(summaries, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving conversation summary to JSON: {e}")
            return False
    
    def get_recent_messages(self, limit: int = 10) -> List[str]:
        """Get the most recent messages"""
        try:
            if not os.path.exists(self.messages_file):
                return []
                
            with open(self.messages_file, 'r') as f:
                messages = json.load(f)
            
            # Return the most recent messages
            return [msg["message"] for msg in messages[-limit:]]
        except Exception as e:
            print(f"Error getting recent messages from JSON: {e}")
            return []
    
    def save_message(self, message: str) -> bool:
        """Save a message to the JSON file"""
        try:
            messages = []
            if os.path.exists(self.messages_file):
                with open(self.messages_file, 'r') as f:
                    messages = json.load(f)
            
            messages.append({
                "message": message,
                "timestamp": time.time()
            })
            
            with open(self.messages_file, 'w') as f:
                json.dump(messages, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving message to JSON: {e}")
            return False
    
    def find_similar_memories(self, embedding: List[float], top_k: int = 5) -> List[Tuple[Memory, float]]:
        """Find memories with similar embeddings"""
        try:
            memories = self.load_memories()
            
            similarities = []
            for memory in memories:
                if memory.embedding:
                    # Calculate cosine similarity
                    norm1 = np.linalg.norm(embedding)
                    norm2 = np.linalg.norm(memory.embedding)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(embedding, memory.embedding) / (norm1 * norm2)
                    else:
                        similarity = 0.0
                        
                    similarities.append((memory, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
        except Exception as e:
            print(f"Error finding similar memories in JSON: {e}")
            return []


class Mem0Agent:
    """Implementation of Mem0 architecture for chatbots with long-term memory"""
    
    def __init__(self, use_postgres: bool = False, postgres_conn_string: str = None):
        # Initialize the appropriate memory store
        if use_postgres and postgres_conn_string:
            try:
                import psycopg2
                self.store = PostgresMemoryStore(postgres_conn_string)
                print("Using PostgreSQL for memory storage")
            except ImportError:
                print("PostgreSQL support not available. Falling back to JSON storage.")
                self.store = JSONMemoryStore()
        else:
            self.store = JSONMemoryStore()
            
        self.db_schema = None
        # Note: memories, conversation_summary, and recent_messages are now loaded per user
        self.current_user_id = None
        self.max_recent_messages = 10
    
    def load_user_context(self, user_id: str):
        """Load context for a specific user"""
        self.current_user_id = user_id
        self.memories = self.store.load_memories(user_id)
        self.conversation_summary = self.store.get_conversation_summary(user_id)
        self.recent_messages = self.store.get_recent_messages(user_id)
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            payload = {
                "input": text,
                "model": EMBEDDING_MODEL
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            )
            
            response_data = response.json()
            if 'data' in response_data and len(response_data['data']) > 0:
                return response_data["data"][0]["embedding"]
            else:
                print(f"Error in embedding response: {response_data}")
                return [0.0] * 1536  # Return empty embedding as fallback
                
        except Exception as e:
            print(f"Error creating embedding: {e}")
            # Return empty embedding vector as fallback
            return [0.0] * 1536
    
    def _find_similar_memories(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Memory, float]]:
        """Find top_k memories most similar to the query embedding"""
        if hasattr(self, 'current_user_id') and self.current_user_id:
            return self.store.find_similar_memories(query_embedding, self.current_user_id, top_k)
        return []
    
    async def extract_memories(self, message_pair: List[str]) -> List[str]:
        """Extract salient memories from a message pair using LLM with enhanced extraction for preferences and terminology"""
        
        prompt = f"""You are an intelligent memory extraction system. Your task is to identify important information from conversations about database queries that should be remembered for future reference.

Context:
Conversation summary so far: {self.conversation_summary}
Recent messages: {self.recent_messages}

Current message pair:
Human: {message_pair[0]}
AI: {message_pair[1]}

Extract 0-3 concise, salient facts that would be useful to remember in future database queries. 

Pay special attention to the following categories and tag them accordingly:
1. [PREFERENCE] User's query preferences or filters (e.g., "User prefers to see loans with interest rates below 5%")
2. [TERM] Custom terminology or abbreviations defined by the user (e.g., "WC means West Coast states: CA, OR, and WA")
3. [ENTITY] Specific tables, columns, or relationships the user is interested in
4. [METRIC] Custom metrics or calculations defined by the user (e.g., "High-risk loans are defined as those with credit score below 650 and loan amount over $200,000")

Format each memory as a single, concise sentence with the appropriate tag prefix. If no important information is worth retaining, return an empty list.

Extracted memories:"""

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            content = response.json()["choices"][0]["message"]["content"]
            
            # Parse the response to extract memories
            memories = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith("Extracted memories:"):
                    # Remove numbering if present
                    if line[0].isdigit() and line[1:3] in ['. ', '- ', ') ']:
                        line = line[3:].strip()
                    memories.append(line)
            
            return memories
        except Exception as e:
            print(f"Error extracting memories: {e}")
            return []
            
    async def update_memories(self, candidate_facts: List[str]):
        """Update memory database with new facts"""
        # Ensure we have a current user
        if not hasattr(self, 'current_user_id') or not self.current_user_id:
            print("Error: No current user set. Call load_user_context first.")
            return
            
        now = time.time()
        
        for fact in candidate_facts:
            # Create embedding for the fact
            embedding = self._create_embedding(fact)
            
            # Find similar existing memories
            similar_memories = self._find_similar_memories(embedding, top_k=3)
            
            # Determine the operation to perform (ADD, UPDATE, DELETE, NOOP)
            operation = self._determine_operation(fact, similar_memories)
            
            if operation == "ADD":
                memory = Memory(content=fact, created_at=now, embedding=embedding)
                memory_id = self.store.save_memory(memory, self.current_user_id)
                if memory_id > 0:  # If save was successful
                    memory.id = memory_id
                    self.memories.append(memory)
                    print(f"Added memory: {fact}")
            
            elif operation == "UPDATE" and similar_memories:
                memory_to_update = similar_memories[0][0]
                # Only update if the new fact contains more information
                if len(fact) > len(memory_to_update.content):
                    memory_to_update.content = fact
                    memory_to_update.embedding = embedding
                    self.store.save_memory(memory_to_update, self.current_user_id)
                    print(f"Updated memory: {fact}")
            
            elif operation == "DELETE" and similar_memories:
                memory_to_delete = similar_memories[0][0]
                if memory_to_delete.id:
                    self.store.delete_memory(memory_to_delete.id, self.current_user_id)
                    self.memories.remove(memory_to_delete)
                    print(f"Deleted memory: {memory_to_delete.content}")
    
    def _determine_operation(self, fact: str, similar_memories: List[Tuple[Memory, float]]) -> str:
        """Determine the memory operation to perform using LLM"""
        
        if not similar_memories:
            return "ADD"
        
        if similar_memories[0][1] > 0.95:  # Very high similarity
            return "NOOP"
        
        if similar_memories[0][1] > 0.85:  # High similarity
            # Check if the new fact contradicts the existing one
            prompt = f"""You are an intelligent memory management system. Determine if these two pieces of information contradict each other or if one enhances the other:

Existing memory: {similar_memories[0][0].content}
New information: {fact}

Respond with one of:
- DELETE if the new information contradicts and supersedes the existing memory
- UPDATE if the new information enhances or adds to the existing memory
- NOOP if the information is redundant or the existing memory is more comprehensive

Decision:"""
            
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {OPENAI_API_KEY}"
                }
                
                payload = {
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                content = response.json()["choices"][0]["message"]["content"].strip().upper()
                
                if "DELETE" in content:
                    return "DELETE"
                elif "UPDATE" in content:
                    return "UPDATE"
                else:
                    return "NOOP"
            except Exception as e:
                print(f"Error determining operation: {e}")
                return "NOOP"
        
        return "ADD"
    
    async def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve memories relevant to the query"""
        if not hasattr(self, 'current_user_id') or not self.current_user_id:
            print("Error: No current user set. Call load_user_context first.")
            return []
            
        query_embedding = self._create_embedding(query)
        similar_memories = self._find_similar_memories(query_embedding, top_k=top_k)
        
        return [memory.content for memory, _ in similar_memories]
    
    async def update_conversation_summary(self):
        """Update the conversation summary based on recent messages"""
        if not hasattr(self, 'current_user_id') or not self.current_user_id:
            print("Error: No current user set. Call load_user_context first.")
            return
            
        if not self.recent_messages:
            return
        
        prompt = f"""Summarize the key points from the following conversation about database queries. 
Focus on important information about the user's query patterns, the tables they're interested in, and any recurrent themes.

Conversation:
{self.recent_messages}

Previous summary:
{self.conversation_summary}

Updated summary:"""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            self.conversation_summary = response.json()["choices"][0]["message"]["content"].strip()
            self.store.save_conversation_summary(self.conversation_summary, self.current_user_id)
        except Exception as e:
            print(f"Error updating conversation summary: {e}")
    
    def add_message_to_history(self, role: str, content: str):
        """Add a message to the recent message history"""
        if not hasattr(self, 'current_user_id') or not self.current_user_id:
            print("Error: No current user set. Call load_user_context first.")
            return
            
        message = f"{role}: {content}"
        self.recent_messages.append(message)
        self.store.save_message(message, self.current_user_id)
        
        # Keep only the most recent messages
        if len(self.recent_messages) > self.max_recent_messages:
            self.recent_messages = self.recent_messages[-self.max_recent_messages:]
    
    def set_db_schema(self, schema: Dict):
        """Store database schema information"""
        self.db_schema = schema

class DenodoAPIClient:
    """Client for interacting with Denodo AI SDK REST API"""
    
    def __init__(self, api_host: str):
        self.api_host = api_host
    
    def _get_basic_auth_header(self, username: str, password: str) -> str:
        """Create Basic Authentication header"""
        auth_string = f"{username}:{password}"
        auth_bytes = auth_string.encode("utf-8")
        auth_b64 = base64.b64encode(auth_bytes).decode("utf-8")
        return f"Basic {auth_b64}"
    
    async def get_metadata(self, database_name: str, username: str, password: str) -> Dict[str, Any]:
        """Get metadata from Denodo platform using getMetadata endpoint"""
        try:
            url = f"{self.api_host}/getMetadata"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": self._get_basic_auth_header(username, password)
            }
            
            params = {
                "vdp_database_names": database_name,
                "insert": False  # Don't insert into vector store, we just want the schema
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                error_detail = f"Status code: {response.status_code}, Response: {response.text}"
                print(f"Error getting metadata from Denodo: {error_detail}")
                return {"error": error_detail}
            
            return response.json()
        except Exception as e:
            print(f"Error accessing Denodo getMetadata endpoint: {e}")
            return {"error": str(e)}
    
    async def answer_question(self, question: str, username: str, password: str) -> Dict[str, Any]:
        """Send a question to the Denodo AI SDK using answerDataQuestion endpoint"""
        try:
            url = f"{self.api_host}/answerDataQuestion"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": self._get_basic_auth_header(username, password)
            }
            
            payload = {
                "question": question,
                "markdown_response": True,
                "disclaimer": True,
                "verbose": True
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                error_detail = f"Status code: {response.status_code}, Response: {response.text}"
                print(f"Error getting answer from Denodo: {error_detail}")
                return {"error": error_detail}
            
            return response.json()
        except Exception as e:
            print(f"Error accessing Denodo answerDataQuestion endpoint: {e}")
            return {"error": str(e)}


class Mem0DenodoChatbot:
    """Chatbot that integrates Mem0 memory with Denodo AI SDK"""
    
    def __init__(self, 
                 use_postgres: bool = False, 
                 postgres_conn_string: str = None,
                 database_name: str = "bank"):
        self.agent = Mem0Agent(use_postgres, postgres_conn_string)
        self.denodo_client = DenodoAPIClient(DENODO_API_HOST)
        self.database_name = database_name
        self.db_loaded = False
        self.current_user = None
    
    def set_user(self, username: str, password: str):
        """Set the current user for the chatbot"""
        self.current_user = {
            "username": username,
            "password": password
        }
        # Load user context from memory store
        self.agent.load_user_context(username)
        # Reset database loaded flag for the new user
        self.db_loaded = False
    
    async def initialize_db(self):
        """Load database metadata at startup"""
        if not self.current_user:
            return False
            
        if not self.db_loaded:
            print(f"Loading database schema for '{self.database_name}' for user {self.current_user['username']}...")
            metadata = await self.denodo_client.get_metadata(
                self.database_name,
                self.current_user["username"],
                self.current_user["password"]
            )
            
            if "error" in metadata:
                print(f"Error loading database schema: {metadata['error']}")
                return False
            
            self.agent.set_db_schema(metadata)
            self.db_loaded = True
            
            # Create an initial memory about the database
            db_description = self._generate_db_description(metadata)
            if db_description:
                now = time.time()
                memory = Memory(
                    content=db_description,
                    created_at=now,
                    source="database_initialization",
                    metadata={"type": "database_schema"}
                )
                memory.embedding = self.agent._create_embedding(db_description)
                self.agent.store.save_memory(memory, self.current_user["username"])
                self.agent.memories.append(memory)
                
            return True
        return True
    
    def _generate_db_description(self, metadata: Dict) -> str:
        """Generate a concise description of the database from its metadata"""
        try:
            tables = []
            for db_schema in metadata.get("db_schema_json", []):
                for view in db_schema.get("views", []):
                    table_name = view.get("tableName", "")
                    if table_name:
                        tables.append(table_name)
            
            if tables:
                return f"The database contains {len(tables)} tables including: {', '.join(tables[:10])}{' and more' if len(tables) > 10 else ''}."
            return ""
        except Exception as e:
            print(f"Error generating database description: {e}")
            return ""
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and generate a response with enhanced context handling"""
        
        # Ensure we have a current user
        if not self.current_user:
            return "Please log in first."
        
        # Initialize database if not already done
        if not self.db_loaded:
            db_initialized = await self.initialize_db()
            if not db_initialized:
                return "I'm having trouble connecting to the database. Please try again later."
        
        # Add user message to history
        self.agent.add_message_to_history("Human", user_message)
        
        # Retrieve relevant memories
        relevant_memories = await self.agent.retrieve_relevant_memories(user_message)
        
        # Process and categorize memories
        preferences = []
        terminology = []
        metrics = []
        general_context = []
        
        for memory in relevant_memories:
            if memory.startswith("[PREFERENCE]"):
                preferences.append(memory.replace("[PREFERENCE]", "").strip())
            elif memory.startswith("[TERM]"):
                terminology.append(memory.replace("[TERM]", "").strip())
            elif memory.startswith("[METRIC]"):
                metrics.append(memory.replace("[METRIC]", "").strip())
            else:
                general_context.append(memory)
        
        # Apply terminology substitutions
        enhanced_question = user_message
        for term in terminology:
            if ":" in term:
                # Extract the abbreviation and its meaning
                try:
                    abbr, meaning = term.split("means", 1)
                    abbr = abbr.strip()
                    meaning = meaning.strip()
                    if abbr in enhanced_question:
                        enhanced_question = enhanced_question.replace(abbr, f"{abbr} ({meaning})")
                except ValueError:
                    pass  # Skip if format doesn't match expectation
        
        # Check if the question is a follow-up
        follow_up_indicators = ["those", "them", "these", "that", "it", "the ones", "their", "them"]
        is_follow_up = any(indicator in " " + user_message.lower() + " " for indicator in 
                        [f" {ind} " for ind in follow_up_indicators])
        
        # Check if the query is about new data or listing items
        new_data_keywords = ["new", "recent", "latest", "today", "this week", "show me", "list", "get", "find"]
        is_new_data_query = any(keyword in user_message.lower() for keyword in new_data_keywords)
        
        # Prepare context parts without applying preferences yet
        context_parts = []
        
        # Add general context for follow-ups
        if is_follow_up and general_context:
            context_parts.append("Based on our previous conversation about: " + "; ".join(general_context))
        
        # Always add custom metrics
        if metrics:
            context_parts.append("Using these definitions: " + "; ".join(metrics))
        
        # NEW: Check if we should ask about applying preferences
        should_ask_about_preferences = is_new_data_query and preferences and not is_follow_up
        
        if should_ask_about_preferences:
            # We need to ask the user first before applying preferences
            preference_question = "Would you like me to "
            for i, pref in enumerate(preferences):
                if i > 0:
                    preference_question += " and "
                # Extract the action from the preference
                # Example: "User prefers to see loans with interest rates below 5%"
                # Extract: "filter for loans with interest rates below 5%"
                if "prefers" in pref:
                    action = pref.split("prefers", 1)[1].strip()
                    if action.startswith("to "):
                        action = action[3:]
                    preference_question += f"{action}"
                else:
                    preference_question += f"apply this preference: {pref}"
            preference_question += " as you typically prefer?"
            
            # Return the question instead of the actual results
            self.agent.add_message_to_history("AI", preference_question)
            return preference_question
        
        # Apply user preferences if they seem relevant or if it's a new query in the same domain
        preference_keywords = ["show", "list", "get", "find", "display", "what", "which", "how many"]
        should_apply_preferences = any(kw in user_message.lower() for kw in preference_keywords)
        
        # NEW: Handle if the user is responding to a preference question
        yes_indicators = ["yes", "yeah", "yep", "sure", "ok", "apply", "please do", "i do"]
        no_indicators = ["no", "nope", "don't", "do not", "negative"]
        
        # Get last AI message
        last_ai_message = next((msg for msg in reversed(self.agent.recent_messages) 
                              if msg.startswith("AI: ") and "would you like me to" in msg.lower()), None)
        
        # Check if this is a response to a preference question
        if last_ai_message and any(indicator in user_message.lower() for indicator in yes_indicators + no_indicators):
            if any(indicator in user_message.lower() for indicator in yes_indicators):
                # User wants to apply preferences
                should_apply_preferences = True
            else:
                # User doesn't want to apply preferences
                should_apply_preferences = False
        
        if should_apply_preferences and preferences:
            context_parts.append("Considering your preferences: " + "; ".join(preferences))
        
        # Combine context parts
        if context_parts:
            context = " ".join(context_parts)
            enhanced_question = f"{context}. {enhanced_question}"
            print(f"Enhanced question: {enhanced_question}")
        
        # Get answer from Denodo with the enhanced question
        response_data = await self.denodo_client.answer_question(
            enhanced_question,
            self.current_user["username"],
            self.current_user["password"]
        )
        
        if "error" in response_data:
            ai_response = f"I'm sorry, I encountered an error: {response_data['error']}"
        else:
            # Extract key information from the response
            answer = response_data.get("answer", "")
            sql_query = response_data.get("sql_query", "")
            tables_used = response_data.get("tables_used", [])
            related_questions = response_data.get("related_questions", [])
            
            # Format a comprehensive response
            ai_response = answer
            
            # If we applied preferences, acknowledge it
            if should_apply_preferences and preferences:
                ai_response += "\n\n(Note: I've applied your previously expressed preferences to this query.)"
            
            # Add attribution if available
            tables_attribution = ""
            if tables_used:
                tables_attribution = f"\n\nThis information was retrieved from: {', '.join(tables_used)}"
                if sql_query:
                    tables_attribution += f"\n\nThe query used was:\n```sql\n{sql_query}\n```"
                    
            ai_response += tables_attribution
        
        # Add AI response to history
        self.agent.add_message_to_history("AI", ai_response)
        
        # Extract memories from the conversation
        memories = await self.agent.extract_memories([user_message, ai_response])
        
        # Update memories
        await self.agent.update_memories(memories)
        
        # Update conversation summary periodically
        if len(self.agent.recent_messages) % 5 == 0:
            await self.agent.update_conversation_summary()
        
        return ai_response
    
async def main():
    """Main function to run the chatbot"""
    # Define PostgreSQL connection string if needed
    pg_conn_string = os.getenv("PG_CONN_STRING", "")
    use_postgres = bool(pg_conn_string)
    
    print("Starting Mem0-Denodo Chatbot...")
    chatbot = Mem0DenodoChatbot(use_postgres=use_postgres, postgres_conn_string=pg_conn_string)
    
    # Initialize database
    await chatbot.initialize_db()
    
    if chatbot.db_loaded:
        print(f"Database '{chatbot.database_name}' loaded successfully. You can now ask questions about the data.")
    else:
        print("Failed to load database schema. The chatbot will continue with limited functionality.")
    
    print("Chatbot is ready. Type 'exit' to quit.")
    
    history = []
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        
        # Add user message to history
        history.append({"role": "user", "content": user_input})
        
        response = await chatbot.process_message(user_input)
        
        # Add assistant response to history
        history.append({"role": "assistant", "content": response})
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    asyncio.run(main())
