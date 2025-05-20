# Smart Query Assistant: Text2SQL Agent with Long-Term Memory

A text-to-SQL agent that remembers user preferences across sessions, implementing user-specific, long-term memory for smarter database interactions.

![Smart Query Assistant](https://img.shields.io/badge/Smart_Query_Assistant-v1.0-blue)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)
![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Overview

The Smart Query Assistant transforms the traditional text-to-SQL experience by implementing long-term memory capabilities that persist across user sessions. Inspired by the Mem0 architecture, this implementation focuses specifically on database query interactions and maintains complete user isolation for personalized, secure experiences.

### Why long-term memory matters for database queries

In traditional text2SQL systems, conversations restart from scratch each session. Preferences, terminology, and context vanish, forcing users to repeatedly re-educate their AI assistant. This creates significant cognitive load as users must:

- Respecify preferences ("only show approved loans") every morning
- Redefine terminology ("luxury properties") each session
- Constantly refresh context for follow-up questions

This implementation solves these problems by enabling your text2SQL agent to remember:

- User preferences across sessions (automatically filtering by criteria)
- Domain-specific terminology that bridges user language and database schema
- Context over time (allowing natural follow-ups like "How many of those were from California?" days later)
- Data interests, improving relevance without explicit instruction

## ✨ Key Features

- **Multi-user memory isolation**: Each user's preferences stored separately, preventing cross-contamination
- **User-targeted extraction**: Identifies SQL-relevant elements (entities, preferences, terminology, metrics) per user
- **Database schema integration**: Connects memories directly to database schema for accurate SQL generation
- **Vector-based similarity search**: Efficiently retrieves relevant memories using embedding similarity
- **Interactive memory management**: View and understand what the system remembers about your preferences
- **Gradio web interface**: User-friendly frontend for interacting with the system

## 🛠️ Architecture

The system consists of two primary phases that work together:

1. **Extraction Phase**:
   - Message Ingestion: Processes new message pairs (user question + AI response)
   - Context Retrieval: Leverages conversation summary and recent messages
   - LLM Processing: Analyzes the conversation to extract relevant information

2. **User-Specific Memory Extraction**:
   - Entity Extraction: Identifies database entities the user frequently references
   - Preference Capture: Records filtering and sorting preferences
   - Terminology Recognition: Maps user-defined terms to database equivalents
   - Metric Definition: Stores custom calculations or criteria

3. **Memory Update Phase**:
   - Similarity Search: Finds similar existing memories
   - Operation Classification: Determines whether to ADD, UPDATE, DELETE, or make NO CHANGE
   - Vector Database Storage: Stores memories with proper user isolation

## 🔧 Installation

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension (for production) or JSON storage (for development)
- Denodo data virtualization platform
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-query-assistant.git
   cd smart-query-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file):
   ```
   DENODO_API_HOST=http://yourdenodoaisdkhost:8080
   DENODO_USERNAME=yourusername
   DENODO_PASSWORD=yourpassword
   OPENAI_API_KEY=your_openai_api_key
   EMBEDDING_MODEL=text-embedding-ada-002
   LLM_MODEL=gpt-4o-mini
   PG_CONN_STRING=postgresql://user:password@localhost:5432/memorydb
   ```

4. Initialize PostgreSQL with pgvector (optional, for production):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

## 🚀 Usage

### Starting the Web Interface

```bash
python memory_frontend.py
```

This will launch a Gradio web interface accessible at http://localhost:7860.

## Remember you can replace DenodoAPIClient with any of your data platform or database where you extract the semanic model

### Workflow

1. **Login**: Enter your Denodo credentials
2. **Load Database**: Specify the database name to connect to
3. **Ask Questions**: Start interacting with your database in natural language
4. **View Memories**: See what the system has learned about your preferences

### Example Interactions

**Setting Preferences**:
```
User: I'm only interested in approved loans going forward.
AI: I understand you want to focus on approved loans. I'll remember this preference.

User: Show me the loans with the highest interest rates.
AI: Here are the approved loans with the highest interest rates.
(Note: I've applied your previously expressed preferences to this query.)
```

**Learning Terminology**:
```
User: Let's define high-risk loans as those with credit scores below 750.
AI: I've saved your definition of high-risk loans (credit scores below 750).

User: How many high-risk loans do we have in California?
AI: There are 28 high-risk loans in California.
(Note: I've applied your definition of high-risk loans as those with credit scores below 750.)
```

## 📁 Project Structure

- `memory_agent.py`: Backend implementation of the memory system
- `memory_frontend.py`: Gradio-based web interface
- `requirements.txt`: Required Python packages
- `docs/`: Additional documentation and architecture diagrams

## 📋 Requirements

```
gradio>=4.0.0
numpy>=1.20.0
openai>=1.0.0
psycopg2-binary>=2.9.5
python-dotenv>=0.21.0
requests>=2.28.1
```

## 🔄 Memory Architecture

The memory system uses:

- **PostgreSQL + pgvector**: Production storage with vector similarity search
- **JSON files**: Development/fallback storage option
- **Hierarchical indexing**: HNSW (Hierarchical Navigable Small World) index for efficient similarity searches
- **User isolation**: Complete separation of memories between users

## ⚙️ Configuration Options

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| DENODO_API_HOST | URL of your Denodo AI SDK API HOST | http://localhost:8080 |
| OPENAI_API_KEY | Your OpenAI API key | None |
| EMBEDDING_MODEL | Embedding model to use | text-embedding-ada-002 |
| LLM_MODEL | Language model to use | gpt-4o-mini |
| PG_CONN_STRING | PostgreSQL connection string | None |

## 🌟 Use Cases

- **Financial Analysis**: Remember analyst preferences for filtering financial data
- **Property Management**: Store custom definitions of property categories
- **Customer Support**: Maintain context about specific customer issues across sessions
- **Data Exploration**: Build user-specific mental models of how database entities relate

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the Mem0 architecture from mem0.ai
- Built on top of the Denodo data virtualization platform
- Uses OpenAI models for embeddings and text processing

## 📬 Contact

For questions or support, please open an issue on GitHub.
