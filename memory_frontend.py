import os
import json
import requests
import asyncio
import gradio as gr
from dotenv import load_dotenv
from memory_agent import Mem0DenodoChatbot, Memory

# Load environment variables
load_dotenv()

# Global variables
chatbot = None
chat_history = []
memories_history = []
database_loaded = False
current_user = None

# Configuration
DENODO_API_HOST = os.getenv("DENODO_API_HOST", "http://localhost:8080")

# Initialize PostgreSQL configuration if available
PG_CONN_STRING = os.getenv("PG_CONN_STRING", "")
USE_POSTGRES = bool(PG_CONN_STRING)

def dummy_login(api_host, username, password):
    """Test login with Denodo API"""
    params = {
        'vdp_database_names': 'fake_vdb',
        'vdp_tag_names': 'fake_tag'
    }
    try:
        response = requests.get(f'{api_host}/getMetadata', params=params, auth=(username, password), verify=False)
        # Accept any successful status code (2xx range)
        return 200 <= response.status_code < 300
    except Exception as e:
        print(f"Login error: {e}")
        return False

def login(username, password):
    """Login and initialize chatbot with user credentials"""
    global chatbot, current_user
    
    # Validate credentials with Denodo API
    login_success = dummy_login(DENODO_API_HOST, username, password)
    
    if login_success:
        # Create the chatbot and set the current user
        chatbot = Mem0DenodoChatbot(
            use_postgres=USE_POSTGRES,
            postgres_conn_string=PG_CONN_STRING
        )
        chatbot.set_user(username, password)
        current_user = username
        
        return f"✅ Login successful. Welcome, {username}!"
    else:
        return "❌ Login failed. Please check your credentials."

async def initialize_chatbot(database_name):
    """Initialize the chatbot with the specified database"""
    global chatbot, database_loaded
    
    if not chatbot or not current_user:
        return "Please log in first."
    
    # Initialize database metadata
    success = await chatbot.initialize_db()
    database_loaded = success
    
    if success:
        return f"✅ Database '{database_name}' successfully loaded for user {current_user}. You can now ask questions about the data."
    else:
        return f"❌ Failed to load database '{database_name}'. Please check your connection settings."

def load_database(database_name):
    """Load the specified database and return status message"""
    if not current_user:
        return "Please log in first."
        
    if not database_name:
        return "Please enter a database name."
    
    result = asyncio.run(initialize_chatbot(database_name))
    return result

async def process_message_async(message):
    """Process a message asynchronously"""
    global chatbot, chat_history, database_loaded, memories_history
    
    if not current_user:
        return "Please log in first."
        
    if not database_loaded:
        return "Please load a database first using the 'Load Database' button."
    
    # Process the message and get the response
    response = await chatbot.process_message(message)
    
    # Clean up the response by removing disclaimers and table information
    response_lines = response.split('\n')
    cleaned_lines = []
    
    for line in response_lines:
        # Skip disclaimer and table information lines
        if not any(skip in line for skip in [
            "DISCLAIMER:",
            "This information was retrieved from:",
            "has been generated based on"
        ]):
            cleaned_lines.append(line)
    
    # Rejoin the cleaned lines
    cleaned_response = '\n'.join(line for line in cleaned_lines if line.strip())
    
    # Get the most recently extracted memories
    if hasattr(chatbot.agent, 'memories') and chatbot.agent.memories:
        recent_memories = sorted(chatbot.agent.memories, key=lambda m: m.created_at, reverse=True)[:5]
        recent_memory_texts = [f"- {memory.content}" for memory in recent_memories]
        memories_history.append("\n".join(recent_memory_texts))
    else:
        memories_history.append("No memories extracted")
    
    return cleaned_response

def process_message(message, history):
    """Process a user message and update the chat history"""
    global chatbot, database_loaded, current_user
    
    if not current_user:
        return [{"role": "assistant", "content": "Please log in first."}]
        
    if not database_loaded:
        return [{"role": "assistant", "content": "Please load a database first using the 'Load Database' button."}]
    
    # Run the async function in a new event loop
    response = asyncio.run(process_message_async(message))
    
    # Format the response as list of message dictionaries
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    
    return history

def get_categorized_memories():
    """Get categorized memories from the database"""
    global chatbot
    if not chatbot or not hasattr(chatbot, 'agent'):
        return "No memories available"
    
    # Get all memories
    memories = chatbot.agent.memories
    if not memories:
        return "No memories available"
    
    # Categorize memories
    categories = {
        'PREFERENCE': [],
        'TERM': [],
        'METRIC': [],
        'ENTITY': []
    }
    
    # Sort memories by creation time, newest first
    sorted_memories = sorted(memories, key=lambda m: m.created_at, reverse=True)
    
    for memory in sorted_memories:
        content = memory.content
        if '[PREFERENCE]' in content:
            categories['PREFERENCE'].append(content.replace('[PREFERENCE]', '').strip())
        elif '[TERM]' in content:
            categories['TERM'].append(content.replace('[TERM]', '').strip())
        elif '[METRIC]' in content:
            categories['METRIC'].append(content.replace('[METRIC]', '').strip())
        elif '[ENTITY]' in content:
            categories['ENTITY'].append(content.replace('[ENTITY]', '').strip())
    
    # Format output with latest 5 entries per category
    output = []
    for category, items in categories.items():
        if items:
            output.append(f"\n{category}:")
            for item in items[:5]:  # Show only latest 5 memories per category
                output.append(f"- {item}")
    
    return '\n'.join(output) if output else "No categorized memories available"

def get_current_memories():
    """Get the current memories from the chatbot"""
    global memories_history
    
    # Get persistent memories from database
    persistent_memories = get_categorized_memories()
    
    # Get current session memories
    current_session = memories_history[-1] if memories_history else "No recent memories in current session"
    
    # Combine both with headers
    return f"""PERSISTENT MEMORIES:\n{persistent_memories}\n\nCURRENT SESSION MEMORIES:\n{current_session}"""

def logout():
    """Log out the current user"""
    global chatbot, current_user, database_loaded, chat_history, memories_history
    
    chatbot = None
    current_user = None
    database_loaded = False
    chat_history = []
    memories_history = []
    
    return "You have been logged out. Please log in to continue."

def build_interface():
    """Build the Gradio interface"""
    with gr.Blocks(css="footer {visibility: hidden}") as interface:
        gr.Markdown("# Smart Query Assistant - Your text to Sql Expert")
        gr.Markdown("This chatbot demonstrates long-term memory capabilities using the Mem0 architecture with Denodo AI SDK.")
        
        # Login section
        with gr.Group():
            gr.Markdown("### Login to Denodo")
            with gr.Row():
                username_input = gr.Textbox(
                    placeholder="Enter your Denodo username",
                    label="Username",
                    value="admin"
                )
                password_input = gr.Textbox(
                    placeholder="Enter your Denodo password",
                    label="Password",
                    value="admin",
                    type="password"
                )
                login_button = gr.Button("Login", variant="primary")
            login_status = gr.Textbox(label="Login Status", interactive=False)
            logout_button = gr.Button("Logout")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Database loading section
                with gr.Group():
                    gr.Markdown("### Load a Database")
                    with gr.Row():
                        db_name_input = gr.Textbox(
                            placeholder="Enter database name (e.g., bank)", 
                            label="Database Name",
                            value="bank"
                        )
                        load_button = gr.Button("Load Database", variant="primary")
                    db_status = gr.Textbox(label="Database Status", interactive=False)
                
                # Chat interface
                chatbot_interface = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    bubble_full_width=False,
                    type="messages"  # Add type parameter
                )
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the data...",
                    label="Your Question",
                    scale=4
                )
                with gr.Row():
                    submit_button = gr.Button("Submit", variant="primary", scale=1)
                    clear_button = gr.Button("Clear Chat", scale=1)
            
            # Memory display section
            with gr.Column(scale=2):
                gr.Markdown("### Memory System")
                gr.Markdown("Showing both persistent memories from database and current session memories:")
                memory_display = gr.Textbox(
                    label="Active Memories",
                    interactive=False,
                    lines=20  # Increased to show more memories
                )
                refresh_memory_btn = gr.Button("Refresh Memories")
        
        # Set up event handlers
        login_button.click(
            fn=login,
            inputs=[username_input, password_input],
            outputs=[login_status]
        )
        
        logout_button.click(
            fn=logout,
            inputs=[],
            outputs=[login_status]
        )
        
        load_button.click(
            fn=load_database,
            inputs=[db_name_input],
            outputs=[db_status]
        )
        
        submit_button.click(
            fn=process_message,
            inputs=[msg_input, chatbot_interface],
            outputs=[chatbot_interface],
            trigger_mode="once"  # Changed from single to once
        ).success(
            fn=lambda: "",
            outputs=[msg_input]
        ).success(
            fn=get_current_memories,
            outputs=[memory_display]
        )
        
        msg_input.submit(
            fn=process_message,
            inputs=[msg_input, chatbot_interface],
            outputs=[chatbot_interface],
            trigger_mode="once"  # Changed from single to once
        )
        
        clear_button.click(
            fn=lambda: [],
            outputs=[chatbot_interface]
        )
        
        refresh_memory_btn.click(
            fn=get_current_memories,
            outputs=[memory_display]
        )
        
        # Add examples
        with gr.Accordion("Example Questions", open=False):
            gr.Markdown("""
            ### Example Questions to Try:
            
            **Basic Queries:**
            - How many approved loans do we have?
            - What is the average interest rate for those?
            - Show me all customers who live in California.
            
            **Follow-up Patterns:**
            - Show me properties in California.
            - Which ones are valued over $400,000?
            - And which of those have loans associated with them?
            
            **Terminology Learning:**
            - I'll refer to West Coast states as WCS, meaning CA, OR, and WA.
            - How many properties do we have in WCS?
            
            **Preference Learning:**
            - I'm only interested in loans with interest rates below 4%.
            - Show me new loans created in the last year.
            
            **Custom Metrics:**
            - Let's define high-value properties as those worth over $500,000.
            - How many high-value properties do we have?
            """)
    
    return interface

# Build and launch the interface
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(server_name="0.0.0.0", share=True)