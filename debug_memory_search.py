#!/usr/bin/env python3
"""
Memory Search Debug Tool
Allows you to test and inspect memory retrieval for any user
Shows internal search process, retrieved memories, and AI context
"""

import os
import sys
import asyncio
import re
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Import modules
try:
    from database import (
        get_or_create_user,
        retrieve_memories,
        retrieve_memories_hybrid
    )
    from services.embedding_service import generate_query_embedding
    from services.memory_service import (
        should_use_long_term_memory,
        get_short_term_memory,
        retrieve_long_term_memories
    )
    from ai import generate_ai_response
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

# Colors for terminal output
class Colors:
    RESET = '\033[0m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'

def print_section(title: str, color: str = Colors.CYAN):
    """Print a section header"""
    print()
    print(f"{color}{'=' * 80}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{color}{'=' * 80}{Colors.RESET}")
    print()

def print_subsection(title: str, color: str = Colors.MAGENTA):
    """Print a subsection header"""
    print()
    print(f"{color}{'─' * 80}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{color}{'─' * 80}{Colors.RESET}")

def print_info(label: str, value: Any, color: str = Colors.BLUE):
    """Print labeled information"""
    print(f"{color}  {label}:{Colors.RESET} {value}")

def print_memory(memory: Dict[str, Any], index: int):
    """Print a memory in a formatted way"""
    content = memory.get('content', 'N/A')
    similarity = memory.get('similarity', 0.0)
    metadata = memory.get('metadata', {})
    mem_type = metadata.get('type', 'unknown')
    created_at = memory.get('created_at', 'N/A')
    
    print(f"\n{Colors.CYAN}  Memory #{index + 1}:{Colors.RESET}")
    print(f"    {Colors.BOLD}Content:{Colors.RESET} {content}")
    print(f"    {Colors.BOLD}Similarity Score:{Colors.RESET} {Colors.GREEN}{similarity:.4f}{Colors.RESET}")
    print(f"    {Colors.BOLD}Type:{Colors.RESET} {mem_type}")
    print(f"    {Colors.BOLD}Created:{Colors.RESET} {created_at}")
    if metadata.get('tags'):
        print(f"    {Colors.BOLD}Tags:{Colors.RESET} {', '.join(metadata.get('tags', []))}")


async def debug_memory_search(
    user_phone: str,
    query: str,
    user_name: Optional[str] = None,
    show_ai_response: bool = False
) -> None:
    """
    Debug memory search process for a user query
    
    Args:
        user_phone: User's phone number
        query: User's query/question
        user_name: Optional user name (will be fetched if not provided)
        show_ai_response: Whether to generate and show AI response
    """
    print_section("🔍 Memory Search Debug Tool", Colors.CYAN)
    
    # Step 1: Get or create user
    print_subsection("Step 1: User Lookup")
    try:
        if not user_name:
            user_name = "User"
        
        user_id = await get_or_create_user(user_phone, user_name)
        if not user_id:
            print(f"{Colors.RED}❌ Failed to get/create user{Colors.RESET}")
            return
        
        print_info("Phone Number", user_phone)
        print_info("User Name", user_name)
        print_info("User ID (UUID)", user_id)
    except Exception as error:
        print(f"{Colors.RED}❌ Error in user lookup: {error}{Colors.RESET}")
        return
    
    # Step 2: Get short-term memory (session buffer)
    print_subsection("Step 2: Short-Term Memory (Session Buffer)")
    try:
        short_term = await get_short_term_memory(user_id)
        print_info("Messages in Buffer", len(short_term))
        if short_term:
            print(f"\n{Colors.BLUE}  Recent Messages:{Colors.RESET}")
            for i, msg in enumerate(short_term[-3:], 1):  # Show last 3
                user_msg = msg.get('userMessage', '')[:60]
                ai_msg = msg.get('aiResponse', '')[:60]
                print(f"    {i}. User: {user_msg}...")
                print(f"       AI: {ai_msg}...")
        else:
            print(f"{Colors.YELLOW}  No messages in session buffer{Colors.RESET}")
    except Exception as error:
        print(f"{Colors.RED}❌ Error getting short-term memory: {error}{Colors.RESET}")
        short_term = []
    
    # Step 3: Check if long-term memory is needed
    print_subsection("Step 3: Memory Retrieval Decision")
    try:
        use_long_term = await should_use_long_term_memory(query, short_term)
        print_info("Should Use Long-Term Memory", f"{Colors.GREEN}YES{Colors.RESET}" if use_long_term else f"{Colors.YELLOW}NO{Colors.RESET}")
        
        if not use_long_term:
            print(f"\n{Colors.YELLOW}⚠️  Long-term memory retrieval was NOT triggered.{Colors.RESET}")
            print(f"{Colors.YELLOW}   This query might not need memory retrieval based on current heuristics.{Colors.RESET}")
    except Exception as error:
        print(f"{Colors.RED}❌ Error in decision logic: {error}{Colors.RESET}")
        use_long_term = True  # Default to true on error
    
    # Step 4: Generate query embedding
    print_subsection("Step 4: Query Embedding Generation")
    query_embedding = None
    try:
        print_info("Query Text", query)
        print(f"{Colors.BLUE}  Generating embedding...{Colors.RESET}")
        query_embedding = await generate_query_embedding(query)
        if query_embedding:
            print_info("Embedding Dimension", len(query_embedding))
            print_info("Embedding Sample", f"[{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ..., {query_embedding[-1]:.4f}]")
        else:
            print(f"{Colors.RED}❌ Failed to generate embedding{Colors.RESET}")
    except Exception as error:
        print(f"{Colors.RED}❌ Error generating embedding: {error}{Colors.RESET}")
    
    # Step 5: Hybrid Search Process
    print_subsection("Step 5: Hybrid Search (Semantic + Keyword)")
    hybrid_memories = []
    if use_long_term and query_embedding:
        try:
            print(f"{Colors.BLUE}  Running hybrid search...{Colors.RESET}")
            print_info("Search Type", "Hybrid (Vector + Keyword)")
            
            # Extract entities for keyword search
            phone_pattern = r'\b\d{7,}\b'
            phone_numbers = re.findall(phone_pattern, query)
            name_pattern = r'\b([A-Z][a-z]+)\b|\b([a-z]{3,})\b'
            potential_names = []
            for match in re.finditer(name_pattern, query.lower()):
                word = match.group(0).lower()
                common_words = {'the', 'what', 'where', 'when', 'who', 'how', 'can', 'you', 'tell', 'me'}
                if word not in common_words and len(word) >= 3:
                    potential_names.append(word)
            
            print_info("Detected Phone Numbers", phone_numbers if phone_numbers else "None")
            print_info("Detected Potential Names", potential_names if potential_names else "None")
            
            hybrid_memories = await retrieve_memories_hybrid(
                user_id=user_id,
                user_message=query,
                query_embedding=query_embedding,
                limit=10,  # Get more for debugging
                similarity_threshold=0.5
            )
            
            print_info("Total Memories Retrieved", len(hybrid_memories))
            
            if hybrid_memories:
                print(f"\n{Colors.GREEN}✅ Retrieved Memories:{Colors.RESET}")
                for i, mem in enumerate(hybrid_memories):
                    print_memory(mem, i)
            else:
                print(f"\n{Colors.YELLOW}⚠️  No memories retrieved{Colors.RESET}")
                
        except Exception as error:
            print(f"{Colors.RED}❌ Error in hybrid search: {error}{Colors.RESET}")
            traceback.print_exc()
    
    # Step 6: Standard Vector Search (for comparison)
    print_subsection("Step 6: Standard Vector Search (Comparison)")
    vector_memories = []
    if use_long_term and query_embedding:
        try:
            print(f"{Colors.BLUE}  Running standard vector search for comparison...{Colors.RESET}")
            thresholds = [0.5, 0.4, 0.0]
            
            for threshold in thresholds:
                vector_memories = await retrieve_memories(
                    user_id=user_id,
                    query_embedding=query_embedding,
                    limit=10,
                    similarity_threshold=threshold
                )
                
                print_info(f"Threshold {threshold}", f"{len(vector_memories)} results")
                
                if len(vector_memories) >= 3:
                    break
            
            if vector_memories:
                print(f"\n{Colors.CYAN}  Top Vector Search Results:{Colors.RESET}")
                for i, mem in enumerate(vector_memories[:5]):
                    print_memory(mem, i)
        except Exception as error:
            print(f"{Colors.RED}❌ Error in vector search: {error}{Colors.RESET}")
    
    # Step 7: Final Memory Context for AI
    print_subsection("Step 7: Final Context Sent to AI")
    final_memories = hybrid_memories[:5] if hybrid_memories else vector_memories[:5]
    
    if final_memories:
        print(f"{Colors.GREEN}✅ Memories included in AI context:{Colors.RESET}")
        memory_context = []
        for i, mem in enumerate(final_memories, 1):
            content = mem.get('content', '')
            similarity = mem.get('similarity', 0.0)
            if similarity >= 0.6:
                formatted = f"{i}. {content} [High confidence: {similarity:.2f}]"
            else:
                formatted = f"{i}. {content}"
            memory_context.append(formatted)
            print(f"  {formatted}")
        
        print(f"\n{Colors.BLUE}  Formatted Memory Context:{Colors.RESET}")
        print(f"  {Colors.CYAN}{'─' * 76}{Colors.RESET}")
        for line in memory_context:
            print(f"  {line}")
        print(f"  {Colors.CYAN}{'─' * 76}{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}⚠️  No memories will be sent to AI{Colors.RESET}")
        print(f"{Colors.YELLOW}   AI will respond without memory context{Colors.RESET}")
    
    # Step 8: Generate AI Response (optional)
    if show_ai_response:
        print_subsection("Step 8: AI Response Generation")
        try:
            print(f"{Colors.BLUE}  Generating AI response with memory context...{Colors.RESET}")
            
            ai_response = await generate_ai_response(
                user_message=query,
                conversation_history=short_term,
                user_name=user_name,
                target_language='en',
                retrieved_memories=final_memories
            )
            
            print(f"\n{Colors.GREEN}✅ AI Response:{Colors.RESET}")
            print(f"  {Colors.BOLD}{ai_response}{Colors.RESET}")
            
            # Check if AI used the memory
            if final_memories:
                memory_used = False
                response_lower = ai_response.lower()
                
                for mem in final_memories:
                    key_content = mem.get('content', '').lower()
                    # Check if key parts of memory appear in response
                    if len(key_content) > 10:
                        # Extract key phrases from memory (words longer than 4 chars)
                        key_phrases = [w for w in key_content.split() if len(w) > 4]
                        # Check if any key phrase appears in response
                        if key_phrases and any(phrase in response_lower for phrase in key_phrases):
                            memory_used = True
                            break
                        # Check for phone numbers
                        phone_in_mem = re.findall(r'\d{7,}', key_content)
                        phone_in_response = re.findall(r'\d{7,}', response_lower)
                        if phone_in_mem and any(p in phone_in_response for p in phone_in_mem):
                            memory_used = True
                            break
                
                if memory_used:
                    print(f"\n{Colors.GREEN}✅ AI successfully used retrieved memory!{Colors.RESET}")
                else:
                    print(f"\n{Colors.YELLOW}⚠️  AI may not have used the retrieved memory{Colors.RESET}")
                    print(f"{Colors.YELLOW}   Response doesn't clearly reference memory content{Colors.RESET}")
        except Exception as error:
            print(f"{Colors.RED}❌ Error generating AI response: {error}{Colors.RESET}")
            traceback.print_exc()
    
    # Summary
    print_section("📊 Summary", Colors.CYAN)
    print_info("User", f"{user_name} ({user_phone})")
    print_info("Query", query)
    print_info("Long-Term Memory Used", f"{Colors.GREEN}YES{Colors.RESET}" if use_long_term else f"{Colors.YELLOW}NO{Colors.RESET}")
    print_info("Memories Retrieved", len(final_memories))
    if final_memories:
        avg_similarity = sum(m.get('similarity', 0) for m in final_memories) / len(final_memories)
        print_info("Average Similarity", f"{avg_similarity:.4f}")
        print_info("Highest Similarity", f"{max(m.get('similarity', 0) for m in final_memories):.4f}")
        print_info("Lowest Similarity", f"{min(m.get('similarity', 0) for m in final_memories):.4f}")
    print()


async def main():
    """Main function to run the debug tool"""
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}Memory Search Debug Tool{Colors.RESET}")
    print(f"{Colors.CYAN}Test memory retrieval for any user query{Colors.RESET}")
    print()
    
    # Get user input
    if len(sys.argv) >= 3:
        user_phone = sys.argv[1]
        query = ' '.join(sys.argv[2:])
        user_name = None
        show_ai = False
    else:
        print(f"{Colors.YELLOW}Usage: python debug_memory_search.py <phone_number> <query> [--name <name>] [--ai]{Colors.RESET}")
        print(f"{Colors.YELLOW}Example: python debug_memory_search.py <phone> \"what's my height\" --name <name> --ai{Colors.RESET}")
        print()
        
        user_phone = input(f"{Colors.BLUE}Enter user phone number: {Colors.RESET}").strip()
        if not user_phone:
            print(f"{Colors.RED}Phone number is required{Colors.RESET}")
            return
        
        query = input(f"{Colors.BLUE}Enter query: {Colors.RESET}").strip()
        if not query:
            print(f"{Colors.RED}Query is required{Colors.RESET}")
            return
        
        user_name = input(f"{Colors.BLUE}Enter user name (optional): {Colors.RESET}").strip() or None
        show_ai_input = input(f"{Colors.BLUE}Generate AI response? (y/n): {Colors.RESET}").strip().lower()
        show_ai = show_ai_input == 'y'
    
    # Check for command line flags
    if '--ai' in sys.argv:
        show_ai = True
    if '--name' in sys.argv:
        name_index = sys.argv.index('--name')
        if name_index + 1 < len(sys.argv):
            user_name = sys.argv[name_index + 1]
    
    # Run debug search
    await debug_memory_search(
        user_phone=user_phone,
        query=query,
        user_name=user_name,
        show_ai_response=show_ai
    )


if __name__ == '__main__':
    asyncio.run(main())

