"""
CoexistAI Startup Banner Module
Displays professional ASCII banner and system information on startup
"""

import os
import sys
from datetime import datetime
from pathlib import Path

def get_ascii_banner():
    """Load ASCII banner from file"""
    return """
 ██████╗ ██████╗ ███████╗██╗  ██╗██╗███████╗████████╗ █████╗ ██╗
██╔════╝██╔═══██╗██╔════╝╚██╗██╔╝██║██╔════╝╚══██╔══╝██╔══██╗██║
██║     ██║   ██║█████╗   ╚███╔╝ ██║███████╗   ██║   ███████║██║
██║     ██║   ██║██╔══╝   ██╔██╗ ██║╚════██║   ██║   ██╔══██║██║
╚██████╗╚██████╔╝███████╗██╔╝ ██╗██║███████║   ██║   ██║  ██║██║
 ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝
        """

def get_system_info():
    """Get basic system information"""
    try:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return {
            "python_version": python_version,
            "platform": sys.platform,
            "startup_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception:
        return {
            "python_version": "Unknown",
            "platform": "Unknown", 
            "startup_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def display_startup_banner(host="localhost", port=8000, mcp_port=None):
    """
    Display the complete startup banner with system information
    
    Args:
        host (str): Server host address
        port (int): FastAPI server port
        mcp_port (int, optional): MCP server port if enabled
    """
    
    # Color codes for terminal output
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Get banner and system info
    banner = get_ascii_banner()
    sys_info = get_system_info()
    
    # Build the complete startup display
    startup_display = f"""
{CYAN}{BOLD}{banner}{RESET}

{MAGENTA}═══════════════════════════════════════════════════════════════════════════════{RESET}
{BOLD}{WHITE}                        🚀 CoexistAI Research Assistant v0.0.2{RESET}
{MAGENTA}═══════════════════════════════════════════════════════════════════════════════{RESET}

{YELLOW}📋 SYSTEM OVERVIEW:{RESET}
{WHITE}   • Modular AI research framework with LLM integration{RESET}
{WHITE}   • Multi-source data exploration: Web, Reddit, YouTube, GitHub, Maps{RESET}
{WHITE}   • Async & parallel processing for optimal performance{RESET}
{WHITE}   • MCP (Model Context Protocol) compatible{RESET}

{BLUE}🔧 CORE FEATURES:{RESET}
{GREEN}   ✓ Web Explorer      {WHITE}- Query web, summarize results with LLMs{RESET}
{GREEN}   ✓ Reddit Explorer   {WHITE}- Search & analyze Reddit content with BM25 ranking{RESET}
{GREEN}   ✓ YouTube Explorer  {WHITE}- Transcript search, summarization & custom prompts{RESET}
{GREEN}   ✓ Map Explorer      {WHITE}- Location search, routing, POI discovery{RESET}
{GREEN}   ✓ GitHub Explorer   {WHITE}- Codebase analysis for GitHub & local repos{RESET}
{GREEN}   ✓ File Explorer     {WHITE}- Local file analysis with vision support{RESET}

{CYAN}🌐 SERVER STATUS:{RESET}
{WHITE}   • FastAPI Server:   {GREEN}http://{host}:{port}{RESET}
{WHITE}   • API Documentation: {GREEN}http://{host}:{port}/docs{RESET}
{WHITE}   • Health Check:     {GREEN}http://{host}:{port}/health{RESET}"""

    if mcp_port:
        startup_display += f"""
{WHITE}   • MCP Server:       {GREEN}mcp://{host}:{mcp_port}{RESET}"""

    startup_display += f"""

{YELLOW}⚙️  SYSTEM INFO:{RESET}
{WHITE}   • Python Version:   {GREEN}{sys_info['python_version']}{RESET}
{WHITE}   • Platform:         {GREEN}{sys_info['platform']}{RESET}
{WHITE}   • Started:          {GREEN}{sys_info['startup_time']}{RESET}
{WHITE}   • Contributor:      {GREEN}Sidhant Pravinkumar Thole{RESET}

{MAGENTA}═══════════════════════════════════════════════════════════════════════════════{RESET}
{BOLD}{CYAN}                    Ready to accelerate your research! 🎯{RESET}
{MAGENTA}═══════════════════════════════════════════════════════════════════════════════{RESET}
"""
    
    print(startup_display)

def display_shutdown_banner():
    """Display shutdown message"""
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    shutdown_msg = f"""
{CYAN}═══════════════════════════════════════════════════════════════════════════════{RESET}
{BOLD}{YELLOW}                    🛑 CoexistAI Server Shutting Down...{RESET}
{CYAN}═══════════════════════════════════════════════════════════════════════════════{RESET}
"""
    print(shutdown_msg)

if __name__ == "__main__":
    # Test the banner
    display_startup_banner()
