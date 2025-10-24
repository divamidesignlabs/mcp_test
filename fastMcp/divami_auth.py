"""
Divami Knowledge Base Authentication Module
"""
import requests
from typing import Optional, Dict
import os
import warnings
from urllib3.exceptions import NotOpenSSLWarning
import webbrowser
from urllib.parse import urlencode
import time

# Suppress SSL warnings
warnings.filterwarnings('ignore', category=NotOpenSSLWarning)

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class DivamiKBAuth:
    """Handle authentication with Divami Knowledge Base"""
    
    def __init__(self, base_url: str = "https://bks.divami.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.is_authenticated = False
        
    def login(self, email: str, password: str, remember_me: bool = False) -> Dict:
        """
        Authenticate with email and password
        
        Args:
            email: User email address
            password: User password
            remember_me: Whether to persist the session
            
        Returns:
            dict: Response containing authentication status and cookies
        """
        login_url = f"{self.base_url}/login"
        
        payload = {
            "email": email,
            "password": password,
            "remember": remember_me
        }
        
        try:
            response = self.session.post(login_url, data=payload)
            
            if response.status_code == 200:
                self.is_authenticated = True
                return {
                    "success": True,
                    "message": "Authentication successful",
                    "cookies": dict(response.cookies)
                }
            else:
                return {
                    "success": False,
                    "message": f"Authentication failed: {response.status_code}",
                    "error": response.text
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": "Connection error",
                "error": str(e)
            }
    
    def login_with_google(self, open_browser: bool = True) -> Dict:
        """
        Initiate Google OAuth login flow
        
        Args:
            open_browser: Whether to automatically open browser for auth
            
        Returns:
            dict: Authentication instructions and URL
        """
        google_login_url = f"{self.base_url}/login/service/google"
        
        if open_browser:
            try:
                webbrowser.open(google_login_url)
                return {
                    "success": True,
                    "message": "Browser opened for Google authentication",
                    "url": google_login_url,
                    "instructions": "Complete authentication in browser, then use set_session_cookies() to set cookies"
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": "Failed to open browser",
                    "url": google_login_url,
                    "error": str(e)
                }
        else:
            return {
                "success": True,
                "message": "Google login URL generated",
                "url": google_login_url,
                "instructions": "Visit this URL in your browser to authenticate"
            }
    
    def login_with_google_popup(self, headless: bool = False, timeout: int = 60) -> Dict:
        """
        Open popup browser for Google OAuth and automatically capture cookies
        
        Args:
            headless: Run browser in headless mode (no visible window)
            timeout: Max seconds to wait for authentication
            
        Returns:
            dict: Authentication result with captured cookies
        """
        if not SELENIUM_AVAILABLE:
            return {
                "success": False,
                "message": "Selenium not installed. Run: pip install selenium",
                "fallback": "Use login_with_google() for manual authentication"
            }
        
        try:
            # Setup Chrome options
            options = webdriver.ChromeOptions()
            if headless:
                options.add_argument('--headless')
            options.add_argument('--window-size=800,600')
            
            # Initialize driver
            driver = webdriver.Chrome(options=options)
            driver.get(f"{self.base_url}/login/service/google")
            
            print("Waiting for Google authentication...")
            print("Please complete the login in the browser window")
            
            # Wait for redirect back to main site after auth
            WebDriverWait(driver, timeout).until(
                lambda d: self.base_url in d.current_url and "/login" not in d.current_url
            )
            
            # Extract cookies
            cookies = driver.get_cookies()
            cookie_dict = {cookie['name']: cookie['value'] for cookie in cookies}
            
            # Set cookies in session
            for cookie in cookies:
                self.session.cookies.set(cookie['name'], cookie['value'])
            
            self.is_authenticated = True
            
            driver.quit()
            
            return {
                "success": True,
                "message": "Google authentication successful",
                "cookies": cookie_dict,
                "cookie_count": len(cookies)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": "Google authentication failed",
                "error": str(e)
            }
    
    def set_session_cookies(self, cookies: Dict[str, str]) -> Dict:
        """
        Set session cookies manually after external authentication
        
        Args:
            cookies: Dictionary of cookie name-value pairs
            
        Returns:
            dict: Status of cookie setting
        """
        try:
            for name, value in cookies.items():
                self.session.cookies.set(name, value)
            
            self.is_authenticated = True
            return {
                "success": True,
                "message": f"Set {len(cookies)} cookies successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": "Failed to set cookies",
                "error": str(e)
            }
    
    def fetch_content(self, endpoint: str) -> Optional[Dict]:
        """
        Fetch content from knowledge base after authentication
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            dict: Response data or None if not authenticated
        """
        if not self.is_authenticated:
            return {"success": False, "message": "Not authenticated. Please login first."}
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.text,
                    "status_code": response.status_code
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to fetch content: {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": "Request failed",
                "error": str(e)
            }
    
    def search_code_details(self, query: str = "") -> Dict:
        """
        Search for code-related content in knowledge base
        
        Args:
            query: Search query for code examples/details
            
        Returns:
            dict: Search results with code details
        """
        if not self.is_authenticated:
            return {"success": False, "message": "Not authenticated. Please login first."}
        
        try:
            # Try common KB endpoints
            endpoints = [
                "",  # Home/dashboard
                "api/articles",
                "api/search",
                "search",
                "articles"
            ]
            
            results = {}
            for endpoint in endpoints:
                url = f"{self.base_url}/{endpoint}"
                if query and endpoint in ["api/search", "search"]:
                    url = f"{url}?q={query}"
                
                response = self.session.get(url)
                if response.status_code == 200:
                    results[endpoint] = {
                        "status": response.status_code,
                        "content": response.text[:1000],  # First 1000 chars
                        "full_url": url
                    }
            
            return {
                "success": True,
                "results": results,
                "message": f"Fetched {len(results)} endpoints"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": "Search failed",
                "error": str(e)
            }
    
    def get_all_articles(self) -> Dict:
        """
        Fetch all articles/code details from knowledge base
        
        Returns:
            dict: All available articles
        """
        return self.fetch_content("api/articles")


if __name__ == "__main__":
    # Usage example
    print("Divami Knowledge Base Authentication")
    print("=" * 40)
    print("\n1. Email/Password login")
    print("2. Google OAuth (manual)")
    print("3. Google OAuth (popup - requires selenium)")
    
    choice = input("\nSelect method (1-3): ")
    
    auth = DivamiKBAuth()
    
    if choice == "1":
        email = os.getenv("DIVAMI_EMAIL") or input("Email: ")
        password = os.getenv("DIVAMI_PASSWORD") or input("Password: ")
        
        result = auth.login(email, password)
        print(f"\nResult: {result}")
        
    elif choice == "2":
        result = auth.login_with_google()
        print(f"\nResult: {result}")
        
    elif choice == "3":
        result = auth.login_with_google_popup()
        print(f"\nResult: {result}")
    else:
        print("Invalid choice")
        exit()
    
    # If authenticated, fetch code details
    if auth.is_authenticated:
        print("\n" + "=" * 40)
        print("Fetching code details...")
        
        # Search for code examples
        search_query = input("\nEnter search query (or press Enter to fetch all): ")
        
        if search_query:
            results = auth.search_code_details(search_query)
        else:
            results = auth.search_code_details()
        
        print(f"\nSearch Results: {results}")
        
        # Try to get all articles
        print("\nFetching all articles...")
        articles = auth.get_all_articles()
        print(f"\nArticles: {articles}")
