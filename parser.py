import re
import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

class EmailParser:
    def extract_body(self, path):
        """
        Extract text body from email file (.eml)
        Handles both plain text and HTML emails
        """
        try:
            with open(path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
            
            body_parts = []
            
            # Add email headers for context
            body_parts.append(f"Subject: {msg.get('subject', 'No Subject')}")
            body_parts.append(f"From: {msg.get('from', 'Unknown Sender')}")
            body_parts.append(f"Date: {msg.get('date', 'Unknown Date')}")
            body_parts.append("")  # Empty line
            
            # Extract main body content
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_maintype() == 'text':
                        body_content = self._extract_part_content(part)
                        if body_content:
                            body_parts.append(body_content)
            else:
                body_content = self._extract_part_content(msg)
                if body_content:
                    body_parts.append(body_content)
            
            # Combine all parts
            full_text = "\n".join(body_parts)
            
            # Clean up text
            cleaned_text = self._clean_text(full_text)
            
            return cleaned_text
            
        except Exception as e:
            raise Exception(f"Failed to parse email: {str(e)}")
    
    def _extract_part_content(self, part):
        """Extract content from a single email part"""
        try:
            # Get content type
            content_type = part.get_content_type()
            
            # Get payload
            payload = part.get_payload(decode=True)
            if not payload:
                return ""
            
            # Try to decode with correct charset
            charset = part.get_content_charset() or 'utf-8'
            
            # Handle different encodings
            try:
                text = payload.decode(charset, errors='replace')
            except (LookupError, UnicodeDecodeError):
                # Fallback to utf-8
                text = payload.decode('utf-8', errors='replace')
            
            # If HTML, extract text
            if content_type == 'text/html':
                text = self._html_to_text(text)
            
            return text.strip()
            
        except Exception as e:
            # If decoding fails, return empty string
            return ""
    
    def _html_to_text(self, html_content):
        """Convert HTML to plain text"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text with proper spacing
            text = soup.get_text(separator=' ')
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except:
            # If BeautifulSoup fails, return original with HTML tags removed
            return re.sub(r'<[^>]+>', ' ', html_content)
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common email signatures patterns
        signature_patterns = [
            r'--+\s*$',
            r'^Sent from my .+$',
            r'^Best regards,.+$',
            r'^Thanks,.+$',
            r'^Regards,.+$',
            r'^Sincerely,.+$',
        ]
        
        for pattern in signature_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def extract_from_txt(self, path):
        """Extract text from plain text file"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self._clean_text(content)
        except Exception as e:
            raise Exception(f"Failed to read text file: {str(e)}")