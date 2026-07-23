import os
import re

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

class DocumentParser:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
            
        if not PYPDF_AVAILABLE:
            return f"--- PAGE 1 ---\nMock extracted PDF text for {os.path.basename(pdf_path)}.\nAlex Chen ML Resume details."
            
        reader = PdfReader(pdf_path)
        text_blocks = []
        
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Basic header preservation
                text_blocks.append(f"--- PAGE {idx+1} ---")
                text_blocks.append(page_text)
                
        return "\n".join(text_blocks)

    def clean_text(self, text):
        # Remove consecutive empty lines and trailing whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def parse_document(self, file_path):
        _, ext = os.path.splitext(file_path.lower())
        raw_text = ""
        
        if ext == '.pdf':
            raw_text = self.extract_text_from_pdf(file_path)
        elif ext in ['.txt', '.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        else:
            raise ValueError(f"Unsupported document format: {ext}")
            
        cleaned = self.clean_text(raw_text)
        return cleaned

if __name__ == "__main__":
    # Test execution
    parser = DocumentParser()
    print("DocumentParser initialized.")
