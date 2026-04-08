import os

def update_impl_files():
    replacements = [
        ("mistralai/Mistral-7B-Instruct-v0.2", "microsoft/Phi-3-mini-4k-instruct"),
        ("Mistral-7B-Instruct-v0.2", "Phi-3-mini-4k-instruct"),
        ("Mistral-7B", "Phi-3-Mini"),
        ("MISTRAL-7B", "PHI-3-MINI"),
    ]
    
    for filename in os.listdir('.'):
        if filename.startswith('impl_part') and filename.endswith('.md'):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            original = content
            for old, new in replacements:
                content = content.replace(old, new)
            if content != original:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Updated {filename}")
                
if __name__ == '__main__':
    update_impl_files()
