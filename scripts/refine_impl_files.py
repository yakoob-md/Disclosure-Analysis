import os
import re

def refine_file(filepath, replacements, additions=None):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} - not found.")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"Applied replacement in {filepath}")
            
    if additions:
        content = additions + "\n\n" + content
        print(f"Applied prepend to {filepath}")
        
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        print(f"No changes made to {filepath}")

def main():
    # PART 4: Phase 6, 7, 8 (DL Models - DeBERTa fixes)
    p4 = 'impl_part4_phase6_7_8.md'
    refine_file(p4, [
        # Fixing token_type_ids in dataset
        ("item = {\n            'input_ids':      self.encodings['input_ids'][i],", 
         "# AMENDMENT 4A: DeBERTa does not use token_type_ids.\n        item = {\n            'input_ids':      self.encodings['input_ids'][i],"),
        # Adding gradient checkpointing
        ("train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)",
         "train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)\n    \n    # AMENDMENT 4B: RTX 2050 VRAM optimization\n    model.deberta.gradient_checkpointing_enable()")
    ])
    
    # PART 5: Phase 9, 10
    p5 = 'impl_part5_phase9_and_phase10.md'
    refine_file(p5, [
        ("=== IMPLEMENTATION PROMPT: PHASE 9 — MISTRAL-7B LLM BASELINE ===",
         "=== IMPLEMENTATION PROMPT: PHASE 9 — MISTRAL-7B LLM BASELINE ===\n# AMENDMENT: Use Entity Overlap instead of BERTScore for LLM evaluation faithfulness where applicable.")
    ])
    
    # PART 6: Phase 11
    p6 = 'impl_part6_phase11_and_index.md'
    refine_file(p6, [
        ("BERTScore", "Entity Overlap Score (Amendment 5A)"),
        ("bert-score", "Entity Overlap")
    ])

    # Prepend a meta-prompt to all files to ensure Gemini 1.5 Flash uses validation scripts
    meta_prompt = "> **GEMINI 1.5 FLASH CRITICAL INSTRUCTION**: The user has mandated stepping validation. For every phase you execute from this document, you MUST ALSO execute the corresponding `scripts/phaseX_results.py` validation script immediately after and verify the outputs are correct before proceeding to the next phase."
    
    for filename in os.listdir('.'):
        if filename.startswith('impl_part') and filename.endswith('.md'):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            if "GEMINI 1.5 FLASH CRITICAL INSTRUCTION" not in content:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(meta_prompt + "\n\n" + content)
                    print(f"Injected meta-prompt into {filename}")

if __name__ == '__main__':
    main()
