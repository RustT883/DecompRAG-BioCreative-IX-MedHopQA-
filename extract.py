import re
import json
import bz2
from urllib.parse import quote
import mwparserfromhell
import mwxml
from tqdm import tqdm
import os

# ===== CONFIGURATION =====
UNWANTED_SECTIONS = {
    'references', 'notes', 'citations', 'sources',
    'external links', 'bibliography',
    'see also', 'footnotes', 'works cited'
}

WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"
CHECKPOINT_INTERVAL = 100000  # Save progress every 100,000 articles
CHECKPOINT_FILE = "checkpoint.json"

def load_health_categories(file_path: str) -> set[str]:
    """Load health categories from file, one per line, with case-insensitive matching."""
    categories = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Only process non-empty lines
                # Add the normalized version (lowercase and stripped)
                normalized = line.lower().strip()
                if normalized:  # Make sure we don't add empty strings
                    categories.add(normalized)
    return categories

def get_wikipedia_url(title: str) -> str:
    """Generate Wikipedia URL from title."""
    return WIKIPEDIA_BASE_URL + quote(title.replace(" ", "_"))

def clean_wikitext(wikitext: str) -> str:
    """Optimized wikitext cleaning function."""
    if not wikitext:
        return ""

    try:
        # First parse with mwparserfromhell to handle templates and tags properly
        parsed = mwparserfromhell.parse(wikitext)
        
        # Remove specific unwanted nodes
        nodes_to_remove = []
        for node in parsed.nodes:
            if isinstance(node, mwparserfromhell.nodes.template.Template):
                # Only remove infoboxes and similar templates, not all templates
                if node.name.lower().strip().startswith(('infobox', 'cite', 'reflist')):
                    nodes_to_remove.append(node)
            elif isinstance(node, mwparserfromhell.nodes.tag.Tag) and node.tag.lower() == 'ref':
                nodes_to_remove.append(node)
            elif isinstance(node, mwparserfromhell.nodes.wikilink.Wikilink):
                if node.title.lower().startswith(('file:', 'image:')):
                    nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            parsed.remove(node)
        
        # Convert to text while preserving section structure
        text = parsed.strip_code()
        
        # Clean up remaining artifacts
        text = re.sub(r'\[\[([^|\]]*?\|)?([^\]]*?)\]\]', r'\2', text)  # Simplify links
        text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove any remaining templates
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'(\s\.)+', '.', text)  # Fix spaced dots
        
        # Remove references section if present
        lines = []
        skip = False
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.startswith('=='):
                section_match = re.match(r'=+\s*(.*?)\s*=+', line)
                if section_match:
                    current_section = section_match.group(1).lower()
                    skip = current_section in UNWANTED_SECTIONS
                    continue
            
            if not skip:
                lines.append(line)
        
        return '\n'.join(lines).strip()
    
    except Exception as e:
        # Fallback to simpler cleaning if parsing fails
        text = re.sub(r'\{\{(Infobox|infobox)[^\}]*?\}\}', '', wikitext, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\[\[(File|Image):.*?\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\[\[([^|\]]*?\|)?([^\]]*?)\]\]', r'\2', text)
        text = re.sub(r'\{\{.*?\}\}', '', text)
        text = re.sub(r'\[[^\]]+\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

def save_checkpoint(data):
    """Save checkpoint data to disk."""
    temp_file = f"{CHECKPOINT_FILE}.tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    os.rename(temp_file, CHECKPOINT_FILE)

def load_checkpoint():
    """Load checkpoint data from disk if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def process_dump(dump_path: str, output_file: str, health_categories: set[str]):
    """Process Wikipedia dump with better category matching."""
    checkpoint = load_checkpoint()
    if checkpoint:
        processed_count = checkpoint['processed_count']
        health_articles_found = checkpoint['health_articles_found']
        print(f"Resuming from checkpoint - previously processed {processed_count} articles")
    else:
        processed_count = 0
        health_articles_found = 0

    output_dir = os.path.dirname(output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    with (bz2.open(dump_path) as dump_file,
          open(output_file, 'a' if checkpoint else 'w', encoding='utf-8') as out_file):
        
        dump = mwxml.Dump.from_file(dump_file)
        
        for page in tqdm(dump, initial=processed_count, desc="Processing articles"):
            processed_count += 1
            if not isinstance(page, mwxml.Page):
                continue

            try:
                revision = next(iter(page))
                text = revision.text or ""
                
                # Improved category extraction
                categories = set()
                for match in re.finditer(r"\[\[Category:(.*?)(?:\|.*?)?\]\]", text, re.IGNORECASE):
                    category = match.group(1).strip()
                    if category:  # Only add non-empty categories
                        categories.add(category.lower())
                
                # Debug output for first few articles
                if processed_count < 10:
                    print(f"\nDebug - Article: {page.title}")
                    print(f"Categories found: {categories}")
                    print(f"Health categories sample: {list(health_categories)[:5]}")
                
                # Check for any overlap between article categories and health categories
                if categories & health_categories:
                    clean_text = clean_wikitext(text)
                    if len(clean_text.split()) >= 50:
                        json.dump({
                            "title": page.title,
                            "url": get_wikipedia_url(page.title),
                            "text": clean_text,
                            "categories": list(categories),
                        }, out_file, ensure_ascii=False)
                        out_file.write("\n")
                        health_articles_found += 1
                        
                        # Debug output for found articles
                        if health_articles_found < 5:
                            print(f"\nFound medical article: {page.title}")
                            print(f"Matching categories: {categories & health_categories}")

                if processed_count % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint({
                        'processed_count': processed_count,
                        'health_articles_found': health_articles_found
                    })
                    print(f"\nCheckpoint saved - Processed: {processed_count}, Found: {health_articles_found}")
                    out_file.flush()

            except Exception as e:
                continue

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    return health_articles_found

def main():
    WIKIPEDIA_DUMP_PATH = "enwiki-20250420-pages-articles-multistream.xml.bz2"
    HEALTH_CATEGORIES_FILE = "medical_categories.txt"
    OUTPUT_FILE = "medical_articles.jsonl"

    print(f"Loading health categories from {HEALTH_CATEGORIES_FILE}...")
    health_categories = load_health_categories(HEALTH_CATEGORIES_FILE)
    
    # Print info about loaded health categories
    print(f"Loaded {len(health_categories)} unique health categories (normalized to lowercase).")
    print("Sample of first 5 categories:")
    print("\n".join(sorted(health_categories)[:5]))

    print(f"\nProcessing Wikipedia dump to {OUTPUT_FILE}...")
    total_health_articles = process_dump(WIKIPEDIA_DUMP_PATH, OUTPUT_FILE, health_categories)

    print(f"\nDone! Found {total_health_articles} health articles in total.")
    print(f"All articles saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
