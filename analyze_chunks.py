import json

with open(r'c:\Users\neeha\Documents\rag\logs\chunks\Placement_RAG_Dataset_Enhanced.jsonl') as f:
    chunks = [json.loads(line) for line in f]

print(f'Total chunks: {len(chunks)}')
print()

# Section distribution
sections = {}
for c in chunks:
    sec = c.get('metadata', {}).get('section', 'N/A')
    sections[sec] = sections.get(sec, 0) + 1
print('Section distribution:')
for k, v in sorted(sections.items()):
    print(f'  {k}: {v}')
print()

# Find Google-specific chunks
google_chunks = [c for c in chunks if 'google' in c.get('text', '').lower()]
print(f'Google-related chunks: {len(google_chunks)}')
print()

for c in google_chunks:
    print('--- CHUNK ---')
    meta = c.get('metadata', {})
    print(f'  section: {meta.get("section", "N/A")}')
    print(f'  content_type: {meta.get("content_type", "N/A")}')
    print(f'  company: {meta.get("company", "N/A")}')
    print(f'  sde: {meta.get("sde", "N/A")}')
    print(f'  analyst: {meta.get("analyst", "N/A")}')
    print(f'  officer: {meta.get("officer", "N/A")}')
    print(f'  intern: {meta.get("intern", "N/A")}')
    print(f'  text: {c.get("text", "")[:400]}')
    print()

# Show all hiring chunks
print('='*60)
print('ALL HIRING SECTION CHUNKS:')
hiring_chunks = [c for c in chunks if c.get('metadata', {}).get('section', '') == 'hiring']
print(f'Total hiring chunks: {len(hiring_chunks)}')
for c in hiring_chunks:
    meta = c.get('metadata', {})
    print(f'  company={meta.get("company","?")} sde={meta.get("sde","?")} analyst={meta.get("analyst","?")} officer={meta.get("officer","?")} intern={meta.get("intern","?")} total={meta.get("total","?")}')
    print(f'    text: {c.get("text","")[:200]}')
