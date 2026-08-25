content = open('components/Sidebar.tsx', encoding='utf-8').read()
content = content.replace('"use client""', '"use client"', 1)
open('components/Sidebar.tsx', 'w', encoding='utf-8', newline='
').write(content)
print('Sidebar:', repr(open('components/Sidebar.tsx', encoding='utf-8').read()[:25]))
raw = open('lib/api.ts', encoding='utf-8').read()
print('api.ts has axios:', 'import axios' in raw)
print('api.ts has authApi:', 'authApi' in raw)