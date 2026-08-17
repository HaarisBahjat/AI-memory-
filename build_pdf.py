import os
import sys
import subprocess
import re
import markdown

def compile_md_to_pdf(md_filename: str, pdf_filename: str, title: str):
    if not os.path.exists(md_filename):
        raise FileNotFoundError(f"Input markdown file not found: {md_filename}")

    temp_html = f"temp_{os.path.splitext(md_filename)[0]}.html"

    with open(md_filename, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown to html with extended features
    html_body = markdown.markdown(
        md_text,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
            "nl2br"
        ]
    )

    # Convert mermaid code blocks (<pre><code class="language-mermaid">...</code></pre>) to (<div class="mermaid">...</div>)
    html_body = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<div class="mermaid">\1</div>',
        html_body,
        flags=re.DOTALL
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  @page {{
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
    @bottom-right {{
      content: counter(page);
      font-size: 8pt;
      color: #64748b;
      font-family: 'Inter', sans-serif;
    }}
  }}

  * {{
    box-sizing: border-box;
  }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    line-height: 1.55;
    font-size: 9.5pt;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
  }}

  h1 {{
    font-size: 18pt;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
  }}

  h2 {{
    font-size: 13pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 22px;
    margin-bottom: 10px;
    page-break-after: avoid;
    letter-spacing: -0.01em;
  }}

  h3 {{
    font-size: 11pt;
    font-weight: 600;
    color: #1e293b;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}

  h4 {{
    font-size: 10pt;
    font-weight: 600;
    color: #2563eb;
    margin-top: 14px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 0;
    margin-bottom: 8px;
  }}

  ul, ol {{
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 20px;
  }}

  li {{
    margin-bottom: 3px;
  }}

  code {{
    font-family: 'JetBrains Mono', "Cascadia Code", Consolas, monospace;
    font-size: 8.5pt;
    background-color: #f1f5f9;
    padding: 1.5px 4.5px;
    border-radius: 4px;
    color: #0f172a;
    border: 1px solid #e2e8f0;
  }}

  pre {{
    background-color: #0f172a;
    color: #f8fafc;
    padding: 10px 14px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 8pt;
    page-break-inside: avoid;
    line-height: 1.45;
    margin: 8px 0 12px 0;
  }}

  pre code {{
    background-color: transparent;
    color: inherit;
    padding: 0;
    border: none;
    font-size: 8pt;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }}

  th, td {{
    border: 1px solid #cbd5e1;
    padding: 6px 9px;
    text-align: left;
    vertical-align: top;
  }}

  th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
  }}

  tr:nth-child(even) {{
    background-color: #f8fafc;
  }}

  blockquote {{
    border-left: 3.5px solid #2563eb;
    margin: 10px 0;
    padding: 6px 12px;
    background-color: #eff6ff;
    color: #1e3a8a;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
  }}

  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
  }}

  a {{
    color: #2563eb;
    text-decoration: none;
    font-weight: 500;
  }}

  .mermaid {{
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
    background: #ffffff;
    padding: 10px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
  }}

  .mermaid svg {{
    max-width: 100%;
    height: auto;
  }}
</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'neutral',
    securityLevel: 'loose',
    flowchart: {{
      curve: 'basis',
      useMaxWidth: true,
      htmlLabels: true
    }}
  }});
</script>
</head>
<body>
{html_body}
</body>
</html>"""

    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    
    edge_path = next((p for p in edge_candidates if os.path.exists(p)), None)
    if not edge_path:
        raise FileNotFoundError("Could not find Microsoft Edge or Google Chrome executable.")

    abs_html = os.path.abspath(temp_html)
    abs_pdf = os.path.abspath(pdf_filename)

    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=9000",
        f"--print-to-pdf={abs_pdf}",
        abs_html
    ]

    print(f"Exporting '{md_filename}' to '{pdf_filename}' via Headless Browser...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Error output:", res.stderr)
        raise RuntimeError(f"Edge exited with return code {res.returncode}")

    # Clean up temporary html file
    if os.path.exists(temp_html):
        try:
            os.remove(temp_html)
        except Exception:
            pass

    if os.path.exists(abs_pdf):
        size_kb = round(os.path.getsize(abs_pdf) / 1024, 1)
        print(f"Successfully generated: {abs_pdf} ({size_kb} KB)")
    else:
        raise RuntimeError(f"PDF creation failed: {abs_pdf}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DEPENDENCY_GRAPH.md"
    
    if target == "all":
        compile_md_to_pdf("CODEBASE_AUDIT.md", "CODEBASE_AUDIT.pdf", "AI Wellness LMS — System Architecture & Codebase Audit")
        compile_md_to_pdf("DEPENDENCY_GRAPH.md", "DEPENDENCY_GRAPH.pdf", "AI Wellness LMS — Dependency Graph & Structural Analysis")
    elif target.endswith(".md"):
        pdf_out = os.path.splitext(target)[0] + ".pdf"
        compile_md_to_pdf(target, pdf_out, f"AI Wellness LMS — {os.path.splitext(target)[0].replace('_', ' ').title()}")
    else:
        print(f"Unknown target: {target}")
