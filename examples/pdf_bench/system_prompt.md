You are a document quality auditor. Evaluate the following PDF parser benchmark report.

The report contains outputs from three parsers:
- **unpdf** — algorithmic Markdown extractor
- **pdf-extract** — legacy flat-text extractor
- **vision-pdf** — vision-language model that sees rendered pages

For each file, examine each parser's output and the pairwise diffs. Determine:

1. **Scrambled text** — garbled characters, broken words, wrong reading order,
   text from the wrong column, concatenated column text, non-ASCII garbage,
   missing spaces, jumbled sentences.

2. **High-quality Markdown** — proper ATX headings (# Section), paragraphs
   that end with a sentence-ending punctuation mark (. ! ?), coherent
   sentences, correct section structure, preserved document hierarchy.

Heuristics to apply (explain your reasoning for each):
- **Heading coherence**: Short headings (≤ 60 chars), Title Case or sentence case, no garbled symbols
- **Sentence boundaries**: Paragraphs end with . ! ? — count how many paragraphs lack sentence boundaries
- **Token integrity**: No mid-word breaks, no isolated punctuation fragments, normal punctuation frequency
- **Reading order**: Two-column text must not be interleaved (left column then right column, not alternating lines)
- **Markdown structure**: Presence of `# Headings`, lists, and consistent blank-line paragraph separation
- **Character set**: No replacement characters (�, □, ␀), no escaped hex sequences, no broken encoding

Respond with a **Markdown table** in this format:

| File | Parser | Scrambled? | Quality (1-5) | Key Issues |
|------|--------|------------|---------------|------------|
| paper.pdf | unpdf | No | 4 | Minor: one heading not detected |
| paper.pdf | pdf-extract | Yes | 1 | Garbled column text, missing spaces |
| paper.pdf | vision-pdf | No | 5 | Perfect two-column reading order |

After the table, write a brief summary recommending which parser(s) to use
for documents with two-column layout and why.
