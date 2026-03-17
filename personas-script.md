# Synthetic Persona Script: Installation & Usage Guide

This script automates the synthetic persona generation process covered in the main article. Instead of manually building persona cards from your data sources, the script does the heavy lifting: it syn[...]

---

## What The Script Generates

You'll get 2-3 persona cards (depending on data volume) that include:

- **Persona name and confidence score:** Based on the amount of supporting data
- **Complete 5-field persona card:** Job-to-be-done, constraints, success metric, decision criteria, and vocabulary
- **Trackable prompts:** 15-30 prompts per persona across low, medium, and high intent levels
- **Evidence tracing:** Documentation of which data sources support each persona field
- **Multiple output formats:** Markdown (for LLMs), JSON (for programmatic use), HTML (for browser viewing)

The script solves the cold-start problem by turning your existing behavioral data into actionable personas without waiting weeks for traditional research.

---

## Setup Instructions

### Step 1: Access Google Colab

- Navigate to `colab.research.google.com`
- Create a new notebook or open an existing one
- Click **Code** to create a new cell
- Copy-paste the entire script into the cell (approximately 2,000 lines)

### Step 2: Run The Script

Two options:

- Scroll to the bottom and click the play button on the left side of the cell
- Or use **Run All** from the **Runtime** menu

The script loads in under a minute and will prompt you for data files.

**Optional:** If you want the script to look a little nicer and come with a Wordcloud, add a code cell before the main script and copy/paste this:

```bash
!pip install tqdm scikit-learn matplotlib wordcloud
```

### Step 3: Upload Your Data
The script accepts three types of input:

#### Required:

- Google Search Console CSV export: Use the last 28 days, filtered by your target country. Export query data using this regex filter to capture question-type queries: (?i)^(who|what|why|how|when|whe[...]
- 
**Optional:**  (but recommended for richer personas):

- Survey data CSV: Newsletter surveys, customer feedback forms, onboarding questionnaires
- Transcript files: Customer conversations, support calls, sales calls, user interviews (TXT format)
The more data sources you provide, the higher the confidence scores for each persona field.

### Step 4: Filter Your Voice (Optional)
If you're uploading transcripts that include your own voice (customer calls, interviews), the script will prompt you to enter your name. This filters out your input so the personas reflect pure custom[...]

You can enter multiple names if several team members appear in transcripts, or skip this step if you're only uploading one-sided data like support tickets.

### Step 5: Process & Download
The script runs through approximately 10 processing steps:

- Data ingestion and cleaning
- Pattern extraction across sources
- Vocabulary analysis
- Job-to-be-done identification
- Constraint mapping
- Success metric synthesis
- Decision criteria extraction
- Persona clustering
- Prompt generation
- Evidence linking

After processing completes (typically 2-5 minutes), you'll see a summary and download options.

### Available downloads:

- Persona card Markdown files (for uploading to ChatGPT/Claude/Gemini)
- Persona card JSON files (for programmatic use)
- Persona card HTML files (for viewing in browser)
- Evidence file (traces each persona field back to source data)
- Run summary (documents what the script processed)
### Download options:

- Type specific numbers separated by commas (e.g., "1, 3, 6")
- Type "all" to download everything
- Type "skip" to just view the summary in Colab
For most use cases, download the Markdown file for your primary persona, plus the HTML files for all personas (for reference), plus the evidence file (for validation).

### Step 6: Generate Trackable Prompts
Once you have your persona cards:

Open the HTML file in your browser to review the complete persona profile
Copy the "How to Generate Trackable Prompts" section from the persona card
Open your LLM of choice (ChatGPT, Claude, or Gemini)
Upload the Markdown file for that persona
Paste the prompt generation instructions
The LLM will output 15-30 prompts organized by intent level:
Low intent: Exploration and problem-space learning
Medium intent: Evaluation and solution comparison
High intent: Final decision and proof point validation
These prompts reflect how that specific persona segment would actually search, using their natural vocabulary and shaped by their real constraints.

Using Your Personas For Prompt Tracking
Feed these prompts into your AI search tracking tool of choice, organized by:

Persona (Enterprise IT Buyer vs. Individual User)
Intent level (Low vs. Medium vs. High)
Topic/product category
Track citation presence, ranking position, and coverage gaps over time. Re-run the same prompts weekly or bi-weekly to measure how content changes affect visibility across different user segments.

### Maintenance & Updates
Synthetic personas stay accurate as long as your input data stays current. Set regeneration triggers:

Quarterly refresh using new GSC data and recent support tickets
Immediate refresh when major product changes ship
Immediate refresh when competitor dynamics shift (new entrant, positioning change)
Immediate refresh when you notice vocabulary shifts in customer conversations
Simply re-run the script with updated data files. The persona confidence scores will tell you if you have enough new data to warrant regeneration.

### Troubleshooting
If you encounter issues:

Reply to your premium subscriber email
Comment on the article
Most issues stem from CSV formatting (ensure UTF-8 encoding) or file size limits in Colab
Expected total time: 5-10 minutes for initial setup and generation, 2-3 minutes for subsequent regenerations.
