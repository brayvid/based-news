# Based News Reader

This Python script fetches the latest Google News RSS headlines for a user-defined list of topics and updates an HTML site every hour. It uses Google Gemini, with a carefully refined prompting strategy, to prioritize headlines based on the user's preferences. Designed to be scheduled with `cron` on any Unix-based system.

View the demo here: **[news.blakerayvid.com](https://news.blakerayvid.com)**

---

## How it works

*   Reads your preferences from a [configuration file](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing) (Google Sheets published as CSV).
*   Retrieves the latest headlines from Google News RSS for each topic.
*   Filters out banned keywords and already-seen headlines using a local `history.json` file.
*   **Employs a detailed and refined prompt with Google Gemini to select and prioritize headlines based on user-defined topic/keyword weights, strict filtering criteria (deduplication, content type, quality), and specific instructions to ensure relevance and objectivity.**
*   Updates styled HTML files (`digest.html` hourly, `summary.html` daily) for Netlify (or any static host).
*   Designed to run hourly using `cron`.

---

## Gemini Prompting Strategy

Achieving precise compliance from Large Language Models (LLMs) like Gemini for specific tasks such as news curation requires careful prompt engineering. The prompt used in this project has been iteratively refined to:

*   **Emphasize Aggressive Deduplication:** Instructing Gemini to critically avoid multiple versions of the same core news event.
*   **Clarify Geographic Focus:** Allowing local news only if it has clear national/international implications.
*   **Strengthen Content Filtering:** Providing more explicit examples of what to reject (e.g., specific stock-picking advice, sensationalism, fluff) and what is acceptable (e.g., broad market trends).
*   **Improve Objectivity:** Prioritizing factual reporting over opinion pieces.
*   **Incorporate Chain-of-Thought (CoT) Cues:** Guiding the model's internal reasoning process to better adhere to complex instructions.

This ongoing refinement is crucial for improving the quality and relevance of the news digest.

---

## Directory Structure

```plaintext
based-news/
├── digest.py              # Main Python script for generating the digest
├── requirements.txt       # Python dependencies
├── history.json           # Stores previously posted headlines to avoid duplicates
├── public/                # Root directory for the static website
│   ├── index.html         # Main landing page (typically static)
│   ├── digest.html        # Dynamically updated hourly with the latest news digest
│   └── summary.html       # Dynamically updated weekly with a news summary
├── .env                   # Contains Gemini API key (excluded from version control)
└── logs/                  # Directory for log files (excluded from version control)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/brayvid/based-news.git
cd based-news
```

### 2. Install dependencies

It's recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Or manually:
```bash
pip install nltk requests python-dotenv google-generativeai
```
*(Note: `nltk` might require additional data download. The script attempts to handle this, but run `python -m nltk.downloader punkt` if you encounter issues.)*

### 3. Set up environment

Create a `.env` file in the `based-news` root directory containing your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key
```

[Get a Gemini API key here](https://ai.google.dev/gemini-api/docs/api-key).

### 4. Configure preferences

The script fetches its configuration from publicly published Google Sheets CSVs.

1.  **Make a copy** of the template [Google Sheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing).
2.  For **each tab** (`Topics`, `Keywords`, `Overrides`, `Parameters`):
    *   Go to `File > Share > Publish to web`.
    *   Select the specific sheet (tab).
    *   Choose `Comma-separated values (.csv)` as the format.
    *   Ensure "Automatically republish when changes are made" is checked.
    *   Click `Publish` and copy the generated URL.
3.  Update the corresponding CSV URLs at the top of the `digest.py` script:
    ```python
    # Configuration URLs from Google Sheets (Publish to Web > CSV)
    TOPICS_CSV_URL = "YOUR_PUBLISHED_TOPICS_CSV_URL"
    KEYWORDS_CSV_URL = "YOUR_PUBLISHED_KEYWORDS_CSV_URL"
    OVERRIDES_CSV_URL = "YOUR_PUBLISHED_OVERRIDES_CSV_URL"
    PARAMETERS_CSV_URL = "YOUR_PUBLISHED_PARAMETERS_CSV_URL"
    ```

#### Configuration Details:

**Topics Sheet:** Assign importance weights to news topics.
```
Topic,Weight
Technology,5
Global Health,4
Ukraine Crisis,5
Artificial Intelligence,4
...
```

**Keywords Sheet:** Assign importance weights to specific keywords found in headlines.
```
Keyword,Weight
nuclear,5
emergency,5
breakthrough,4
...
```

**Overrides Sheet:** Define actions for specific terms found in headlines.
```
Override,Action
buzzfeed,ban       # Headlines containing 'buzzfeed' will be rejected
celebrity,demote   # Headlines containing 'celebrity' will be deprioritized
listicle,ban
horoscope,ban
...
```

**Parameters Sheet:** Control script behavior.
| Parameter                | Description                                                                 | Example Value      |
| ------------------------ | --------------------------------------------------------------------------- | ------------------ |
| `MAX_ARTICLE_HOURS`      | Maximum age of news items (in hours) to be considered for the digest.       | `72`               |
| `MAX_TOPICS`             | Maximum number of topics to include in the digest.                          | `10`               |
| `MAX_ARTICLES_PER_TOPIC` | Maximum number of articles to show per selected topic.                      | `5`                |
| `DEMOTE_FACTOR`          | Multiplier (0.0 to 1.0) for the importance of headlines with 'demote' terms. The prompt translates this to a qualitative instruction for Gemini. | `0.2`              |
| `TIMEZONE`               | Timezone for date/time display in the digest (e.g., `America/New_York`).    | `America/New_York` |
| `GEMINI_MODEL_NAME`      | The specific Gemini model to use for generation.                            | `gemini-1.5-flash-latest` |
| `MAX_OUTPUT_TOKENS_GEMINI` | Max tokens for Gemini's response.                                           | `8192`             |
| `TEMPERATURE_GEMINI`     | Creativity level for Gemini (0.0-1.0). Lower is more deterministic.         | `0.3`              |
| `TOP_P_GEMINI`           | Nucleus sampling parameter for Gemini.                                      | `0.9`              |
| `TOP_K_GEMINI`           | Top-K sampling parameter for Gemini.                                        | `40`               |

---

## Running the Script

Ensure your virtual environment is active if you used one.

```bash
python3 digest.py
```

This will generate/update `public/digest.html` and `public/summary.html` (if it's the day for summary generation).

To schedule the script to run every hour using `cron`:

1.  Open your crontab for editing:
    ```bash
    crontab -e
    ```
2.  Add an entry similar to the following, adjusting paths as necessary:

    ```cron
    0 * * * * cd /full/path/to/based-news && /full/path/to/based-news/venv/bin/python3 /full/path/to/based-news/digest.py >> /full/path/to/based-news/logs/digest.log 2>&1
    ```
    *   Replace `/full/path/to/based-news` with the absolute path to your project directory.
    *   If not using a virtual environment, adjust the python interpreter path accordingly (e.g., `/usr/bin/env python3`).
    *   This example runs the script at the start of every hour and appends standard output and errors to a log file.

---

<br>

![](images/example.png)
