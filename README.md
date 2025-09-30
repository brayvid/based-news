
# Based News Reader

This project is a fully automated news digest generator. It fetches headlines from Google News, uses Google's Gemini AI to intelligently select and rank them based on user preferences, and generates a high-performance, statically-served website. The entire process is designed to be run on an automated schedule (e.g., hourly via `cron` or a GitHub Action).

View the live demo: **[news.blakerayvid.com](https://news.blakerayvid.com)**

---

## Key Features & How It Works

*   **Configurable via Google Sheets:** Easily manage topics, keywords, banned terms, and script parameters by editing a [Google Sheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing), which is then fetched as CSV.
*   **Intelligent Curation with Gemini:** A carefully engineered, multi-step prompt instructs the Gemini AI to:
    *   Perform aggressive cross-topic deduplication to find the single best headline for each news event.
    *   Filter out low-quality content, clickbait, ads, and sensationalism.
    *   Strictly adhere to banned and demoted keywords.
    *   Prioritize headlines based on user-defined weights for topics and keywords.
    *   Avoid selecting headlines that have appeared in recent historical digests.
*   **High-Performance Static Site Generation:**
    *   The latest news digest is injected directly into `index.html` during the build process. This eliminates client-side fetching for the initial view, resulting in excellent PageSpeed scores and a fast user experience.
    *   The front end is pure, dependency-free HTML, CSS, and vanilla JavaScript.
*   **Historical Digest Slider:**
    *   The website features a swipeable/clickable slider that allows users to browse through previous digests.
    *   Older digests are lazy-loaded on demand, keeping the initial page load light.
*   **Automated Deployment:** Includes logic to automatically commit and push the updated website files to a Git repository, making it perfect for CI/CD workflows like GitHub Actions.

## Tech Stack

*   **Backend:** Python 3
*   **AI:** Google Gemini API (`gemini-2.5-flash`)
*   **Data/NLP:** `requests`, `nltk`
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript
*   **Configuration:** Google Sheets (published as CSV)
*   **Automation:** `cron` or GitHub Actions
*   **Hosting:** Any static hosting provider (e.g., GitHub Pages, Netlify, Vercel)

---

## Directory Structure

```plaintext
based-news/
├── digest.py                   # Main Python script for generating the digest
├── requirements.txt            # Python dependencies
├── .env                        # Stores API keys (excluded from version control)
├── .github/workflows/          # (Optional) For GitHub Actions automation
│   └── digest.yml
└── public/                     # Root directory for the static website
    ├── index.html              # The final, generated main page with the latest digest embedded
    ├── index.template.html     # The template used by the script to generate index.html
    ├── digest-manifest.json    # A list of historical digests for the front-end slider
    └── digests/                # Folder containing all historical digest HTML files
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/brayvid/based-news.git
cd based-news
```

### 2. Install Dependencies

It's highly recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
The script will attempt to download necessary `nltk` data on first run.

### 3. Set Up Environment Variables

Create a `.env` file in the project's root directory. At a minimum, it needs your Gemini API key. If you plan to use automated Git pushes, add your GitHub credentials.

```env
# Required for news generation
GEMINI_API_KEY="your_gemini_api_key_here"

# Required for automated Git push from script or GitHub Actions
GITHUB_TOKEN="your_github_personal_access_token"
GITHUB_REPOSITORY="your_username/based-news"
GITHUB_USER="Your GitHub Username"
GITHUB_EMAIL="your-email@example.com"
```

*   [Get a Gemini API key here](https://ai.google.dev/gemini-api/docs/api-key).
*   For `GITHUB_TOKEN`, create a [Personal Access Token (classic)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) with the `repo` scope.

### 4. Configure Your Preferences

The script is configured via public Google Sheets.

1.  **Make a copy** of the template [Google Sheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing).
2.  For **each tab** (`Topics`, `Keywords`, `Overrides`, `Config`):
    *   Go to `File > Share > Publish to web`.
    *   Select the specific sheet (tab) and `Comma-separated values (.csv)`.
    *   Ensure "Automatically republish when changes are made" is checked.
    *   Click `Publish` and copy the generated URL.
3.  Update the corresponding URLs at the top of the `digest.py` script.

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
| `GEMINI_MODEL_NAME`      | The specific Gemini model to use for generation.                            | `gemini-2.5-flash-lite` |
| `MAX_OUTPUT_TOKENS_GEMINI` | Max tokens for Gemini's response.                                           | `8192`             |
| `TEMPERATURE_GEMINI`     | Creativity level for Gemini (0.0-1.0). Lower is more deterministic.         | `0.3`              |
| `TOP_P_GEMINI`           | Nucleus sampling parameter for Gemini.                                      | `0.9`              |
| `TOP_K_GEMINI`           | Top-K sampling parameter for Gemini.                                        | `40`               |

---

## Running the Script

### Manual Run

Ensure your virtual environment is active.
```bash
python3 digest.py
```
This will generate/update all necessary files in the `public/` directory.

### Automated Run (GitHub Actions)

This is the recommended method for a "set it and forget it" setup.

1.  In your GitHub repository, go to `Settings > Secrets and variables > Actions`.
2.  Create repository secrets for each of the environment variables in your `.env` file (`GEMINI_API_KEY`, `GITHUB_TOKEN`, etc.).
3.  Create the file `.github/workflows/digest.yml` with the following content:

```yaml
name: Generate Daily News Digest

on:
  schedule:
    # Runs every hour at the top of the hour
    - cron: '0 * * * *'
  workflow_dispatch: # Allows you to run this workflow manually from the Actions tab

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11' # Or your preferred version

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Digest Script
        run: python digest.py
        env:
          # These secrets are configured in your repository settings
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ secrets.GITHUB_REPOSITORY }}
          GITHUB_USER: ${{ secrets.GITHUB_USER }}
          GITHUB_EMAIL: ${{ secrets.GITHUB_EMAIL }}

```
This action will check out your code, run the Python script, and the script's internal Git logic will commit and push the updated `public/` directory back to the repository. If you host your site with GitHub Pages, it will be updated automatically.

---

<br>

![](images/example.png)


