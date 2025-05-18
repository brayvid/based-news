# BasedNews

This Python script fetches the latest Google News RSS headlines for a user-specified set of topics and generates a static HTML summary page. It runs hourly, filters and prioritizes headlines using Gemini based on the user's preferences, and updates a Netlify-hosted page with the results.

---

## How it works

* Reads your preferences from a [configuration spreadsheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing)
* Retrieves the latest headlines from Google News RSS for each topic
* Filters out banned keywords and already-seen headlines using a local history file
* Scores headlines using Google Gemini based on topic and keyword weights
* Outputs a clean, static `index.html` file for Netlify (or any static host)
* Designed to run hourly using `cron`

---

## Directory Structure

```plaintext
newspagebot/
├── newspagebot.py         # Main script
├── requirements.txt       # Python dependencies
├── history.json           # Stores previously seen headlines
├── index.html             # Static output file
├── .env                   # Contains Gemini API key (excluded from version control)
├── logs/                  # Log directory (excluded from version control)
│   └── newspagebot.log    # Cron and runtime logs
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/newspagebot.git
cd newspagebot
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

Or manually:

```bash
pip3 install nltk requests python-dotenv google-generativeai
```

### 3. Set up environment

Create a `.env` file containing your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key
```

[Get a Gemini API key](https://ai.google.dev/gemini-api/docs/api-key).

### 4. Configure preferences

Make a copy of [this Google Sheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing), publish each tab as CSV, and update the relevant URLs in `newspagebot.py`.

#### Topics (scored 1-5)

```
Topic,Weight
Technology,5
Global Health,4
```

#### Keywords (scored 1-5)

```
Keyword,Weight
breakthrough,5
emergency,4
```

#### Overrides

```
Override,Action
buzzfeed,ban
opinion,demote
```

#### Parameters

| Parameter                | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `MAX_ARTICLE_HOURS`      | Max age of news items in hours                   |
| `MAX_TOPICS`             | Max number of topics to show                     |
| `MAX_ARTICLES_PER_TOPIC` | Max number of articles per topic                 |
| `DEMOTE_FACTOR`          | 0–1 importance multiplier for 'demote' overrides |
| `TIMEZONE`               | Timezone string, e.g. `America/New_York`         |

---

## Running the Script

```bash
python3 newspagebot.py
```

To run every hour:

```bash
crontab -e
```

Example crontab entry:

```cron
0 * * * * cd /path/to/newspagebot && /usr/bin/env python3 newspagebot.py >> /path/to/newspagebot/logs/newspagebot.log 2>&1
```

---

## Output

The script overwrites `index.html` with a summary page that can be deployed to any static host (e.g., Netlify). Use Netlify CLI or auto-deploy from Git for updates.

---

## Lockfile Notice

If interrupted, the script may leave behind a lockfile (`newspagebot.lock`). Remove it manually if needed:

```bash
rm newspagebot.lock
```

---

## Logging

Logs are stored in `logs/based_news.log` for monitoring and debugging.