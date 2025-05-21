# Based News

This Python script is designed to fetch the latest Google News RSS headlines for a user-defined list of topics and update a Netlify webpage every hour. It uses Google Gemini to prioritize headlines based on the user's preferences.

---

## How it works

* Reads your preferences from a [configuration file](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing)
* Retrieves the latest headlines from Google News RSS for each topic
* Filters out banned keywords and already-seen headlines using a local history file
* Prioritizes headlines using Google Gemini based on topic and keyword weights
* Updates a styled `index.html` file for Netlify (or any static host)
* Designed to run hourly using `cron`

---

## Directory Structure

```plaintext
based-news/
├── based_news.py         # Main script
├── requirements.txt       # Python dependencies
├── history.json           # Stores previously posted headlines
├── public/                # Webpage directory
│   ├── index.html         # Main page, does not change
│   ├── digest.html        # Gets updated hourly
│   └── summary.html       # Gets updated biweekly
├── .env                   # Contains Gemini API key (excluded from version control)
└── logs/                  # Log directory (excluded from version control)
    └── based_news.log     # Cron and runtime logs
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/brayvid/based-news.git
cd based-news
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

Make a copy of [this Google Sheet](https://docs.google.com/spreadsheets/d/1OjpsQEnrNwcXEWYuPskGRA5Jf-U8e_x0x3j2CKJualg/edit?usp=sharing), publish each tab as CSV, and update the relevant URLs in `based_news.py`.

#### Topics (scored 1-5)

```
Topic,Weight
Technology,5
Global Health,4
...
```

#### Keywords (scored 1-5)

```
Keyword,Weight
nuclear,5
emergency,5
...
```

#### Overrides

```
Override,Action
buzzfeed,ban
celebrity,demote
...
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
python3 based_news.py
```

To run every hour:

```bash
crontab -e
```

Example crontab entry:

```cron
0 * * * * cd /path/to/based_news && /usr/bin/env python3 based_news.py >> /path/to/based_news/logs/based_news.log 2>&1
```

---

## Lockfile Notice

If interrupted, the script may leave behind a lockfile (`based_news.lock`). Remove it manually if needed:

```bash
rm based_news.lock
```

---

## Logging

Logs are stored in `logs/based_news.log` for monitoring and debugging.